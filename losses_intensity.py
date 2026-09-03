"""
Place this file in the ROOT of the kla-restoration repo (next to train.py),
so it can be imported directly as `from losses_intensity import IntensityWeightedLoss`.

Motivation: the noise-distribution analysis on the full 4,785-pair dataset found
a strong correlation (0.864) between GT pixel intensity and local noise level --
i.e. the noise is multiplicative/speckle-dominant, not additive/Gaussian. The
original CombinedLoss (Charbonnier + SSIM) weights every pixel's error equally
regardless of local intensity, which mismatches this. This loss reweights the
Charbonnier term by inverse local GT intensity, so errors in naturally
noisier/brighter regions aren't over-penalized relative to genuinely fixable
errors elsewhere -- while keeping the SSIM term unchanged (SSIM's local
contrast normalization already partially handles this, so leave it as-is
and only change the pixel-fidelity term).

Self-contained: implements its own differentiable SSIM (standard 11x11
Gaussian window) rather than depending on the repo's internal src/losses.py,
so it can be dropped in without needing to know that file's exact internals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window_1d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_map(img1, img2, window_size=11, sigma=1.5):
    channel = img1.shape[1]
    device, dtype = img1.device, img1.dtype
    g1d = _gaussian_window_1d(window_size, sigma, device, dtype)
    window_2d = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return ssim_n / ssim_d


class IntensityWeightedLoss(nn.Module):
    def __init__(self, weight_ssim=0.2, intensity_eps=0.05, charbonnier_eps=1e-3):
        """
        weight_ssim: same role/scale as the original CombinedLoss's weight_ssim,
            so swapping loss types doesn't require re-tuning this value.
        intensity_eps: floor added to GT intensity before inverting, so
            near-zero-intensity (dark) regions don't get an unbounded weight.
            0.05 keeps the max per-pixel weight multiplier bounded to ~20x
            relative to a fully-bright (intensity=1.0) pixel.
        charbonnier_eps: standard Charbonnier smoothing epsilon, matches
            typical SR/denoising literature default.
        """
        super().__init__()
        self.weight_ssim = weight_ssim
        self.intensity_eps = intensity_eps
        self.charbonnier_eps = charbonnier_eps

    def forward(self, pred, gt):
        diff = pred - gt
        charbonnier = torch.sqrt(diff * diff + self.charbonnier_eps ** 2)

        # Inverse-intensity weighting: downweight loss in bright (naturally
        # noisier, per the confirmed corr=0.864 finding) regions, upweight
        # loss in dark regions where errors are more likely genuinely fixable
        # detail rather than irreducible speckle noise.
        weight = 1.0 / (gt.detach() + self.intensity_eps)
        weight = weight / weight.mean()  # renormalize so total loss scale ~matches plain Charbonnier
        weighted_charbonnier = (weight * charbonnier).mean()

        ssim_val = _ssim_map(pred, gt).mean()
        ssim_loss = 1.0 - ssim_val

        total = weighted_charbonnier + self.weight_ssim * ssim_loss
        parts = {
            'charbonnier_weighted': float(weighted_charbonnier.detach()),
            'ssim_loss': float(ssim_loss.detach()),
        }
        return total, parts
