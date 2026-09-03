"""
Loss functions for the restoration model.

- CharbonnierLoss: smooth L1-like loss, standard in SR/denoising literature
  (Restormer, SwinIR, MPRNet all use it) -- more robust to outliers than L2,
  sharper reconstructions than L1.
- SSIMLoss: structural similarity term, directly optimizes toward one of
  KLA's reported metrics.
- CombinedLoss: weighted sum, the common recipe for restoration training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM), single-scale, Gaussian window."""

    def __init__(self, window_size=11, sigma=1.5, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._make_window(window_size, sigma, channels))

    @staticmethod
    def _gaussian(window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _make_window(self, window_size, sigma, channels):
        g1d = self._gaussian(window_size, sigma).unsqueeze(1)
        g2d = g1d @ g1d.t()
        window = g2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        c = pred.shape[1]
        if c != self.channels:
            window = self._make_window(self.window_size, 1.5, c).to(pred.device)
        else:
            window = self.window.to(pred.device)

        pad = self.window_size // 2
        mu_p = F.conv2d(pred, window, padding=pad, groups=c)
        mu_t = F.conv2d(target, window, padding=pad, groups=c)

        mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

        sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """weight_charbonnier * Charbonnier + weight_ssim * (1 - SSIM)"""

    def __init__(self, channels=1, weight_charbonnier=1.0, weight_ssim=0.2):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(channels=channels)
        self.w_char = weight_charbonnier
        self.w_ssim = weight_ssim

    def forward(self, pred, target):
        pred_c = pred.clamp(0, 1)
        target_c = target.clamp(0, 1)
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred_c, target_c)
        total = self.w_char * l_char + self.w_ssim * l_ssim
        return total, {"charbonnier": l_char.item(), "ssim_loss": l_ssim.item()}
