"""
PSNR, SSIM, LPIPS metrics for validation and reporting, as required by the
KLA problem statement.

LPIPS requires the `lpips` package (pip install lpips) and downloads a
small pretrained AlexNet/VGG feature extractor on first use -- this needs
internet access once (fine in Colab/most training environments).
"""

import torch
import torch.nn.functional as F

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """pred, target: (B,C,H,W) or (C,H,W), values expected in [0, max_val]."""
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Single-scale SSIM, matches the standard formulation."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    c = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, c, pred.device)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=c)
    mu_t = F.conv2d(target, window, padding=pad, groups=c)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


class LPIPSMetric:
    """Wraps the `lpips` package. Call once, reuse across the validation loop
    (loading the feature network per-call is expensive)."""

    def __init__(self, net="alex", device="cpu"):
        if not _LPIPS_AVAILABLE:
            raise ImportError("pip install lpips  # required for LPIPS metric")
        self.model = lpips_lib.LPIPS(net=net).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """pred, target: (B,C,H,W) or (C,H,W) in [0,1]. LPIPS expects 3
        channels in [-1,1]; grayscale is repeated to 3 channels."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred = pred.clamp(0, 1) * 2 - 1
        target = target.clamp(0, 1) * 2 - 1
        return self.model(pred.to(self.device), target.to(self.device)).mean().item()


def compute_all_metrics(pred, target, lpips_metric=None):
    """Returns a dict with psnr, ssim, and (if lpips_metric provided) lpips."""
    out = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
    if lpips_metric is not None:
        out["lpips"] = lpips_metric(pred, target)
    return out
