"""
Synthetic degradation pipeline: speckle noise + additive Gaussian noise +
downsampling, applied in random order -- matching the KLA problem statement
("The three degradations may have been applied in any order").

Use this to generate EXTRA synthetic (NoisyLR, GT) training pairs from your
clean GT images, to augment whatever paired dataset KLA provides.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F


def add_speckle_noise(img: torch.Tensor, level: float) -> torch.Tensor:
    """Multiplicative speckle noise: img + img * N(0, level^2)."""
    noise = torch.randn_like(img) * level
    return img + img * noise


def add_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive Gaussian noise with the given std (in [0,1]-normalized units)."""
    return img + torch.randn_like(img) * sigma


def downsample(img: torch.Tensor, scale: int, mode: str = "area") -> torch.Tensor:
    """Downsample by `scale` via area averaging (mimics sensor binning /
    optical resolution loss better than naive nearest/bicubic downsampling)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
        out = F.interpolate(img, scale_factor=1.0 / scale, mode=mode)
        return out.squeeze(0)
    return F.interpolate(img, scale_factor=1.0 / scale, mode=mode)


def degrade(
    clean: torch.Tensor,
    scale: int = 2,
    speckle_level_range=(0.05, 0.25),
    gaussian_sigma_range=(0.01, 0.08),
    seed: int = None,
) -> torch.Tensor:
    """
    Apply speckle noise, additive Gaussian noise, and downsampling to a
    clean (C, H, W) or (B, C, H, W) image tensor in [0, 1], in a RANDOM
    order each call (matching the undisclosed order in the real dataset).

    Output values are intentionally NOT clipped to [0,1] -- this matches
    KLA's stated NoisyLR value range ("may extend slightly outside [0,1];
    this is intentional").
    """
    if seed is not None:
        rng_state = torch.get_rng_state()
        py_state = random.getstate()
        torch.manual_seed(seed)
        random.seed(seed)

    speckle_level = random.uniform(*speckle_level_range)
    gaussian_sigma = random.uniform(*gaussian_sigma_range)

    ops = [
        ("speckle", lambda t: add_speckle_noise(t, speckle_level)),
        ("gaussian", lambda t: add_gaussian_noise(t, gaussian_sigma)),
        ("downsample", lambda t: downsample(t, scale)),
    ]
    random.shuffle(ops)

    out = clean
    for name, fn in ops:
        out = fn(out)

    if seed is not None:
        torch.set_rng_state(rng_state)
        random.setstate(py_state)

    return out


def make_synthetic_pair(clean_img: torch.Tensor, scale: int = 2, seed: int = None):
    """Convenience wrapper: returns (noisy_lr, clean_gt) for one image."""
    noisy_lr = degrade(clean_img, scale=scale, seed=seed)
    return noisy_lr, clean_img
