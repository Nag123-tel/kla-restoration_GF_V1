"""
Run from the kla-restoration repo root in Colab.

Generates synthetic (NoisyLR, GT) pairs from real SEM GT images using the
repo's own src/degradations.py, then runs the SAME noise-vs-intensity
analysis we ran on the real data -- to check whether the synthetic
generator's default parameters (speckle_level ~ U(0.05,0.25), gaussian_sigma
~ U(0.01,0.08)) actually reproduce the measured real noise characteristic
(corr=0.864 between GT intensity and noise level, i.e. multiplicative/
speckle-dominant) before we trust it to generate extra training pairs.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.degradations import degrade

GT_DIR = '/content/local_training_data/GT'
N_SAMPLES = 500  # subset is enough for a stable correlation estimate

gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith('.npy'))[:N_SAMPLES]
print(f'Generating synthetic degraded pairs from {len(gt_files)} real GT images...')

gt_means, residual_stds = [], []

for f in gt_files:
    gt = np.load(os.path.join(GT_DIR, f)).astype(np.float32)
    gt_t = torch.from_numpy(gt).unsqueeze(0)  # (1, H, W) -- degrade() expects (C, H, W) or (B, C, H, W)

    synthetic_noisy = degrade(gt_t, scale=2)  # (1, H/2, W/2)
    synthetic_noisy = synthetic_noisy.squeeze(0).numpy()

    # Same residual estimate as the real-data analysis: compare against an
    # area-average downsample of GT (the "clean, downsampled" reference)
    gt_down = F.avg_pool2d(gt_t.unsqueeze(0), kernel_size=2, stride=2)[0, 0].numpy()
    residual = synthetic_noisy - gt_down

    gt_means.append(float(gt_down.mean()))
    residual_stds.append(float(residual.std()))

gt_means = np.array(gt_means)
residual_stds = np.array(residual_stds)

corr = np.corrcoef(gt_means, residual_stds)[0, 1]

print(f'\n=== Synthetic degradation pipeline noise characteristic ===')
print(f'Correlation between GT intensity and residual std: {corr:.3f}')
print(f'(Real data measured: 0.864)')
print(f'Mean residual std (synthetic): {residual_stds.mean():.4f}')
print(f'(Real data measured, global mean: ~0.10, ranging 0.086-0.104 by category)')

if abs(corr - 0.864) < 0.15:
    print('\n-> Synthetic noise characteristic is a reasonably close match to real data. '
          'Safe to use --synth_from_gt_dir for extra training pairs.')
else:
    print('\n-> Synthetic noise characteristic differs meaningfully from real data. '
          'Consider tuning speckle_level_range/gaussian_sigma_range in degrade() '
          'before relying on synthetic pairs, or use them cautiously (e.g. small mixing ratio).')
