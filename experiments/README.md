# Phase 2 Experiment Log

Final submitted checkpoint: `models/best.pt` (tiny NAFNetSR, trained with:
1.5x oversampling on sparse_particles_on_substrate + fiber_mesh categories,
random 90/180/270-degree rotation augmentation, and calibrated synthetic
degradation pairs generated from training-only GT images).

## Official metrics (evaluate.py)

**Internal stratified held-out val set (475 images, from the same 4,785-image
training pool, never trained on):**
PSNR: 23.561 dB | SSIM: 0.6167 | LPIPS: 0.3433
See `results/metrics_summary_internal_val.json`.

**External test set (297 images, genuinely separate data provided alongside
training data -- confirmed zero content overlap via hash comparison):**
PSNR: 23.624 dB | SSIM: 0.6311 | LPIPS: 0.3267
See `results/metrics_summary_external_test.json`.

The external test set scoring as well or slightly better than the internal
val set on every metric is a strong signal against overfitting to our own
validation split through iterative tuning this session.

Bicubic baseline (both sets, near-identical): PSNR ~20.45 dB | SSIM ~0.50.

See `results/per_category_eval.json` for the per-category (7-cluster)
breakdown on the internal val set.

## What was tried, in order
1. **Stratified train/val split** by visual content category (ResNet18
   features + K-means, 7 categories: wires/rods, fiber mesh, dense
   particle aggregates, porous/foam, faceted grain boundary, grainy/blurry
   particles, sparse particles on substrate) instead of a plain random
   split -- ensures validation reflects the true content mix. IMPORTANT:
   Phase 1's original leaderboard number (27.81 dB) was found to be
   inflated by a train/val leak; every number in this log is measured on
   a genuinely disjoint, verified-leak-free split.
2. **Noise distribution analysis**: confirmed multiplicative/speckle-
   dominant noise (corr=0.864 between GT intensity and noise level) across
   the full 4,785-image dataset.
3. **Intensity-weighted loss** (targeting the multiplicative noise
   finding): did not improve results (-0.053 dB vs baseline). NOT used.
4. **Category-weighted oversampling**: 3.0x factor gave real gains on
   target categories but damaged wires_rods/faceted_grain_boundary
   (net +0.012 dB). 1.5x factor gave a cleaner trade-off (+0.029 dB,
   minimal damage to other categories). 1.5x used in final checkpoint.
5. **Model capacity test** (small, 3.1M params vs tiny, 1.0M params):
   flat-to-negative quality change (-0.016 dB) at ~2x the inference
   latency (batch=8: 7.09ms vs 3.42ms/image). Small model NOT used.
6. **Cross-domain generalization check**: the SEM-trained model scores
   27.6 dB PSNR on Phase 1's natural-photo validation set (vs 23.5 dB on
   its own SEM domain) -- strong evidence against needing to merge Phase 1
   data into training.
7. **Rotation augmentation** (random 90/180/270 deg per training sample,
   valid for SEM imagery which has no canonical orientation): broad
   improvement across 6/7 categories, +0.027 dB overall. USED.
8. **Synthetic degradation calibration**: the repo's default degrade()
   (random op ordering, speckle level ~U(0.05,0.25)) measured corr=0.33-0.44
   between GT intensity and noise level on synthetic pairs, vs the real
   data's measured 0.864 -- a poor match, traced to (a) op ordering
   diluting the correlation when downsampling happens after noise, and
   (b) wide per-image level randomization adding unrelated variance.
   Calibrated to forced downsample-first ordering + narrow speckle level
   range U(0.15,0.20), achieving corr=0.916 (close match). Synthetic pairs
   generated ONLY from training-split GT images (verified to exclude all
   475 held-out val images, avoiding leakage). USED -- broad improvement
   across 6/7 categories, +0.034 dB overall on top of rotation augmentation.

Cumulative improvement over the plain (non-augmented) baseline: +0.090 dB
PSNR, achieved through additive, individually-verified changes rather than
a single large architecture or loss change.

See individual scripts in this folder for full methodology and the
verification scripts (test_degradation_ordering.py, test_fixed_noise_level.py,
calibrate_synthetic_range.py) for how the calibration was derived.
