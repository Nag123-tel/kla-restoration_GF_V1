# AI-Based Restoration of Degraded Images for Semiconductor Inspection
### KLA Problem Statement — Hackathon 2026 (SEMICON India)

Joint denoising + super-resolution model that restores NoisyLR images
(degraded by speckle noise, additive Gaussian noise, and downsampling, in
an undisclosed order) back to their GT resolution and quality.

---

## Phase 2 Results Summary

Final checkpoint: `models/best.pt` (NAFNetSR-tiny, 1.0M params), trained with
content-stratified data splitting, rotation augmentation, and calibrated
synthetic degradation. Full experiment log and methodology in
[`experiments/README.md`](experiments/README.md).

| | Internal held-out val (475 img) | External test (297 img, zero overlap) |
|---|---|---|
| PSNR | 23.561 dB | 23.624 dB |
| SSIM | 0.6167 | 0.6311 |
| LPIPS | 0.3433 | 0.3267 |
| Bicubic baseline PSNR | 20.444 dB | 20.455 dB |

Real batch-8 throughput (matching `run.py`'s exact grading pipeline): **3.42 ms/image**.

---


- **Architecture**: [NAFNet](https://arxiv.org/abs/2204.04676) (Chen et al.,
  ECCV 2022, *"Simple Baselines for Image Restoration"*) encoder-decoder,
  with a PixelShuffle super-resolution head appended so denoising and
  upsampling happen in a single forward pass. NAFNet was chosen over
  heavier transformer baselines (Restormer, SwinIR, Uformer) because it
  matches or beats them on real-world denoising benchmarks while being
  substantially cheaper to run — directly relevant to KLA's H100
  end-to-end throughput scoring axis. See `src/model.py` for full
  architectural rationale and code-level comments.
- **Losses**: Charbonnier (robust L1-like reconstruction loss, standard in
  SR/denoising literature) + SSIM loss, combined (`src/losses.py`).
- **Metrics**: PSNR, SSIM, LPIPS (`src/metrics.py`), matching KLA's
  required reporting set exactly.
- **Data**: paired NoisyLR/GT `.npy` files, loaded at their **native
  resolutions** (no downsampling of GT to match NoisyLR — the model
  upsamples internally, so no detail is thrown away before training).
- **Synthetic augmentation** (optional): `src/degradations.py` implements
  speckle + Gaussian + downsampling in random order, exactly matching the
  KLA-confirmed degradation spec, so you can generate additional
  (NoisyLR, GT) pairs from any GT-only images.

## ⚠️ Mandatory submission entry point: `run.py`

 Run it exactly as:

```bash
python run.py <input-dir> <output-dir>
```

- Positional arguments only, no flags.
- Reads every `.npy` file in `<input-dir>`, creates `<output-dir>` if missing.
- Writes one restored `.npy` file per input file, same filename.
- Output shape `(H, W)`, values in `[0,1]`, NaN/Inf sanitized.
- Checkpoint loaded from `models/best.pt` (local only, no internet).
- Runs on GPU if available, batches by shape (batch size 8), falls back to CPU / batch size 1.

`inference.py` remains the dev/eval script (configurable, timing reports, un-clipped output) — it is not the graded script.

## Repository structure

```
repository/
  README.md
  requirements.txt
  train.py                # training script
  inference.py             # standalone inference script (input_dir -> output_dir)
  validate_data.py          # run BEFORE training: checks the whole dataset for problems
  evaluate.py                # run AFTER training: full-res PSNR/SSIM/LPIPS + baseline + examples
  configs/
    default.yaml              # default hyperparameters
  src/
    model.py                   # NAFNetSR architecture
    dataset.py                  # paired dataset loader
    degradations.py               # speckle/Gaussian/downsampling synthesis
    losses.py                      # Charbonnier + SSIM training loss
    metrics.py                      # PSNR / SSIM / LPIPS
  weights/                # dev checkpoints (used by train.py / inference.py)
  models/                 # GRADED checkpoint lives here (models/best.pt) -- see the
                           # "Mandatory submission entry point: run.py" section below
  results/                # metric summaries, per-epoch logs, example restorations
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.1+, CUDA 11.8/12.x. Runs on CPU too
(slower), for debugging without a GPU.

## Data layout expected

```
data/
  train/
    NoisyLR/000001.npy, 000002.npy, ...   # e.g. 128x128, values may exceed [0,1]
    GT/000001.npy, 000002.npy, ...         # e.g. 256x256, values in [0,1]
```

Files are matched between `NoisyLR/` and `GT/` **by filename stem**
(`000001.npy` in both folders = same scene). Preserve the official
dataset's folder structure and filenames as instructed by KLA.

If your GT/NoisyLR resolution ratio differs from 2×, pass `--scale <ratio>`
to `train.py` — the model and dataset both derive their upsampling factor
from this single argument.

## Step 0 — Validate the whole dataset BEFORE training

```bash
python validate_data.py --noisy_dir data/train/NoisyLR --gt_dir data/train/GT
```

Checks, across the **entire** dataset (not a sample):
- every NoisyLR file has a matching GT file by filename stem, and vice versa
- every file loads without error (catches corrupt/truncated `.npy`)
- shapes are consistent and reports the implied scale factor (tells you
  exactly what `--scale` to pass to `train.py`) — flags any inconsistent
  pairs if your dataset accidentally mixes resolutions
- GT values fall within [0,1] as specified; NoisyLR out-of-range values are
  reported as informational (expected per spec) unless wildly extreme

Run this once on the full official dataset before committing to a long
training run — catching a bad file or a resolution mismatch here takes
seconds; catching it 80 epochs in wastes hours.

## Training

```bash
python train.py \
  --noisy_dir data/train/NoisyLR \
  --gt_dir data/train/GT \
  --scale 2 \
  --patch_size 64 \
  --batch_size 16 \
  --epochs 100 \
  --model_size tiny \
  --ckpt_dir weights \
  --results_dir results
```

`train.py` above reproduces the ORIGINAL (unaugmented, randomly-split)
baseline pipeline. **It does NOT produce the submitted checkpoint.**

The actual submitted checkpoint (`models/best.pt`) was produced by
`train_stratified.py`, a wrapper that adds content-stratified splitting,
category-weighted oversampling, rotation augmentation, and calibrated
synthetic degradation on top of the same model/loss/logging as `train.py`.
To reproduce the submitted checkpoint exactly:

```bash
python train_stratified.py \
  --noisy_dir data/train/NoisyLR --gt_dir data/train/GT \
  --manifest <path_to_val_split_FINAL.json> \
  --scale 2 --patch_size 64 --batch_size 16 --epochs 100 --model_size tiny \
  --lr 2e-4 --weight_ssim 0.2 --seed 42 --loss_type combined \
  --oversample_categories "sparse_particles_on_substrate,fiber_mesh" \
  --oversample_factor 1.5 --rotate_augment \
  --synth_from_gt_dir <training-only_GT_folder> --calibrated_synth \
  --ckpt_dir weights --results_dir results
```

`--manifest` needs the content-stratification manifest (built via the
scripts in `experiments/`, using ResNet feature clustering + K-means --
see `experiments/README.md` for the full methodology). Category
stratification and oversampling are training-time-only concerns: they
affect which images the model sees and how often, but nothing at
inference/test time. `run.py`/`inference.py` have no category logic at
all -- they process every input image identically, regardless of which
content category it belongs to. A A
`small` (3.1M params) variant was also trained and benchmarked as a comparison — it
scored marginally higher on PSNR/SSIM/LPIPS but at ~1.8x the inference latency; `tiny`
was chosen as the final submission for its stronger quality/throughput trade-off (see
the presentation, Slide 6, for the full comparison). Pass `--model_size small` if you
want to reproduce that comparison run instead.

Or via config file:
```bash
python train.py --config configs/default.yaml
```

- A clean validation split (`--val_split`, default 10%) is held out from
  the **real** paired data only — never used for training or model
  selection leakage.
- Optional: add `--synth_from_gt_dir data/extra_gt` to mix in synthetic
  (NoisyLR, GT) pairs generated on-the-fly from GT-only images, for extra
  training diversity (matches KLA's "you may create extra synthetic
  degraded pairs" allowance). Synthetic samples are used for training only.
- The best checkpoint by validation PSNR is saved to `weights/best.pt`;
  periodic checkpoints are also saved every `--save_every` epochs.
- **Experiment tracking**: each run writes `results/train_config_<run_id>.json`
  (full hyperparameters, seed, PyTorch/CUDA/GPU info) and
  `results/train_log_<run_id>.csv` (per-epoch loss, lr, val PSNR/SSIM, wall
  time) — this is your record for KLA's "track every experiment" requirement.

## Inference (standalone, mandatory format)

```bash
python inference.py \
  --input_dir /path/to/NoisyLR_test \
  --output_dir /path/to/restored \
  --checkpoint weights/best.pt \
  --batch_size 8
```

- Accepts only `--input_dir` / `--output_dir` / `--checkpoint` as required
  arguments — no source-code edits needed to run on new data.
- Loads every `.npy` file in `input_dir`, restores it, and writes an
  identically-named `.npy` file to `output_dir` (filename preserved).
- Automatically groups inputs by shape so batches of 256×256 and 512×512
  test images (both mentioned in the KLA spec) are handled correctly in
  the same run.
- Outputs are **not clipped or renormalized by default**, matching KLA's
  stated scoring behavior ("KLA does not clip or renormalize outputs").
  Pass `--clip` if you want [0,1]-bounded outputs for your own visual
  inspection — do not use it for the actual scored submission unless KLA's
  instructions say otherwise.
- **Timing matches KLA's exact runtime definition**: disk read →
  preprocessing → CPU→GPU transfer → model execution → GPU→CPU transfer →
  post-processing → **saving to disk**, all inside the measured window.
  Per-batch and aggregate timing are printed, and a full
  `inference_runtime_report.json` (runtime, batch size, ms/image, hardware,
  PyTorch/CUDA versions, timing methodology) is written to `output_dir`.
- Runs on GPU automatically if available (`torch.cuda.is_available()`),
  falls back to CPU otherwise; override with `--device cuda`/`--device cpu`.

## Evaluation — full-resolution PSNR/SSIM/LPIPS + baseline + examples

```bash
python evaluate.py \
  --noisy_dir data/val/NoisyLR --gt_dir data/val/GT \
  --checkpoint weights/best.pt \
  --results_dir results \
  --num_examples 3
```

Point `--noisy_dir`/`--gt_dir` at a held-out validation set with known GT
(NOT the hidden test set, which has no GT). This script:

- Computes PSNR, SSIM, and LPIPS on **full-resolution images** (not
  training crops) — satisfies "show restored examples at full image
  resolution."
- Computes the **same metrics for a bicubic-upsampling-only baseline** on
  the same images, satisfying "compare at least one baseline with the
  final method."
- Writes `results/metrics_summary.json` (aggregate) and
  `results/metrics_per_image.csv` (per-image), so both numeric summaries
  and full detail are available.
- Exports the best-N and worst-N restorations by PSNR (noisy / restored /
  baseline / GT quadruplets, as `.npy`) to
  `results/example_restorations/{best,worst}/` — satisfies "including
  successful and failed cases."
- LPIPS requires `pip install lpips` (already in `requirements.txt`) and a
  one-time download of a small pretrained feature extractor on first run
  (needs internet access once). Use `--no_lpips` to skip it if unavailable.

## Design notes / limitations

- The NAFNet backbone ships in `tiny` and `small` sizes (see
  `build_model()` in `src/model.py`); `tiny` (~1.0M params) is the
  submitted default, chosen for its stronger quality/throughput trade-off
  after benchmarking both against the validation set (`small` scored
  marginally higher on PSNR/SSIM/LPIPS but at ~1.8x the inference latency
  — see the presentation for the full comparison).
- The model assumes a single, fixed upsampling `scale` (inferred from your
  GT/NoisyLR resolution ratio) for the whole dataset. `validate_data.py`
  will flag it if your dataset mixes multiple scale factors — if so,
  extend `dataset.py`/`model.py` to handle per-sample scale, or train
  separate checkpoints per scale and route inputs by shape in
  `inference.py` (the shape-grouping logic is already there to build on).
- **No pretrained/external weights or public datasets are used** in this
  reference implementation. 
- LPIPS requires `pip install lpips` and a one-time download of a small
  pretrained feature extractor (needs internet access once).

## Baseline comparison

The required baseline is bicubic upsampling with no
denoising at all — `evaluate.py` computes this automatically alongside the
trained model's metrics on the same validation images (see above). This is
also exactly what the model's own global residual connection uses as its
starting point (see `NAFNetSR.forward()` in `src/model.py`).



