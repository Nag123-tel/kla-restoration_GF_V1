"""
Validate the entire dataset before committing to a training run.

Checks:
  - every NoisyLR file has a matching GT file (by filename stem) and vice versa
  - every file loads without error (catches corrupt/truncated .npy)
  - shapes are consistent and the implied scale factor (GT/NoisyLR resolution
    ratio) is the same across the whole dataset (flags any inconsistent pairs)
  - value ranges: GT should be within [0,1] per KLA spec; NoisyLR may exceed
    [0,1] (flagged as informational, not an error) but wildly out-of-range
    values (e.g. NaN, Inf, or huge magnitudes) are flagged as real problems
  - reports summary statistics so you know your dataset is clean before a
    long training run, rather than finding out mid-way through

Usage:
    python validate_data.py --noisy_dir data/train/NoisyLR --gt_dir data/train/GT
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--ext", type=str, default=".npy")
    p.add_argument("--sample_only", type=int, default=None,
                    help="If set, only fully load+check this many pairs (still checks all filenames/shapes cheaply).")
    return p.parse_args()


def peek_shape(path):
    """Read just the .npy header (fast, no full load into memory) via mmap."""
    arr = np.load(path, mmap_mode="r")
    return arr.shape, arr.dtype


def main():
    args = get_args()

    noisy_files = {Path(p).stem: p for p in glob.glob(os.path.join(args.noisy_dir, f"*{args.ext}"))}
    gt_files = {Path(p).stem: p for p in glob.glob(os.path.join(args.gt_dir, f"*{args.ext}"))}

    print(f"NoisyLR dir: {args.noisy_dir}  ({len(noisy_files)} files)")
    print(f"GT dir:      {args.gt_dir}  ({len(gt_files)} files)")
    print()

    # --- 1. Filename matching ---
    only_noisy = set(noisy_files) - set(gt_files)
    only_gt = set(gt_files) - set(noisy_files)
    matched = sorted(set(noisy_files) & set(gt_files))

    print(f"Matched pairs: {len(matched)}")
    if only_noisy:
        print(f"  [WARN] {len(only_noisy)} NoisyLR file(s) with no matching GT: "
              f"{sorted(only_noisy)[:5]}{' ...' if len(only_noisy) > 5 else ''}")
    if only_gt:
        print(f"  [WARN] {len(only_gt)} GT file(s) with no matching NoisyLR: "
              f"{sorted(only_gt)[:5]}{' ...' if len(only_gt) > 5 else ''}")
    if not only_noisy and not only_gt:
        print("  All files matched 1:1. Good.")
    print()

    if len(matched) == 0:
        print("[FATAL] No matched pairs found. Check your directory paths.")
        return

    # --- 2. Shape / scale consistency (cheap: header-only reads) ---
    print("Checking shapes and scale consistency (header-only, fast)...")
    scales = {}
    shape_errors = []
    noisy_shapes, gt_shapes = set(), set()

    for stem in matched:
        try:
            n_shape, n_dtype = peek_shape(noisy_files[stem])
            g_shape, g_dtype = peek_shape(gt_files[stem])
        except Exception as e:
            shape_errors.append((stem, f"could not read header: {e}"))
            continue

        noisy_shapes.add(n_shape)
        gt_shapes.add(g_shape)

        if len(n_shape) < 2 or len(g_shape) < 2:
            shape_errors.append((stem, f"unexpected ndim: noisy={n_shape}, gt={g_shape}"))
            continue

        if g_shape[0] % n_shape[0] != 0 or g_shape[1] % n_shape[1] != 0:
            shape_errors.append((stem, f"GT shape {g_shape} is not an integer multiple of NoisyLR shape {n_shape}"))
            continue

        scale_h = g_shape[0] / n_shape[0]
        scale_w = g_shape[1] / n_shape[1]
        if scale_h != scale_w:
            shape_errors.append((stem, f"non-uniform scale: h={scale_h}, w={scale_w}"))
            continue

        scales[stem] = scale_h

    unique_scales = set(scales.values())
    print(f"  Distinct NoisyLR shapes seen: {sorted(noisy_shapes)}")
    print(f"  Distinct GT shapes seen:      {sorted(gt_shapes)}")
    print(f"  Distinct scale factors implied: {sorted(unique_scales)}")
    if len(unique_scales) > 1:
        print(f"  [WARN] Multiple scale factors found in the dataset! "
              f"Your model/dataset code currently assumes ONE fixed scale -- "
              f"decide how to handle this (separate checkpoints per scale, or "
              f"resize to a common scale) before training.")
    elif len(unique_scales) == 1:
        print(f"  -> Use --scale {int(list(unique_scales)[0])} for train.py")

    if shape_errors:
        print(f"  [WARN] {len(shape_errors)} pair(s) had shape problems:")
        for stem, msg in shape_errors[:10]:
            print(f"    {stem}: {msg}")
        if len(shape_errors) > 10:
            print(f"    ... and {len(shape_errors) - 10} more")
    print()

    # --- 3. Value range / integrity checks (full load, can be slow -> optionally sampled) ---
    check_stems = matched if args.sample_only is None else matched[:args.sample_only]
    print(f"Checking value ranges and integrity on {len(check_stems)} pair(s) "
          f"({'all' if args.sample_only is None else 'sampled'})...")

    corrupt = []
    gt_out_of_range = []
    noisy_extreme = []
    gt_min_seen, gt_max_seen = float("inf"), float("-inf")
    noisy_min_seen, noisy_max_seen = float("inf"), float("-inf")

    for stem in check_stems:
        try:
            noisy = np.load(noisy_files[stem])
            gt = np.load(gt_files[stem])
        except Exception as e:
            corrupt.append((stem, str(e)))
            continue

        if not np.isfinite(noisy).all():
            corrupt.append((stem, "NoisyLR contains NaN/Inf"))
            continue
        if not np.isfinite(gt).all():
            corrupt.append((stem, "GT contains NaN/Inf"))
            continue

        gt_min_seen = min(gt_min_seen, float(gt.min()))
        gt_max_seen = max(gt_max_seen, float(gt.max()))
        noisy_min_seen = min(noisy_min_seen, float(noisy.min()))
        noisy_max_seen = max(noisy_max_seen, float(noisy.max()))

        if gt.min() < -1e-3 or gt.max() > 1 + 1e-3:
            gt_out_of_range.append((stem, float(gt.min()), float(gt.max())))

        # NoisyLR "may extend slightly outside [0,1]" per spec -- flag only if wildly off
        if noisy.min() < -5 or noisy.max() > 5:
            noisy_extreme.append((stem, float(noisy.min()), float(noisy.max())))

    print(f"  GT value range across dataset:      [{gt_min_seen:.4f}, {gt_max_seen:.4f}]")
    print(f"  NoisyLR value range across dataset: [{noisy_min_seen:.4f}, {noisy_max_seen:.4f}]")

    if corrupt:
        print(f"  [FATAL] {len(corrupt)} corrupt/unreadable file(s):")
        for stem, msg in corrupt[:10]:
            print(f"    {stem}: {msg}")
    if gt_out_of_range:
        print(f"  [WARN] {len(gt_out_of_range)} GT file(s) outside expected [0,1] range "
              f"(spec says GT should be normalized to [0,1]):")
        for stem, lo, hi in gt_out_of_range[:10]:
            print(f"    {stem}: [{lo:.4f}, {hi:.4f}]")
    if noisy_extreme:
        print(f"  [INFO] {len(noisy_extreme)} NoisyLR file(s) with unusually large values "
              f"(spec allows slightly outside [0,1], but these look extreme -- worth a manual look):")
        for stem, lo, hi in noisy_extreme[:10]:
            print(f"    {stem}: [{lo:.4f}, {hi:.4f}]")

    print()
    print("=" * 60)
    n_problems = len(only_noisy) + len(only_gt) + len(shape_errors) + len(corrupt) + len(gt_out_of_range)
    if n_problems == 0:
        print("Dataset looks clean. Safe to proceed with training.")
    else:
        print(f"Found {n_problems} total issue(s) above -- review before a long training run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
