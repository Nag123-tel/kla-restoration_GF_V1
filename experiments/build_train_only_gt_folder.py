"""
Run from the kla-restoration repo root in Colab.

Builds a folder with symlinks to ONLY the training-split GT images
(excluding the 475 held-out val images), safe to pass as
--synth_from_gt_dir without leaking validation content into synthetic
training augmentation.
"""

import json
import os

RUN_VAL_SPLIT = 'results/val_split_20260903_064417_stratified.json'  # rotation-augmented run
OUT_DIR = 'train_only_gt_for_synth'

with open(RUN_VAL_SPLIT) as f:
    run_val = json.load(f)

val_stems = set(run_val['val_file_stems'])
gt_dir = run_val['gt_dir']

os.makedirs(OUT_DIR, exist_ok=True)

all_gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
n_linked = 0
for f in all_gt_files:
    stem = f[:-4]
    if stem in val_stems:
        continue  # skip val images -- do not include in synth source
    dst = os.path.join(OUT_DIR, f)
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(os.path.join(gt_dir, f)), dst)
    n_linked += 1

print(f'Linked {n_linked} training-only GT images into {OUT_DIR}')
print(f'(Excluded {len(val_stems)} held-out val images)')
print(f'Verify: {len(os.listdir(OUT_DIR))} files in {OUT_DIR}')
