"""
Run from the kla-restoration repo root in Colab.

Breaks down PSNR/SSIM by SEM morphology category (the 7 clusters we built:
wires_rods, fiber_mesh, dense_particle_aggregates, porous_foam_membrane,
faceted_grain_boundary, grainy_blurry_particles, sparse_particles_on_substrate)
for both the trained model and the bicubic baseline, on the same held-out
475-image val set used throughout.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import psnr, ssim
from src.model import build_model

# ---------- Paths ----------
RUN_VAL_SPLIT = 'results/val_split_20260903_045504_stratified.json'
FINAL_MANIFEST = '/content/drive/MyDrive/Naga Vyshnavi/prepared_data/results/val_split_FINAL.json'
CHECKPOINT = 'weights_small_1p5x/best.pt'
BICUBIC_RESULTS = 'results/bicubic_baseline_new_data.json'  # from the previous step, reused

with open(RUN_VAL_SPLIT) as f:
    run_val = json.load(f)
with open(FINAL_MANIFEST) as f:
    manifest = json.load(f)
with open(BICUBIC_RESULTS) as f:
    bicubic_results = json.load(f)

val_stems = sorted(run_val['val_file_stems'])
noisy_dir = run_val['noisy_dir']
gt_dir = run_val['gt_dir']
tags = manifest['tags']  # stem -> 'cluster_N'
category_names = manifest['category_names']  # cluster_N -> human name

bicubic_per_image = {d['stem']: d for d in bicubic_results['per_image']}

# ---------- Load model ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(CHECKPOINT, map_location=device)
args = ckpt['args']
model = build_model(in_ch=args['in_ch'], scale=args['scale'], size=args['model_size']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_psnr={ckpt.get('val_psnr', 'n/a')}")

# ---------- Run model + bicubic per image, grouped by category ----------
by_category_model = defaultdict(lambda: {'psnr': [], 'ssim': []})
by_category_bicubic = defaultdict(lambda: {'psnr': [], 'ssim': []})

with torch.no_grad():
    for stem in val_stems:
        noisy = np.load(os.path.join(noisy_dir, stem + '.npy')).astype(np.float32)
        gt = np.load(os.path.join(gt_dir, stem + '.npy')).astype(np.float32)

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

        pred = model(noisy_t).clamp(0, 1)
        p = float(psnr(pred[0], gt_t[0]))
        s = float(ssim(pred[0], gt_t[0]))

        cat_id = tags.get(stem, 'unknown')
        cat_name = category_names.get(cat_id, cat_id)

        by_category_model[cat_name]['psnr'].append(p)
        by_category_model[cat_name]['ssim'].append(s)

        if stem in bicubic_per_image:
            by_category_bicubic[cat_name]['psnr'].append(bicubic_per_image[stem]['psnr'])
            by_category_bicubic[cat_name]['ssim'].append(bicubic_per_image[stem]['ssim'])

# ---------- Report ----------
print(f"\n{'Category':<32} {'N':>5} {'Bicubic PSNR':>13} {'Model PSNR':>11} {'Gain':>7} | {'Bicubic SSIM':>13} {'Model SSIM':>11} {'Gain':>7}")
print('-' * 115)

summary = {}
for cat_name in sorted(by_category_model.keys(), key=lambda c: -len(by_category_model[c]['psnr'])):
    n = len(by_category_model[cat_name]['psnr'])
    mp = np.mean(by_category_model[cat_name]['psnr'])
    ms = np.mean(by_category_model[cat_name]['ssim'])
    bp = np.mean(by_category_bicubic[cat_name]['psnr']) if by_category_bicubic[cat_name]['psnr'] else float('nan')
    bs = np.mean(by_category_bicubic[cat_name]['ssim']) if by_category_bicubic[cat_name]['ssim'] else float('nan')
    print(f"{cat_name:<32} {n:>5} {bp:>13.3f} {mp:>11.3f} {mp-bp:>+7.3f} | {bs:>13.4f} {ms:>11.4f} {ms-bs:>+7.4f}")
    summary[cat_name] = {
        'n': n,
        'bicubic_psnr': float(bp), 'model_psnr': float(mp), 'psnr_gain': float(mp - bp),
        'bicubic_ssim': float(bs), 'model_ssim': float(ms), 'ssim_gain': float(ms - bs),
    }

overall_model_psnr = np.mean([v for cat in by_category_model.values() for v in cat['psnr']])
overall_model_ssim = np.mean([v for cat in by_category_model.values() for v in cat['ssim']])
print('-' * 115)
print(f"{'OVERALL':<32} {len(val_stems):>5} {'':>13} {overall_model_psnr:>11.3f} {'':>7} | {'':>13} {overall_model_ssim:>11.4f}")

with open('results/per_category_eval.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved results/per_category_eval.json")
print("\nRanked worst-to-best by model PSNR:")
for cat_name, s in sorted(summary.items(), key=lambda kv: kv[1]['model_psnr']):
    print(f"  {cat_name:<32} {s['model_psnr']:.3f} dB (n={s['n']})")
