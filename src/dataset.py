"""
Dataset for the KLA paired restoration task: (NoisyLR, GT) .npy pairs,
matched by filename stem, at their NATIVE resolutions (e.g. NoisyLR
128x128, GT 256x256). No resizing of GT -- the model performs joint
denoise+super-resolve, matching the model's built-in scale factor.

Also supports optional on-the-fly synthetic degradation: if you only have
GT images (or want more training pairs), set `synth_from_gt_dir` and the
dataset will generate NoisyLR from GT using src/degradations.py.
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.degradations import degrade


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        noisy_dir=None,
        gt_dir=None,
        synth_from_gt_dir=None,
        patch_size=64,
        scale=2,
        train=True,
        ext=".npy",
    ):
        """
        Two modes (can be combined):
          1. Real pairs: pass `noisy_dir` + `gt_dir`, matched by filename stem.
          2. Synthetic pairs: pass `synth_from_gt_dir` (a folder of GT-only
             .npy images); NoisyLR is generated on-the-fly each epoch via
             src.degradations.degrade(), giving fresh noise realizations
             every time (a form of augmentation).

        `patch_size` is the NOISY (input) crop size; the GT crop is
        `patch_size * scale`.
        """
        self.patch_size = patch_size
        self.scale = scale
        self.train = train
        self.ext = ext

        self.real_ids = []
        self.noisy_paths, self.gt_paths = {}, {}
        if noisy_dir is not None and gt_dir is not None:
            self.noisy_paths = {Path(p).stem: p for p in glob.glob(os.path.join(noisy_dir, f"*{ext}"))}
            self.gt_paths = {Path(p).stem: p for p in glob.glob(os.path.join(gt_dir, f"*{ext}"))}
            self.real_ids = sorted(set(self.noisy_paths) & set(self.gt_paths))
            missing = set(self.gt_paths) ^ set(self.noisy_paths)
            if missing:
                print(f"[warn] {len(missing)} file(s) present in only one of noisy_dir/gt_dir, skipped.")

        self.synth_gt_paths = []
        if synth_from_gt_dir is not None:
            self.synth_gt_paths = sorted(glob.glob(os.path.join(synth_from_gt_dir, f"*{ext}")))

        total = len(self.real_ids) + len(self.synth_gt_paths)
        assert total > 0, "No data found: provide (noisy_dir + gt_dir) and/or synth_from_gt_dir."
        print(f"Dataset: {len(self.real_ids)} real pairs, {len(self.synth_gt_paths)} synth-source GT images.")

    def __len__(self):
        return len(self.real_ids) + len(self.synth_gt_paths)

    def _load(self, path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        if idx < len(self.real_ids):
            scene_id = self.real_ids[idx]
            noisy = self._load(self.noisy_paths[scene_id])
            gt = self._load(self.gt_paths[scene_id])
        else:
            gt_path = self.synth_gt_paths[idx - len(self.real_ids)]
            gt = self._load(gt_path)
            noisy = degrade(gt, scale=self.scale)

        if self.train:
            noisy, gt = self._random_crop_and_flip(noisy, gt)

        return noisy, gt

    def _random_crop_and_flip(self, noisy, gt):
        _, h, w = noisy.shape
        s = self.scale
        ps = min(self.patch_size, h, w)

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        noisy = noisy[:, top:top + ps, left:left + ps]
        gt = gt[:, top * s:(top + ps) * s, left * s:(left + ps) * s]

        if random.random() < 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
        if random.random() < 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            gt = torch.flip(gt, dims=[-2])

        return noisy.contiguous(), gt.contiguous()
