"""
Training script for the KLA Hackathon 2026 restoration model.

Usage:
    python train.py --config configs/default.yaml

Or override individual settings on the command line, e.g.:
    python train.py --noisy_dir data/train/NoisyLR --gt_dir data/train/GT \
        --epochs 100 --batch_size 16
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from src.dataset import PairedRestorationDataset
from src.losses import CombinedLoss
from src.metrics import psnr, ssim
from src.model import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file; CLI args override its values.")

    p.add_argument("--noisy_dir", type=str, default=None)
    p.add_argument("--gt_dir", type=str, default=None)
    p.add_argument("--synth_from_gt_dir", type=str, default=None,
                    help="Optional: folder of GT-only images to synthesize extra training pairs from.")
    p.add_argument("--val_split", type=float, default=0.1,
                    help="Fraction of REAL pairs held out for validation (no leakage into training).")

    p.add_argument("--patch_size", type=int, default=64, help="NoisyLR crop size.")
    p.add_argument("--scale", type=int, default=2, help="GT resolution / NoisyLR resolution.")
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"])

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_ssim", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ckpt_dir", type=str, default="weights")
    p.add_argument("--results_dir", type=str, default="results",
                    help="Where per-epoch training logs (CSV/JSON) and the run config are saved.")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)

    args = p.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            p.set_defaults(**{k: v})
        args = p.parse_args()

    return args


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device)
        pred = model(noisy).clamp(0, 1)
        for i in range(pred.shape[0]):
            psnrs.append(psnr(pred[i], gt[i]))
            ssims.append(ssim(pred[i], gt[i]))
    model.train()
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    print("Config:", vars(args))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(args.results_dir, f"train_config_{run_id}.json")
    with open(config_path, "w") as f:
        json.dump({
            **vars(args),
            "run_id": run_id,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }, f, indent=2)
    print(f"Saved run config to: {config_path}")

    log_path = os.path.join(args.results_dir, f"train_log_{run_id}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "lr", "val_psnr", "val_ssim", "epoch_time_seconds"])
    print(f"Logging per-epoch metrics to: {log_path}")

    assert args.noisy_dir or args.synth_from_gt_dir, \
        "Provide --noisy_dir (+ --gt_dir) and/or --synth_from_gt_dir"

    full_ds = PairedRestorationDataset(
        noisy_dir=args.noisy_dir,
        gt_dir=args.gt_dir,
        synth_from_gt_dir=args.synth_from_gt_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        train=True,
    )

    n_val = max(1, int(len(full_ds) * args.val_split)) if args.val_split > 0 else 0
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_ds.dataset.train = True
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    if n_val > 0:
        val_indices = val_ds.indices
        real_ids = full_ds.real_ids
        val_stems = [real_ids[i] for i in val_indices if i < len(real_ids)]
        val_manifest_path = os.path.join(args.results_dir, f"val_split_{run_id}.json")
        with open(val_manifest_path, "w") as f:
            json.dump({
                "seed": args.seed,
                "val_split_fraction": args.val_split,
                "num_val_files": len(val_stems),
                "val_file_stems": sorted(val_stems),
                "noisy_dir": args.noisy_dir,
                "gt_dir": args.gt_dir,
            }, f, indent=2)
        print(f"Saved validation file list to: {val_manifest_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if n_val > 0 else None

    model = build_model(in_ch=args.in_ch, scale=args.scale, size=args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NAFNetSR ({args.model_size}), params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss(channels=args.in_ch, weight_ssim=args.weight_ssim)

    best_psnr = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss, parts = loss_fn(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

        scheduler.step()
        epoch_loss = running_loss / n_train
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | loss: {epoch_loss:.5f} | "
              f"lr: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        val_psnr_val, val_ssim_val = "", ""
        if val_loader is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            val_psnr, val_ssim = validate(model, val_loader, device)
            val_psnr_val, val_ssim_val = val_psnr, val_ssim
            print(f"  [val] PSNR: {val_psnr:.3f} dB | SSIM: {val_ssim:.4f}")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                }, os.path.join(args.ckpt_dir, "best.pt"))
                print(f"  -> saved new best checkpoint (PSNR={val_psnr:.3f} dB)")

        log_writer.writerow([epoch, epoch_loss, scheduler.get_last_lr()[0], val_psnr_val, val_ssim_val, dt])
        log_file.flush()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            }, os.path.join(args.ckpt_dir, f"epoch{epoch}.pt"))

    log_file.close()
    print("Training complete. Best val PSNR:", best_psnr)
    print(f"Full training log saved to: {log_path}")


if __name__ == "__main__":
    main()
