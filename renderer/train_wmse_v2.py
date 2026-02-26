#!/usr/bin/env python3
"""
MIP-splatting training: Weighted MSE loss ONLY (with proper train/val split).

This retrains WMSE with the same train/val split (90/10, seed=42) as the other
ablation configurations (WMSE+SSIM, WMSE+SSIM+Edge, WMSE+SSIM+Edge+Inte).

Run with:
    cd /workspace/hisnegs/src/renderer
    nohup python train_wmse_v2.py > train_wmse_v2.log 2>&1 &

Or inside tmux/screen:
    conda activate neurogs
    cd /workspace/hisnegs/src/renderer
    python train_wmse_v2.py
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Ensure the renderer package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.rendering import (
    Camera,
    MIPSplattingTrainer,
    compute_aspect_scales,
    generate_camera_poses_from_config,
    generate_mip_dataset,
    load_config,
    load_volume,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config("config_splat.yml")

    # ── Load from GMF checkpoint (same starting point as original WMSE) ──
    GMF_CKPT = "/workspace/neurogs_v7/gmf_refined_best.pt"
    ckpt = torch.load(GMF_CKPT, map_location=device)

    means_raw = ckpt["means"].to(device)
    log_scales_raw = ckpt["log_scales"].to(device)
    quaternions_raw = ckpt["quaternions"].to(device)
    log_amplitudes_raw = ckpt.get("log_amplitudes", ckpt.get("log_intensities")).to(device)

    K = means_raw.shape[0]
    print(f"\nCheckpoint: {GMF_CKPT}")
    print(f"  Source:     GMF refined checkpoint")
    print(f"  Gaussians:  K = {K}")

    # ── Load GT volume ───────────────────────────────────────────────
    VOL_PATH = os.path.abspath("../../dataset/10-2900-control-cell-05_cropped_corrected.tif")
    vol_np = load_volume(VOL_PATH)
    vol_gpu = torch.from_numpy(vol_np).to(device)
    print(f"\nGT Volume: shape={vol_np.shape}, range=[{vol_np.min():.3f}, {vol_np.max():.3f}]")

    aspect_scales = compute_aspect_scales(vol_np.shape)
    print(f"  Aspect scales: {aspect_scales.tolist()}")

    # ── Generate GT MIP dataset ──────────────────────────────────────
    print("\nGenerating GT MIP dataset...")
    cam_train = Camera.from_fov(
        fov_x_deg=cfg["camera"]["fov_x_deg"],
        width=256, height=256,
        near=cfg["camera"]["near"],
        far=cfg["camera"]["far"],
    )

    poses = generate_camera_poses_from_config(cfg)
    print(f"  Camera poses: {len(poses)} viewpoints")

    rm_cfg = cfg["ray_marching"]
    dataset = generate_mip_dataset(
        vol_gpu, cam_train, poses,
        n_ray_samples=rm_cfg["n_samples"],
        near=rm_cfg["near"],
        far=rm_cfg["far"],
        aspect_scales=aspect_scales,
    )
    print(f"  Dataset ready: {len(dataset)} GT MIP projections")

    # ── Modify config for WMSE only (no SSIM, no edge, no sparsity) ──
    cfg_wmse = cfg.copy()
    cfg_wmse["loss"] = dict(cfg["loss"])
    cfg_wmse["loss"]["lambda_ssim"] = 0.0   # WMSE only - no SSIM
    cfg_wmse["loss"]["lambda_edge"] = 0.0   # No edge loss
    cfg_wmse["loss"]["lambda_sparse"] = 0.0 # No sparsity

    # ── Create trainer ───────────────────────────────────────────────
    trainer = MIPSplattingTrainer(
        means=means_raw.clone(),
        log_scales=log_scales_raw.clone(),
        quaternions=quaternions_raw.clone(),
        log_intensities=log_amplitudes_raw.clone(),
        cfg=cfg_wmse,
        aspect_scales=aspect_scales,
    )
    print(f"\nTrainer initialized from GMF checkpoint:")
    print(f"  K = {trainer.means.shape[0]} Gaussians")
    print(f"  fg_weight    = {trainer.fg_weight}")
    print(f"  lambda_ssim  = {trainer.lambda_ssim}  (WMSE only)")
    print(f"  lambda_edge  = {trainer.lambda_edge}")
    print(f"  lambda_sparse= {trainer.lambda_sparse}")

    # ── Training config ──────────────────────────────────────────────
    TOTAL_EPOCHS = 2000
    SAVE_DIR = "checkpoints_wmse_v2"
    
    cfg_train = cfg_wmse.copy()
    cfg_train["training"] = dict(cfg_wmse["training"])
    cfg_train["training"]["start_epoch"] = 0
    cfg_train["training"]["total_epochs"] = TOTAL_EPOCHS
    cfg_train["training"]["n_epochs"] = TOTAL_EPOCHS
    cfg_train["training"]["save_every"] = 100
    cfg_train["training"]["log_every"] = 50
    # Use same train/val split as other models
    cfg_train["training"]["val_ratio"] = 0.1
    cfg_train["training"]["val_seed"] = 42
    cfg_train["training"]["val_min_views"] = 8

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_template = os.path.join(SAVE_DIR, "splat_wmse_v2_ep{epoch}.pt")

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\nStarting WMSE training (with train/val split)...")
    print(f"  Epochs: 0 → {TOTAL_EPOCHS}")
    print(f"  Checkpoints → {SAVE_DIR}/")
    history = trainer.train(cam_train, dataset, cfg_train, save_path=save_template)
    elapsed = time.time() - t0

    # ── Save best checkpoint ─────────────────────────────────────────
    best_psnr = max(h["psnr"] for h in history)
    best_ssim = max(h["ssim"] for h in history)
    best_loss = min(h["loss"] for h in history)

    best_path = os.path.join(SAVE_DIR, "splat_wmse_v2_best.pt")
    torch.save({
        "means": trainer.means.data.cpu(),
        "log_scales": trainer.log_scales.data.cpu(),
        "quaternions": trainer.quaternions.data.cpu(),
        "log_intensities": trainer.log_intensities.data.cpu(),
        "epoch": TOTAL_EPOCHS,
    }, best_path)

    # ── Save training history ────────────────────────────────────────
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"WMSE Training complete in {elapsed / 60:.1f} min")
    print(f"  Final K:    {trainer.means.shape[0]}")
    print(f"  Best PSNR:  {best_psnr:.2f} dB")
    print(f"  Best SSIM:  {best_ssim:.4f}")
    print(f"  Best loss:  {best_loss:.6f}")
    print(f"  Best ckpt:  {best_path}")
    print(f"  History:    {history_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
