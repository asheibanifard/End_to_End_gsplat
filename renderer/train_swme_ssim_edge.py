#!/usr/bin/env python3
"""
MIP-splatting training with Weighted MSE + SSIM + Edge loss.
Then continues with Intensity Sparsity Regularization.

Phase 1: WMSE + SSIM + Edge loss
- Penalizes blurry neurite boundaries using Sobel-like finite differences
- Particularly valuable for thin processes that MSE tends to smooth over
- Logs: train_swme_ssim_edge.log, checkpoints: train_swme_ssim_edge_checkpoint/

Phase 2: WMSE + SSIM + Edge + Sparsity regularization  
- Encourages Gaussians to be clearly on or clearly off
- Improves densification decisions downstream
- Logs: train_swme_ssim_edge_inte.log, checkpoints: train_swme_ssim_edge_inte_checkpoint/

Run with:
    cd /workspace/hisnegs/src/renderer
    nohup python train_swme_ssim_edge.py > train_swme_ssim_edge.log 2>&1 &

Or inside tmux/screen:
    conda activate neurogs
    cd /workspace/hisnegs/src/renderer
    python train_swme_ssim_edge.py
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


def train_phase(
    device,
    cfg,
    vol_np,
    vol_gpu,
    aspect_scales,
    cam_train,
    poses,
    dataset,
    ckpt_path,
    start_epoch,
    total_epochs,
    save_dir,
    phase_name,
    lambda_edge=0.1,
    lambda_sparse=0.0,
):
    """Run a single training phase with specified loss weights."""
    
    ckpt = torch.load(ckpt_path, map_location=device)
    means_raw = ckpt["means"].to(device)
    log_scales_raw = ckpt["log_scales"].to(device)
    quaternions_raw = ckpt["quaternions"].to(device)
    log_amplitudes_raw = ckpt.get("log_amplitudes", ckpt.get("log_intensities")).to(device)

    K = means_raw.shape[0]
    print(f"\n{'=' * 60}")
    print(f"PHASE: {phase_name}")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"  Gaussians:  K = {K}")

    # ── Modify config for this phase ─────────────────────────────────
    cfg_phase = cfg.copy()
    cfg_phase["loss"] = dict(cfg["loss"])
    cfg_phase["loss"]["lambda_edge"] = lambda_edge
    cfg_phase["loss"]["lambda_sparse"] = lambda_sparse

    # ── Create trainer ───────────────────────────────────────────────
    trainer = MIPSplattingTrainer(
        means=means_raw.clone(),
        log_scales=log_scales_raw.clone(),
        quaternions=quaternions_raw.clone(),
        log_intensities=log_amplitudes_raw.clone(),
        cfg=cfg_phase,
        aspect_scales=aspect_scales,
    )
    print(f"\nTrainer initialized:")
    print(f"  K = {trainer.means.shape[0]} Gaussians")
    print(f"  fg_weight    = {trainer.fg_weight}")
    print(f"  lambda_ssim  = {trainer.lambda_ssim}")
    print(f"  lambda_edge  = {trainer.lambda_edge}")
    print(f"  lambda_sparse= {trainer.lambda_sparse}")

    # ── Training config ──────────────────────────────────────────────
    cfg_train = cfg_phase.copy()
    cfg_train["training"] = dict(cfg_phase["training"])
    cfg_train["training"]["start_epoch"] = start_epoch
    cfg_train["training"]["total_epochs"] = total_epochs
    cfg_train["training"]["n_epochs"] = total_epochs - start_epoch
    cfg_train["training"]["save_every"] = 100
    cfg_train["training"]["log_every"] = 50

    os.makedirs(save_dir, exist_ok=True)
    save_template = os.path.join(save_dir, f"splat_{phase_name}_ep{{epoch}}.pt")

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\nStarting training...")
    print(f"  Epochs: {start_epoch} → {total_epochs}")
    print(f"  Checkpoints → {save_dir}/")
    history = trainer.train(cam_train, dataset, cfg_train, save_path=save_template)
    elapsed = time.time() - t0

    # ── Save best checkpoint ─────────────────────────────────────────
    best_psnr = max(h["psnr"] for h in history)
    best_ssim = max(h["ssim"] for h in history)
    best_loss = min(h["loss"] for h in history)

    best_path = os.path.join(save_dir, f"splat_{phase_name}_best.pt")
    torch.save({
        "means": trainer.means.data.cpu(),
        "log_scales": trainer.log_scales.data.cpu(),
        "quaternions": trainer.quaternions.data.cpu(),
        "log_intensities": trainer.log_intensities.data.cpu(),
        "epoch": total_epochs,
    }, best_path)

    # ── Save training history ────────────────────────────────────────
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{phase_name} complete in {elapsed / 60:.1f} min")
    print(f"  Final K:    {trainer.means.shape[0]}")
    print(f"  Best PSNR:  {best_psnr:.2f} dB")
    print(f"  Best SSIM:  {best_ssim:.4f}")
    print(f"  Best loss:  {best_loss:.6f}")
    print(f"  Best ckpt:  {best_path}")
    print(f"  History:    {history_path}")
    print(f"{'=' * 60}")

    return best_path, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load config ──────────────────────────────────────────────────
    cfg = load_config("config_splat.yml")

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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: WMSE + SSIM + Edge loss
    # ══════════════════════════════════════════════════════════════════
    PHASE1_CKPT = os.path.join("checkpoints_wsme_ssim", "splat_wsme_ssim_best.pt")
    PHASE1_EPOCHS = 2000
    PHASE1_SAVE_DIR = "train_swme_ssim_edge_checkpoint"
    
    phase1_best_path, phase1_history = train_phase(
        device=device,
        cfg=cfg,
        vol_np=vol_np,
        vol_gpu=vol_gpu,
        aspect_scales=aspect_scales,
        cam_train=cam_train,
        poses=poses,
        dataset=dataset,
        ckpt_path=PHASE1_CKPT,
        start_epoch=0,
        total_epochs=PHASE1_EPOCHS,
        save_dir=PHASE1_SAVE_DIR,
        phase_name="swme_ssim_edge",
        lambda_edge=0.1,
        lambda_sparse=0.0,
    )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: WMSE + SSIM + Edge + Sparsity regularization
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Starting Phase 2: Adding Intensity Sparsity Regularization")
    print("=" * 60)
    
    # Redirect output to phase 2 log file
    phase2_log = open("train_swme_ssim_edge_inte.log", "w")
    import contextlib
    
    # Continue logging to stdout but also write phase 2 info
    PHASE2_CKPT = phase1_best_path  # Use best from Phase 1
    PHASE2_EPOCHS = 2000
    PHASE2_SAVE_DIR = "train_swme_ssim_edge_inte_checkpoint"
    
    phase2_best_path, phase2_history = train_phase(
        device=device,
        cfg=cfg,
        vol_np=vol_np,
        vol_gpu=vol_gpu,
        aspect_scales=aspect_scales,
        cam_train=cam_train,
        poses=poses,
        dataset=dataset,
        ckpt_path=PHASE2_CKPT,
        start_epoch=0,
        total_epochs=PHASE2_EPOCHS,
        save_dir=PHASE2_SAVE_DIR,
        phase_name="swme_ssim_edge_inte",
        lambda_edge=0.1,
        lambda_sparse=0.001,  # Intensity sparsity regularization
    )

    # ══════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ALL TRAINING PHASES COMPLETE")
    print("=" * 60)
    print(f"\nPhase 1 (Edge loss):")
    print(f"  Best PSNR: {max(h['psnr'] for h in phase1_history):.2f} dB")
    print(f"  Best SSIM: {max(h['ssim'] for h in phase1_history):.4f}")
    print(f"  Checkpoint: {phase1_best_path}")
    print(f"\nPhase 2 (Edge + Sparsity):")
    print(f"  Best PSNR: {max(h['psnr'] for h in phase2_history):.2f} dB")
    print(f"  Best SSIM: {max(h['ssim'] for h in phase2_history):.4f}")
    print(f"  Checkpoint: {phase2_best_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
