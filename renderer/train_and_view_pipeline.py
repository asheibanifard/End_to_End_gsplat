#!/usr/bin/env python3
"""
End-to-End MIP Splatting Pipeline

Pipeline:
1. Load volume data (TIFF)
2. Randomly initialize 3D Gaussians
3. Generate ground truth MIP projections from volume
4. Train Gaussians to match GT MIPs
5. (Optional) Launch interactive viewer

Usage:
    python train_and_view_pipeline.py --volume path/to/volume.tif --num_gaussians 10000 --steps 5000
    python train_and_view_pipeline.py --volume path/to/volume.tif --config config.yml
"""
import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import math

from rendering import (
    Camera,
    GaussianParameters,
    render_mip_projection,
    render_gt_mip,
    load_volume,
    compute_aspect_scales,
    generate_camera_poses,
    mse_loss,
    weighted_mse_loss,
    ssim_loss_fn,
)


def initialize_gaussians_random(
    num_gaussians: int,
    vol_shape: tuple,
    device: torch.device,
    init_scale: float = 0.05,
    bounds: tuple = ((-1, 1), (-1, 1), (-1, 1))
) -> tuple:
    """
    Randomly initialize Gaussian parameters.
    
    Returns:
        means, log_scales, quaternions, log_intensities (all require_grad=True)
    """
    print(f"\nInitializing {num_gaussians} Gaussians randomly...")
    
    # Means: uniform random in bounds
    means = torch.zeros(num_gaussians, 3, device=device)
    for i in range(3):
        lo, hi = bounds[i]
        means[:, i] = torch.rand(num_gaussians, device=device) * (hi - lo) + lo
    
    # Scales: isotropic, log-space
    log_scales = torch.ones(num_gaussians, 3, device=device) * np.log(init_scale)
    
    # Quaternions: identity rotation
    quaternions = torch.zeros(num_gaussians, 4, device=device)
    quaternions[:, 0] = 1.0  # w=1, x=y=z=0
    
    # Intensities: moderate initial values in log space
    log_intensities = torch.ones(num_gaussians, device=device) * np.log(0.1)
    
    # Make parameters
    means = torch.nn.Parameter(means)
    log_scales = torch.nn.Parameter(log_scales)
    quaternions = torch.nn.Parameter(quaternions)
    log_intensities = torch.nn.Parameter(log_intensities)
    
    print(f"  Means range: [{means.min().item():.3f}, {means.max().item():.3f}]")
    print(f"  Initial scale: {np.exp(log_scales[0, 0].item()):.4f}")
    print(f"  Initial intensity: {torch.sigmoid(log_intensities[0]).item():.4f}")
    
    return means, log_scales, quaternions, log_intensities


def initialize_gaussians_from_volume(
    volume: torch.Tensor,
    num_gaussians: int,
    vol_shape: tuple,
    device: torch.device,
    init_scale: float = 0.03,
    intensity_threshold: float = 0.1,
) -> tuple:
    """
    Initialize Gaussians near bright voxels in the volume.
    This prevents 2D collapse by starting with a proper 3D distribution.
    
    Returns:
        means, log_scales, quaternions, log_intensities (all require_grad=True)
    """
    print(f"\nInitializing {num_gaussians} Gaussians from volume structure...")
    
    Z, Y, X = vol_shape
    
    # Find bright voxels
    vol_flat = volume.flatten()
    bright_mask = vol_flat > intensity_threshold
    n_bright = bright_mask.sum().item()
    print(f"  Found {n_bright} voxels above threshold {intensity_threshold}")
    
    if n_bright < num_gaussians:
        # Fall back to sampling all voxels weighted by intensity
        print(f"  Using intensity-weighted sampling (not enough bright voxels)")
        weights = vol_flat.clamp(min=1e-6)
    else:
        # Sample only from bright voxels
        weights = vol_flat * bright_mask.float()
    
    # Sample voxel indices weighted by intensity
    # Use numpy to avoid torch.multinomial's 2^24 category limit
    weights_np = weights.cpu().numpy().astype(np.float64)
    weights_np = weights_np / weights_np.sum()  # Normalize to probability distribution
    indices_np = np.random.choice(len(weights_np), size=num_gaussians, replace=True, p=weights_np)
    indices = torch.from_numpy(indices_np).to(device)
    
    # Convert flat indices to 3D coordinates
    z_idx = indices // (Y * X)
    y_idx = (indices % (Y * X)) // X
    x_idx = indices % X
    
    # Convert to normalized coordinates [-1, 1]³ (uniform training space)
    # Aspect correction is applied later at render time by apply_aspect_correction()
    x_norm = (x_idx.float() / (X - 1) * 2 - 1)
    y_norm = (y_idx.float() / (Y - 1) * 2 - 1)
    z_norm = (z_idx.float() / (Z - 1) * 2 - 1)
    
    means = torch.stack([x_norm, y_norm, z_norm], dim=-1).to(device)
    
    # Add small jitter to avoid exact voxel centers
    means = means + torch.randn_like(means) * 0.01
    
    # Initialize scales - smaller for volume-based init since positions are better
    log_scales = torch.ones(num_gaussians, 3, device=device) * np.log(init_scale)
    
    # Quaternions: random rotations for variety
    quaternions = torch.randn(num_gaussians, 4, device=device)
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    
    # Intensities: initialize from sampled voxel values
    sampled_intensities = volume.flatten()[indices]
    # Convert to log-odds for sigmoid parametrization
    log_intensities = torch.log(sampled_intensities.clamp(0.01, 0.99) / (1 - sampled_intensities.clamp(0.01, 0.99)))
    log_intensities = log_intensities.to(device)
    
    # Make parameters
    means = torch.nn.Parameter(means)
    log_scales = torch.nn.Parameter(log_scales)
    quaternions = torch.nn.Parameter(quaternions)
    log_intensities = torch.nn.Parameter(log_intensities)
    
    print(f"  Means range: x=[{means[:,0].min().item():.3f}, {means[:,0].max().item():.3f}]")
    print(f"               y=[{means[:,1].min().item():.3f}, {means[:,1].max().item():.3f}]")
    print(f"               z=[{means[:,2].min().item():.3f}, {means[:,2].max().item():.3f}]")
    print(f"  Initial scale: {np.exp(log_scales[0, 0].item()):.4f}")
    print(f"  Intensity range: [{torch.sigmoid(log_intensities).min().item():.4f}, {torch.sigmoid(log_intensities).max().item():.4f}]")
    
    return means, log_scales, quaternions, log_intensities


def build_gaussians_from_params(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quaternions: torch.Tensor,
    log_intensities: torch.Tensor,
) -> GaussianParameters:
    """Build GaussianParameters from learnable parameters."""
    K = means.shape[0]
    device = means.device
    
    # Scales: exp and clamp
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    
    # Quaternions: normalize
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Build rotation matrices from quaternions
    R = torch.zeros(K, 3, 3, device=device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    # Build covariances: Σ = R @ S² @ Rᵀ
    S2 = torch.diag_embed(scales ** 2)
    covariances = R @ S2 @ R.transpose(-2, -1)
    
    # Intensities: sigmoid to [0, 1]
    intensities = torch.sigmoid(log_intensities)
    
    return GaussianParameters(
        means=means,
        covariances=covariances,
        intensities=intensities
    )


def generate_dataset(
    volume: torch.Tensor,
    camera: Camera,
    num_views: int,
    aspect_scales: torch.Tensor,
    n_ray_samples: int = 256,
    near: float = 0.5,
    far: float = 6.0,
    device: torch.device = None
) -> list:
    """Generate ground truth MIP projections from volume."""
    print(f"\nGenerating {num_views} GT MIP projections...")
    
    if device is None:
        device = volume.device
    
    # Generate camera poses - include views from below to prevent Z-collapse
    poses = generate_camera_poses(
        n_azimuth=max(num_views // 5, 4),
        n_elevation=min(num_views, 7),
        elevation_range=(-60.0, 75.0),  # Include negative elevations!
        radius=3.5,
        include_axis_aligned=True  # Add axis-aligned views for better coverage
    )[:num_views]
    
    dataset = []
    for i, pose_dict in enumerate(poses):
        R = pose_dict['R'].to(device)
        T = pose_dict['T'].to(device)
        
        # Render GT MIP
        gt_img = render_gt_mip(
            volume,
            camera,
            R,
            T,
            n_samples=n_ray_samples,
            near=near,
            far=far,
            aspect_scales=aspect_scales
        )
        
        dataset.append({
            'R': R,
            'T': T,
            'gt_mip': gt_img
        })
        
        if (i + 1) % max(num_views // 5, 1) == 0:
            print(f"  Generated {i+1}/{num_views} views...")
    
    print(f"  Dataset ready: {len(dataset)} views")
    return dataset


# =============================================================================
# Densification and Pruning
# =============================================================================

def prune_gaussians(
    means, log_scales, quaternions, log_intensities,
    grad_accum, grad_count,
    intensity_thresh: float = 0.01,
    min_gaussians: int = 100,
    step: int = 0
):
    """Prune Gaussians with low intensity."""
    with torch.no_grad():
        intensities = torch.sigmoid(log_intensities)
        keep = intensities > intensity_thresh
        n_before = keep.shape[0]
        n_keep = int(keep.sum().item())
        
        if n_keep >= n_before or n_keep < min_gaussians:
            return means, log_scales, quaternions, log_intensities, grad_accum, grad_count, 0
        
        means_new = torch.nn.Parameter(means.data[keep].clone())
        log_scales_new = torch.nn.Parameter(log_scales.data[keep].clone())
        quaternions_new = torch.nn.Parameter(quaternions.data[keep].clone())
        log_intensities_new = torch.nn.Parameter(log_intensities.data[keep].clone())
        grad_accum_new = grad_accum[keep].clone()
        grad_count_new = grad_count[keep].clone()
        
        n_pruned = n_before - n_keep
        print(f"  [Prune @ step {step}] {n_before} → {n_keep} (removed {n_pruned})")
        
        return means_new, log_scales_new, quaternions_new, log_intensities_new, grad_accum_new, grad_count_new, n_pruned


def densify_and_prune(
    means, log_scales, quaternions, log_intensities,
    grad_accum, grad_count,
    grad_thresh: float = 0.001,
    scale_thresh: float = 0.05,
    intensity_thresh: float = 0.01,
    max_gaussians: int = 50000,
    min_gaussians: int = 100,
    split_factor: float = 1.6,
    max_clone_per_step: int = 200,
    step: int = 0
):
    """Densify (split/clone) and prune Gaussians based on gradients."""
    with torch.no_grad():
        K = means.shape[0]
        device = means.device
        
        # Average gradient magnitude
        avg_grad = grad_accum / grad_count.clamp(min=1)
        high_grad = avg_grad > grad_thresh
        
        # Get max scale per Gaussian
        scales = torch.exp(log_scales).clamp(1e-5, 1e2)
        max_scale = scales.max(dim=-1).values
        
        # Split mask: high gradient AND large scale
        split_mask = high_grad & (max_scale > scale_thresh)
        # Clone mask: high gradient AND small scale
        clone_mask = high_grad & ~split_mask
        
        n_split = int(split_mask.sum().item())
        n_clone = int(clone_mask.sum().item())
        
        # Cap clones per step to prevent explosion
        if n_clone > max_clone_per_step:
            clone_indices = torch.where(clone_mask)[0]
            # Keep only top-gradient candidates
            clone_grads = avg_grad[clone_mask]
            _, top_idx = torch.topk(clone_grads, max_clone_per_step)
            new_clone_mask = torch.zeros_like(clone_mask)
            new_clone_mask[clone_indices[top_idx]] = True
            clone_mask = new_clone_mask
            n_clone = max_clone_per_step
        
        # Check if we'd exceed max - skip densification but STILL do pruning
        skip_densify = (K + n_split + n_clone > max_gaussians)
        if skip_densify:
            n_split = 0
            n_clone = 0
            split_mask = torch.zeros_like(split_mask)
            clone_mask = torch.zeros_like(clone_mask)
        
        new_m, new_ls, new_q, new_li = [], [], [], []
        
        # Split: create 2 smaller Gaussians along principal axis
        if n_split > 0:
            s_m = means.data[split_mask]
            s_ls = log_scales.data[split_mask]
            s_q = quaternions.data[split_mask]
            s_li = log_intensities.data[split_mask]
            reduced_ls = s_ls - math.log(split_factor)
            s_scales = torch.exp(s_ls)
            
            # Build rotation matrix to find principal axis
            q = F.normalize(s_q, p=2, dim=-1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            col0 = torch.stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)], dim=-1)
            col1 = torch.stack([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)], dim=-1)
            col2 = torch.stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)], dim=-1)
            R_cols = torch.stack([col0, col1, col2], dim=-1)
            
            # Find principal (largest) axis
            max_axis = s_scales.argmax(dim=-1)
            principal = R_cols[torch.arange(len(max_axis), device=device), :, max_axis]
            offset = principal * s_scales.max(dim=-1, keepdim=True).values
            
            for sign in (1.0, -1.0):
                new_m.append(s_m + sign * offset)
                new_ls.append(reduced_ls.clone())
                new_q.append(s_q.clone())
                new_li.append(s_li.clone())
        
        # Clone: duplicate Gaussians in place
        if n_clone > 0:
            new_m.append(means.data[clone_mask].clone())
            new_ls.append(log_scales.data[clone_mask].clone())
            new_q.append(quaternions.data[clone_mask].clone())
            new_li.append(log_intensities.data[clone_mask].clone())
        
        # Combine: keep non-split, add new
        keep_mask = ~split_mask
        all_m = torch.cat([means.data[keep_mask]] + new_m, dim=0)
        all_ls = torch.cat([log_scales.data[keep_mask]] + new_ls, dim=0)
        all_q = torch.cat([quaternions.data[keep_mask]] + new_q, dim=0)
        all_li = torch.cat([log_intensities.data[keep_mask]] + new_li, dim=0)
        
        # Prune low intensity
        intensities = torch.sigmoid(all_li)
        alive = intensities > intensity_thresh
        if alive.sum() < min_gaussians:
            alive = torch.ones(all_m.shape[0], dtype=torch.bool, device=device)
        n_pruned = int((~alive).sum().item())
        
        means_new = torch.nn.Parameter(all_m[alive])
        log_scales_new = torch.nn.Parameter(all_ls[alive])
        quaternions_new = torch.nn.Parameter(all_q[alive])
        log_intensities_new = torch.nn.Parameter(all_li[alive])
        
        # New gradient accumulators
        K_new = means_new.shape[0]
        grad_accum_new = torch.zeros(K_new, device=device)
        grad_count_new = torch.zeros(K_new, device=device)
        
        print(f"  [Densify @ step {step}] K: {K} → {K_new} (split {n_split}, clone {n_clone}, pruned {n_pruned})")
        
        return means_new, log_scales_new, quaternions_new, log_intensities_new, grad_accum_new, grad_count_new, n_split, n_clone, n_pruned


def train_step(
    means, log_scales, quaternions, log_intensities,
    dataset_batch,
    camera,
    beta,
    optimizer,
    loss_weights: dict = None,
    lambda_scale_reg: float = 0.01  # Scale anisotropy regularization
):
    """Single training step over a batch of views."""
    if loss_weights is None:
        loss_weights = {'mse': 1.0}
    
    optimizer.zero_grad()
    
    # Build Gaussians from current parameters
    gaussians = build_gaussians_from_params(means, log_scales, quaternions, log_intensities)
    
    total_loss = 0.0
    loss_components = {}
    
    # Render each view and accumulate loss
    for view in dataset_batch:
        R = view['R']
        T = view['T']
        gt_mip = view['gt_mip']
        
        # Render with current Gaussians
        pred_mip, n_vis = render_mip_projection(
            gaussians,
            camera,
            R,
            T,
            beta=beta
        )
        
        # Compute loss
        if 'mse' in loss_weights:
            loss_mse = mse_loss(pred_mip, gt_mip)
            total_loss += loss_weights['mse'] * loss_mse
            loss_components['mse'] = loss_mse.item()
        
        if 'wmse' in loss_weights:
            loss_wmse = weighted_mse_loss(pred_mip, gt_mip, fg_weight=5.0)
            total_loss += loss_weights['wmse'] * loss_wmse
            loss_components['wmse'] = loss_wmse.item()
        
        if 'ssim' in loss_weights:
            loss_ssim = 1.0 - ssim_loss_fn(pred_mip, gt_mip)
            total_loss += loss_weights['ssim'] * loss_ssim
            loss_components['ssim'] = loss_ssim.item()
    
    # Scale anisotropy regularization - penalize flat Gaussians (prevents 2D collapse)
    if lambda_scale_reg > 0:
        scales = torch.exp(log_scales).clamp(1e-5, 1e2)
        # Ratio of max to min scale: 1.0 = isotropic, >1 = anisotropic/flat
        scale_ratio = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp(min=1e-6)
        scale_reg = (scale_ratio - 1.0).mean()  # Penalize deviation from isotropic
        total_loss = total_loss + lambda_scale_reg * scale_reg
        loss_components['scale_reg'] = scale_reg.item()
    
    # Backward pass
    total_loss.backward()
    
    # Capture gradient magnitude on means for densification
    grad_magnitude = None
    if means.grad is not None:
        grad_magnitude = means.grad.norm(dim=-1).detach()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([means, log_scales, quaternions, log_intensities], max_norm=1.0)
    
    optimizer.step()
    
    return total_loss.item(), loss_components, n_vis, grad_magnitude


def train_pipeline(
    volume: torch.Tensor,
    num_gaussians: int,
    num_steps: int,
    num_views: int,
    batch_size: int,
    lr: float,
    beta: float,
    vol_shape: tuple,
    device: torch.device,
    checkpoint_dir: str = None,
    save_every: int = 500,
    # Densification/Pruning parameters
    densify_every: int = 100,
    densify_start: int = 500,
    densify_stop: int = 15000,
    grad_thresh: float = 0.001,
    scale_thresh: float = 0.05,
    intensity_thresh: float = 0.01,
    max_gaussians: int = 50000,
    split_factor: float = 1.6,
    init_from_volume: bool = True,  # Use volume-informed initialization
    lambda_scale_reg: float = 0.01,  # Scale anisotropy regularization
):
    """Full training pipeline."""
    print("\n" + "="*60)
    print("MIP SPLATTING TRAINING PIPELINE")
    print("="*60)
    
    # Initialize Gaussians
    if init_from_volume:
        means, log_scales, quaternions, log_intensities = initialize_gaussians_from_volume(
            volume, num_gaussians, vol_shape, device
        )
    else:
        means, log_scales, quaternions, log_intensities = initialize_gaussians_random(
            num_gaussians, vol_shape, device
        )
    
    # Compute aspect correction
    aspect_scales = compute_aspect_scales(vol_shape).to(device)
    print(f"\nAspect scales (X,Y,Z): {aspect_scales.tolist()}")
    
    # Setup camera
    Z, Y, X = vol_shape
    camera = Camera.from_fov(
        fov_x_deg=50.0,
        width=int(X),
        height=int(Y),
        near=0.01,
        far=1000.0
    )
    print(f"\nCamera: {camera.width}×{camera.height}, FOV={50.0}°")
    
    # Generate dataset
    dataset = generate_dataset(
        volume, camera, num_views, aspect_scales,
        n_ray_samples=256, near=0.5, far=6.0, device=device
    )
    
    # Setup optimizer
    params = [means, log_scales, quaternions, log_intensities]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01
    )
    
    # Gradient accumulators for densification
    grad_accum = torch.zeros(num_gaussians, device=device)
    grad_count = torch.zeros(num_gaussians, device=device)
    
    # Training loop
    print(f"\n" + "="*60)
    print(f"TRAINING: {num_steps} steps, batch_size={batch_size}")
    print(f"Densify: every {densify_every} steps, from step {densify_start} to {densify_stop}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch of views
        batch_indices = torch.randperm(len(dataset))[:batch_size]
        batch = [dataset[i] for i in batch_indices]
        
        # Training step
        loss, loss_comp, n_vis, grad_mag = train_step(
            means, log_scales, quaternions, log_intensities,
            batch, camera, beta, optimizer,
            loss_weights={'wmse': 1.0, 'ssim': 0.2},
            lambda_scale_reg=lambda_scale_reg
        )
        
        # Accumulate gradients for densification
        if grad_mag is not None and len(grad_mag) == len(grad_accum):
            grad_accum += grad_mag
            grad_count += 1
        
        scheduler.step()
        
        # Densification and pruning
        if step >= densify_start and step < densify_stop and step % densify_every == 0:
            means, log_scales, quaternions, log_intensities, grad_accum, grad_count, n_split, n_clone, n_pruned = densify_and_prune(
                means, log_scales, quaternions, log_intensities,
                grad_accum, grad_count,
                grad_thresh=grad_thresh,
                scale_thresh=scale_thresh,
                intensity_thresh=intensity_thresh,
                max_gaussians=max_gaussians,
                min_gaussians=100,
                split_factor=split_factor,
                step=step
            )
            # Rebuild optimizer with new parameters
            params = [means, log_scales, quaternions, log_intensities]
            optimizer = torch.optim.Adam(params, lr=scheduler.get_last_lr()[0], betas=(0.9, 0.999))
            # Update num_gaussians for logging
            num_gaussians = means.shape[0]
        
        # Logging
        if step % 10 == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (num_steps - step - 1)
            
            print(f"Step {step:5d}/{num_steps} | "
                  f"Loss (wMSE): {loss:.6f} | "
                  f"Vis: {n_vis}/{num_gaussians} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA: {eta/60:.1f}min")
        
        # Save checkpoint
        if checkpoint_dir and (step % save_every == 0 or step == num_steps - 1):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
            torch.save({
                'step': step,
                'means': means.detach().cpu(),
                'log_scales': log_scales.detach().cpu(),
                'quaternions': quaternions.detach().cpu(),
                'log_intensities': log_intensities.detach().cpu(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")
    
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Final loss: {loss:.6f}")
    print("="*60 + "\n")
    
    # Save final model
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save({
            'step': num_steps,
            'means': means.detach().cpu(),
            'log_scales': log_scales.detach().cpu(),
            'quaternions': quaternions.detach().cpu(),
            'log_intensities': log_intensities.detach().cpu(),
            'vol_shape': vol_shape,
            'num_gaussians': num_gaussians,
        }, final_path)
        print(f"Final model saved: {final_path}")
        return final_path
    
    return None


def launch_viewer(checkpoint_path: str, port: int = 8090, beta: float = 50.0):
    """Launch interactive viewer with trained model."""
    print(f"\nLaunching interactive viewer...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Port: {port}")
    
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "interactive_mip_viewer.py"),
        "--ckpt", checkpoint_path,
        "--port", str(port),
        "--beta", str(beta)
    ]
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="End-to-End MIP Splatting Pipeline")
    parser.add_argument("--volume", required=True, help="Path to volume TIFF file")
    parser.add_argument("--num_gaussians", type=int, default=10000, help="Number of Gaussians")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--num_views", type=int, default=50, help="Number of training views")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (views per step)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta", type=float, default=50.0, help="MIP temperature")
    parser.add_argument("--output_dir", default="./checkpoints/pipeline", help="Output directory")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    # Densification args
    parser.add_argument("--densify_every", type=int, default=100, help="Densify every N steps")
    parser.add_argument("--densify_start", type=int, default=500, help="Start densification at step")
    parser.add_argument("--densify_stop", type=int, default=15000, help="Stop densification at step")
    parser.add_argument("--grad_thresh", type=float, default=0.001, help="Gradient threshold for densification")
    parser.add_argument("--scale_thresh", type=float, default=0.05, help="Scale threshold for split vs clone")
    parser.add_argument("--intensity_thresh", type=float, default=0.01, help="Intensity threshold for pruning")
    parser.add_argument("--max_gaussians", type=int, default=50000, help="Maximum number of Gaussians")
    parser.add_argument("--init_random", action="store_true", help="Use random init instead of volume-informed")
    parser.add_argument("--lambda_scale_reg", type=float, default=0.01, help="Scale anisotropy regularization")
    parser.add_argument("--launch_viewer", action="store_true", help="Launch viewer after training")
    parser.add_argument("--viewer_port", type=int, default=8090, help="Viewer port")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be slow!")
    
    # Load volume
    print(f"\nLoading volume: {args.volume}")
    vol_np = load_volume(args.volume)
    vol_tensor = torch.from_numpy(vol_np).to(device)
    vol_shape = vol_np.shape
    print(f"  Shape: {vol_shape}")
    print(f"  Range: [{vol_np.min():.3f}, {vol_np.max():.3f}]")
    print(f"  Memory: {vol_tensor.element_size() * vol_tensor.nelement() / 1e9:.2f} GB")
    
    # Train
    checkpoint_path = train_pipeline(
        volume=vol_tensor,
        num_gaussians=args.num_gaussians,
        num_steps=args.steps,
        num_views=args.num_views,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        vol_shape=vol_shape,
        device=device,
        checkpoint_dir=args.output_dir,
        save_every=args.save_every,
        densify_every=args.densify_every,
        densify_start=args.densify_start,
        densify_stop=args.densify_stop,
        grad_thresh=args.grad_thresh,
        scale_thresh=args.scale_thresh,
        intensity_thresh=args.intensity_thresh,
        max_gaussians=args.max_gaussians,
        init_from_volume=not args.init_random,
        lambda_scale_reg=args.lambda_scale_reg,
    )
    
    # Launch viewer if requested
    if args.launch_viewer and checkpoint_path:
        launch_viewer(checkpoint_path, args.viewer_port, args.beta)
    else:
        print(f"\nTo view the trained model, run:")
        print(f"  python interactive_mip_viewer.py --ckpt {checkpoint_path} --port {args.viewer_port}")


if __name__ == "__main__":
    main()
