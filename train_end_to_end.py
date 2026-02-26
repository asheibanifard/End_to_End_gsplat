#!/usr/bin/env python3
"""
End-to-End Training: NeuRoGS v7 → MIP Renderer
==============================================

Optimizes 3D Gaussian parameters in NeuRoGS v7 model based on
multi-view MIP rendering supervision.

Architecture:
    [3D Gaussians] → [MIP Renderer] → [Multi-view Projections] → [L1 + SSIM Loss]
         ↑_____________________ gradient descent _____________________________↓

The 3D Gaussians are parameterized by:
    - means:          (K, 3)   centers in 3D space
    - log_scales:     (K, 3)   anisotropic sizes
    - quaternions:    (K, 4)   rotations
    - log_amplitudes: (K,)     emission intensities

Loss:
    L = (1-λ_dssim) * L1(rendered, target) + λ_dssim * (1 - SSIM(rendered, target))
    Uses photometric_loss from losses.py
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

# Add neurogs_v7 to path
sys.path.insert(0, str(Path(__file__).parent / "neurogs" / "neurogs_v7"))
from neurogs_v7 import GaussianMixtureField

# Import renderer
from renderer import Camera, render

# Import loss functions
from losses import photometric_loss, opacity_sparsity_loss, scale_regularisation


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss computation uses photometric_loss from losses.py
# ═══════════════════════════════════════════════════════════════════════════════
# The photometric_loss combines L1 + (1 - SSIM) per channel.
# lambda_dssim controls the SSIM weight: loss = (1-λ)*L1 + λ*(1-SSIM)
# With lambda_dssim=1.0, it's pure SSIM loss (1-SSIM)
# With lambda_dssim=0.0, it's pure L1 loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Gaussian Adapter (NeuRoGS → Renderer)
# ═══════════════════════════════════════════════════════════════════════════════

class NeuRoGStoRenderer:
    """
    Adapter to convert GaussianMixtureField parameters to renderer format.
    
    The renderer expects:
        - means:     [N, 3]
        - cov3d:     [N, 6] (upper triangle: xx, xy, xz, yy, yz, zz)
        - intensity: [N, C]  emission strength
    
    NeuRoGS v7 provides:
        - means:          [K, 3]
        - log_scales:     [K, 3]
        - quaternions:    [K, 4]
        - log_amplitudes: [K]
    """
    
    def __init__(self, gaussian_field: GaussianMixtureField):
        self.field = gaussian_field
        # Fix normalization bounds at initialization to prevent coordinate drift
        with torch.no_grad():
            means_raw = self.field.means
            self.norm_min = means_raw.min(dim=0)[0]
            self.norm_max = means_raw.max(dim=0)[0]
            self.norm_range = (self.norm_max - self.norm_min).clamp(min=1e-6)
            self.norm_scale = self.norm_range.mean()
    
    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R
    
    def build_cov3d_torch(self) -> torch.Tensor:
        """
        Build 3D covariance matrices from scales and quaternions.
        Returns upper triangle [N, 6]: [xx, xy, xz, yy, yz, zz]
        """
        scales = torch.exp(self.field.log_scales).clamp(1e-5, 1e2)  # [K, 3]
        R = self.quat_to_rotmat(self.field.quaternions)  # [K, 3, 3]
        
        # Σ = R @ diag(s²) @ R^T
        S2_diag = scales ** 2  # [K, 3]
        S2 = torch.diag_embed(S2_diag)  # [K, 3, 3]
        Sigma = R @ S2 @ R.transpose(-2, -1)  # [K, 3, 3]
        
        # Extract upper triangle
        cov3d = torch.stack([
            Sigma[:, 0, 0],  # xx
            Sigma[:, 0, 1],  # xy
            Sigma[:, 0, 2],  # xz
            Sigma[:, 1, 1],  # yy
            Sigma[:, 1, 2],  # yz
            Sigma[:, 2, 2],  # zz
        ], dim=-1)  # [K, 6]
        
        return cov3d
    
    def get_renderer_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract parameters needed for rendering.
        
        Returns
        -------
        means : torch.Tensor
            Gaussian centers [K, 3] normalized to [0, 1]
        cov3d : torch.Tensor
            Covariance matrices [K, 6] (upper triangle)
        intensity : torch.Tensor
            Emission intensities [K, 1]
        """
        # Get raw means and normalize using FIXED bounds from initialization
        means_raw = self.field.means  # [K, 3]
        means = (means_raw - self.norm_min) / self.norm_range  # [K, 3] in [0, 1]
        
        # Covariances scaled by FIXED normalization factor
        cov3d = self.build_cov3d_torch()  # [K, 6]
        cov3d = cov3d / (self.norm_scale ** 2)
        
        intensity = torch.exp(self.field.log_amplitudes).unsqueeze(-1)  # [K, 1]
        
        return means, cov3d, intensity


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera Generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_multiview_cameras(
    H: int,
    W: int,
    num_views: int = 4,
    fov_deg: float = 60.0,
    device: str = "cuda"
) -> List[Tuple[Camera, torch.Tensor]]:
    """
    Create multiple camera views around the volume.
    
    Parameters
    ----------
    H, W : int
        Image dimensions
    num_views : int
        Number of camera views
    fov_deg : float
        Field of view in degrees
    device : str
        Device to create tensors on
    
    Returns
    -------
    cameras : List[Tuple[Camera, torch.Tensor]]
        List of (camera, view_matrix) pairs
    """
    cameras = []
    
    # Standard orthographic views
    cam = Camera(H, W, fov_deg=fov_deg, device=device)
    
    # Top-down (Z-axis)
    view_mat_z = cam.view_matrix_orthographic(axis=2)
    cameras.append((cam, view_mat_z))
    
    # Side views (X-axis, Y-axis)
    if num_views >= 2:
        view_mat_x = cam.view_matrix_orthographic(axis=0)
        cameras.append((cam, view_mat_x))
    
    if num_views >= 3:
        view_mat_y = cam.view_matrix_orthographic(axis=1)
        cameras.append((cam, view_mat_y))
    
    # Additional perspective views if needed
    if num_views >= 4:
        angles = np.linspace(0, 2 * np.pi, num_views - 2, endpoint=False)
        radius = 2.0
        for angle in angles[-(num_views - 3):]:
            eye = torch.tensor([
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle),
                0.5 + radius * 0.5
            ], device=device)
            view_mat = cam.view_matrix_perspective(eye)
            cameras.append((cam, view_mat))
    
    return cameras


# ═══════════════════════════════════════════════════════════════════════════════
#  Synthetic Target Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_targets(
    gaussian_field: GaussianMixtureField,
    cameras: List[Tuple[Camera, torch.Tensor]],
    adapter: NeuRoGStoRenderer,
    add_noise: float = 0.0
) -> List[torch.Tensor]:
    """
    Generate synthetic target projections from initial Gaussian field.
    
    This creates "ground truth" MIP projections that we'll optimize towards.
    """
    targets = []
    
    with torch.no_grad():
        means, cov3d, intensity = adapter.get_renderer_params()
        
        print(f"  Target generation params:")
        print(f"    Means range: [{means.min():.3f}, {means.max():.3f}]")
        print(f"    Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
        
        # Create temporary gaussians object
        class TempGaussians:
            def __init__(self, m, cov, inten):
                self.means = m
                self.intensity = inten
                self.N = m.shape[0]
                self.quats = torch.zeros(self.N, 4, device=m.device)
                self.quats[:, 0] = 1.0
                self._log_scales = torch.zeros(self.N, 3, device=m.device)
                self._cov3d = cov
            
            def build_cov3d_torch(self):
                return self._cov3d
        
        gaussians = TempGaussians(means, cov3d, intensity)
        
        for camera, view_mat in cameras:
            img, _, _ = render(gaussians, camera, view_mat, use_cuda_cov=False)
            
            if add_noise > 0:
                noise = torch.randn_like(img) * add_noise
                img = (img + noise).clamp(0, 1)
            
            targets.append(img.detach())
    
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_end_to_end(
    gaussian_field: GaussianMixtureField,
    cameras: List[Tuple[Camera, torch.Tensor]],
    targets: List[torch.Tensor],
    num_iterations: int = 1000,
    lr: float = 1e-3,
    lambda_mse: float = 0.0,
    lambda_ssim: float = 1.0,
    ssim_warmup: int = 100,
    mse_fadeout: int = 0,
    device: str = "cuda"
):
    """
    Train the Gaussian field end-to-end using multi-view MIP rendering.
    
    Parameters
    ----------
    gaussian_field : GaussianMixtureField
        The 3D Gaussian model to optimize
    cameras : List[Tuple[Camera, torch.Tensor]]
        List of (camera, view_matrix) pairs
    targets : List[torch.Tensor]
        Target MIP projections for each view
    num_iterations : int
        Number of training iterations
    lr : float
        Learning rate
    lambda_mse : float
        Weight for MSE loss
    lambda_ssim : float
        Weight for SSIM loss
    device : str
        Device to train on
    """
    
    # Move model to device
    gaussian_field = gaussian_field.to(device)
    
    # Create adapter
    adapter = NeuRoGStoRenderer(gaussian_field)
    
    # Setup optimizer with learning rate scheduler
    optimizer = torch.optim.Adam([
        {'params': gaussian_field.means, 'lr': lr},
        {'params': gaussian_field.log_scales, 'lr': lr * 0.1},
        {'params': gaussian_field.quaternions, 'lr': lr * 0.1},
        {'params': gaussian_field.log_amplitudes, 'lr': lr},
    ])
    
    # Learning rate scheduler: exponential decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    # Training loop
    pbar = tqdm(range(num_iterations), desc="Training")
    
    # Learning rate warmup parameters
    lr_warmup_iters = min(50, num_iterations // 10)  # Warmup for first 10% of training
    base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    
    for iteration in pbar:
        # Learning rate warmup: linearly increase LR from 0 to base_lr
        if iteration < lr_warmup_iters:
            warmup_factor = (iteration + 1) / lr_warmup_iters
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = base_lrs[i] * warmup_factor
        
        total_loss = 0.0
        mse_total = 0.0
        ssim_total = 0.0
        
        # SSIM warmup schedule: gradually increase SSIM weight
        if ssim_warmup > 0 and iteration < ssim_warmup:
            ssim_weight = lambda_ssim * (iteration / ssim_warmup)
        else:
            ssim_weight = lambda_ssim
        
        # MSE fadeout: gradually decrease MSE weight to emphasize structure
        if mse_fadeout > 0 and iteration >= mse_fadeout:
            fade_progress = min(1.0, (iteration - mse_fadeout) / (num_iterations - mse_fadeout))
            mse_weight = lambda_mse * (1.0 - fade_progress)
        else:
            mse_weight = lambda_mse
        
        # Get current renderer parameters
        means, cov3d, intensity = adapter.get_renderer_params()
        
        # Create temporary gaussians object for renderer
        class TempGaussians:
            def __init__(self, m, cov, inten):
                self.means = m
                self.intensity = inten
                self.N = m.shape[0]
                self.quats = torch.zeros(self.N, 4, device=m.device)
                self.quats[:, 0] = 1.0
                self._log_scales = torch.zeros(self.N, 3, device=m.device)
                self._cov3d = cov
            
            def build_cov3d_torch(self):
                return self._cov3d
        
        gaussians = TempGaussians(means, cov3d, intensity)
        
        # Render all views and compute loss
        for view_idx, ((camera, view_mat), target) in enumerate(zip(cameras, targets)):
            # Render current state
            rendered, _, _ = render(gaussians, camera, view_mat, use_cuda_cov=False)
            
            # Debug first iteration
            if iteration == 0:
                print(f"    View {view_idx}: rendered range=[{rendered.min():.3f}, {rendered.max():.3f}], "
                      f"target range=[{target.min():.3f}, {target.max():.3f}]")
            
            # Compute loss using photometric_loss (L1 + SSIM)
            # Map our lambda_ssim to losses.py lambda_dssim parameter
            # If we want pure SSIM (lambda_mse=0, lambda_ssim=1), use lambda_dssim=1.0
            # If we want L1+SSIM mix, adjust lambda_dssim accordingly
            if ssim_weight > 0 and mse_weight == 0:
                # Pure SSIM mode
                lambda_dssim = 1.0
            elif ssim_weight == 0 and mse_weight > 0:
                # Pure L1 mode
                lambda_dssim = 0.0
            else:
                # Mixed mode: balance L1 and SSIM
                total_weight = mse_weight + ssim_weight
                lambda_dssim = ssim_weight / total_weight if total_weight > 0 else 0.2
            
            loss, loss_dict = photometric_loss(rendered, target, lambda_dssim=lambda_dssim)
            
            # Scale by warmup/fadeout weights
            combined_weight = max(mse_weight, ssim_weight) if (mse_weight + ssim_weight) > 0 else 1.0
            loss = loss * combined_weight
            
            # Accumulate
            total_loss += loss
            mse_total += loss_dict.get('l1', 0.0)  # L1 loss instead of MSE
            ssim_total += loss_dict.get('ssim', 0.0)
        
        # Add regularization losses for stability
        # Opacity sparsity: encourage clear opacity values
        opacity_loss = opacity_sparsity_loss(
            torch.sigmoid(gaussian_field.log_amplitudes.unsqueeze(-1)),
            target_mean=0.15
        )
        
        # Scale regularization: keep Gaussian sizes reasonable
        scale_loss = scale_regularisation(
            gaussian_field.log_scales,
            min_log_s=-8.0,
            max_log_s=-1.0
        )
        
        # Add regularization to total loss (small weights)
        reg_loss = 0.001 * opacity_loss + 0.001 * scale_loss
        total_loss = total_loss + reg_loss
        
        # Average over views
        total_loss = total_loss / len(cameras)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(gaussian_field.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Apply constraints
        with torch.no_grad():
            gaussian_field.apply_aabb_clamp(margin=0.0)
            gaussian_field.clamp_log_scales_(-8.0, -1.0)
            gaussian_field.clamp_log_amplitudes_(-10.0, 6.0)
        
        # Logging
        if iteration % 10 == 0:
            ssim_similarity = ssim_total / len(cameras)  # SSIM similarity (higher is better)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'l1': f'{mse_total / len(cameras):.4f}',
                'ssim': f'{ssim_similarity:.3f}',
                'lr': f'{current_lr:.1e}',
                'λ_dssim': f'{lambda_dssim:.2f}'
            })
        
        # Save checkpoint
        if iteration % 500 == 0 and iteration > 0:
            checkpoint_path = f"checkpoint_iter{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'model_state_dict': gaussian_field.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_path)
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n✓ Training complete!")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training: NeuRoGS v7 → MIP Renderer"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to NeuRoGS v7 checkpoint (or None to initialize new)"
    )
    parser.add_argument(
        "--num_gaussians",
        type=int,
        default=2000,
        help="Number of Gaussians (if initializing new model)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=0.0,
        help="Weight for MSE loss (0.0 = disabled, recommended to avoid structural degradation)"
    )
    parser.add_argument(
        "--lambda_ssim",
        type=float,
        default=1.0,
        help="Weight for SSIM loss (structural similarity - primary loss)"
    )
    parser.add_argument(
        "--ssim_warmup",
        type=int,
        default=100,
        help="Number of iterations to warmup SSIM loss (0 to disable)"
    )
    parser.add_argument(
        "--mse_fadeout",
        type=int,
        default=0,
        help="Iteration to start fading out MSE loss (0 to disable, helps preserve structure)"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=3,
        help="Number of projection views [1-6]"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image resolution [H W]"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    device = args.device
    H, W = args.resolution
    
    print("═" * 70)
    print("End-to-End Training: NeuRoGS v7 → MIP Renderer")
    print("═" * 70)
    
    # Initialize or load Gaussian field
    if args.checkpoint:
        print(f"\n→ Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Check if it's a wrapped state_dict or direct tensors
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            K = state_dict['means'].shape[0]
            aabb = checkpoint.get('aabb', torch.tensor([[0., 1.], [0., 1.], [0., 1.]]))
        else:
            # Direct tensor format (NeuRoGS v7 checkpoint)
            state_dict = checkpoint
            K = state_dict['means'].shape[0]
            aabb = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        
        gaussian_field = GaussianMixtureField(
            num_gaussians=K,
            init_amplitude=0.1,
            aabb=aabb
        )
        gaussian_field.load_state_dict(state_dict)
        print(f"  ✓ Loaded {K} Gaussians")
    else:
        print(f"\n→ Initializing new model with {args.num_gaussians} Gaussians")
        gaussian_field = GaussianMixtureField(
            num_gaussians=args.num_gaussians,
            init_amplitude=0.1,
            aabb=torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        )
    
    gaussian_field = gaussian_field.to(device)
    
    # Create multi-view cameras
    print(f"\n→ Creating {args.num_views} camera views")
    cameras = create_multiview_cameras(
        H=H, W=W,
        num_views=args.num_views,
        device=device
    )
    print(f"  ✓ {len(cameras)} cameras created")
    
    # Generate synthetic targets
    print(f"\n→ Generating synthetic target projections")
    adapter = NeuRoGStoRenderer(gaussian_field)
    targets = generate_synthetic_targets(
        gaussian_field,
        cameras,
        adapter,
        add_noise=0.01
    )
    print(f"  ✓ {len(targets)} targets generated, shape: {targets[0].shape}")
    
    # Train
    print(f"\n→ Starting training for {args.iterations} iterations")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • λ_MSE: {args.lambda_mse}, λ_SSIM: {args.lambda_ssim}")
    print(f"  • SSIM warmup: {args.ssim_warmup} iterations")
    if args.mse_fadeout > 0:
        print(f"  • MSE fadeout: starts at iteration {args.mse_fadeout}")
    print()
    
    train_end_to_end(
        gaussian_field=gaussian_field,
        cameras=cameras,
        targets=targets,
        num_iterations=args.iterations,
        lr=args.lr,
        lambda_mse=args.lambda_mse,
        lambda_ssim=args.lambda_ssim,
        ssim_warmup=args.ssim_warmup,
        mse_fadeout=args.mse_fadeout,
        device=device
    )
    
    # Save final model
    final_path = "neurogs_renderer_optimized.pt"
    torch.save({
        'model_state_dict': gaussian_field.state_dict(),
        'num_gaussians': gaussian_field.num_gaussians,
        'aabb': gaussian_field.aabb,
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}")


if __name__ == "__main__":
    main()
