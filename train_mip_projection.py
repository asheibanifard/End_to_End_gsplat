"""
End-to-End MIP Projection Training
====================================

Optimizes 3D Gaussian mixture field by comparing rendered MIP projections 
with ground truth MIP projections from the input volume.

Architecture:
1. 3D Gaussians (from neurogs_v7.py GaussianMixtureField)
2. MIP Splatting Renderer (from renderer.py)
3. Multi-view projection loss

At each iteration:
- Render N MIP projections from different viewpoints
- Compare with GT MIP projections
- Optimize 3D Gaussians based on projection errors
"""

from __future__ import annotations
import os, sys, math, time, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Add neurogs_v7 to path
sys.path.insert(0, str(Path(__file__).parent / "neurogs" / "neurogs_v7"))
from neurogs_v7 import GaussianMixtureField

# Local modules
from data_io import load_tif_volume, load_swc, normalise_swc_to_voxel
from renderer import Camera, render, CUDA_AVAILABLE
from config import load_config


# ═══════════════════════════════════════════════════════════════════════════
#  Convert GaussianMixtureField to renderer-compatible format
# ═══════════════════════════════════════════════════════════════════════════

class GaussianFieldToRenderer:
    """Adapter to make GaussianMixtureField compatible with MIP splatting renderer."""
    
    def __init__(self, field: GaussianMixtureField):
        self.field = field
    
    @property
    def N(self) -> int:
        return self.field.num_gaussians
    
    @property
    def means(self) -> torch.Tensor:
        """[N, 3] positions"""
        return self.field.means
    
    @property
    def quats(self) -> torch.Tensor:
        """[N, 4] normalized quaternions"""
        return F.normalize(self.field.quaternions, p=2, dim=-1)
    
    @property
    def _log_scales(self) -> torch.Tensor:
        """[N, 3] log scales"""
        return self.field.log_scales
    
    @property
    def scales(self) -> torch.Tensor:
        """[N, 3] scales"""
        return torch.exp(self.field.log_scales).clamp(1e-5, 1e2)
    
    @property
    def intensity(self) -> torch.Tensor:
        """[N, C] emission intensity - derived from amplitudes"""
        amp = torch.exp(self.field.log_amplitudes).clamp(0, 1)  # [N]
        return amp.unsqueeze(-1)  # [N, 1] for grayscale
    
    def build_cov3d_torch(self) -> torch.Tensor:
        """Build 3D covariance matrices [N, 6] upper triangle"""
        cov = self.field.get_covariance_matrices()  # [N, 3, 3]
        # Extract upper triangle: [c00, c01, c02, c11, c12, c22]
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
        ], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Ground-truth MIP generator
# ═══════════════════════════════════════════════════════════════════════════

def compute_gt_mips(volume: torch.Tensor) -> dict:
    """
    Compute ground-truth Maximum Intensity Projections (XY, XZ, YZ planes).
    
    Returns dict: {axis_name: Tensor [H_out, W_out, 1]}
    """
    if volume.ndim == 3:
        vol = volume.unsqueeze(0)   # [1, D, H, W]
    else:
        vol = volume                # [C, D, H, W]
    
    mips = {}
    # XY plane (top-down, along Z/depth axis=1)
    mips["xy"] = vol.max(dim=1).values.permute(1, 2, 0)   # [H, W, C]
    # XZ plane (along Y axis=2)
    mips["xz"] = vol.max(dim=2).values.permute(1, 2, 0)   # [D, W, C]
    # YZ plane (along X axis=3)
    mips["yz"] = vol.max(dim=3).values.permute(1, 2, 0)   # [D, H, C]
    
    # Ensure single channel and normalize
    out = {}
    for k, m in mips.items():
        if m.shape[-1] == 0:
            m = m.unsqueeze(-1)
        out[k] = m[..., :1].clamp(0, 1)
    return out


def resize_mip(mip: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinearly resize a [Hm, Wm, C] MIP to [H, W, C]."""
    m = mip.permute(2, 0, 1).unsqueeze(0)      # [1, C, Hm, Wm]
    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    return m.squeeze(0).permute(1, 2, 0)        # [H, W, C]


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-view camera schedule
# ═══════════════════════════════════════════════════════════════════════════

def get_training_views(n_views: int, device: str) -> list[torch.Tensor]:
    """Generate N camera view matrices for training (orthogonal + orbit views)."""
    views = []
    
    # 1. Three orthogonal views (XY, XZ, YZ)
    views.append(topdown_camera_view(device=device))      # XY (z-axis)
    views.append(front_camera_view(device=device))        # XZ (y-axis)
    views.append(side_camera_view(device=device))         # YZ (x-axis)
    
    # 2. Additional orbit views
    n_orbit = max(0, n_views - 3)
    for i in range(n_orbit):
        t = i / max(n_orbit, 1)
        views.append(orbit_camera_view(t, device=device))
    
    return views


def topdown_camera_view(radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """Top-down view (XY plane MIP) looking along -Z axis."""
    eye = torch.tensor([0.5, 0.5, 0.5 + radius], device=device)
    at  = torch.tensor([0.5, 0.5, 0.5], device=device)
    up  = torch.tensor([0.0, 1.0, 0.0], device=device)
    return _build_view_matrix(eye, at, up)


def front_camera_view(radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """Front view (XZ plane MIP) looking along +Y axis."""
    eye = torch.tensor([0.5, 0.5 - radius, 0.5], device=device)
    at  = torch.tensor([0.5, 0.5, 0.5], device=device)
    up  = torch.tensor([0.0, 0.0, 1.0], device=device)
    return _build_view_matrix(eye, at, up)


def side_camera_view(radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """Side view (YZ plane MIP) looking along +X axis."""
    eye = torch.tensor([0.5 + radius, 0.5, 0.5], device=device)
    at  = torch.tensor([0.5, 0.5, 0.5], device=device)
    up  = torch.tensor([0.0, 1.0, 0.0], device=device)
    return _build_view_matrix(eye, at, up)


def orbit_camera_view(t: float, radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """Orbiting camera around Y-axis with elevation variation."""
    theta = t * 2 * math.pi
    phi   = math.pi / 6 * math.sin(t * math.pi * 4)
    eye = torch.tensor([
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(phi) + 0.5,
        radius * math.cos(theta) * math.cos(phi),
    ], device=device) + 0.5
    
    at = torch.tensor([0.5, 0.5, 0.5], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    return _build_view_matrix(eye, at, up)


def _build_view_matrix(eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Build 4x4 view matrix from camera parameters."""
    z = F.normalize(at - eye, dim=0)
    x = F.normalize(torch.cross(up, z, dim=0), dim=0)
    y = torch.cross(z, x, dim=0)
    
    M = torch.eye(4, device=eye.device)
    M[0, :3] = x;  M[0, 3] = -(x * eye).sum()
    M[1, :3] = y;  M[1, 3] = -(y * eye).sum()
    M[2, :3] = z;  M[2, 3] = -(z * eye).sum()
    return M


# ═══════════════════════════════════════════════════════════════════════════
#  Projection loss
# ═══════════════════════════════════════════════════════════════════════════

class ProjectionLoss(nn.Module):
    """Multi-view MIP projection reconstruction loss."""
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_scale_reg: float = 1e-4,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_scale_reg = lambda_scale_reg
    
    def forward(
        self,
        rendered: list[torch.Tensor],  # List of [H, W, C] rendered projections
        targets: list[torch.Tensor],   # List of [H, W, C] GT projections
        field: GaussianMixtureField,
    ) -> tuple[torch.Tensor, dict]:
        """Compute projection loss over multiple views."""
        
        total_loss = 0.0
        losses = {"l1": 0.0, "ssim": 0.0, "scale_reg": 0.0}
        
        # Photometric loss over all views
        for rend, tgt in zip(rendered, targets):
            # L1 loss
            l1 = F.l1_loss(rend, tgt)
            losses["l1"] += l1
            
            # SSIM loss (if images large enough)
            if self.lambda_ssim > 0 and min(rend.shape[:2]) >= 11:
                ssim_loss = 1.0 - self._ssim(
                    rend.permute(2, 0, 1).unsqueeze(0),
                    tgt.permute(2, 0, 1).unsqueeze(0)
                )
                losses["ssim"] += ssim_loss
        
        # Average over views
        n_views = len(rendered)
        losses["l1"] /= n_views
        losses["ssim"] /= n_views
        
        # Scale regularization (prevent Gaussians from growing too large)
        if self.lambda_scale_reg > 0:
            scales = torch.exp(field.log_scales)
            scale_reg = (scales - 0.03).clamp(min=0).pow(2).mean()
            losses["scale_reg"] = scale_reg
        
        # Total weighted loss
        total_loss = (
            self.lambda_l1 * losses["l1"] +
            self.lambda_ssim * losses["ssim"] +
            self.lambda_scale_reg * losses["scale_reg"]
        )
        
        losses["total"] = total_loss
        return total_loss, losses
    
    def _ssim(self, x, y, window_size=11, C1=0.01**2, C2=0.03**2):
        """Simplified SSIM calculation."""
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
        
        sigma_x = F.avg_pool2d(x**2, window_size, stride=1, padding=window_size//2) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, window_size, stride=1, padding=window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, window_size, stride=1, padding=window_size//2) - mu_x*mu_y
        
        ssim_num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
        ssim_den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        return (ssim_num / ssim_den).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MIP-Projection] device={device}  CUDA_EXT={'yes' if CUDA_AVAILABLE else 'no'}")
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    
    # ── 1. Load data ──────────────────────────────────────────────────────
    print("[MIP-Projection] Loading TIF volume …")
    volume, vol_meta = load_tif_volume(args.tif, normalise=True, device=device)
    print(f"           shape={vol_meta['shape']}  range=[{vol_meta['voxel_min']:.3f}, {vol_meta['voxel_max']:.3f}]")
    
    # Load SWC if provided
    swc_xyz, swc_radii = None, None
    if args.swc:
        print("[MIP-Projection] Loading SWC morphology …")
        swc_xyz_raw, swc_radii_raw, _ = load_swc(args.swc, device="cpu")
        vol_shape = vol_meta["shape"]
        swc_xyz, swc_radii = normalise_swc_to_voxel(
            swc_xyz_raw, swc_radii_raw, vol_shape,
            voxel_size=tuple(args.voxel_size) if args.voxel_size else None,
        )
        swc_xyz = swc_xyz.cpu().numpy()
        swc_radii = swc_radii.cpu().numpy()
        print(f"           SWC nodes={swc_xyz.shape[0]}")
    
    # ── 2. Compute GT MIP projections ────────────────────────────────────
    print("[MIP-Projection] Computing GT MIP projections …")
    gt_mips_raw = compute_gt_mips(volume)
    
    # Resize to render resolution
    gt_mips_resized = {}
    for name, mip in gt_mips_raw.items():
        gt_mips_resized[name] = resize_mip(mip, args.H, args.W)
    
    # Save GT projections
    gt_dir = out_dir / "gt_projections"
    gt_dir.mkdir(exist_ok=True)
    for name, mip in gt_mips_resized.items():
        _save_image(mip.detach().cpu().clamp(0, 1).numpy(), gt_dir / f"gt_mip_{name}.png")
    print(f"           GT projections saved to {gt_dir}")
    
    # ── 3. Initialize 3D Gaussian field ───────────────────────────────────
    print(f"[MIP-Projection] Initializing GaussianMixtureField (K={args.num_gaussians}) …")
    
    # Define normalized bounds [0, 1]^3 for the volume
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    
    field = GaussianMixtureField(
        num_gaussians=args.num_gaussians,
        init_scale=args.init_scale,
        init_amplitude=args.init_amplitude,
        bounds=bounds,
        swc_coords=swc_xyz,
        swc_radii=swc_radii,
    ).to(device)
    
    # Wrap for renderer
    gaussians = GaussianFieldToRenderer(field)
    print(f"           Initialized K={gaussians.N} Gaussians")
    
    # ── 4. Camera and views ───────────────────────────────────────────────
    camera = Camera(args.H, args.W, fov_deg=args.fov, device=device)
    training_views = get_training_views(args.n_views, device=device)
    print(f"           Using {len(training_views)} training views")
    
    # Match GT projections to first 3 views (orthogonal)
    gt_targets = [
        gt_mips_resized["xy"],  # view 0: top-down
        gt_mips_resized["xz"],  # view 1: front
        gt_mips_resized["yz"],  # view 2: side
    ]
    # For orbit views, use XY as approximate target
    for _ in range(len(training_views) - 3):
        gt_targets.append(gt_mips_resized["xy"])
    
    # ── 5. Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam([
        {"params": [field.means], "lr": args.lr_pos, "name": "means"},
        {"params": [field.quaternions], "lr": args.lr_rot, "name": "quats"},
        {"params": [field.log_scales], "lr": args.lr_scale, "name": "scales"},
        {"params": [field.log_amplitudes], "lr": args.lr_amp, "name": "amplitudes"},
    ], eps=1e-15)
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay ** (1 / args.iterations)
    )
    
    # ── 6. Loss ───────────────────────────────────────────────────────────
    criterion = ProjectionLoss(
        lambda_l1=1.0,
        lambda_ssim=0.2,
        lambda_scale_reg=args.lambda_scale,
    )
    
    # ── 7. Training loop ──────────────────────────────────────────────────
    print("[MIP-Projection] Starting training …")
    t0 = time.time()
    
    for it in range(1, args.iterations + 1):
        optimizer.zero_grad()
        
        # Render all training views
        rendered_views = []
        for view_mat in training_views:
            img, weight, depth = render(gaussians, camera, view_mat)
            rendered_views.append(img)
        
        # Compute loss
        loss, loss_dict = criterion(rendered_views, gt_targets, field)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        
        # Clamp parameters
        field.clamp_log_scales_(math.log(5e-4), math.log(0.3))
        field.clamp_log_amplitudes_(math.log(1e-4), math.log(1.0))
        field.apply_aabb_clamp()
        
        # ── Logging ───────────────────────────────────────────────────────
        if it % args.log_interval == 0:
            elapsed = time.time() - t0
            fps = it / elapsed
            print(
                f"  iter={it:6d}/{args.iterations}  "
                f"loss={loss_dict['total']:.4f}  "
                f"l1={loss_dict['l1']:.4f}  "
                f"ssim={loss_dict['ssim']:.4f}  "
                f"K={gaussians.N}  "
                f"{fps:.1f} it/s"
            )
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss/{k}", v, it)
            writer.add_scalar("gaussians/N", gaussians.N, it)
        
        if it % args.save_interval == 0 or it == args.iterations:
            # Save checkpoint
            ckpt_path = out_dir / f"field_{it:06d}.pt"
            torch.save({
                'iteration': it,
                'model_state_dict': field.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(ckpt_path))
            
            # Save rendered images
            for i, (img, name) in enumerate(zip(rendered_views[:3], ["xy", "xz", "yz"])):
                img_np = img.detach().cpu().clamp(0, 1).numpy()
                _save_image(img_np, out_dir / f"render_{name}_{it:06d}.png")
    
    writer.close()
    print(f"[MIP-Projection] Done. K={gaussians.N}  time={time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_image(img_np: np.ndarray, path: Path):
    """Save float [H, W, C] numpy array as PNG."""
    try:
        from PIL import Image
        if img_np.shape[-1] == 1:
            img_np = img_np[..., 0]
        im = Image.fromarray((img_np * 255).clip(0, 255).astype(np.uint8))
        im.save(str(path))
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end MIP projection training")
    
    # Data
    parser.add_argument("--tif", type=str, required=True, help="Path to .tif volume")
    parser.add_argument("--swc", type=str, default=None, help="Optional .swc morphology")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None)
    parser.add_argument("--out", type=str, default="outputs/mip_proj", help="Output directory")
    
    # Model
    parser.add_argument("--num_gaussians", type=int, default=2000, help="Number of 3D Gaussians")
    parser.add_argument("--init_scale", type=float, default=0.05, help="Initial Gaussian scale")
    parser.add_argument("--init_amplitude", type=float, default=0.1, help="Initial amplitude")
    
    # Rendering
    parser.add_argument("--H", type=int, default=512, help="Render height")
    parser.add_argument("--W", type=int, default=512, help="Render width")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view (degrees)")
    parser.add_argument("--n_views", type=int, default=8, help="Number of training views")
    
    # Training
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--lr_pos", type=float, default=1e-3, help="Learning rate for positions")
    parser.add_argument("--lr_rot", type=float, default=1e-3, help="Learning rate for rotations")
    parser.add_argument("--lr_scale", type=float, default=5e-3, help="Learning rate for scales")
    parser.add_argument("--lr_amp", type=float, default=1e-2, help="Learning rate for amplitudes")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="LR decay factor")
    
    # Loss weights
    parser.add_argument("--lambda_scale", type=float, default=1e-4, help="Scale regularization weight")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
