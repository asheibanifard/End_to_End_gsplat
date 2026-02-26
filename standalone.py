"""
neuro3dgs/train.py

End-to-end training pipeline:
  .tif fluorescence volume + .swc morphology  →  3D Gaussian model  →  MIP renderer

Usage:
    python train.py \
        --tif  data/neurite_stack.tif \
        --swc  data/neurite_morphology.swc \
        --out  outputs/run_01 \
        --iterations 30000 \
        --H 512 --W 512

Tested on: PyTorch 2.2, CUDA 12.1, A100 / RTX 3090.
"""

from __future__ import annotations
import os, sys, math, time, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ── local modules ──────────────────────────────────────────────────────────────
from data_io         import (load_tif_volume, load_swc, normalise_swc_to_voxel,
                              extract_seeds_from_volume)
from gaussian_model  import NeuriteGaussians
from renderer        import Camera, render, CUDA_AVAILABLE
from losses          import NeuriteReconLoss
from config          import load_config


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground-truth MIP generator
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gt_mips(volume: torch.Tensor, C_feat: int) -> dict:
    """
    Compute ground-truth Maximum Intensity Projections (XY, XZ, YZ planes).
    If volume is single-channel [D,H,W], replicate to [D,H,W,C_feat].

    Returns dict: {axis_name: Tensor [H_out, W_out, C_feat]}
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

    # Replicate C_feat times if single channel
    out = {}
    for k, m in mips.items():
        if m.shape[-1] == 1 and C_feat > 1:
            m = m.expand(-1, -1, C_feat)
        elif m.shape[-1] > C_feat:
            m = m[..., :C_feat]
        out[k] = m.clamp(0, 1)
    return out


def resize_mip(mip: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinearly resize a [Hm, Wm, C] MIP to [H, W, C]."""
    m = mip.permute(2, 0, 1).unsqueeze(0)      # [1, C, Hm, Wm]
    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    return m.squeeze(0).permute(1, 2, 0)        # [H, W, C]


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera schedule: rotating views around the volume
# ═══════════════════════════════════════════════════════════════════════════════

def orbit_camera_view(t: float, radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """
    Generate a view matrix for an orbiting camera.
    t ∈ [0, 1) → full orbit around Y-axis with slight elevation.
    """
    theta = t * 2 * math.pi
    phi   = math.pi / 6 * math.sin(t * math.pi * 4)   # gentle elevation oscillation
    eye = torch.tensor([
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(phi) + 0.5,
        radius * math.cos(theta) * math.cos(phi),
    ], device=device) + 0.5

    at = torch.tensor([0.5, 0.5, 0.5], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)

    z = F.normalize(eye - at, dim=0)
    x = F.normalize(torch.cross(up, z, dim=0), dim=0)
    y = torch.cross(z, x, dim=0)

    M = torch.eye(4, device=device)
    M[0, :3] = x;  M[0, 3] = -(x * eye).sum()
    M[1, :3] = y;  M[1, 3] = -(y * eye).sum()
    M[2, :3] = z;  M[2, 3] = -(z * eye).sum()
    return M


def topdown_camera_view(radius: float = 2.5, device: str = "cpu") -> torch.Tensor:
    """
    Fixed top-down camera looking along -Z axis (XY plane view).
    This view corresponds to the XY MIP projection.
    
    Uses convention where points in front of camera have positive Z in camera space.
    """
    eye = torch.tensor([0.5, 0.5, 0.5 + radius], device=device)
    at  = torch.tensor([0.5, 0.5, 0.5], device=device)
    up  = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Camera looks FROM eye TO at, so z-axis points at - eye (forward direction)
    # This ensures objects in front have positive Z in camera space
    z = F.normalize(at - eye, dim=0)
    x = F.normalize(torch.cross(up, z, dim=0), dim=0)
    y = torch.cross(z, x, dim=0)

    M = torch.eye(4, device=device)
    M[0, :3] = x;  M[0, 3] = -(x * eye).sum()
    M[1, :3] = y;  M[1, 3] = -(y * eye).sum()
    M[2, :3] = z;  M[2, 3] = -(z * eye).sum()
    return M


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NeuroSGM] device={device}  CUDA_EXT={'yes' if CUDA_AVAILABLE else 'no (fallback)'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[NeuroSGM] Loading TIF volume …")
    volume, vol_meta = load_tif_volume(args.tif, normalise=True, device=device)
    print(f"           shape={vol_meta['shape']}  range=[{vol_meta['voxel_min']:.3f}, {vol_meta['voxel_max']:.3f}]")

    print("[NeuroSGM] Loading SWC morphology …")
    swc_xyz_raw, swc_radii, swc_parents = load_swc(args.swc, device=device)
    vol_shape = vol_meta["shape"]
    swc_xyz, swc_radii_norm = normalise_swc_to_voxel(
        swc_xyz_raw, swc_radii, vol_shape,
        voxel_size=tuple(args.voxel_size) if args.voxel_size else None,
    )
    print(f"           SWC nodes={swc_xyz.shape[0]}")

    # ── 2. Extract voxel seeds ────────────────────────────────────────────────
    print("[NeuroSGM] Extracting voxel seeds …")
    vol_seeds, vol_intensities = extract_seeds_from_volume(
        volume,
        threshold=args.seed_threshold,
        max_seeds=args.max_seeds,
        strategy=args.seed_strategy,
    )
    print(f"           seeds={vol_seeds.shape[0]}")

    # ── 3. Initialise Gaussians ───────────────────────────────────────────────
    C_int = args.intensity_dim
    print(f"[NeuroSGM] Initialising Gaussians  (intensity_dim={C_int}) …")
    gaussians = NeuriteGaussians.from_swc_and_volume(
        swc_xyz, swc_radii_norm,
        vol_seeds, vol_intensities,
        intensity_dim=C_int,
    )
    print(f"           initial N={gaussians.N}")

    # ── 4. GT MIP targets ─────────────────────────────────────────────────────
    gt_mips_raw = compute_gt_mips(volume, C_int)
    gt_mip_xy = resize_mip(gt_mips_raw["xy"], args.H, args.W)   # primary target

    # Save GT projections
    gt_dir = out_dir / "gt_projections"
    gt_dir.mkdir(exist_ok=True)
    for name, mip in gt_mips_raw.items():
        mip_np = mip.detach().cpu().clamp(0, 1).numpy()
        _save_image(mip_np, gt_dir / f"gt_mip_{name}.png")
    _save_image(gt_mip_xy.detach().cpu().clamp(0, 1).numpy(), gt_dir / "gt_mip_xy_resized.png")
    print(f"           GT projections saved to {gt_dir}")

    # ── 5. Camera ─────────────────────────────────────────────────────────────
    camera = Camera(args.H, args.W, fov_deg=args.fov, device=device)

    # ── 6. Optimiser ─────────────────────────────────────────────────────────
    #  Different learning rates for position, rotation, scale, opacity, features
    optimizer = torch.optim.Adam([
        {"params": [gaussians._means],          "lr": args.lr_pos,       "name": "means"},
        {"params": [gaussians._quats],          "lr": args.lr_rot,       "name": "quats"},
        {"params": [gaussians._log_scales],     "lr": args.lr_scale,     "name": "scales"},
        {"params": [gaussians._logit_intensity], "lr": args.lr_intensity, "name": "intensity"},
    ], eps=1e-15)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay ** (1 / args.iterations)
    )

    # ── 7. Loss ───────────────────────────────────────────────────────────────
    criterion = NeuriteReconLoss(
        lambda_photometric=1.0,
        lambda_mip_consist=args.lambda_mip,
        lambda_intensity_sparse=args.lambda_intensity,
        lambda_scale_reg=args.lambda_scale,
        lambda_depth_smooth=args.lambda_depth,
        lambda_feat_smooth=args.lambda_feat,
        lambda_dssim=0.2,
    )

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print("[NeuroSGM] Starting optimisation …")
    t0 = time.time()

    for it in range(1, args.iterations + 1):
        # Fixed top-down camera matching XY MIP projection
        # (orbit views don't match the fixed XY target, causing loss plateau)
        view_mat = topdown_camera_view(radius=2.5, device=device)

        # Forward render
        rendered, weight, depth = render(gaussians, camera, view_mat)

        # Determine GT view: for orbit views, use the XY MIP as target
        # (In a full pipeline you'd pre-render GT from all angles)
        target_view = gt_mip_xy

        # Loss
        loss, loss_dict = criterion(
            rendered, target_view, depth, weight,
            gaussians,
            gt_mip=gt_mip_xy,
            swc_xyz=swc_xyz,
            iteration=it,
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [gaussians._means, gaussians._quats,
             gaussians._log_scales, gaussians._logit_intensity],
            max_norm=1.0,
        )

        # Accumulate grad stats for densification
        if gaussians._means.grad is not None:
            try:
                import neuro3dgs_cuda as _C
                grad_mag = _C.compute_grad_magnitude(
                    gaussians._means.grad.contiguous(),
                    view_mat.flatten().contiguous(),
                )
            except Exception:
                grad_mag = gaussians._means.grad.norm(dim=-1)

            # Approx screen-space radius from scale
            screen_rad = gaussians.scales.max(dim=-1).values.detach()
            gaussians.accumulate_grad(grad_mag, screen_rad)

        optimizer.step()
        lr_scheduler.step()

        # ── Adaptive density control ────────────────────────────────────────
        if (it % NeuriteGaussians.DENSIFY_INTERVAL == 0
                and it < args.densify_until):
            gaussians.densify_and_prune(optimizer, img_size=float(max(args.H, args.W)))
            print(f"  [densify] iter={it}  N={gaussians.N}")

        if it % NeuriteGaussians.INTENSITY_RESET_INTERVAL == 0:
            gaussians.reset_intensity()

        # ── Logging ─────────────────────────────────────────────────────────
        if it % args.log_interval == 0:
            elapsed = time.time() - t0
            fps = it / elapsed
            print(
                f"  iter={it:6d}/{args.iterations}  "
                f"loss={loss_dict['total']:.4f}  "
                f"photo={loss_dict['photo']:.4f}  "
                f"N={gaussians.N}  "
                f"{fps:.1f} it/s"
            )
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss/{k}", v, it)
            writer.add_scalar("gaussians/N", gaussians.N, it)
            writer.add_scalar("gaussians/mean_intensity", gaussians.intensity.mean().item(), it)

        if it % args.save_interval == 0 or it == args.iterations:
            ckpt_path = out_dir / f"gaussians_{it:06d}.pt"
            gaussians.save(str(ckpt_path))

            # Save rendered MIP image
            img_np = rendered.detach().cpu().clamp(0, 1).numpy()
            _save_image(img_np, out_dir / f"render_{it:06d}.png")

    writer.close()
    print(f"[NeuroSGM] Done. Final N={gaussians.N}  "
          f"time={time.time()-t0:.1f}s")

    # ── 9. Final multi-view MIP visualisation ─────────────────────────────────
    print("[NeuroSGM] Generating final MIP renders …")
    _render_mip_gallery(gaussians, camera, out_dir, device)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_image(img_np: np.ndarray, path: Path):
    """Save float [H, W, C] numpy array as PNG."""
    try:
        from PIL import Image
        if img_np.shape[-1] == 1:
            img_np = img_np[..., 0]
        elif img_np.shape[-1] > 3:
            img_np = img_np[..., :3]
        im = Image.fromarray((img_np * 255).clip(0, 255).astype(np.uint8))
        im.save(str(path))
    except ImportError:
        pass  # PIL not required


def _render_mip_gallery(
    gaussians: NeuriteGaussians,
    camera: Camera,
    out_dir: Path,
    device: str,
    n_views: int = 36,
):
    """Render a gallery of orbit views and save as PNG sequence."""
    import os
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i in range(n_views):
            t = i / n_views
            vm = orbit_camera_view(t, device=device)
            img, _, _ = render(gaussians, camera, vm)
            img_np = img.cpu().clamp(0, 1).numpy()
            _save_image(img_np, gallery_dir / f"view_{i:03d}.png")

    print(f"  Gallery saved to {gallery_dir}  ({n_views} views)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NeuroSGM: 3DGS for dense neurite volumes")

    # ── Config ────────────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (overrides CLI defaults)")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--tif",  default=None, help="Path to .tif fluorescence volume")
    p.add_argument("--swc",  default=None, help="Path to .swc morphology file")
    p.add_argument("--out",  default="outputs/run", help="Output directory")
    p.add_argument("--voxel-size", nargs=3, type=float, default=None,
                   metavar=("DZ", "DY", "DX"),
                   help="Physical voxel size in SWC units (microns)")

    # ── Seeding ───────────────────────────────────────────────────────────────
    p.add_argument("--seed-threshold", type=float, default=0.1)
    p.add_argument("--max-seeds",      type=int,   default=200_000)
    p.add_argument("--seed-strategy",  default="threshold",
                   choices=["threshold", "grid", "topk"])

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--intensity-dim", type=int, default=1,
                   help="Per-Gaussian intensity channels (typically 1 for fluorescence)")

    # ── Rendering ─────────────────────────────────────────────────────────────
    p.add_argument("--H",   type=int,   default=512)
    p.add_argument("--W",   type=int,   default=512)
    p.add_argument("--fov", type=float, default=60.0)

    # ── Optimisation ──────────────────────────────────────────────────────────
    p.add_argument("--iterations",    type=int,   default=30_000)
    p.add_argument("--densify-until", type=int,   default=15_000)
    p.add_argument("--lr-pos",     type=float, default=1.6e-4)
    p.add_argument("--lr-rot",     type=float, default=1e-3)
    p.add_argument("--lr-scale",   type=float, default=5e-3)
    p.add_argument("--lr-intensity", type=float, default=2.5e-2)
    p.add_argument("--lr-decay",   type=float, default=0.01,
                   help="Total LR decay factor over all iterations")

    # ── Loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--lambda-mip",     type=float, default=0.5)
    p.add_argument("--lambda-intensity", type=float, default=0.05)
    p.add_argument("--lambda-scale",   type=float, default=0.01)
    p.add_argument("--lambda-depth",   type=float, default=0.005)
    p.add_argument("--lambda-feat",    type=float, default=0.001)
    p.add_argument("--lambda-swc",     type=float, default=0.01)

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log-interval",  type=int, default=100)
    p.add_argument("--save-interval", type=int, default=5000)

    return p.parse_args()


def args_from_config(cfg, args):
    """Merge config values into args namespace (config takes priority for unset args)."""
    # Data
    if args.tif is None:
        args.tif = cfg.data.tif
    if args.swc is None:
        args.swc = cfg.data.swc
    if hasattr(cfg.data, 'voxel_size') and cfg.data.voxel_size and args.voxel_size is None:
        args.voxel_size = cfg.data.voxel_size
    
    # Output
    if hasattr(cfg, 'logging') and hasattr(cfg.logging, 'out_dir'):
        args.out = cfg.logging.out_dir
    
    # Init/Seeding
    if hasattr(cfg, 'init'):
        args.seed_threshold = cfg.init.seed_threshold
        args.max_seeds = cfg.init.max_seeds
        args.seed_strategy = cfg.init.seed_strategy
        args.intensity_dim = cfg.init.feature_dim
    
    # Rendering
    if hasattr(cfg, 'render'):
        args.H = cfg.render.H
        args.W = cfg.render.W
        args.fov = cfg.render.fov_deg
    
    # Optimisation
    if hasattr(cfg, 'optim'):
        args.iterations = cfg.optim.iterations
        if hasattr(cfg.optim, 'lr'):
            args.lr_pos = cfg.optim.lr.means
            args.lr_rot = cfg.optim.lr.quats
            args.lr_scale = cfg.optim.lr.scales
            args.lr_intensity = cfg.optim.lr.intensity if hasattr(cfg.optim.lr, 'intensity') else cfg.optim.lr.features
        if hasattr(cfg.optim, 'lr_decay'):
            args.lr_decay = cfg.optim.lr_decay
    
    # ADC
    if hasattr(cfg, 'adc'):
        args.densify_until = cfg.adc.densify_until
    
    # Loss weights
    if hasattr(cfg, 'loss'):
        if hasattr(cfg.loss, 'mip_consistency'):
            args.lambda_mip = cfg.loss.mip_consistency.weight
        if hasattr(cfg.loss, 'opacity_sparsity'):
            args.lambda_intensity = cfg.loss.opacity_sparsity.weight
        if hasattr(cfg.loss, 'scale_reg'):
            args.lambda_scale = cfg.loss.scale_reg.weight
        if hasattr(cfg.loss, 'depth_smooth'):
            args.lambda_depth = cfg.loss.depth_smooth.weight
        if hasattr(cfg.loss, 'feature_smooth'):
            args.lambda_feat = cfg.loss.feature_smooth.weight
        if hasattr(cfg.loss, 'swc_proximity'):
            args.lambda_swc = cfg.loss.swc_proximity.weight
    
    # Logging
    if hasattr(cfg, 'logging'):
        args.log_interval = cfg.logging.log_interval
        args.save_interval = cfg.logging.save_interval
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Load config if provided
    if args.config:
        cfg = load_config(args.config, validate=True)
        args = args_from_config(cfg, args)
    
    # Validate required args
    if args.tif is None or args.swc is None:
        raise ValueError("--tif and --swc are required (via CLI or --config)")
    
    train(args)
