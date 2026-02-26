#!/usr/bin/env python3
"""
NeuroSGM — Neurite Sparse Gaussian Mixture: MIP Rendering Pipeline
====================================================================

Renders 3D Gaussians G(μ, Σ, intensity) as Maximum Intensity Projections.

Formulation
-----------
Each Gaussian is parameterised by:
    μ          : (3,)    centre in world space
    Σ          : (3, 3)  anisotropic covariance (per-axis scales + rotation)
    intensity  : scalar  signal amplitude in [0, 1]   ← NO opacity

Splatting pipeline for one projection view k:

    1) Transform to camera frame:
           μ_cam = R·μ + T
           Σ_cam = R·Σ·Rᵀ

    2) Project to 2D image plane:
           μ₂D = (fx·μx/μz + cx,  fy·μy/μz + cy)
           J   = [[fx/z,  0,  -fx·x/z²],
                  [0,  fy/z,  -fy·y/z²]]
           Σ₂D = J·Σ_cam·Jᵀ

    3) Evaluate 2D Gaussian at every pixel p = (u, v):
           G₂D(p; μ₂D, Σ₂D) = exp(-½ (p-μ₂D)ᵀ Σ₂D⁻¹ (p-μ₂D))

    4) MIP splatting — replaces alpha compositing entirely:
           I_k(u,v) = max_i [ intensity_i · G₂D_i(u,v) ]

       Loss over all M projection views (matches formulation):
           L = Σ_{k=1}^{M} ‖ I_k(u,v) − max_i [ intensity_i · G₂D_i(u,v) ] ‖²

Design decisions
----------------
- No opacity parameter. Intensity is the sole scalar per Gaussian.
- MIP naturally models fluorescence microscopy: brightest structure wins.
- Training iterates over ALL M projection views per epoch — directly matches
  the Σ_k outer sum in the formulation (not random pixel sampling).
- log_scales is (K, 3): per-axis anisotropic scales essential for neurites.
- Differentiable MIP via soft-max approximation (β controls sharpness).
- All hyperparameters are loaded from config.yml — no hardcoded values.
"""

from __future__ import annotations

import math
import os
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None  # handled at load time

try:
    from .splat_cuda_wrapper import CUDASplattingTrainer
except ImportError:
    try:
        from splat_cuda_wrapper import CUDASplattingTrainer
    except ImportError:
        CUDASplattingTrainer = None

try:
    from .splat_mip_cuda_wrapper import HAS_MIP_CUDA, splat_mip_grid_cuda
except ImportError:
    try:
        from splat_mip_cuda_wrapper import HAS_MIP_CUDA, splat_mip_grid_cuda
    except ImportError:
        HAS_MIP_CUDA = False
        splat_mip_grid_cuda = None


# ===================================================================
#  Config loader
# ===================================================================
def load_config(path: str = "config_splat.yml") -> dict:
    """
    Load YAML config. Falls back to an empty dict if file is missing
    so the codebase stays importable without a config present.
    """
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ===================================================================
#  Camera (pinhole)
# ===================================================================
@dataclass
class Camera:
    """
    Pinhole camera intrinsics.

    Attributes
    ----------
    fx, fy        : focal lengths in pixels
    cx, cy        : principal point in pixels
    width, height : image resolution
    near, far     : depth clipping planes
    """
    fx:     float
    fy:     float
    cx:     float
    cy:     float
    width:  int
    height: int
    near:   float = 0.1
    far:    float = 100.0

    @property
    def K(self) -> torch.Tensor:
        """3×3 intrinsic matrix."""
        return torch.tensor([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0 ],
        ], dtype=torch.float32)

    @classmethod
    def from_fov(
        cls,
        fov_x_deg: float,
        width:     int,
        height:    int,
        near:      float = 0.1,
        far:       float = 100.0,
    ) -> "Camera":
        """Construct from horizontal field-of-view (degrees)."""
        fx = width / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        return cls(
            fx=fx, fy=fx,
            cx=width  / 2.0,
            cy=height / 2.0,
            width=width, height=height,
            near=near, far=far,
        )

    @classmethod
    def from_config(cls, cfg: dict, width: int, height: int) -> "Camera":
        """Construct from the 'camera' section of config.yml."""
        c = cfg["camera"]
        return cls.from_fov(
            fov_x_deg = c["fov_x_deg"],
            width     = width,
            height    = height,
            near      = c["near"],
            far       = c["far"],
        )


# ===================================================================
#  Gaussian parameter container  (intensity-only, no opacity)
# ===================================================================
@dataclass
class GaussianParameters:
    """
    Intensity-only 3D Gaussian primitives — no opacity field.

    Attributes
    ----------
    means       : (K, 3)    world-space centres μ_k
    covariances : (K, 3, 3) anisotropic covariance matrices Σ_k
    intensities : (K,)      signal amplitude in [0, 1]
    """
    means:       torch.Tensor   # (K, 3)
    covariances: torch.Tensor   # (K, 3, 3)
    intensities: torch.Tensor   # (K,)


# ===================================================================
#  Step 1 — Transform to camera frame
# ===================================================================
def transform_to_camera(
    means:       torch.Tensor,
    covariances: torch.Tensor,
    R:           torch.Tensor,
    T:           torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    means_cam = means @ R.T + T.unsqueeze(0)
    cov_cam   = R.unsqueeze(0) @ covariances @ R.T.unsqueeze(0)
    return means_cam, cov_cam


# ===================================================================
#  Step 2 — Project to 2D
# ===================================================================
def compute_projection_jacobian(
    means_cam: torch.Tensor,
    fx:        float,
    fy:        float,
) -> torch.Tensor:
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    z_s  = z.clamp(min=1e-6)
    z2_s = (z * z).clamp(min=1e-12)
    K = x.shape[0]
    J = torch.zeros(K, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] =  fx / z_s
    J[:, 0, 2] = -fx * x / z2_s
    J[:, 1, 1] =  fy / z_s
    J[:, 1, 2] = -fy * y / z2_s
    return J


def project_to_2d(
    means_cam: torch.Tensor,
    cov_cam:   torch.Tensor,
    camera:    Camera,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    z_s = z.clamp(min=1e-6)
    u = camera.fx * x / z_s + camera.cx
    v = camera.fy * y / z_s + camera.cy
    means_2d = torch.stack([u, v], dim=-1)
    J      = compute_projection_jacobian(means_cam, camera.fx, camera.fy)
    cov_2d = J @ cov_cam @ J.transpose(-2, -1)
    eye    = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype).unsqueeze(0)
    cov_2d = cov_2d + 1e-4 * eye
    return means_2d, cov_2d, z


# ===================================================================
#  Step 3 — Evaluate 2D Gaussian
# ===================================================================
def evaluate_gaussian_2d(
    pixels:   torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d:   torch.Tensor,
) -> torch.Tensor:
    diff = pixels[:, None, :] - means_2d[None, :, :]
    a, b    = cov_2d[:, 0, 0], cov_2d[:, 0, 1]
    c, d    = cov_2d[:, 1, 0], cov_2d[:, 1, 1]
    inv_det = 1.0 / (a * d - b * c).clamp(min=1e-12)
    cov_inv = torch.stack([
        torch.stack([ d * inv_det, -b * inv_det], dim=-1),
        torch.stack([-c * inv_det,  a * inv_det], dim=-1),
    ], dim=-2)
    tmp   = torch.einsum('nki,kij->nkj', diff, cov_inv)
    mahal = (tmp * diff).sum(dim=-1)
    return torch.exp(-0.5 * mahal)


# ===================================================================
#  Step 4 — MIP Splatting
# ===================================================================
def _invert_cov_2x2(cov_2d: torch.Tensor) -> torch.Tensor:
    """Batch-invert (K, 2, 2) symmetric covariance matrices."""
    a, b    = cov_2d[:, 0, 0], cov_2d[:, 0, 1]
    c, d_v  = cov_2d[:, 1, 0], cov_2d[:, 1, 1]
    inv_det = 1.0 / (a * d_v - b * c).clamp(min=1e-12)
    return torch.stack([
        torch.stack([ d_v * inv_det, -b   * inv_det], dim=-1),
        torch.stack([-c   * inv_det,  a   * inv_det], dim=-1),
    ], dim=-2)


def splat_mip(
    pixels:      torch.Tensor,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 50.0,
    chunk_size:  int   = 4096,
) -> torch.Tensor:
    N   = pixels.shape[0]
    out = torch.zeros(N, device=pixels.device, dtype=pixels.dtype)
    cov_inv = _invert_cov_2x2(cov_2d)
    for i in range(0, N, chunk_size):
        pix   = pixels[i : i + chunk_size]
        diff  = pix[:, None, :] - means_2d[None, :, :]
        tmp   = torch.einsum('nki,kij->nkj', diff, cov_inv)
        mahal = (tmp * diff).sum(-1)
        del diff, tmp
        gauss = torch.exp(-0.5 * mahal)
        del mahal
        gauss = gauss * intensities[None, :]
        sm    = torch.softmax(beta * gauss, dim=-1)
        out[i : i + chunk_size] = (sm * gauss).sum(-1)
        del gauss, sm
    return out


def splat_mip_grid(
    H:           int,
    W:           int,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 50.0,
    chunk_size:  int   = 4096,
    device:      torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = means_2d.device
    else:
        device = torch.device(device)
    assert device.type == means_2d.device.type, (
        f"device mismatch: device={device}, means_2d.device={means_2d.device}")

    # ── CUDA fast path ──────────────────────────────────────────────
    if HAS_MIP_CUDA and means_2d.device.type == 'cuda':
        return splat_mip_grid_cuda(H, W, means_2d, cov_2d, intensities, beta)

    # ── Python fallback (chunked) ───────────────────────────────────
    N   = H * W
    out = torch.zeros(N, device=device, dtype=means_2d.dtype)
    cov_inv = _invert_cov_2x2(cov_2d)
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    for row_start in range(0, H, chunk_size // max(W, 1)):
        row_end = min(row_start + chunk_size // max(W, 1), H)
        ys      = torch.arange(row_start, row_end, device=device, dtype=torch.float32) + 0.5
        gy, gx  = torch.meshgrid(ys, xs, indexing='ij')
        pix     = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        i, j    = row_start * W, row_end * W
        diff    = pix[:, None, :] - means_2d[None, :, :]
        tmp     = torch.einsum('nki,kij->nkj', diff, cov_inv)
        mahal   = (tmp * diff).sum(-1)
        del diff, tmp, pix
        gauss   = torch.exp(-0.5 * mahal)
        del mahal
        gauss   = gauss * intensities[None, :]
        sm      = torch.softmax(beta * gauss, dim=-1)
        out[i:j] = (sm * gauss).sum(-1)
        del gauss, sm
    return out


# ===================================================================
#  Full MIP projection renderer
# ===================================================================
def render_mip_projection(
    gaussians:  GaussianParameters,
    camera:     Camera,
    R:          torch.Tensor,
    T:          torch.Tensor,
    beta:       float = 50.0,
    chunk_size: int   = 4096,
) -> Tuple[torch.Tensor, int]:
    device = gaussians.means.device
    H, W   = camera.height, camera.width
    means_cam, cov_cam = transform_to_camera(
        gaussians.means, gaussians.covariances, R, T)
    z   = means_cam[:, 2]
    vis = (z > camera.near) & (z < camera.far)
    if vis.sum() == 0:
        return torch.zeros(H, W, device=device), 0
    means_cam_v = means_cam[vis]
    cov_cam_v   = cov_cam[vis]
    intens_v    = gaussians.intensities[vis]
    means_2d, cov_2d, _ = project_to_2d(means_cam_v, cov_cam_v, camera)
    with torch.no_grad():
        a_c, b_c, d_c = cov_2d[:, 0, 0], cov_2d[:, 0, 1], cov_2d[:, 1, 1]
        tr     = a_c + d_c
        disc   = (tr * tr - 4.0 * (a_c * d_c - b_c * b_c)).clamp(min=0.0)
        radius = 3.0 * torch.sqrt(0.5 * (tr + torch.sqrt(disc)).clamp(min=1e-8))
        u_2d, v_2d = means_2d[:, 0], means_2d[:, 1]
        in_img = ((u_2d + radius > 0) & (u_2d - radius < W) &
                  (v_2d + radius > 0) & (v_2d - radius < H))
    means_2d  = means_2d[in_img]
    cov_2d    = cov_2d[in_img]
    intens_v  = intens_v[in_img]
    n_visible = int(in_img.sum().item())
    if n_visible == 0:
        return torch.zeros(H, W, device=device), 0
    image = splat_mip_grid(H, W, means_2d, cov_2d, intens_v,
                           beta=beta, chunk_size=chunk_size, device=device)
    return image.reshape(H, W), n_visible


# ===================================================================
#  Loss functions
# ===================================================================
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def weighted_mse_loss(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    fg_weight: float = 5.0,
) -> torch.Tensor:
    """MSE with foreground emphasis — brighter GT pixels weighted more.

    For fluorescence microscopy the neurites occupy ~5-15 % of pixels;
    standard MSE under-penalises errors on those sparse bright structures.
    ``fg_weight=5.0`` linearly ramps the per-pixel weight from 1 (background)
    to ``fg_weight`` (brightest foreground).
    """
    weight = 1.0 + (fg_weight - 1.0) * target   # (H, W) or flat
    return (weight * (pred - target) ** 2).mean()


def ssim_loss_fn(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    window_size: int   = 11,
    C1:          float = 1e-4,
    C2:          float = 9e-4,
) -> torch.Tensor:
    p   = pred.unsqueeze(0).unsqueeze(0)
    g   = target.unsqueeze(0).unsqueeze(0)
    coords = torch.arange(window_size, device=pred.device, dtype=torch.float32)
    coords -= window_size // 2
    win    = torch.exp(-coords ** 2 / (2.0 * 1.5 ** 2))
    win    = win.unsqueeze(1) * win.unsqueeze(0)
    win    = (win / win.sum()).unsqueeze(0).unsqueeze(0)
    pad    = window_size // 2
    mu_p   = F.conv2d(p,     win, padding=pad)
    mu_g   = F.conv2d(g,     win, padding=pad)
    sig_p  = F.conv2d(p * p, win, padding=pad) - mu_p ** 2
    sig_g  = F.conv2d(g * g, win, padding=pad) - mu_g ** 2
    sig_x  = F.conv2d(p * g, win, padding=pad) - mu_p * mu_g
    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sig_x + C2)) / \
               ((mu_p**2 + mu_g**2 + C1) * (sig_p + sig_g + C2))
    return 1.0 - ssim_map.mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient/edge loss for fine process preservation.
    
    Penalizes blurry neurite boundaries using Sobel-like finite differences.
    Particularly valuable for thin processes that MSE tends to smooth over.
    """
    # Sobel-like finite differences
    dy_pred = pred[1:, :] - pred[:-1, :]
    dx_pred = pred[:, 1:] - pred[:, :-1]
    dy_gt = target[1:, :] - target[:-1, :]
    dx_gt = target[:, 1:] - target[:, :-1]
    return F.mse_loss(dx_pred, dx_gt) + F.mse_loss(dy_pred, dy_gt)


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def lpips_metric(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> float:
    if lpips_model is None:
        return float("nan")
    with torch.no_grad():
        p = pred.detach().float().clamp(0, 1)[None, None].repeat(1, 3, 1, 1) * 2 - 1
        t = target.detach().float().clamp(0, 1)[None, None].repeat(1, 3, 1, 1) * 2 - 1
        return float(lpips_model(p, t).mean().item())


# ===================================================================
#  Aspect-ratio helpers
# ===================================================================
def compute_aspect_scales(vol_shape: Tuple[int, int, int]) -> torch.Tensor:
    Z, Y, X = vol_shape
    m = float(max(X, Y, Z))
    return torch.tensor([X / m, Y / m, Z / m], dtype=torch.float32)


def apply_aspect_correction(
    gaussians:     GaussianParameters,
    aspect_scales: torch.Tensor,
) -> GaussianParameters:
    s = aspect_scales.to(gaussians.means.device)
    S = torch.diag(s)
    return GaussianParameters(
        means       = gaussians.means * s.unsqueeze(0),
        covariances = S.unsqueeze(0) @ gaussians.covariances @ S.unsqueeze(0).transpose(-2, -1),
        intensities = gaussians.intensities,
    )


# ===================================================================
#  GT MIP dataset generation from raw 3D volume
# ===================================================================
def load_volume(tif_path: str) -> np.ndarray:
    import tifffile
    vol = tifffile.imread(tif_path).astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def _sample_volume_trilinear(vol: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)
    N    = points.shape[0]
    grid = points.reshape(1, 1, 1, N, 3)
    return F.grid_sample(vol, grid, mode='bilinear',
                         padding_mode='zeros',
                         align_corners=True).reshape(N)


def render_gt_mip(
    vol:           torch.Tensor,
    camera:        Camera,
    R:             torch.Tensor,
    T:             torch.Tensor,
    n_samples:     int                     = 256,
    near:          float                   = 0.5,
    far:           float                   = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = vol.device
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)
    H, W   = camera.height, camera.width
    R_cw   = R.T
    T_cw   = -R_cw @ T
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dirs_cam = torch.stack([
        (gx - camera.cx) / camera.fx,
        (gy - camera.cy) / camera.fy,
        torch.ones_like(gx),
    ], dim=-1)
    dirs_world = (R_cw @ dirs_cam.reshape(-1, 3).T).T
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)
    origin = T_cw.unsqueeze(0)
    t_vals = torch.linspace(near, far, n_samples, device=device)
    points = origin.unsqueeze(1) + t_vals[None, :, None] * dirs_world.unsqueeze(1)
    N_rays   = H * W
    mip_flat = torch.zeros(N_rays, device=device)
    chunk    = 1024
    for i in range(0, N_rays, chunk):
        j   = min(i + chunk, N_rays)
        pts = points[i:j].reshape(-1, 3)
        if aspect_scales is not None:
            inv_s = 1.0 / aspect_scales.to(pts.device)
            pts   = pts * inv_s.unsqueeze(0)
        vals = _sample_volume_trilinear(vol, pts).reshape(j - i, n_samples)
        mip_flat[i:j] = vals.max(dim=1)[0]
    return mip_flat.reshape(H, W)


def _orbit_pose(
    elevation_deg: float,
    azimuth_deg:   float,
    radius:        float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    el, az  = math.radians(elevation_deg), math.radians(azimuth_deg)
    cam_pos = torch.tensor([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=torch.float32)
    forward = -cam_pos / (cam_pos.norm() + 1e-8)
    up_y    = torch.tensor([0.0, 1.0, 0.0])
    up_z    = torch.tensor([0.0, 0.0, 1.0])
    pole_weight = forward[1].abs()
    world_up    = (1.0 - pole_weight) * up_y + pole_weight * up_z
    world_up    = world_up / (world_up.norm() + 1e-8)
    right = torch.linalg.cross(forward, world_up)
    right = right / (right.norm() + 1e-8)
    up    = torch.linalg.cross(right, forward)
    up    = up    / (up.norm()    + 1e-8)
    R = torch.stack([right, -up, forward], dim=0)
    T = -R @ cam_pos
    return R, T


def generate_camera_poses(
    n_azimuth:            int                   = 12,
    n_elevation:          int                   = 5,
    elevation_range:      Tuple[float, float]   = (-60.0, 60.0),
    radius:               float                 = 3.5,
    include_axis_aligned: bool                  = True,
) -> List[dict]:
    poses = []
    for el in np.linspace(elevation_range[0], elevation_range[1], n_elevation):
        for az in np.linspace(0, 360, n_azimuth, endpoint=False):
            R, T = _orbit_pose(float(el), float(az), radius)
            poses.append({'R': R, 'T': T,
                          'elevation': float(el), 'azimuth': float(az)})
    if include_axis_aligned:
        for el, az in [(0,0),(0,180),(0,90),(0,-90),(89,0),(-89,0)]:
            R, T = _orbit_pose(float(el), float(az), radius)
            poses.append({'R': R, 'T': T,
                          'elevation': float(el), 'azimuth': float(az)})
    return poses


def generate_camera_poses_from_config(cfg: dict) -> List[dict]:
    """Generate camera poses from the 'poses' section of config.yml."""
    p = cfg["poses"]
    return generate_camera_poses(
        n_azimuth            = p["n_azimuth"],
        n_elevation          = p["n_elevation"],
        elevation_range      = (p["elevation_min"], p["elevation_max"]),
        radius               = p["radius"],
        include_axis_aligned = p["include_axis_aligned"],
    )


def generate_mip_dataset(
    vol:           torch.Tensor,
    camera:        Camera,
    poses:         List[dict],
    n_ray_samples: int                     = 256,
    near:          float                   = 0.5,
    far:           float                   = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> List[dict]:
    dataset = []
    device  = vol.device
    for idx, pose in enumerate(poses):
        R, T = pose['R'].to(device), pose['T'].to(device)
        mip  = render_gt_mip(vol, camera, R, T,
                             n_samples=n_ray_samples,
                             near=near, far=far,
                             aspect_scales=aspect_scales)
        dataset.append({
            'image':     mip,
            'R':         R,
            'T':         T,
            'elevation': pose['elevation'],
            'azimuth':   pose['azimuth'],
        })
        if (idx + 1) % 10 == 0 or idx == len(poses) - 1:
            print(f"  GT MIP: {idx+1}/{len(poses)} projections rendered")
    return dataset


# ===================================================================
#  MIPSplattingTrainer
# ===================================================================
class MIPSplattingTrainer:
    """
    Train intensity-only 3D Gaussians against multi-view MIP ground truth.
    All hyperparameters are supplied via a config dict (from config.yml).
    """

    def __init__(
        self,
        means:           torch.Tensor,
        log_scales:      torch.Tensor,
        quaternions:     torch.Tensor,
        log_intensities: torch.Tensor,
        cfg:             dict,
        aspect_scales:   Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        means           : (K, 3)
        log_scales      : (K, 3) or (K,)/(K,1) — auto-upgraded to (K, 3)
        quaternions     : (K, 4)
        log_intensities : (K,)
        cfg             : full config dict from load_config()
        aspect_scales   : (3,) from compute_aspect_scales()
        """
        self.device        = means.device
        self.aspect_scales = aspect_scales

        # Pull sub-sections once for readability
        t   = cfg["training"]
        l   = cfg["loss"]
        sc  = cfg["scale_clamp"]
        li  = cfg["log_intensity_clamp"]
        pr  = cfg["pruning"]
        dn  = cfg["densification"]

        self.beta_mip         = t.get("beta_mip_init", t["beta_mip"])
        self.beta_mip_final   = t["beta_mip"]
        self.beta_mip_init    = t.get("beta_mip_init", self.beta_mip_final)
        self.beta_warmup_epochs = t.get("beta_warmup_epochs", 0)
        self.chunk_size       = t["chunk_size"]
        self.views_per_step   = t.get("views_per_step", 4)

        self.fg_weight        = l.get("fg_weight", 5.0)
        self.lambda_ssim      = l.get("lambda_ssim", 0.2)
        self.lambda_edge      = l.get("lambda_edge", 0.0)  # gradient/edge loss for fine processes
        self.lambda_sparse    = l.get("lambda_sparse", 0.0)  # intensity sparsity regularization
        self.lambda_scale     = l["lambda_scale"]
        self.scale_min        = l["scale_min"]
        self.lambda_scale_max = l["lambda_scale_max"]
        self.scale_max        = l["scale_max"]

        self.log_scale_min    = sc["log_min"]
        self.log_scale_max    = sc["log_max"]
        self.log_intens_min   = li["min"]
        self.log_intens_max   = li["max"]

        self.prune_intens_thresh  = pr["intens_thresh"]
        self.prune_min_gaussians  = pr["min_gaussians"]

        self.densify_every        = dn["densify_every"]
        self.densify_start_epoch  = dn["start_epoch"]
        self.densify_stop_epoch   = dn["stop_epoch"]
        self.densify_grad_thresh  = dn["grad_thresh"]
        self.densify_scale_thresh = dn["scale_thresh"]
        self.max_gaussians        = dn["max_gaussians"]
        self.split_factor         = dn["split_factor"]

        lr = t["lr"]
        self.lr_init  = lr
        self.lr_final = t.get("lr_final", lr * 0.01)
        self._lr_multipliers = [1.0, 0.5, 0.3, 1.0]

        # Auto-upgrade isotropic → anisotropic log_scales
        if log_scales.ndim == 1 or (log_scales.ndim == 2 and log_scales.shape[1] == 1):
            ls = log_scales.reshape(-1, 1).expand(-1, 3).clone().contiguous()
            print(f"  [MIPSplattingTrainer] log_scales upgraded "
                  f"{log_scales.shape} → {ls.shape}  (anisotropic)")
        else:
            ls = log_scales.clone()

        self.means           = nn.Parameter(means.clone())
        self.log_scales      = nn.Parameter(ls)
        self.quaternions     = nn.Parameter(quaternions.clone())
        self.log_intensities = nn.Parameter(log_intensities.clone())

        self.optimizer = torch.optim.Adam([
            {'params': [self.means],           'lr': lr},
            {'params': [self.log_scales],      'lr': lr * 0.5},
            {'params': [self.quaternions],     'lr': lr * 0.3},
            {'params': [self.log_intensities], 'lr': lr},
        ])

        K = self.means.shape[0]
        self._grad_accum = torch.zeros(K, device=self.device)
        self._grad_count = torch.zeros(K, device=self.device)

    # ------------------------------------------------------------------
    #  Parameter construction
    # ------------------------------------------------------------------
    def _build_gaussians(self) -> GaussianParameters:
        K = self.means.shape[0]
        scales = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        q = F.normalize(self.quaternions, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        Rm = torch.zeros(K, 3, 3, device=self.device, dtype=q.dtype)
        Rm[:, 0, 0] = 1 - 2*(y*y + z*z);  Rm[:, 0, 1] = 2*(x*y - w*z);  Rm[:, 0, 2] = 2*(x*z + w*y)
        Rm[:, 1, 0] = 2*(x*y + w*z);      Rm[:, 1, 1] = 1 - 2*(x*x+z*z);Rm[:, 1, 2] = 2*(y*z - w*x)
        Rm[:, 2, 0] = 2*(x*z - w*y);      Rm[:, 2, 1] = 2*(y*z + w*x);  Rm[:, 2, 2] = 1 - 2*(x*x+y*y)
        S2  = torch.diag_embed(scales ** 2)
        cov = Rm @ S2 @ Rm.transpose(-2, -1)
        intensities = torch.sigmoid(self.log_intensities)
        return GaussianParameters(means=self.means, covariances=cov, intensities=intensities)

    def _build_gaussians_corrected(self) -> GaussianParameters:
        g = self._build_gaussians()
        if self.aspect_scales is not None:
            g = apply_aspect_correction(g, self.aspect_scales)
        return g

    # ------------------------------------------------------------------
    #  Single-projection forward + backward
    # ------------------------------------------------------------------
    def _forward_projection(
        self,
        camera:    Camera,
        gt_mip:    torch.Tensor,
        R:         torch.Tensor,
        T:         torch.Tensor,
        gaussians: GaussianParameters,
    ) -> dict:
        pred_mip, n_vis = render_mip_projection(
            gaussians, camera, R, T,
            beta=self.beta_mip, chunk_size=self.chunk_size)

        mse  = weighted_mse_loss(pred_mip, gt_mip, fg_weight=self.fg_weight)
        # PSNR from unweighted MSE for fair metric reporting
        mse_unw = F.mse_loss(pred_mip, gt_mip)
        psnr = -10.0 * torch.log10(mse_unw.clamp(min=1e-12))

        # SSIM loss — differentiable, contributes to gradient
        ssim_loss = ssim_loss_fn(pred_mip, gt_mip)   # 1 - SSIM
        ssim_val  = 1.0 - ssim_loss.item()

        # Edge loss — gradient preservation for fine processes
        edge_l = edge_loss(pred_mip, gt_mip) if self.lambda_edge > 0 else torch.tensor(0.0, device=pred_mip.device)
        edge_val = edge_l.item() if self.lambda_edge > 0 else 0.0

        with torch.no_grad():
            mae_val  = F.l1_loss(pred_mip, gt_mip)

        scales      = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_small = self.lambda_scale * torch.clamp(
            self.scale_min - scales, min=0.0).mean()
        scale_big   = self.lambda_scale_max * torch.clamp(
            scales - self.scale_max, min=0.0).mean()
        scale_reg   = scale_small + scale_big

        # Intensity sparsity regularization — push Gaussians toward clearly on or off
        # This improves densification decisions downstream
        intensities = torch.sigmoid(self.log_intensities)
        sparsity_reg = self.lambda_sparse * intensities.mean() if self.lambda_sparse > 0 else torch.tensor(0.0, device=pred_mip.device)
        sparsity_val = sparsity_reg.item() if self.lambda_sparse > 0 else 0.0

        loss = mse + scale_reg
        # + self.lambda_ssim * ssim_loss + self.lambda_edge * edge_l + scale_reg + sparsity_reg
        loss.backward()
        del pred_mip

        return {
            'loss':      loss.item(),
            'mse':       mse.item(),
            'psnr':      psnr.item(),
            'ssim':      ssim_val,
            'mae':       mae_val.item(),
            'edge':      edge_val,
            'sparsity':  sparsity_val,
            'scale_reg': scale_reg.item(),
            'n_visible': n_vis,
        }

    # ------------------------------------------------------------------
    #  Epoch  (mini-batch SGD: multiple optimizer steps per epoch)
    # ------------------------------------------------------------------
    def train_epoch(self, camera: Camera, dataset: List[dict]) -> dict:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        vps = self.views_per_step
        all_metrics = []
        params = [self.means, self.log_scales,
                  self.quaternions, self.log_intensities]

        for batch_start in range(0, len(indices), vps):
            batch_idx = indices[batch_start : batch_start + vps]
            self.optimizer.zero_grad()

            for vi in batch_idx:
                view = dataset[vi]
                gaussians = self._build_gaussians_corrected()
                m = self._forward_projection(
                    camera, view['image'], view['R'], view['T'], gaussians)
                
                all_metrics.append(m)

            B = len(batch_idx)
            if B > 1:
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(B)

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if self.means.grad is not None:
                with torch.no_grad():
                    gn = self.means.grad.norm(dim=-1)
                    self._grad_accum += gn
                    self._grad_count += 1

            self.optimizer.step()

        with torch.no_grad():
            self.log_scales.data.clamp_(self.log_scale_min, self.log_scale_max)
            self.log_intensities.data.clamp_(self.log_intens_min, self.log_intens_max)
            self.means.data.clamp_(-1.0, 1.0)

        if self.means.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0]}

    def validate_epoch(self, camera: Camera, dataset: List[dict]) -> dict:
        if not dataset:
            return {'psnr': 0.0, 'ssim': 0.0, 'mae': 0.0, 'n_visible': 0.0}

        all_metrics = []
        with torch.no_grad():
            gaussians = self._build_gaussians_corrected()
            for view in dataset:
                pred_mip, n_vis = render_mip_projection(
                    gaussians, camera, view['R'], view['T'],
                    beta=self.beta_mip, chunk_size=self.chunk_size)
                assert pred_mip.requires_grad, \
                "CUDA kernel broke autograd — pred_mip has no grad_fn"
                print(pred_mip.grad_fn)  # should show a non-None grad_fn

                gt_mip = view['image']
                mse_unw = F.mse_loss(pred_mip, gt_mip)
                psnr = -10.0 * torch.log10(mse_unw.clamp(min=1e-12))
                ssim_val = 1.0 - ssim_loss_fn(pred_mip, gt_mip).item()
                mae_val = F.l1_loss(pred_mip, gt_mip)
                all_metrics.append({
                    'psnr': psnr.item(),
                    'ssim': ssim_val,
                    'mae': mae_val.item(),
                    'n_visible': float(n_vis),
                })
                del pred_mip

        if self.means.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0]}

    # ------------------------------------------------------------------
    #  Optimizer / accumulator helpers
    # ------------------------------------------------------------------
    def _rebuild_optimizer(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam([
            {'params': [self.means],           'lr': lr},
            {'params': [self.log_scales],      'lr': lr * 0.5},
            {'params': [self.quaternions],     'lr': lr * 0.3},
            {'params': [self.log_intensities], 'lr': lr},
        ])

    def _reset_grad_accum(self):
        K = self.means.shape[0]
        self._grad_accum = torch.zeros(K, device=self.device)
        self._grad_count = torch.zeros(K, device=self.device)

    # ------------------------------------------------------------------
    #  Pruning
    # ------------------------------------------------------------------
    def prune_gaussians(self, epoch: int = 0) -> int:
        with torch.no_grad():
            intens   = torch.sigmoid(self.log_intensities)
            keep     = intens > self.prune_intens_thresh
            n_before = keep.shape[0]
            n_keep   = int(keep.sum().item())
            if n_keep >= n_before or n_keep < self.prune_min_gaussians:
                return 0
            self.means           = nn.Parameter(self.means.data[keep].clone())
            self.log_scales      = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions     = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_intensities = nn.Parameter(self.log_intensities.data[keep].clone())
            self._rebuild_optimizer()
            self._reset_grad_accum()
            n_pruned = n_before - n_keep
            print(f"  [Prune @ epoch {epoch}] {n_before} → {n_keep} "
                  f"(removed {n_pruned}, intensity < {self.prune_intens_thresh})")
            return n_pruned

    # ------------------------------------------------------------------
    #  Densification
    # ------------------------------------------------------------------
    def densify_and_prune(self, epoch: int = 0):
        with torch.no_grad():
            K        = self.means.shape[0]
            avg_grad = self._grad_accum / self._grad_count.clamp(min=1)
            high_grad  = avg_grad > self.densify_grad_thresh
            scales     = torch.exp(self.log_scales).clamp(1e-5, 1e2)
            max_scale  = scales.max(dim=-1).values
            split_mask = high_grad & (max_scale > self.densify_scale_thresh)
            clone_mask = high_grad & ~split_mask
            n_split    = int(split_mask.sum().item())
            n_clone    = int(clone_mask.sum().item())

            if K + n_split + n_clone > self.max_gaussians:
                self._reset_grad_accum()
                return 0, 0

            new_m, new_ls, new_q, new_li = [], [], [], []

            if n_split > 0:
                s_m  = self.means.data[split_mask]
                s_ls = self.log_scales.data[split_mask]
                s_q  = self.quaternions.data[split_mask]
                s_li = self.log_intensities.data[split_mask]
                reduced_ls = s_ls - math.log(self.split_factor)
                s_scales   = torch.exp(s_ls)

                q = F.normalize(s_q, p=2, dim=-1)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                col0   = torch.stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)], dim=-1)
                col1   = torch.stack([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)], dim=-1)
                col2   = torch.stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)], dim=-1)
                R_cols = torch.stack([col0, col1, col2], dim=-1)
                max_axis  = s_scales.argmax(dim=-1)
                principal = R_cols[torch.arange(len(max_axis), device=self.device), :, max_axis]
                offset    = principal * s_scales.max(dim=-1, keepdim=True).values

                for sign in (1.0, -1.0):
                    new_m .append(s_m + sign * offset)
                    new_ls.append(reduced_ls.clone())
                    new_q .append(s_q.clone())
                    new_li.append(s_li.clone())

            if n_clone > 0:
                new_m .append(self.means.data[clone_mask].clone())
                new_ls.append(self.log_scales.data[clone_mask].clone())
                new_q .append(self.quaternions.data[clone_mask].clone())
                new_li.append(self.log_intensities.data[clone_mask].clone())

            keep_mask = ~split_mask
            all_m  = torch.cat([self.means.data[keep_mask]]           + new_m,  dim=0)
            all_ls = torch.cat([self.log_scales.data[keep_mask]]      + new_ls, dim=0)
            all_q  = torch.cat([self.quaternions.data[keep_mask]]     + new_q,  dim=0)
            all_li = torch.cat([self.log_intensities.data[keep_mask]] + new_li, dim=0)

            intens = torch.sigmoid(all_li)
            alive  = intens > self.prune_intens_thresh
            if alive.sum() < self.prune_min_gaussians:
                alive = torch.ones(all_m.shape[0], dtype=torch.bool, device=self.device)
            n_pruned = int((~alive).sum().item())

            self.means           = nn.Parameter(all_m[alive])
            self.log_scales      = nn.Parameter(all_ls[alive])
            self.quaternions     = nn.Parameter(all_q[alive])
            self.log_intensities = nn.Parameter(all_li[alive])

            self._rebuild_optimizer()
            self._reset_grad_accum()

            K_new = self.means.shape[0]
            print(f"  [Densify @ epoch {epoch}] K: {K} → {K_new}  "
                  f"(split {n_split}, clone {n_clone}, pruned {n_pruned})")
            return n_split, n_clone

    # ------------------------------------------------------------------
    #  Full training loop
    # ------------------------------------------------------------------
    def train(
        self,
        camera:    Camera,
        dataset:   List[dict],
        cfg:       dict,
        save_path: Optional[str] = None,
    ) -> List[dict]:
        t          = cfg["training"]
        pr         = cfg["pruning"]
        n_epochs   = t["n_epochs"]
        start_epoch = t.get("start_epoch", 0)
        total_epochs = t.get("total_epochs", start_epoch + n_epochs)
        log_every  = t["log_every"]
        save_every = t["save_every"]
        prune_every = pr["prune_every"]

        M       = len(dataset)
        history = []

        val_ratio = t.get("val_ratio", 0.1)
        val_seed = t.get("val_seed", 42)
        val_min_views = t.get("val_min_views", 8)

        train_dataset = dataset
        val_dataset = []
        if M >= 2 and val_ratio > 0.0:
            n_val = max(val_min_views, int(round(M * val_ratio)))
            n_val = min(max(1, n_val), M - 1)
            idx = list(range(M))
            rnd = random.Random(val_seed)
            rnd.shuffle(idx)
            val_idx = set(idx[:n_val])
            train_dataset = [dataset[i] for i in range(M) if i not in val_idx]
            val_dataset = [dataset[i] for i in range(M) if i in val_idx]

        M_train = len(train_dataset)
        M_val = len(val_dataset)
        steps_per_epoch = max(1, -(-M_train // self.views_per_step))  # ceil div

        print(f"\nNeuroSGM — MIP Splatting Training")
        print(f"  Epochs         : {start_epoch + 1} → {start_epoch + n_epochs}  "
              f"(total schedule {total_epochs})")
        if M_val > 0:
            print(f"  Projections    : train={M_train}  val={M_val}  total={M}")
        else:
            print(f"  Projections    : M = {M_train}")
        print(f"  Views/step     : {self.views_per_step}  "
              f"→ {steps_per_epoch} optimizer steps/epoch  "
              f"({steps_per_epoch * n_epochs} total this run)")
        print(f"  Gaussians      : K = {self.means.shape[0]}")
        print(f"  log_scales     : {self.log_scales.shape}  (anisotropic)")
        print(f"  LR             : {self.lr_init} → {self.lr_final}  (cosine)")
        print(f"  beta_mip       : {self.beta_mip_init} → {self.beta_mip_final}  "
              f"(warmup {self.beta_warmup_epochs} ep)")
        print(f"  densify_every  : {self.densify_every}  "
              f"(epochs {self.densify_start_epoch}–{self.densify_stop_epoch})")
        print(f"  prune_every    : {prune_every}")
        print("-" * 60)

        pbar = tqdm(range(start_epoch + 1, start_epoch + n_epochs + 1), desc="Training", unit="ep",
                    dynamic_ncols=True)
        best = {'loss': float('inf'), 'psnr': 0.0, 'ssim': 0.0, 'mae': float('inf')}

        for epoch in pbar:
            # ── Cosine LR annealing ──
            frac = (epoch - 1) / max(total_epochs - 1, 1)
            cosine_lr = self.lr_final + 0.5 * (self.lr_init - self.lr_final) * (
                1.0 + math.cos(math.pi * frac))
            for pg, mult in zip(self.optimizer.param_groups,
                                self._lr_multipliers):
                pg['lr'] = cosine_lr * mult

            # ── Beta warmup ──
            if self.beta_warmup_epochs > 0 and epoch <= self.beta_warmup_epochs:
                self.beta_mip = self.beta_mip_init + (
                    self.beta_mip_final - self.beta_mip_init
                ) * (epoch / self.beta_warmup_epochs)
            else:
                self.beta_mip = self.beta_mip_final

            if (self.densify_every > 0
                    and epoch % self.densify_every == 0
                    and self.densify_start_epoch <= epoch <= self.densify_stop_epoch):
                self.densify_and_prune(epoch)
            elif prune_every > 0 and epoch % prune_every == 0:
                self.prune_gaussians(epoch)

            metrics = self.train_epoch(camera, train_dataset)
            history.append(metrics)

            best['loss'] = min(best['loss'], metrics['loss'])
            best['psnr'] = max(best['psnr'], metrics['psnr'])
            best['ssim'] = max(best['ssim'], metrics['ssim'])
            best['mae']  = min(best['mae'],  metrics['mae'])

            pbar.set_postfix({
                'loss': f"{best['loss']:.5f}",
                'psnr': f"{best['psnr']:.2f}",
                'ssim': f"{best['ssim']:.4f}",
                'mae':  f"{best['mae']:.5f}",
                'K':    self.means.shape[0],
            })

            if epoch % log_every == 0:
                tqdm.write(
                    f"  Epoch {epoch:>4d}/{total_epochs}  "
                    f"loss={metrics['loss']:.5f}  psnr={metrics['psnr']:.2f} dB  "
                    f"ssim={metrics['ssim']:.4f}  K={self.means.shape[0]}"
                )

            if save_path and epoch % save_every == 0:
                tqdm.write(f"  Checkpoint → {save_path.format(epoch=epoch)}")
                self._save_checkpoint(save_path.format(epoch=epoch), epoch)

                if M_val > 0:
                    val_metrics = self.validate_epoch(camera, val_dataset)
                    history[-1]['val_psnr'] = val_metrics['psnr']
                    history[-1]['val_ssim'] = val_metrics['ssim']
                    history[-1]['val_mae'] = val_metrics['mae']
                    tqdm.write(
                        f"  Validation @ {epoch:>4d}  "
                        f"psnr={val_metrics['psnr']:.2f} dB  "
                        f"ssim={val_metrics['ssim']:.4f}  "
                        f"mae={val_metrics['mae']:.5f}"
                    )

        pbar.close()
        if save_path:
            self._save_checkpoint(save_path.format(epoch=start_epoch + n_epochs),
                                  start_epoch + n_epochs)
        return history

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'means':           self.means.data.cpu(),
            'log_scales':      self.log_scales.data.cpu(),
            'quaternions':     self.quaternions.data.cpu(),
            'log_intensities': self.log_intensities.data.cpu(),
            'epoch':           epoch,
        }, path)
        print(f"  Checkpoint → {path}")


# ===================================================================
#  Analysis utilities
# ===================================================================
def save_training_analysis(
    history:            List[dict],
    validation_metrics: List[dict],
    out_dir:            str,
    validation_renders: Optional[List[dict]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if history:
        keys = sorted({k for h in history for k in h.keys()})
        with open(os.path.join(out_dir, "training_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
            w.writeheader()
            for i, h in enumerate(history, 1):
                w.writerow({"epoch": i, **h})
    if validation_metrics:
        keys = sorted({k for v in validation_metrics for k in v.keys()})
        with open(os.path.join(out_dir, "validation_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(validation_metrics)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable ({exc})"); return
    if history:
        epochs = np.arange(1, len(history) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(epochs, [h['mse']       for h in history], color='tab:blue')
        axes[0].set_title('MSE Loss');  axes[0].set_xlabel('epoch')
        axes[1].plot(epochs, [h['psnr']      for h in history], color='tab:green')
        axes[1].set_title('PSNR (dB)'); axes[1].set_xlabel('epoch')
        axes[2].plot(epochs, [h.get('ssim', 0.0) for h in history], color='tab:orange', label='ssim')
        axes[2].plot(epochs, [h['scale_reg'] for h in history], color='tab:red', label='scale')
        axes[2].set_title('Regularisers'); axes[2].set_xlabel('epoch'); axes[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_metrics.png"), dpi=160)
        plt.close(fig)
    if validation_renders:
        n = min(6, len(validation_renders))
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.0))
        if n == 1:
            axes = np.array(axes).reshape(2, 1)
        for i in range(n):
            item = validation_renders[i]
            axes[0, i].imshow(item['gt'],   cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"GT {item.get('label', i)}")
            axes[0, i].axis('off')
            axes[1, i].imshow(item['pred'], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Pred {item.get('label', i)}")
            axes[1, i].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "validation_renders.png"), dpi=180)
        plt.close(fig)


# ===================================================================
#  Main
# ===================================================================
if __name__ == "__main__":
    print("NeuroSGM — MIP-supervised Gaussian Training")
    print("=" * 60)

    # ── 0. Load config ───────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config_splat.yml")
    cfg      = load_config(cfg_path)
    print(f"Config loaded: {cfg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load volume ───────────────────────────────────────────
    vol_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), cfg["dataset"]["vol_path"]))
    print(f"\nLoading volume: {vol_path}")
    vol_np  = load_volume(vol_path)
    Z, Y, X = vol_np.shape
    vol_gpu = torch.from_numpy(vol_np).to(device)
    print(f"  Shape (Z,Y,X): ({Z},{Y},{X})")

    aspect_scales = compute_aspect_scales((Z, Y, X))
    print(f"  Aspect scales (x,y,z): {aspect_scales.tolist()}")

    # ── 2. Load Gaussian checkpoint ──────────────────────────────
    ckpt_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), cfg["dataset"]["ckpt_path"]))
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    means           = ckpt["means"]
    log_scales      = ckpt["log_scales"]
    quaternions     = ckpt["quaternions"]
    log_intensities = ckpt.get("log_intensities", ckpt.get("log_amplitudes"))
    if log_intensities is None:
        raise KeyError("Checkpoint missing both 'log_intensities' and 'log_amplitudes'")
    print(f"  {means.shape[0]} Gaussians  |  log_scales: {log_scales.shape}")

    # ── 3. Camera ────────────────────────────────────────────────
    H, W   = int(Y), int(X)
    camera = Camera.from_config(cfg, width=W, height=H)
    print(f"\nCamera: {W}×{H}  fx={camera.fx:.1f}")

    # ── 4. Generate GT MIP dataset ───────────────────────────────
    print("\nGenerating camera poses...")
    poses = generate_camera_poses_from_config(cfg)
    print(f"  M = {len(poses)} projection views")

    rm = cfg["ray_marching"]
    print("\nRendering GT MIP dataset...")
    dataset = generate_mip_dataset(
        vol_gpu, camera, poses,
        n_ray_samples = rm["n_samples"],
        near          = rm["near"],
        far           = rm["far"],
        aspect_scales = aspect_scales,
    )

    # ── 5. Train ─────────────────────────────────────────────────
    out_cfg  = cfg["output"]
    base_dir = os.path.dirname(__file__)

    print(f"\n  CUDA MIP kernel: {'ACTIVE' if HAS_MIP_CUDA else 'not available (Python fallback)'}")

    print("\nInitialising MIPSplattingTrainer...")
    trainer = MIPSplattingTrainer(
        means           = means,
        log_scales      = log_scales,
        quaternions     = quaternions,
        log_intensities = log_intensities,
        cfg             = cfg,
        aspect_scales   = aspect_scales,
    )
    save_tmpl = os.path.abspath(os.path.join(
        base_dir, out_cfg["mip_ckpt_dir"], out_cfg["epoch_template"]))
    history = trainer.train(
        camera    = camera,
        dataset   = dataset,
        cfg       = cfg,
        save_path = save_tmpl,
    )

    # ── 6. Validation ────────────────────────────────────────────
    print("\nValidation renders...")
    lpips_model = None
    try:
        import lpips as _lpips
        lpips_model = _lpips.LPIPS(net='alex').to(device).eval()
    except Exception as exc:
        print(f"  LPIPS unavailable: {exc}")

    val_metrics, val_renders = [], []
    gaussians = trainer._build_gaussians_corrected()
    if not hasattr(gaussians, "intensities") and hasattr(gaussians, "weights"):
        gaussians = GaussianParameters(
            means=gaussians.means,
            covariances=gaussians.covariances,
            intensities=gaussians.weights,
        )
    beta_mip = getattr(trainer, "beta_mip", cfg["training"]["beta_mip"])

    for vi in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
        view = dataset[vi]
        pred_mip, n_vis = render_mip_projection(
            gaussians, camera, view['R'], view['T'], beta=beta_mip)
        gt      = view['image']
        p_score = psnr_metric(pred_mip, gt)
        s_score = float(1.0 - ssim_loss_fn(pred_mip, gt).item())
        lp      = lpips_metric(pred_mip, gt, lpips_model)
        val_metrics.append({
            'view_idx':  vi,
            'elevation': view['elevation'],
            'azimuth':   view['azimuth'],
            'psnr':      p_score,
            'ssim':      s_score,
            'lpips':     lp,
            'n_visible': n_vis,
        })
        val_renders.append({
            'view_idx': vi,
            'label':    f"el={view['elevation']:.0f}°",
            'gt':       gt.detach().float().clamp(0, 1).cpu().numpy(),
            'pred':     pred_mip.detach().float().clamp(0, 1).cpu().numpy(),
        })
        print(f"  View {vi:>3d}  PSNR={p_score:.2f} dB  "
              f"SSIM={s_score:.4f}  LPIPS={lp:.4f}  visible={n_vis}")

    figure_dir = os.path.join(base_dir, out_cfg["figure_dir"])
    save_training_analysis(history, val_metrics, figure_dir,
                           validation_renders=val_renders)
    print(f"\nSaved analysis → {figure_dir}")
    print("\n✓ NeuroSGM MIP training complete!")