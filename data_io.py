"""
neuro3dgs/data_io.py

Loaders for fluorescence microscopy .tif stacks and .swc neurite morphology files.
Produces normalised voxel grids and SWC node coordinates for Gaussian initialisation.
"""

from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
import warnings


# ─────────────────────────────────────────────────────────────────────────────
#  TIF / TIFF stack loader
# ─────────────────────────────────────────────────────────────────────────────

def load_tif_volume(
    path: str | Path,
    normalise: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[torch.Tensor, dict]:
    """
    Load a 3-D fluorescence microscopy TIFF stack.

    Returns
    -------
    volume : Tensor [D, H, W]  or  [C, D, H, W] if multi-channel
    meta   : dict with keys 'shape', 'dtype_orig', 'voxel_min', 'voxel_max'
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("Install tifffile: pip install tifffile")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIF not found: {path}")

    raw = tifffile.imread(str(path))   # numpy, shape varies
    meta = {"path": str(path), "shape_raw": raw.shape, "dtype_orig": str(raw.dtype)}

    # Normalise axes to [D, H, W] or [C, D, H, W]
    if raw.ndim == 2:
        raw = raw[np.newaxis, np.newaxis]          # 1 slice → [1,1,H,W]... no
        raw = raw[np.newaxis]                       # [1,H,W]
    elif raw.ndim == 3:
        pass                                        # [D,H,W] assumed
    elif raw.ndim == 4:
        pass                                        # [C,D,H,W] or [D,H,W,C]
        if raw.shape[-1] in (1, 2, 3, 4) and raw.shape[-1] < raw.shape[0]:
            raw = raw.transpose(3, 0, 1, 2)        # → [C,D,H,W]
    else:
        warnings.warn(f"Unexpected TIF shape {raw.shape}, trying to proceed.")

    vol = raw.astype(np.float32)

    vmin, vmax = float(vol.min()), float(vol.max())
    meta.update({"voxel_min": vmin, "voxel_max": vmax})

    if normalise and vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    tensor = torch.from_numpy(vol).to(dtype=dtype, device=device)
    meta["shape"] = tuple(tensor.shape)
    return tensor, meta


# ─────────────────────────────────────────────────────────────────────────────
#  SWC morphology loader
# ─────────────────────────────────────────────────────────────────────────────

def load_swc(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Parse an SWC file.

    Standard SWC columns:
        id  type  x  y  z  radius  parent_id

    Returns
    -------
    xyz     : Tensor [N, 3]  — node positions (x, y, z)
    radii   : Tensor [N]     — node radii
    parents : ndarray [N]    — parent IDs (-1 for roots)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SWC not found: {path}")

    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append([float(p) for p in parts])

    if not rows:
        raise ValueError(f"Empty or unreadable SWC: {path}")

    data = np.array(rows, dtype=np.float64)  # [N, 7+]
    # columns: 0=id, 1=type, 2=x, 3=y, 4=z, 5=radius, 6=parent_id
    xyz     = torch.tensor(data[:, 2:5], dtype=torch.float32, device=device)
    radii   = torch.tensor(data[:, 5],   dtype=torch.float32, device=device)
    parents = data[:, 6].astype(np.int64)

    return xyz, radii, parents


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate normalisation utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalise_swc_to_voxel(
    xyz: torch.Tensor,
    radii: torch.Tensor,
    vol_shape: Tuple[int, ...],
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map SWC world coordinates (microns) to voxel-space [0,1]^3.

    Parameters
    ----------
    xyz         : [N,3] in physical units (z, y, x ordering from SWC)
    radii       : [N]   in physical units
    vol_shape   : (D, H, W) in voxels
    voxel_size  : (dz, dy, dx) in same units as SWC; if None uses auto-scale

    Returns
    -------
    xyz_norm   : [N,3] in [0,1]^3
    radii_norm : [N]   as fraction of volume diagonal
    """
    D, H, W = vol_shape[-3], vol_shape[-2], vol_shape[-1]

    if voxel_size is not None:
        dz, dy, dx = voxel_size
        # Convert physical → voxel index
        xyz_vox = xyz.clone()
        xyz_vox[:, 0] = xyz[:, 0] / dx   # x
        xyz_vox[:, 1] = xyz[:, 1] / dy   # y
        xyz_vox[:, 2] = xyz[:, 2] / dz   # z
    else:
        # Fit SWC bounding box into voxel grid
        mn = xyz.min(dim=0).values
        mx = xyz.max(dim=0).values
        rng = (mx - mn).clamp(min=1e-6)
        target = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=xyz.device)
        xyz_vox = (xyz - mn) / rng * target

    # Normalise to [0,1]
    scale = torch.tensor([W - 1, H - 1, D - 1], dtype=torch.float32, device=xyz.device)
    xyz_norm   = xyz_vox / scale
    radius_scale = (scale.norm() + 1e-6)
    radii_norm = radii / radius_scale

    return xyz_norm, radii_norm


# ─────────────────────────────────────────────────────────────────────────────
#  Voxel → candidate Gaussian seed extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_seeds_from_volume(
    volume: torch.Tensor,
    threshold: float = 0.1,
    max_seeds: int = 200_000,
    strategy: str = "threshold",   # "threshold" | "grid" | "topk"
    grid_stride: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract candidate positions and intensities from a voxel volume.

    Returns
    -------
    coords      : [M, 3]  normalised (z,y,x) in [0,1]
    intensities : [M]     voxel values at those coords
    """
    # Volume can be [D,H,W] or [C,D,H,W]; use max-intensity projection across C
    if volume.ndim == 4:
        vol3d = volume.max(dim=0).values  # [D,H,W]
    else:
        vol3d = volume

    D, H, W = vol3d.shape

    if strategy == "threshold":
        mask = vol3d > threshold
        zyx  = mask.nonzero(as_tuple=False).float()  # [M, 3]
        vals = vol3d[mask]
        if len(zyx) > max_seeds:
            idx = torch.randperm(len(zyx))[:max_seeds]
            zyx, vals = zyx[idx], vals[idx]

    elif strategy == "grid":
        zz = torch.arange(0, D, grid_stride, device=vol3d.device)
        yy = torch.arange(0, H, grid_stride, device=vol3d.device)
        xx = torch.arange(0, W, grid_stride, device=vol3d.device)
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        zyx  = torch.stack([gz.flatten(), gy.flatten(), gx.flatten()], dim=1).float()
        vals = vol3d[gz.flatten().long(), gy.flatten().long(), gx.flatten().long()]

    elif strategy == "topk":
        flat_vals, flat_idx = vol3d.flatten().topk(min(max_seeds, vol3d.numel()))
        z = flat_idx // (H * W)
        y = (flat_idx % (H * W)) // W
        x = flat_idx % W
        zyx  = torch.stack([z, y, x], dim=1).float()
        vals = flat_vals

    else:
        raise ValueError(f"Unknown seed strategy: {strategy}")

    # Normalise to [0,1]
    scale = torch.tensor([D - 1, H - 1, W - 1], dtype=torch.float32, device=vol3d.device)
    coords_norm = zyx / scale.clamp(min=1e-6)

    return coords_norm, vals
