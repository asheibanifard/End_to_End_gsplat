"""
neuro3dgs/visualise.py

Post-training visualisation utilities:
  1.  Render XY / XZ / YZ max-intensity projections from a saved Gaussian model
  2.  Export as .ply point cloud (compatible with Blender / MeshLab / napari)
  3.  Matplotlib interactive viewer (no GPU required)

Usage:
    python visualise.py --ckpt outputs/run_01/gaussians_030000.pt \
                        --tif  data/neurite_stack.tif \
                        --mode all
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_model import NeuriteGaussians
from renderer       import Camera, render


# ═════════════════════════════════════════════════════════════════════════════
#  Orthographic MIP renders
# ═════════════════════════════════════════════════════════════════════════════

def render_mip_trio(
    gaussians: NeuriteGaussians,
    H: int = 512, W: int = 512,
    fov: float = 60.0,
    device: str = "cuda",
) -> dict:
    """
    Render XY, XZ, YZ soft-MIP projections.
    Returns dict of {name: np.ndarray [H,W,C]}.
    """
    camera = Camera(H, W, fov_deg=fov, device=device)
    views  = {}

    # XY — top-down (camera looks down -Z)
    M_xy = torch.eye(4, device=device)
    M_xy[2, 3] = -2.5  # push scene forward
    img, _, _ = render(gaussians, camera, M_xy)
    views["xy"] = img.detach().cpu().clamp(0, 1).numpy()

    # XZ — side view (rotate 90° around X)
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    M_xz = torch.tensor([
        [1, 0,  0, -0.5],
        [0, c, -s,  0.0],
        [0, s,  c, -2.5],
        [0, 0,  0,  1.0],
    ], device=device, dtype=torch.float32)
    img, _, _ = render(gaussians, camera, M_xz)
    views["xz"] = img.detach().cpu().clamp(0, 1).numpy()

    # YZ — front view (rotate 90° around Y)
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    M_yz = torch.tensor([
        [ c, 0, s, 0.0],
        [ 0, 1, 0, -0.5],
        [-s, 0, c, -2.5],
        [ 0, 0, 0,  1.0],
    ], device=device, dtype=torch.float32)
    img, _, _ = render(gaussians, camera, M_yz)
    views["yz"] = img.detach().cpu().clamp(0, 1).numpy()

    return views


# ═════════════════════════════════════════════════════════════════════════════
#  PLY export
# ═════════════════════════════════════════════════════════════════════════════

def export_ply(gaussians: NeuriteGaussians, path: str):
    """
    Export Gaussians as a coloured point cloud (PLY format).
    Colour = first 3 feature channels; alpha = opacity.
    """
    path = Path(path)
    means = gaussians.means.detach().cpu().numpy()        # [N, 3]
    feats = gaussians.features.detach().cpu().numpy()     # [N, C]
    op    = gaussians.opacity.detach().cpu().numpy()      # [N, 1]
    scales = gaussians.scales.detach().cpu().numpy()      # [N, 3]
    quats  = gaussians.quats.detach().cpu().numpy()       # [N, 4]
    N = means.shape[0]

    rgb = feats[:, :3] if feats.shape[1] >= 3 else np.repeat(feats[:, :1], 3, axis=1)
    rgb = (rgb * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    header = (
        f"ply\n"
        f"format ascii 1.0\n"
        f"element vertex {N}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"property float opacity\n"
        f"property float scale_x\n"
        f"property float scale_y\n"
        f"property float scale_z\n"
        f"property float rot_w\n"
        f"property float rot_x\n"
        f"property float rot_y\n"
        f"property float rot_z\n"
        f"end_header\n"
    )

    with open(path, "w") as f:
        f.write(header)
        for i in range(N):
            f.write(
                f"{means[i,0]:.6f} {means[i,1]:.6f} {means[i,2]:.6f} "
                f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]} "
                f"{op[i,0]:.4f} "
                f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} "
                f"{quats[i,0]:.6f} {quats[i,1]:.6f} {quats[i,2]:.6f} {quats[i,3]:.6f}\n"
            )
    print(f"  PLY exported → {path}  ({N} Gaussians)")


# ═════════════════════════════════════════════════════════════════════════════
#  Matplotlib viewer
# ═════════════════════════════════════════════════════════════════════════════

def show_mip_trio(views: dict, title: str = "NeuroSGM MIP"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    for ax, (name, img) in zip(axes, views.items()):
        if img.shape[-1] == 1:
            ax.imshow(img[..., 0], cmap="gray")
        elif img.shape[-1] >= 3:
            ax.imshow(img[..., :3])
        else:
            ax.imshow(img[..., 0], cmap="gray")
        ax.set_title(f"{name.upper()} projection")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def interactive_orbit(
    gaussians: NeuriteGaussians,
    H: int = 512, W: int = 512,
    fov: float = 60.0,
    n_frames: int = 72,
):
    """
    Simple matplotlib animation of an orbiting camera.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    device = next(iter([gaussians.means.device])).__str__() if False else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    camera = Camera(H, W, fov_deg=fov, device=device)

    def _view_mat(t):
        theta = t * 2 * math.pi
        radius = 2.5
        eye = torch.tensor([
            radius * math.sin(theta) + 0.5,
            0.5,
            radius * math.cos(theta) + 0.5,
        ], device=device)
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    with torch.no_grad():
        img0, _, _ = render(gaussians, camera, _view_mat(0))
        im_plot = ax.imshow(img0.cpu().clamp(0, 1).numpy()[..., :3])

    def update(frame):
        t = frame / n_frames
        with torch.no_grad():
            img, _, _ = render(gaussians, camera, _view_mat(t))
        im_plot.set_data(img.cpu().clamp(0, 1).numpy()[..., :3])
        return [im_plot]

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=50, blit=True)
    plt.title("NeuroSGM — orbit view")
    plt.show()
    return ani


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="Path to saved .pt Gaussian model")
    p.add_argument("--out",    default=None,  help="Directory to save outputs")
    p.add_argument("--H",      type=int,   default=512)
    p.add_argument("--W",      type=int,   default=512)
    p.add_argument("--fov",    type=float, default=60.0)
    p.add_argument("--mode",   default="all",
                   choices=["mip", "ply", "orbit", "all"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Visualise] Loading {args.ckpt} …")
    gaussians = NeuriteGaussians.load(args.ckpt, device=device)
    print(f"            N={gaussians.N}  C={gaussians.features.shape[1]}")

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("mip", "all"):
        print("[Visualise] Rendering MIP trio …")
        views = render_mip_trio(gaussians, args.H, args.W, args.fov, device)
        # Save
        try:
            from PIL import Image
            for name, img in views.items():
                arr = (img[..., :3] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(str(out_dir / f"mip_{name}.png"))
        except ImportError:
            pass
        show_mip_trio(views)

    if args.mode in ("ply", "all"):
        export_ply(gaussians, str(out_dir / "gaussians.ply"))

    if args.mode in ("orbit", "all"):
        interactive_orbit(gaussians, args.H, args.W, args.fov)
