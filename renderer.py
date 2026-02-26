"""
neuro3dgs/renderer.py

Differentiable MIP splatting renderer.
Wraps the CUDA kernels (or a pure-PyTorch fallback) behind a clean
torch.autograd.Function interface so the entire pipeline is differentiable.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# ─── Try to import compiled CUDA extension ────────────────────────────────────
try:
    import neuro3dgs_cuda as _C
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(
        "neuro3dgs_cuda not compiled. Falling back to pure-PyTorch renderer "
        "(10–100× slower). Run `pip install -e .` to build the CUDA extension.",
        RuntimeWarning,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Camera utilities
# ═════════════════════════════════════════════════════════════════════════════

class Camera:
    """Pinhole camera for projecting normalised-volume Gaussians."""

    def __init__(
        self,
        H: int, W: int,
        fov_deg: float = 60.0,
        znear: float = 0.01,
        zfar:  float = 10.0,
        device: str = "cpu",
    ):
        self.H, self.W = H, W
        self.fx = W / (2 * math.tan(math.radians(fov_deg / 2)))
        self.fy = self.fx
        self.cx = W / 2.0
        self.cy = H / 2.0
        self.znear, self.zfar = znear, zfar
        self.device = device

    def view_matrix_orthographic(self, axis: int = 2) -> torch.Tensor:
        """
        Return a 4×4 view matrix for orthographic projection along
        axis 0 (X→Z), 1 (Y→Z), or 2 (Z→Z, top-down MIP).
        Centres the volume in camera view.
        """
        perms = {
            0: [2, 1, 0, 3],   # look along X
            1: [0, 2, 1, 3],   # look along Y
            2: [0, 1, 2, 3],   # look along Z (top-down)
        }
        M = torch.eye(4, device=self.device)
        # Translate centre (0.5,0.5,0.5) → camera space origin
        M[:3, 3] = -0.5
        # Push scene along camera +Z so it's in front of camera (z > 0)
        M[2, 3] += 2.0
        return M

    def view_matrix_perspective(
        self,
        eye: torch.Tensor,     # [3]
        at: Optional[torch.Tensor] = None,
        up: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if at is None:
            at = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=self.device)

        z = F.normalize(eye - at, dim=0)
        x = F.normalize(torch.cross(up, z, dim=0), dim=0)
        y = torch.cross(z, x, dim=0)

        M = torch.eye(4, device=self.device)
        M[0, :3] = x;  M[0, 3] = -(x * eye).sum()
        M[1, :3] = y;  M[1, 3] = -(y * eye).sum()
        M[2, :3] = z;  M[2, 3] = -(z * eye).sum()
        return M


# ═════════════════════════════════════════════════════════════════════════════
#  Autograd wrapper
# ═════════════════════════════════════════════════════════════════════════════

class MIPSplatFunction(torch.autograd.Function):
    """
    Custom autograd Function that dispatches to the CUDA kernels.
    Falls back to a PyTorch implementation when CUDA extension is absent.
    """

    @staticmethod
    def forward(ctx, means3d, cov3d, opacity, features, view_mat,
                H, W, fx, fy, cx, cy):
        N = means3d.shape[0]
        # Handle edge case of no Gaussians
        if N == 0:
            C = features.shape[1] if features.numel() > 0 else 3
            device = means3d.device
            img = torch.zeros(H, W, C, device=device)
            weight = torch.zeros(H, W, device=device)
            depth = torch.zeros(H, W, device=device)
            ctx.save_for_backward(means3d, cov3d, opacity, features, view_mat,
                                  img, weight, depth)
            ctx.render_params = (H, W, fx, fy, cx, cy)
            return img, weight, depth

        if CUDA_AVAILABLE:
            img, weight, depth = _C.mip_splat_forward(
                means3d.contiguous(), cov3d.contiguous(),
                opacity.contiguous(), features.contiguous(),
                view_mat.contiguous(),
                H, W, fx, fy, cx, cy,
            )
        else:
            img, weight, depth = _mip_splat_pytorch(
                means3d, cov3d, opacity, features, view_mat,
                H, W, fx, fy, cx, cy,
            )

        ctx.save_for_backward(means3d, cov3d, opacity, features, view_mat,
                              img, weight, depth)
        ctx.render_params = (H, W, fx, fy, cx, cy)
        return img, weight, depth

    @staticmethod
    def backward(ctx, dL_dimg, dL_dweight, dL_ddepth):
        (means3d, cov3d, opacity, features, view_mat,
         out_img, out_weight, out_depth) = ctx.saved_tensors
        H, W, fx, fy, cx, cy = ctx.render_params

        if CUDA_AVAILABLE:
            grads = _C.mip_splat_backward(
                means3d.contiguous(), cov3d.contiguous(),
                opacity.contiguous(), features.contiguous(),
                view_mat.contiguous(),
                out_img.contiguous(), out_weight.contiguous(), out_depth.contiguous(),
                dL_dimg.contiguous(), dL_dweight.contiguous(), dL_ddepth.contiguous(),
                H, W, fx, fy, cx, cy,
            )
            dL_dmeans, dL_dcov3d, dL_dopacity, dL_dfeatures = grads
        else:
            # Finite-difference fallback (for debugging only — very slow)
            dL_dmeans = dL_dopacity = dL_dfeatures = dL_dcov3d = None

        return (dL_dmeans, dL_dcov3d, dL_dopacity, dL_dfeatures,
                None, None, None, None, None, None, None)


def render(
    gaussians,          # NeuriteGaussians
    camera: Camera,
    view_mat: torch.Tensor,  # [4,4]
    use_cuda_cov: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full render pass: build cov3d → splat → return (image, weight, depth).
    
    For fluorescence MIP, uses emission-only model:
    - `intensity` is the emission strength per Gaussian [N, C]
    - No absorption/opacity

    Returns
    -------
    image  : [H, W, C]
    weight : [H, W]
    depth  : [H, W]
    """
    means     = gaussians.means
    intensity = gaussians.intensity  # emission intensity in [0, 1]
    
    # For API compat with kernel, pass ones for opacity (unused internally)
    # and intensity for features
    dummy_opacity = torch.ones(gaussians.N, 1, device=means.device)

    if CUDA_AVAILABLE and use_cuda_cov:
        cov3d = _C.build_cov3d(
            gaussians.quats.contiguous(),
            gaussians._log_scales.contiguous(),
        )
    else:
        cov3d = gaussians.build_cov3d_torch()

    vm = view_mat.flatten().contiguous()

    image, weight, depth = MIPSplatFunction.apply(
        means, cov3d, dummy_opacity, intensity, vm,
        camera.H, camera.W,
        camera.fx, camera.fy, camera.cx, camera.cy,
    )
    return image, weight, depth


# ═════════════════════════════════════════════════════════════════════════════
#  Pure-PyTorch fallback renderer  (readable reference, not speed-optimised)
# ═════════════════════════════════════════════════════════════════════════════

def _mip_splat_pytorch(
    means3d, cov3d, opacity, features, view_mat_flat,
    H, W, fx, fy, cx, cy,
    soft_beta: float = 0.01,  # Reduced from 0.5 to avoid numerical overflow
):
    """
    Differentiable soft-MIP splatting in pure PyTorch.
    O(N×H×W) — only use for small scenes or debugging.
    """
    device = means3d.device
    N, C = means3d.shape[0], features.shape[1]
    M = view_mat_flat.view(4, 4)

    # Project all Gaussians
    ones = torch.ones(N, 1, device=device)
    pts_h = torch.cat([means3d, ones], dim=1)  # [N, 4]
    pc = (M @ pts_h.T).T                        # [N, 4]
    xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]

    valid = zc > 0.01
    inv_z  = 1.0 / zc.clamp(min=0.01)
    inv_z2 = inv_z ** 2

    u0 = xc * inv_z * fx + cx   # [N]
    v0 = yc * inv_z * fy + cy   # [N]

    # 2-D pixel grid
    pv, pu = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )   # [H, W]

    # Broadcast: [H, W, N]
    du = pu.unsqueeze(-1) - u0.unsqueeze(0).unsqueeze(0)   # [H,W,N]
    dv = pv.unsqueeze(-1) - v0.unsqueeze(0).unsqueeze(0)

    # 2-D covariance from 3-D cov (simplified — ignoring off-diagonals for clarity)
    # a = fx^2/z^2 * Sigma_xx_cam + 0.3 (low-pass)
    # Using upper-tri cov3d: [c00,c01,c02,c11,c12,c22]
    s00 = cov3d[:, 0]; s01 = cov3d[:, 1]; s02 = cov3d[:, 2]
    s11 = cov3d[:, 3]; s12 = cov3d[:, 4]; s22 = cov3d[:, 5]

    J00 = fx * inv_z; J02 = -fx * xc * inv_z2
    J11 = fy * inv_z; J12 = -fy * yc * inv_z2

    a = J00**2 * s00 + 2*J00*J02 * s02 + J02**2 * s22 + 0.3
    b = J00*J11 * s01 + J00*J12 * s02 + J02*J11 * s12 + J02*J12 * s22
    c = J11**2 * s11 + 2*J11*J12 * s12 + J12**2 * s22 + 0.3

    det  = (a * c - b**2).clamp(min=1e-10)  # [N]
    inv_det = 1.0 / det

    maha = (
        c * du**2 - 2.0 * b * du * dv + a * dv**2
    ) * inv_det.unsqueeze(0).unsqueeze(0)   # [H,W,N]

    footprint = torch.exp(-0.5 * maha) * valid.float().unsqueeze(0).unsqueeze(0)  # [H,W,N]
    
    # Emission strength per Gaussian: footprint × mean intensity
    mean_intensity = features.mean(dim=-1, keepdim=True)  # [N,1]
    emission = footprint * mean_intensity.view(1, 1, N)  # [H,W,N]
    
    # Soft-max weights: emphasize higher emissions (true MIP, no depth bias)
    soft_w = torch.exp(soft_beta * emission)  # [H,W,N]
    soft_w_sum = soft_w.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # [H,W,1]
    
    # Soft-max weighted intensity per channel
    soft_max_num = (soft_w.unsqueeze(-1) * features.unsqueeze(0).unsqueeze(0))  # [H,W,N,C]
    image = soft_max_num.sum(dim=2) / soft_w_sum  # [H,W,C]
    
    # Depth weighted by soft-max weights
    depth_img  = (soft_w * zc.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / soft_w_sum.squeeze(-1)
    weight_img = emission.sum(dim=-1)

    return image, weight_img, depth_img
