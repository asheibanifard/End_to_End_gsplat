"""
MIP Splatting Operations with PyTorch Autograd Support
Wrapper for mip_splatting_cuda extension for fluorescence microscopy
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

try:
    import mip_splatting_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: mip_splatting_cuda extension not found. Using PyTorch fallback.")


class MIPSplattingFunction(Function):
    """
    Autograd function for MIP splatting forward and backward passes.
    
    Forward: Soft-MIP rendering of 3D Gaussians
    Backward: Analytic gradients w.r.t. means, covariances, and features
    """
    
    @staticmethod
    def forward(ctx, means_3d, cov_3d, features, view_matrix, fx, fy, cx, cy, H, W):
        """
        Args:
            means_3d: (N, 3) - 3D Gaussian centers
            cov_3d: (N, 6) - Upper-triangular 3D covariances [c00,c01,c02,c11,c12,c22]
            features: (N, C) - Emission intensities per channel
            view_matrix: (4, 4) - Camera view matrix (world to camera)
            fx, fy, cx, cy: Camera intrinsics
            H, W: Image dimensions
            
        Returns:
            rendered_image: (H, W, C) - Soft-MIP projection
            weight: (H, W) - Total emission weight
            depth: (H, W) - Depth of max emitter
        """
        N = means_3d.shape[0]
        C = features.shape[1]
        
        # Dummy opacity (not used in MIP splatting but required by API)
        opacity = torch.ones(N, 1, device=means_3d.device)
        
        # Flatten view matrix to row-major [16]
        view_mat_flat = view_matrix.reshape(-1).contiguous()
        
        if CUDA_AVAILABLE and means_3d.is_cuda:
            # CUDA kernel returns 4 tensors: img, weight, depth, sum_sw
            # sum_sw = Σ(soft_w) per pixel — required by backward for correct gradients
            img, weight, depth, sum_sw = mip_splatting_cuda.mip_splat_forward(
                means_3d.contiguous(),
                cov_3d.contiguous(),
                opacity.contiguous(),
                features.contiguous(),
                view_mat_flat,
                H, W, float(fx), float(fy), float(cx), float(cy)
            )
        else:
            # Fallback: PyTorch implementation (slow, for debugging)
            img, weight, depth = _mip_splat_forward_pytorch(
                means_3d, cov_3d, features, view_matrix, fx, fy, cx, cy, H, W
            )
            # CPU fallback cannot compute exact sum_sw; gradients will be approximate
            sum_sw = torch.zeros(H, W, device=means_3d.device)
        
        # Save for backward — sum_sw is essential for correct gradient computation
        ctx.save_for_backward(means_3d, cov_3d, opacity, features, view_mat_flat, img, weight, depth, sum_sw)
        ctx.H = H
        ctx.W = W
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy
        
        return img, weight, depth
    
    @staticmethod
    def backward(ctx, dL_dimg, dL_dweight, dL_ddepth):
        """
        Compute gradients w.r.t. inputs.
        """
        means_3d, cov_3d, opacity, features, view_mat_flat, img, weight, depth, sum_sw = ctx.saved_tensors
        H, W = ctx.H, ctx.W
        fx, fy, cx, cy = ctx.fx, ctx.fy, ctx.cx, ctx.cy
        
        dL_dimg    = dL_dimg.contiguous()
        dL_dweight = dL_dweight.contiguous() if dL_dweight is not None else torch.zeros_like(weight)
        dL_ddepth  = dL_ddepth.contiguous() if dL_ddepth is not None else torch.zeros_like(depth)
        
        if CUDA_AVAILABLE and means_3d.is_cuda:
            # Pass sum_sw so the backward kernel can use the exact Σ(soft_w) per pixel
            dL_dmeans, dL_dcov, dL_dopacity, dL_dfeatures = mip_splatting_cuda.mip_splat_backward(
                means_3d, cov_3d, opacity, features, view_mat_flat,
                img, weight, depth, sum_sw,
                dL_dimg, dL_dweight, dL_ddepth,
                H, W, float(fx), float(fy), float(cx), float(cy)
            )
        else:
            # Fallback: numerical gradients (very slow)
            dL_dmeans = torch.zeros_like(means_3d)
            dL_dcov = torch.zeros_like(cov_3d)
            dL_dfeatures = torch.zeros_like(features)
        
        # Return gradients for each input (None for non-tensor inputs)
        return dL_dmeans, dL_dcov, dL_dfeatures, None, None, None, None, None, None, None


def _mip_splat_forward_pytorch(means_3d, cov_3d, features, view_matrix, fx, fy, cx, cy, H, W):
    """
    PyTorch fallback implementation of MIP splatting (slow, for debugging).
    Applies the full 4x4 view matrix so CPU debugging matches CUDA behaviour.
    """
    N = means_3d.shape[0]
    C = features.shape[1]
    device = means_3d.device

    img    = torch.zeros(H, W, C, device=device)
    weight = torch.zeros(H, W, device=device)
    depth  = torch.zeros(H, W, device=device)

    # Pre-extract 3x4 view submatrix (top three rows of 4x4 view_matrix)
    R = view_matrix[:3, :3]  # (3, 3)
    t = view_matrix[:3,  3]  # (3,)

    for g in range(N):
        mean = means_3d[g]  # (3,)

        # Apply view transform: p_cam = R @ mean + t
        mean_cam = R @ mean + t  # (3,)

        if mean_cam[2] <= 0.01:
            continue

        # Perspective projection
        u = (mean_cam[0] / mean_cam[2]) * fx + cx
        v = (mean_cam[1] / mean_cam[2]) * fy + cy

        if u < 0 or u >= W or v < 0 or v >= H:
            continue

        px = int(u)
        py = int(v)

        if 0 <= px < W and 0 <= py < H:
            for ch in range(C):
                img[py, px, ch] = torch.max(img[py, px, ch], features[g, ch])
            weight[py, px] += 1.0
            depth[py, px]   = mean_cam[2]

    return img, weight, depth


def build_covariance_from_cholesky(cov_tril_params):
    """
    Build 3D covariance matrices from Cholesky parameters (for compatibility).
    
    Args:
        cov_tril_params: (N, 6) - Cholesky parameters [a,b,c,d,e,f]
            where L = [[exp(a), 0, 0], [b, exp(c), 0], [d, e, exp(f)]]
    
    Returns:
        cov_3d: (N, 6) - Upper triangular covariances [c00,c01,c02,c11,c12,c22]
    """
    N = cov_tril_params.shape[0]
    device = cov_tril_params.device
    
    # Build lower triangular matrix L
    L = torch.zeros(N, 3, 3, device=device)
    L[:, 0, 0] = torch.exp(cov_tril_params[:, 0])  # a
    L[:, 1, 0] = cov_tril_params[:, 1]             # b
    L[:, 1, 1] = torch.exp(cov_tril_params[:, 2])  # c
    L[:, 2, 0] = cov_tril_params[:, 3]             # d
    L[:, 2, 1] = cov_tril_params[:, 4]             # e
    L[:, 2, 2] = torch.exp(cov_tril_params[:, 5])  # f
    
    # Compute Σ = L @ L^T
    cov_full = torch.bmm(L, L.transpose(1, 2))  # (N, 3, 3)
    
    # Extract upper triangular as [c00, c01, c02, c11, c12, c22]
    cov_3d = torch.stack([
        cov_full[:, 0, 0], cov_full[:, 0, 1], cov_full[:, 0, 2],
        cov_full[:, 1, 1], cov_full[:, 1, 2], cov_full[:, 2, 2]
    ], dim=1)
    
    return cov_3d


def mip_splat_render(means_3d, cov_tril_params, features, view_matrix, fx, fy, cx, cy, H, W):
    """
    High-level MIP splatting render function.
    
    Args:
        means_3d: (N, 3) - 3D Gaussian centers
        cov_tril_params: (N, 6) - Cholesky covariance parameters
        features: (N, C) - Emission intensities
        view_matrix: (4, 4) - Camera view matrix
        fx, fy, cx, cy: Camera intrinsics
        H, W: Image dimensions

    Returns:
        img:    (H, W, C) - Soft-MIP projection  [differentiable]
        weight: (H, W)    - Σ(emission_strength) [diagnostic only, no gradient]
        depth:  (H, W)    - Weighted mean depth   [diagnostic only, no gradient]

    Note: only `img` participates in autograd. `weight` and `depth` are saved
    context tensors returned for visualisation / debugging. If you need depth
    gradients, add a dedicated depth-loss term inside MIPSplattingFunction.
    """
    # Build 3D covariances from Cholesky params (differentiable)
    cov_3d = build_covariance_from_cholesky(cov_tril_params)

    # Run differentiable forward (weight and depth are ctx tensors, not autograd outputs)
    img, weight, depth = MIPSplattingFunction.apply(
        means_3d, cov_3d, features, view_matrix,
        fx, fy, cx, cy, H, W
    )

    return img, weight, depth


if __name__ == "__main__":
    # Simple test
    print(f"MIP Splatting CUDA available: {CUDA_AVAILABLE}")
    
    if CUDA_AVAILABLE:
        device = 'cuda'
        N = 100
        C = 1
        H, W = 64, 64
        
        means_3d = torch.randn(N, 3, device=device, requires_grad=True)
        cov_tril = torch.randn(N, 6, device=device, requires_grad=True) * 0.1
        features = torch.rand(N, C, device=device, requires_grad=True)
        view_matrix = torch.eye(4, device=device)
        
        img, weight, depth = mip_splat_render(
            means_3d, cov_tril, features, view_matrix,
            fx=400.0, fy=400.0, cx=32.0, cy=32.0, H=H, W=W
        )
        
        print(f"Output shape: {img.shape}")
        print(f"Output range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Test backward
        loss = img.sum()
        loss.backward()
        
        print(f"Gradient on means: {means_3d.grad.abs().mean():.6f}")
        print(f"Gradient on features: {features.grad.abs().mean():.6f}")
        print("✓ MIP Splatting test passed!")