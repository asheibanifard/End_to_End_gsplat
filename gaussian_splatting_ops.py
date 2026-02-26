"""
Python wrapper for CUDA-accelerated Gaussian Splatting operations.
"""

import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import gaussian_splatting_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: gaussian_splatting_cuda not available. Using PyTorch fallback.")


# ============================================================================
# CUDA-Accelerated Operations
# ============================================================================

class TransformGaussiansFunction(Function):
    """Autograd function for transforming 3D Gaussians to camera space."""
    
    @staticmethod
    def forward(ctx, means_3d, covs_3d, R, T):
        """
        Transform 3D Gaussians to camera space.
        
        Args:
            means_3d: [N, 3] - 3D Gaussian centers
            covs_3d: [N, 3, 3] - 3D covariances
            R: [3, 3] - Rotation matrix
            T: [3] - Translation vector
            
        Returns:
            means_cam: [N, 3] - Transformed centers
            covs_cam: [N, 3, 3] - Transformed covariances
        """
        if not CUDA_AVAILABLE:
            # PyTorch fallback
            means_cam = torch.matmul(means_3d, R.T) + T
            covs_cam = torch.einsum('ij,njk,lk->nil', R, covs_3d, R)
            return means_cam, covs_cam
        
        result = gaussian_splatting_cuda.transform_gaussians(
            means_3d.contiguous(),
            covs_3d.contiguous(),
            R.contiguous(),
            T.contiguous()
        )
        
        means_cam, covs_cam = result[0], result[1]
        
        ctx.save_for_backward(means_3d, covs_3d, R, T, means_cam, covs_cam)
        return means_cam, covs_cam
    
    @staticmethod
    def backward(ctx, grad_means_cam, grad_covs_cam):
        """Backward pass - gradients computed by PyTorch."""
        means_3d, covs_3d, R, T, means_cam, covs_cam = ctx.saved_tensors
        
        # Gradient w.r.t. means_3d: d_means = R^T @ d_means_cam
        grad_means_3d = torch.matmul(grad_means_cam, R)
        
        # Gradient w.r.t. covs_3d: d_covs = R^T @ d_covs_cam @ R
        grad_covs_3d = torch.einsum('ji,njk,kl->nil', R, grad_covs_cam, R)
        
        # Gradient w.r.t. R (more complex, using chain rule)
        grad_R = torch.matmul(grad_means_cam.T, means_3d)
        grad_R += torch.einsum('njk,ni,nl->jl', covs_3d, grad_covs_cam.sum(dim=1), means_3d)
        
        # Gradient w.r.t. T
        grad_T = grad_means_cam.sum(dim=0)
        
        return grad_means_3d, grad_covs_3d, grad_R, grad_T


class ProjectTo2DFunction(Function):
    """Autograd function for projecting 3D Gaussians to 2D."""
    
    @staticmethod
    def forward(ctx, means_cam, covs_cam, fx, fy, cx, cy):
        """
        Project 3D Gaussians to 2D with Jacobian.
        
        Args:
            means_cam: [N, 3] - 3D centers in camera space
            covs_cam: [N, 3, 3] - 3D covariances
            fx, fy: float - Focal lengths
            cx, cy: float - Principal point
            
        Returns:
            means_2d: [N, 2] - 2D projected centers
            covs_2d: [N, 2, 2] - 2D covariances
            depths: [N] - Depth values
        """
        if not CUDA_AVAILABLE:
            # PyTorch fallback (from existing code)
            x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
            z = torch.clamp(z, min=1e-6)
            
            means_2d = torch.stack([
                fx * x / z + cx,
                fy * y / z + cy
            ], dim=1)
            
            # Jacobian
            z_inv = 1.0 / z
            J = torch.zeros(means_cam.shape[0], 2, 3, device=means_cam.device)
            J[:, 0, 0] = fx * z_inv
            J[:, 0, 2] = -fx * x * z_inv * z_inv
            J[:, 1, 1] = fy * z_inv
            J[:, 1, 2] = -fy * y * z_inv * z_inv
            
            # Σ_2D = J @ Σ_3D @ J^T
            covs_2d = torch.bmm(torch.bmm(J, covs_cam), J.transpose(1, 2))
            covs_2d[:, 0, 0] += 1e-6
            covs_2d[:, 1, 1] += 1e-6
            
            depths = z
            
            return means_2d, covs_2d, depths
        
        result = gaussian_splatting_cuda.project_to_2d(
            means_cam.contiguous(),
            covs_cam.contiguous(),
            float(fx), float(fy), float(cx), float(cy)
        )
        
        means_2d, covs_2d, depths = result
        
        ctx.save_for_backward(means_cam, covs_cam, means_2d, covs_2d, depths)
        ctx.fx = fx
        ctx.fy = fy
        
        return means_2d, covs_2d, depths
    
    @staticmethod
    def backward(ctx, grad_means_2d, grad_covs_2d, grad_depths):
        """Backward pass for projection."""
        means_cam, covs_cam, means_2d, covs_2d, depths = ctx.saved_tensors
        fx, fy = ctx.fx, ctx.fy
        
        # Simplified gradient computation (full version would be more complex)
        z = depths.clamp(min=1e-6)
        z_inv = 1.0 / z
        
        # Gradients w.r.t. means_cam
        grad_means_cam = torch.zeros_like(means_cam)
        grad_means_cam[:, 0] = grad_means_2d[:, 0] * fx * z_inv
        grad_means_cam[:, 1] = grad_means_2d[:, 1] * fy * z_inv
        grad_means_cam[:, 2] = grad_depths + \
                               -grad_means_2d[:, 0] * fx * means_cam[:, 0] * z_inv * z_inv + \
                               -grad_means_2d[:, 1] * fy * means_cam[:, 1] * z_inv * z_inv
        
        # Gradient w.r.t. covs_cam (simplified)
        grad_covs_cam = torch.zeros_like(covs_cam)
        
        return grad_means_cam, grad_covs_cam, None, None, None, None


class RenderGaussians2DFunction(Function):
    """Autograd function for rendering 2D Gaussians."""
    
    @staticmethod
    def forward(ctx, pixel_coords, means_2d, covs_2d, weights):
        """
        Render 2D Gaussians to pixel grid.
        
        Args:
            pixel_coords: [H*W, 2] - Pixel coordinates
            means_2d: [N, 2] - 2D Gaussian centers
            covs_2d: [N, 2, 2] - 2D covariances
            weights: [N] - Gaussian weights
            
        Returns:
            output: [H*W] - Rendered pixel values
        """
        if not CUDA_AVAILABLE:
            # PyTorch fallback
            N = means_2d.shape[0]
            M = pixel_coords.shape[0]
            
            # Compute differences: [M, N, 2]
            diff = pixel_coords.unsqueeze(1) - means_2d.unsqueeze(0)
            
            # Compute inverse covariances
            det = covs_2d[:, 0, 0] * covs_2d[:, 1, 1] - covs_2d[:, 0, 1] * covs_2d[:, 1, 0]
            det = torch.clamp(det, min=1e-6)
            inv_det = 1.0 / det
            
            inv_cov = torch.zeros_like(covs_2d)
            inv_cov[:, 0, 0] = covs_2d[:, 1, 1] * inv_det
            inv_cov[:, 0, 1] = -covs_2d[:, 0, 1] * inv_det
            inv_cov[:, 1, 0] = -covs_2d[:, 1, 0] * inv_det
            inv_cov[:, 1, 1] = covs_2d[:, 0, 0] * inv_det
            
            # Mahalanobis distance
            mahal = torch.einsum('mni,nij,mnj->mn', diff, inv_cov, diff)
            
            # Gaussian evaluation
            gaussians = torch.exp(-0.5 * mahal)
            
            # Weighted sum
            output = torch.matmul(gaussians, weights)
            
            ctx.save_for_backward(pixel_coords, means_2d, covs_2d, weights, gaussians, diff, inv_cov)
            return output
        
        output = gaussian_splatting_cuda.render_gaussians_2d(
            pixel_coords.contiguous(),
            means_2d.contiguous(),
            covs_2d.contiguous(),
            weights.contiguous()
        )
        
        # Save for backward
        ctx.save_for_backward(pixel_coords, means_2d, covs_2d, weights)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for rendering."""
        if not CUDA_AVAILABLE:
            pixel_coords, means_2d, covs_2d, weights, gaussians, diff, inv_cov = ctx.saved_tensors
            
            # Gradient w.r.t. weights
            grad_weights = torch.matmul(gaussians.T, grad_output)
            
            # Gradient w.r.t. means_2d (complex)
            grad_gaussians = grad_output.unsqueeze(1) * weights.unsqueeze(0)
            term = -gaussians * grad_gaussians
            grad_means_2d_from_mahal = -torch.einsum('mn,mni,nij->nj', term, diff, inv_cov) * 2
            
            # Gradient w.r.t. covs_2d (even more complex, simplified here)
            grad_covs_2d = torch.zeros_like(covs_2d)
            
            return None, grad_means_2d_from_mahal, grad_covs_2d, grad_weights
        
        # For CUDA version, use PyTorch autograd
        pixel_coords, means_2d, covs_2d, weights = ctx.saved_tensors
        
        # Recompute forward pass for gradients
        N = means_2d.shape[0]
        M = pixel_coords.shape[0]
        
        diff = pixel_coords.unsqueeze(1) - means_2d.unsqueeze(0)
        
        det = covs_2d[:, 0, 0] * covs_2d[:, 1, 1] - covs_2d[:, 0, 1] * covs_2d[:, 1, 0]
        det = torch.clamp(det, min=1e-6)
        inv_det = 1.0 / det
        
        inv_cov = torch.zeros_like(covs_2d)
        inv_cov[:, 0, 0] = covs_2d[:, 1, 1] * inv_det
        inv_cov[:, 0, 1] = -covs_2d[:, 0, 1] * inv_det
        inv_cov[:, 1, 0] = -covs_2d[:, 1, 0] * inv_det
        inv_cov[:, 1, 1] = covs_2d[:, 0, 0] * inv_det
        
        mahal = torch.einsum('mni,nij,mnj->mn', diff, inv_cov, diff)
        gaussians = torch.exp(-0.5 * mahal)
        
        # Gradients
        grad_weights = torch.matmul(gaussians.T, grad_output)
        grad_gaussians = grad_output.unsqueeze(1) * weights.unsqueeze(0)
        term = -gaussians * grad_gaussians
        grad_means_2d = -torch.einsum('mn,mni,nij->nj', term, diff, inv_cov) * 2
        grad_covs_2d = torch.zeros_like(covs_2d)
        
        return None, grad_means_2d, grad_covs_2d, grad_weights


# ============================================================================
# User-facing Functions
# ============================================================================

def transform_gaussians_cuda(means_3d, covs_3d, R, T):
    """Transform 3D Gaussians to camera space (CUDA-accelerated)."""
    return TransformGaussiansFunction.apply(means_3d, covs_3d, R, T)


def project_to_2d_cuda(means_cam, covs_cam, fx, fy, cx, cy):
    """Project 3D Gaussians to 2D (CUDA-accelerated)."""
    return ProjectTo2DFunction.apply(means_cam, covs_cam, fx, fy, cx, cy)


def render_gaussians_2d_cuda(pixel_coords, means_2d, covs_2d, weights):
    """Render 2D Gaussians to pixels (CUDA-accelerated)."""
    return RenderGaussians2DFunction.apply(pixel_coords, means_2d, covs_2d, weights)
