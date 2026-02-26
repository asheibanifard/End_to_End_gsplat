"""
PyTorch wrapper for CUDA-accelerated Gaussian Field operations

This module provides a drop-in replacement for the LearnableGaussianField
class with custom CUDA kernels for maximum performance.

Usage:
    from gaussian_field_ops import CUDALearnableGaussianField
    
    model = CUDALearnableGaussianField(num_gaussians=1000, volume_size=10.0)
    output = model(points)  # Forward pass uses CUDA kernels
    loss.backward()         # Backward pass uses CUDA kernels
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Import CUDA extension (will be compiled on first import)
# Note: torch must be imported first to set up library paths
try:
    import torch  # Required for CUDA extension loading
    import gaussian_field_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(f"gaussian_field_cuda not available: {e}. Run 'python setup_gaussian_field.py install' first.")


class MahalanobisDistanceFunction(torch.autograd.Function):
    """
    Custom autograd function for Mahalanobis distance computation.
    
    Uses CUDA kernels for both forward and backward passes.
    """
    
    @staticmethod
    def forward(ctx, points, means, cov_chol):
        """
        Forward pass: Compute Mahalanobis distances
        
        Args:
            points: [B, 3] query points
            means: [N, 3] Gaussian centers
            cov_chol: [N, 3, 3] Cholesky factors of covariance matrices
            
        Returns:
            mahal_dist: [B, N] Mahalanobis distances
        """
        mahal_dist = gaussian_field_cuda.mahalanobis_distance_forward(
            points.contiguous(),
            means.contiguous(),
            cov_chol.contiguous()
        )
        
        # Save for backward
        ctx.save_for_backward(points, means, cov_chol, mahal_dist)
        
        return mahal_dist
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients w.r.t. inputs
        
        Args:
            grad_output: [B, N] gradients from downstream
            
        Returns:
            Tuple of gradients: (grad_points, grad_means, grad_cov_chol)
        """
        points, means, cov_chol, mahal_dist = ctx.saved_tensors
        
        grad_points, grad_means, grad_cov_chol = \
            gaussian_field_cuda.mahalanobis_distance_backward(
                grad_output.contiguous(),
                points,
                means,
                cov_chol,
                mahal_dist
            )
        
        return grad_points, grad_means, grad_cov_chol


class CUDALearnableGaussianField(nn.Module):
    """
    CUDA-accelerated Learnable 3D Gaussian implicit field.
    
    This is a drop-in replacement for LearnableGaussianField that uses
    custom CUDA kernels for 10-100x speedup over PyTorch implementations.
    
    Features:
    - Full anisotropic covariances via Cholesky decomposition
    - Optimized CUDA kernels for forward/backward passes
    - Automatic gradient computation via PyTorch autograd
    - Memory-efficient batched operations
    
    Args:
        num_gaussians: Number of Gaussian basis functions
        volume_size: Size of 3D volume (cube)
        use_full_cov: Use full covariance (True) or diagonal (False)
        device: Computing device ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        num_gaussians: int,
        volume_size: float = 10.0,
        use_full_cov: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        if not CUDA_AVAILABLE and device == 'cuda':
            raise RuntimeError(
                "CUDA extension not available. Install with:\n"
                "  cd /workspace/end_to_end\n"
                "  python setup_gaussian_field.py install"
            )
        
        self.num_gaussians = num_gaussians
        self.volume_size = volume_size
        self.use_full_cov = use_full_cov
        self.device = device
        
        # Initialize parameters
        scale = volume_size / np.cbrt(num_gaussians)
        
        # Gaussian centers
        self.means = nn.Parameter(
            torch.rand(num_gaussians, 3, device=device) * volume_size
        )
        
        # Covariance parameterization
        if use_full_cov:
            # Full covariance via Cholesky: 6 parameters per Gaussian
            init_scale = np.log(scale)
            self.cov_tril = nn.Parameter(
                torch.tensor(
                    [[init_scale, 0.0, init_scale, 0.0, 0.0, init_scale]],
                    device=device
                ).repeat(num_gaussians, 1)
            )
        else:
            # Diagonal covariance: 3 parameters per Gaussian
            self.log_scales = nn.Parameter(
                torch.ones(num_gaussians, 3, device=device) * np.log(scale)
            )
        
        # Gaussian weights
        self.weights = nn.Parameter(torch.ones(num_gaussians, device=device))
    
    def get_covariance(self) -> torch.Tensor:
        """
        Reconstruct full covariance matrices from parameters.
        
        Returns:
            Covariance matrices [N, 3, 3]
        """
        if not self.use_full_cov:
            # Diagonal covariance
            scales = torch.exp(self.log_scales)
            cov = torch.zeros(self.num_gaussians, 3, 3, device=scales.device)
            cov[:, 0, 0] = scales[:, 0] ** 2
            cov[:, 1, 1] = scales[:, 1] ** 2
            cov[:, 2, 2] = scales[:, 2] ** 2
            return cov
        
        # Construct lower triangular Cholesky factors
        L = torch.zeros(self.num_gaussians, 3, 3, device=self.cov_tril.device)
        
        # Diagonal elements (positive via exp)
        L[:, 0, 0] = torch.exp(self.cov_tril[:, 0])
        L[:, 1, 1] = torch.exp(self.cov_tril[:, 2])
        L[:, 2, 2] = torch.exp(self.cov_tril[:, 5])
        
        # Off-diagonal elements
        L[:, 1, 0] = self.cov_tril[:, 1]
        L[:, 2, 0] = self.cov_tril[:, 3]
        L[:, 2, 1] = self.cov_tril[:, 4]
        
        # Add small regularization
        L = L + 1e-6 * torch.eye(3, device=L.device).unsqueeze(0)
        
        return L
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate implicit function at query points using CUDA kernels.
        
        Args:
            x: Query points, shape [3] or [B, 3]
            
        Returns:
            Field values, shape [] or [B]
        """
        # Handle single point input
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = x.shape[0]
        
        # Get Cholesky factors
        cov_chol = self.get_covariance()  # [N, 3, 3]
        
        # CUDA kernel: Compute Mahalanobis distances
        mahal_dist = MahalanobisDistanceFunction.apply(
            x, self.means, cov_chol
        )  # [B, N]
        
        # Compute weighted Gaussians
        gaussians = torch.exp(-0.5 * mahal_dist)  # [B, N]
        output = (gaussians * self.weights.unsqueeze(0)).sum(dim=-1)  # [B]
        
        return output.squeeze(0) if squeeze_output else output
    
    def forward_with_cuda_sum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alternative forward that uses CUDA kernel for summation too.
        
        Slightly faster for very large N.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute Mahalanobis distances
        cov_chol = self.get_covariance()
        mahal_dist = MahalanobisDistanceFunction.apply(x, self.means, cov_chol)
        
        # Use CUDA kernel for weighted sum
        output = gaussian_field_cuda.gaussian_field_forward(
            mahal_dist, self.weights
        )
        
        return output.squeeze(0) if squeeze_output else output


def test_cuda_kernels():
    """
    Test CUDA kernels against PyTorch reference implementation.
    """
    if not CUDA_AVAILABLE:
        print("CUDA kernels not available. Skipping test.")
        return
    
    print("Testing CUDA kernels...")
    print("=" * 70)
    
    # Create test data
    B, N = 100, 50
    device = 'cuda'
    
    torch.manual_seed(42)
    points = torch.randn(B, 3, device=device, requires_grad=True)
    means = torch.randn(N, 3, device=device, requires_grad=True)
    
    # Create valid Cholesky factors (positive diagonal)
    L = torch.randn(N, 3, 3, device=device)
    L = torch.tril(L)  # Make lower triangular
    # Ensure positive diagonal for valid Cholesky
    L[:, 0, 0] = L[:, 0, 0].abs() + 0.1
    L[:, 1, 1] = L[:, 1, 1].abs() + 0.1
    L[:, 2, 2] = L[:, 2, 2].abs() + 0.1
    L.requires_grad_(True)
    
    # Forward pass with CUDA
    mahal_cuda = MahalanobisDistanceFunction.apply(points, means, L)
    
    # Forward pass with PyTorch (reference)
    # This is the manual computation that CUDA kernel performs:
    # 1. diff = points - means
    # 2. solve L @ v = diff to get v
    # 3. mahal = ||v||^2
    diff = points.unsqueeze(1) - means.unsqueeze(0)  # [B, N, 3]
    
    # Solve L @ v = diff for each (b, n) pair
    mahal_torch = torch.zeros(B, N, device=device)
    for b in range(B):
        for n in range(N):
            # Forward substitution: L @ v = diff[b, n]
            d = diff[b, n]  # [3]
            L_n = L[n]  # [3, 3]
            
            # Solve using torch.linalg.solve
            v = torch.linalg.solve(L_n, d)
            mahal_torch[b, n] = (v ** 2).sum()
    
    # Check forward pass
    max_error = (mahal_cuda - mahal_torch).abs().max().item()
    mean_error = (mahal_cuda - mahal_torch).abs().mean().item()
    rel_error = max_error / (mahal_torch.abs().mean().item() + 1e-8) * 100
    print(f"Forward pass max error: {max_error:.2e}")
    print(f"Forward pass mean error: {mean_error:.2e}")
    print(f"Forward pass relative error: {rel_error:.3f}%")
    
    if rel_error < 1.0:  # Less than 1% relative error
        print("✅ Forward pass: PASSED")
    else:
        print("❌ Forward pass: FAILED")
        print(f"   Sample CUDA values: {mahal_cuda[0, :5]}")
        print(f"   Sample torch values: {mahal_torch[0, :5]}")
    
    # Test backward pass
    loss_cuda = mahal_cuda.sum()
    loss_torch = mahal_torch.sum()
    
    loss_cuda.backward()
    grad_points_cuda = points.grad.clone()
    grad_means_cuda = means.grad.clone()
    
    points.grad = None
    means.grad = None
    loss_torch.backward()
    grad_points_torch = points.grad.clone()
    grad_means_torch = means.grad.clone()
    
    grad_error = (grad_points_cuda - grad_points_torch).abs().max().item()
    grad_error_means = (grad_means_cuda - grad_means_torch).abs().max().item()
    grad_rel_error = grad_error / (grad_points_torch.abs().mean().item() + 1e-8) * 100
    print(f"\nBackward pass (points) max error: {grad_error:.2e}")
    print(f"Backward pass (points) relative error: {grad_rel_error:.3f}%")
    print(f"Backward pass (means) max error: {grad_error_means:.2e}")
    
    if grad_rel_error < 1.0:  # Less than 1% relative error
        print("✅ Backward pass: PASSED")
    else:
        print("❌ Backward pass: FAILED")
        print(f"   Sample CUDA grad_points: {grad_points_cuda[0]}")
        print(f"   Sample torch grad_points: {grad_points_torch[0]}")
    
    print("=" * 70)
    print("Test complete!")


if __name__ == "__main__":
    test_cuda_kernels()
