"""
CUDA-accelerated MIP (Maximum Intensity Projection) Splatting
==============================================================

Wraps the ``splat_mip_cuda`` extension in a PyTorch custom autograd
``Function`` so gradients flow transparently through the CUDA kernel back
to ``means_2d``, ``cov_inv``, and ``intensities``.

Public API
----------
* ``HAS_MIP_CUDA`` — ``True`` if the extension is importable.
* ``cuda_splat_mip(means_2d, cov_inv, intensities, pixels, beta)``
  — differentiable MIP splatting on arbitrary pixels.
* ``splat_mip_grid_cuda(H, W, means_2d, cov_2d, intensities, beta)``
  — drop-in replacement for ``splat_mip_grid`` in rendering.py.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function

try:
    import splat_mip_cuda as _C
    HAS_MIP_CUDA = True
except ImportError:
    HAS_MIP_CUDA = False


# ── helpers ──────────────────────────────────────────────────────────

def _invert_cov_2x2_packed(cov_2d: torch.Tensor) -> torch.Tensor:
    """
    Batch-invert (K, 2, 2) symmetric covariance matrices and return
    packed (K, 3) format ``[inv_a, inv_b, inv_d]``.

    Fully differentiable through PyTorch autograd.
    """
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    det = (a * d - b * b).clamp(min=1e-12)
    inv_det = 1.0 / det
    return torch.stack([d * inv_det, -b * inv_det, a * inv_det], dim=-1)


# ── Custom autograd function ────────────────────────────────────────

class SplatMIPFunction(Function):
    """
    Forward:  (means_2d, cov_inv, intensities, pixels, beta) → rendered
    Backward: grad_rendered → (grad_means_2d, grad_cov_inv, grad_intensities, None, None)
    """

    @staticmethod
    def forward(ctx, means_2d, cov_inv, intensities, pixels, beta):
        """
        Parameters
        ----------
        means_2d    : (K, 2) float32
        cov_inv     : (K, 3) float32 packed [inv_a, inv_b, inv_d]
        intensities : (K,) float32
        pixels      : (N, 2) float32
        beta        : float

        Returns
        -------
        rendered : (N,) float32
        """
        rendered, max_bg, sum_exp = _C.forward(
            means_2d, cov_inv, intensities, pixels, float(beta)
        )
        ctx.save_for_backward(means_2d, cov_inv, intensities, pixels,
                              rendered, max_bg, sum_exp)
        ctx.beta = float(beta)
        return rendered

    @staticmethod
    def backward(ctx, grad_rendered):
        means_2d, cov_inv, intensities, pixels, \
            rendered, max_bg, sum_exp = ctx.saved_tensors

        grad_means_2d, grad_cov_inv, grad_intensities = _C.backward(
            grad_rendered.contiguous(),
            means_2d, cov_inv, intensities, pixels,
            rendered, max_bg, sum_exp,
            ctx.beta,
        )
        # Returns match positional args: means_2d, cov_inv, intensities, pixels, beta
        return grad_means_2d, grad_cov_inv, grad_intensities, None, None


def cuda_splat_mip(means_2d, cov_inv, intensities, pixels, beta=50.0):
    """Differentiable MIP splatting via CUDA kernel."""
    return SplatMIPFunction.apply(means_2d, cov_inv, intensities, pixels, beta)


# ── Grid-based MIP splatting (drop-in for splat_mip_grid) ───────────

def splat_mip_grid_cuda(
    H:           int,
    W:           int,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 50.0,
) -> torch.Tensor:
    """
    CUDA-accelerated drop-in replacement for ``splat_mip_grid``.

    Parameters
    ----------
    H, W         : image height / width
    means_2d     : (K, 2)   projected Gaussian centres
    cov_2d       : (K, 2, 2) projected covariance matrices
    intensities  : (K,)     signal amplitudes
    beta         : softmax temperature

    Returns
    -------
    out : (H*W,) rendered MIP image (flat)
    """
    device = means_2d.device

    # Invert covariance → packed (K, 3).  This is a normal autograd op so
    # gradients flow through it back to cov_2d and beyond.
    cov_inv = _invert_cov_2x2_packed(cov_2d)

    # Build pixel grid  (x + 0.5, y + 0.5)
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2)

    return cuda_splat_mip(means_2d, cov_inv, intensities, pixels, beta)
