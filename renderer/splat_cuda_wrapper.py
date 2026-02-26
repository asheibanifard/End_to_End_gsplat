"""
CUDA-accelerated Gaussian Splatting for Training
=================================================

Fuses the N×K inner splatting loop in CUDA while using PyTorch autograd
for the projection chain (quaternion→rotation→covariance→camera→2D).

Architecture:
  PyTorch autograd:  means, log_scales, quaternions, log_amplitudes
                     → covariances, opacities
                     → camera transform → 2D means, 2D covariance
                     → invert 2×2 → cov_inv (K,3)
  CUDA kernel:       cov_inv, means_2d, opacities, colors, pixels
                     → rendered (N,)   [forward]
                     → grad_means_2d, grad_cov_inv, grad_opacities, grad_colors [backward]
  PyTorch autograd:  chain rule back through projection to learnable params

This avoids materialising the (N, K) Gaussian evaluation tensor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional, List
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

try:
    import splat_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("WARNING: splat_cuda not found. Build with: python setup_splat_cuda.py build_ext --inplace")


# ── Windowed SSIM (matches rendering.py's ssim_loss_fn) ─────────────
def _windowed_ssim_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    window_size: int   = 11,
    C1:          float = 1e-4,
    C2:          float = 9e-4,
) -> torch.Tensor:
    """1 − SSIM with Gaussian-weighted sliding window on (H, W) images."""
    p = pred.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
    g = target.unsqueeze(0).unsqueeze(0)

    coords = torch.arange(window_size, device=pred.device, dtype=torch.float32)
    coords -= window_size // 2
    win = torch.exp(-coords ** 2 / (2.0 * 1.5 ** 2))
    win = win.unsqueeze(1) * win.unsqueeze(0)
    win = (win / win.sum()).unsqueeze(0).unsqueeze(0)   # (1,1,ws,ws)
    pad = window_size // 2

    mu_p  = F.conv2d(p,     win, padding=pad)
    mu_g  = F.conv2d(g,     win, padding=pad)
    sig_p = F.conv2d(p * p, win, padding=pad) - mu_p ** 2
    sig_g = F.conv2d(g * g, win, padding=pad) - mu_g ** 2
    sig_x = F.conv2d(p * g, win, padding=pad) - mu_p * mu_g

    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sig_x + C2)) / \
               ((mu_p**2 + mu_g**2 + C1) * (sig_p + sig_g + C2))
    return 1.0 - ssim_map.mean()


class SplatAlphaFunction(Function):
    """
    Custom autograd function wrapping the CUDA splatting kernel.

    Forward: (means_2d, cov_inv, opacities, colors, pixels) → rendered
    Backward: grad_rendered → (grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None)
    """

    @staticmethod
    def forward(ctx, means_2d, cov_inv, opacities, colors, pixels):
        """
        Parameters
        ----------
        means_2d : (K, 2) float32 — sorted by depth
        cov_inv : (K, 3) float32 — [inv_a, inv_b, inv_d] sorted
        opacities : (K,) float32 — sorted
        colors : (K,) float32 — sorted (grayscale)
        pixels : (N, 2) float32

        Returns
        -------
        rendered : (N,) float32
        """
        rendered, T_out = splat_cuda.forward(
            means_2d, cov_inv, opacities, colors, pixels
        )
        ctx.save_for_backward(means_2d, cov_inv, opacities, colors, pixels)
        return rendered

    @staticmethod
    def backward(ctx, grad_rendered):
        means_2d, cov_inv, opacities, colors, pixels = ctx.saved_tensors

        grad_means_2d, grad_cov_inv, grad_opacities, grad_colors = splat_cuda.backward(
            grad_rendered.contiguous(),
            means_2d, cov_inv, opacities, colors, pixels
        )

        return grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None


def cuda_splat_alpha(means_2d, cov_inv, opacities, colors, pixels):
    """Differentiable alpha-compositing splatting via CUDA."""
    return SplatAlphaFunction.apply(means_2d, cov_inv, opacities, colors, pixels)


# ============================================================================
# Full differentiable pipeline: learnable params → rendered pixels
# ============================================================================

def build_covariances(quaternions, log_scales):
    """
    Build 3D covariance matrices from quaternions and log-scales.

    Parameters
    ----------
    quaternions : (K, 4)
    log_scales : (K, 3)

    Returns
    -------
    covariances : (K, 3, 3)
    """
    K = quaternions.shape[0]

    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    S2 = torch.diag_embed(scales ** 2)
    covariances = R @ S2 @ R.transpose(-2, -1)

    return covariances


def transform_to_camera(means, covariances, R_cam, T_cam):
    """Transform Gaussians from world to camera frame."""
    means_cam = (R_cam @ means.unsqueeze(-1)).squeeze(-1) + T_cam.unsqueeze(0)
    cov_cam = R_cam.unsqueeze(0) @ covariances @ R_cam.T.unsqueeze(0)
    return means_cam, cov_cam


def project_to_2d(means_cam, cov_cam, fx, fy, cx, cy):
    """
    Project 3D Gaussians to 2D.

    Returns means_2d (K,2), cov_2d (K,2,2), depths (K,).
    """
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2]
    z_safe = z.clamp(min=1e-6)

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    means_2d = torch.stack([u, v], dim=-1)

    # Jacobian
    z_sq = z_safe * z_safe
    K_g = means_cam.shape[0]
    J = torch.zeros(K_g, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * x / z_sq
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * y / z_sq

    cov_2d = J @ cov_cam @ J.transpose(-2, -1)
    eps = 1e-4
    cov_2d = cov_2d + eps * torch.eye(2, device=cov_2d.device).unsqueeze(0)

    return means_2d, cov_2d, z


def invert_2x2(cov_2d):
    """
    Invert 2×2 symmetric matrices.

    Parameters
    ----------
    cov_2d : (K, 2, 2)

    Returns
    -------
    cov_inv_packed : (K, 3) — [inv_a, inv_b, inv_d]
    """
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]

    det = a * d - b * b
    det_safe = det.clamp(min=1e-12)
    inv_det = 1.0 / det_safe

    inv_a = d * inv_det
    inv_b = -b * inv_det
    inv_d = a * inv_det

    return torch.stack([inv_a, inv_b, inv_d], dim=-1)


def cull_gaussians(means_cam, means_2d, cov_2d, near, far, width, height, radius_mult=3.0):
    """Visibility culling: depth + frustum."""
    z = means_cam[:, 2]
    depth_ok = (z > near) & (z < far)

    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    tr = a + d
    det = a * d - b * b
    disc = (tr * tr - 4.0 * det).clamp(min=0.0)
    lambda_max = 0.5 * (tr + torch.sqrt(disc))
    radius = radius_mult * torch.sqrt(lambda_max.clamp(min=1e-8))

    u = means_2d[:, 0]
    v = means_2d[:, 1]
    in_frustum = (
        (u + radius > 0) & (u - radius < width) &
        (v + radius > 0) & (v - radius < height)
    )

    return depth_ok & in_frustum


def apply_aspect_correction(means, covariances, aspect_scales):
    """Scale means and covariances by aspect ratios."""
    s = aspect_scales.to(means.device)
    means_c = means * s.unsqueeze(0)
    S = torch.diag(s)
    cov_c = S.unsqueeze(0) @ covariances @ S.unsqueeze(0)
    return means_c, cov_c


class CUDASplattingRenderer:
    """
    CUDA-accelerated Gaussian splatting renderer for training.

    Uses PyTorch autograd for projection chain + CUDA kernel for splatting.
    """

    def __init__(self, near=0.01, far=10.0, radius_mult=3.0, max_visible_gaussians=0, pixel_window_pad=8.0):
        self.near = near
        self.far = far
        self.radius_mult = radius_mult
        self.max_visible_gaussians = int(max_visible_gaussians)
        self.pixel_window_pad = float(pixel_window_pad)
        assert HAS_CUDA, "splat_cuda extension required. Build with setup_splat_cuda.py"

    def render_at_pixels(
        self,
        means, covariances, opacities, colors,
        R_cam, T_cam,
        fx, fy, cx, cy, width, height,
        pixels,
    ):
        """
        Render at specific pixel locations using CUDA splatting.

        Parameters
        ----------
        means : (K, 3) - world space (aspect-corrected)
        covariances : (K, 3, 3)
        opacities : (K,) - differentiable
        colors : (K,) - grayscale color (differentiable)
        R_cam, T_cam : camera extrinsics
        fx, fy, cx, cy : camera intrinsics
        width, height : image dimensions
        pixels : (N, 2) pixel coordinates

        Returns
        -------
        rendered : (N,) - differentiable w.r.t. all inputs
        """
        # Step 1: World → Camera (differentiable via autograd)
        means_cam, cov_cam = transform_to_camera(means, covariances, R_cam, T_cam)

        # Step 2: 3D → 2D projection (differentiable via autograd)
        means_2d, cov_2d, depths = project_to_2d(means_cam, cov_cam, fx, fy, cx, cy)

        # Visibility culling (no grad)
        with torch.no_grad():
            visible = cull_gaussians(
                means_cam, means_2d, cov_2d,
                self.near, self.far, width, height, self.radius_mult
            )

            px_min = pixels[:, 0].min() - self.pixel_window_pad
            px_max = pixels[:, 0].max() + self.pixel_window_pad
            py_min = pixels[:, 1].min() - self.pixel_window_pad
            py_max = pixels[:, 1].max() + self.pixel_window_pad

            a = cov_2d[:, 0, 0]
            b = cov_2d[:, 0, 1]
            d = cov_2d[:, 1, 1]
            tr = a + d
            det = a * d - b * b
            disc = (tr * tr - 4.0 * det).clamp(min=0.0)
            lambda_max = 0.5 * (tr + torch.sqrt(disc))
            radius = self.radius_mult * torch.sqrt(lambda_max.clamp(min=1e-8))

            u = means_2d[:, 0]
            v = means_2d[:, 1]
            overlap_window = (
                (u + radius >= px_min) & (u - radius <= px_max) &
                (v + radius >= py_min) & (v - radius <= py_max)
            )
            visible = visible & overlap_window

        n_vis = visible.sum().item()
        if n_vis == 0:
            return opacities.sum() * 0.0 + torch.zeros(
                pixels.shape[0], device=pixels.device)

        # Filter to visible
        means_2d_vis = means_2d[visible]
        cov_2d_vis = cov_2d[visible]
        opacities_vis = opacities[visible]
        colors_vis = colors[visible]
        depths_vis = depths[visible]

        if self.max_visible_gaussians > 0 and depths_vis.shape[0] > self.max_visible_gaussians:
            keep_idx = torch.topk(depths_vis, k=self.max_visible_gaussians, largest=False).indices
            means_2d_vis = means_2d_vis[keep_idx]
            cov_2d_vis = cov_2d_vis[keep_idx]
            opacities_vis = opacities_vis[keep_idx]
            colors_vis = colors_vis[keep_idx]
            depths_vis = depths_vis[keep_idx]

        # Sort by depth (differentiable: just reordering)
        order = torch.argsort(depths_vis)
        means_2d_sorted = means_2d_vis[order]
        cov_2d_sorted = cov_2d_vis[order]
        opacities_sorted = opacities_vis[order]
        colors_sorted = colors_vis[order]

        # Invert 2×2 covariance (differentiable via autograd)
        cov_inv = invert_2x2(cov_2d_sorted)  # (K_vis, 3)

        # Step 3+4: CUDA splatting (custom autograd backward)
        rendered = cuda_splat_alpha(
            means_2d_sorted, cov_inv, opacities_sorted, colors_sorted, pixels
        )

        return rendered


class CUDASplattingTrainer:
    """
    Training loop with CUDA-accelerated splatting and scale regularization.
    """

    def __init__(
        self,
        means, log_scales, quaternions, log_amplitudes,
        aspect_scales,
        lr=5e-4,
        pixels_per_step=16384,
        sampling_mode="tile",
        max_visible_gaussians=4096,
    ):
        self.device = means.device
        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.quaternions = nn.Parameter(quaternions.clone())
        self.log_amplitudes = nn.Parameter(log_amplitudes.clone())
        self.aspect_scales = aspect_scales.to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': lr},
            {'params': [self.log_scales], 'lr': lr * 0.5},
            {'params': [self.quaternions], 'lr': lr * 0.3},
            {'params': [self.log_amplitudes], 'lr': lr},
        ])

        self.pixels_per_step = pixels_per_step
        self.sampling_mode = sampling_mode
        self.renderer = CUDASplattingRenderer(max_visible_gaussians=max_visible_gaussians)

        # Regularization
        self.lambda_scale = 0.001
        self.scale_min_target = 0.005
        self.scale_max_target = 0.05       # penalise scales above this
        self.lambda_scale_max = 0.01       # weight for max-scale penalty

        # Intensity-based pruning
        self.prune_every = 2000
        self.prune_intensity_thresh = 0.01
        self.prune_min_gaussians = 2000

        # Densification (split / clone)
        self.densify_every       = 500
        self.densify_start_step  = 500
        self.densify_stop_step   = 15000
        self.densify_grad_thresh = 0.0002
        self.densify_scale_thresh = 0.01   # split if max_scale > this
        self.max_gaussians       = 50000

        # Gradient accumulators for densification
        K = self.means.shape[0]
        self._grad_accum = torch.zeros(K, device=self.device)
        self._grad_count = torch.zeros(K, device=self.device)

    def _build_params(self):
        """Build covariances and intensity from learnable parameters."""
        covariances = build_covariances(self.quaternions, self.log_scales)
        intensity = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
        return covariances, intensity

    def _build_gaussians_corrected(self):
        """Compatibility helper for rendering.py final validation path."""
        covariances, intensity = self._build_params()
        means = self.means

        if self.aspect_scales is not None:
            s = self.aspect_scales.to(means.device)
            means = means * s.unsqueeze(0)
            S = torch.diag(s)
            covariances = S.unsqueeze(0) @ covariances @ S.unsqueeze(0)

        return SimpleNamespace(
            means=means,
            covariances=covariances,
            weights=intensity,
            colors=intensity.unsqueeze(-1),
        )

    def train_step(
        self,
        camera,
        gt_image,
        R_cam,
        T_cam,
        precomputed: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        backward_scale: Optional[float] = None,
        retain_graph: bool = False,
    ):
        """
        One training step.

        Parameters
        ----------
        camera : Camera-like object with fx, fy, cx, cy, width, height
        gt_image : (H, W) ground truth
        R_cam, T_cam : camera extrinsics
        """
        H = int(camera.height)
        W = int(camera.width)

        # Build/reuse differentiable params
        if precomputed is None:
            covariances, intensity = self._build_params()
            means_c, cov_c = apply_aspect_correction(
                self.means, covariances, self.aspect_scales
            )
        else:
            means_c, cov_c, intensity = precomputed

        # Full projection pixels (no random pixel sampling)
        ys = torch.arange(H, device=self.device, dtype=torch.float32) + 0.5
        xs = torch.arange(W, device=self.device, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        gt_vals = gt_image.reshape(-1)

        # Opacity-free mode: use constant opacity for visibility, intensity in color
        opacities = torch.ones_like(intensity)
        colors = intensity

        # CUDA splatting render
        rendered = self.renderer.render_at_pixels(
            means_c, cov_c, opacities, colors,
            R_cam, T_cam,
            float(camera.fx), float(camera.fy),
            float(camera.cx), float(camera.cy),
            W, H, pixels,
        )

        # Projection reconstruction loss — pure MSE per the splatting formula
        mse = F.mse_loss(rendered, gt_vals)
        l1  = F.l1_loss(rendered, gt_vals).detach()    # monitoring only
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-12))

        # Windowed SSIM for monitoring only (not in loss)
        with torch.no_grad():
            rendered_2d = rendered.reshape(H, W)
            gt_2d       = gt_vals.reshape(H, W)
            ssim_val    = 1.0 - _windowed_ssim_loss(rendered_2d, gt_2d)

        # Scale regularization (min + max)
        scales_cur = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_small_pen = torch.clamp(self.scale_min_target - scales_cur, min=0.0).mean()
        scale_big_pen   = torch.clamp(scales_cur - self.scale_max_target, min=0.0).mean()
        scale_reg = (self.lambda_scale * scale_small_pen
                     + self.lambda_scale_max * scale_big_pen)

        loss = mse + scale_reg
        if backward_scale is not None:
            (loss * float(backward_scale)).backward(retain_graph=retain_graph)

        return {
            'loss': loss.item(),
            'l1': l1.item(),
            'mse': mse.item(),
            'psnr': psnr.item(),
            'ssim': ssim_val.item(),
            'scale_reg': scale_reg.item(),
        }

    def train(
        self,
        camera,
        dataset: List[dict],
        n_steps: int = 10000,
        log_every: int = 50,
        views_per_step: Optional[int] = 16,
        save_path: Optional[str] = None,
        save_every: int = 2000,
    ) -> List[dict]:
        n_views = len(dataset)
        history = []

        if views_per_step is None or views_per_step <= 0:
            n_views_step = n_views
        else:
            n_views_step = min(int(views_per_step), n_views)

        print(f"\nCUDA splatting training: {n_steps} steps, {n_views} views, "
              f"{n_views_step} views/step (full projection per view)")
        print(f"  Gaussians: {self.means.shape[0]}")
        print("-" * 60)

        pbar = tqdm(range(1, n_steps + 1), desc="Training", unit="step",
                    dynamic_ncols=True)
        best = {'loss': float('inf'), 'psnr': 0.0, 'ssim': 0.0, 'mae': float('inf')}
        for step in pbar:
            # Adaptive density control: densify (split/clone) + prune
            if (self.densify_every > 0
                    and step % self.densify_every == 0
                    and self.densify_start_step <= step <= self.densify_stop_step):
                self.densify_and_prune(step)
            elif self.prune_every > 0 and step % self.prune_every == 0:
                self.prune_gaussians(step)

            self.optimizer.zero_grad()
            per_view_loss = []
            per_view_mse = []
            per_view_l1 = []
            per_view_psnr = []
            per_view_ssim = []

            covariances, intensity = self._build_params()
            means_c, cov_c = apply_aspect_correction(
                self.means, covariances, self.aspect_scales
            )
            precomputed = (means_c, cov_c, intensity)

            view_order = torch.randperm(n_views, device=self.device)[:n_views_step].tolist()
            for view_idx in view_order:
                view = dataset[view_idx]
                is_last_view = (view_idx == view_order[-1])
                metrics_v = self.train_step(
                    camera,
                    view['image'],
                    view['R'],
                    view['T'],
                    precomputed=precomputed,
                    backward_scale=(1.0 / float(n_views_step)),
                    retain_graph=not is_last_view,
                )
                per_view_loss.append(float(metrics_v['loss']))

                per_view_mse.append(float(metrics_v['mse']))
                per_view_l1.append(float(metrics_v['l1']))
                per_view_psnr.append(float(metrics_v['psnr']))
                per_view_ssim.append(float(metrics_v['ssim']))

            # Track position gradients for densification BEFORE clipping —
            # clip_grad_norm_ distorts per-Gaussian magnitudes and would
            # suppress the signal densification depends on.
            # Track from step 1 so the first densification has data.
            if self.means.grad is not None:
                with torch.no_grad():
                    gn = self.means.grad.norm(dim=-1)      # (K,)
                    self._grad_accum += gn
                    self._grad_count += 1

            torch.nn.utils.clip_grad_norm_(
                [self.means, self.log_scales, self.quaternions, self.log_amplitudes],
                max_norm=1.0,
            )

            self.optimizer.step()

            with torch.no_grad():
                self.log_scales.data.clamp_(-7.6, -2.3)   # max σ ≈ e⁻²·³ ≈ 0.10
                self.log_amplitudes.data.clamp_(-9.2, 0.0)
                self.means.data.clamp_(-1.0, 1.0)

            metrics = {
                'loss': float(np.mean(per_view_loss)),
                'l1': float(np.mean(per_view_l1)),
                'mse': float(np.mean(per_view_mse)),
                'psnr': float(np.mean(per_view_psnr)),
                'ssim': float(np.mean(per_view_ssim)),
                'scale_reg': float(self.lambda_scale * torch.clamp(
                    self.scale_min_target - torch.exp(self.log_scales).clamp(1e-5, 1e2),
                    min=0.0,
                ).mean().detach().item()),
            }
            history.append(metrics)

            # Track best metrics
            best['loss'] = min(best['loss'], metrics['loss'])
            best['psnr'] = max(best['psnr'], metrics['psnr'])
            best['ssim'] = max(best['ssim'], metrics['ssim'])
            best['mae']  = min(best['mae'],  metrics['l1'])

            # Update progress bar with best metrics
            pbar.set_postfix({
                'loss': f"{best['loss']:.5f}",
                'psnr': f"{best['psnr']:.2f}",
                'ssim': f"{best['ssim']:.4f}",
                'mae':  f"{best['mae']:.5f}",
                'K':    self.means.shape[0],
            })

            if save_path and step % save_every == 0:
                path = save_path.format(step=step)
                self.save_checkpoint(path, step)
                tqdm.write(f"  Checkpoint saved → {path}")

        pbar.close()

        if save_path:
            path = save_path.format(step=n_steps)
            self.save_checkpoint(path, n_steps)
            print(f"  Final checkpoint → {path}")

        return history

    # ------------------------------------------------------------------
    #  Optimizer rebuild helper
    # ------------------------------------------------------------------
    def _rebuild_optimizer(self):
        """Rebuild Adam optimizer after parameter count changes."""
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam([
            {'params': [self.means],          'lr': lr},
            {'params': [self.log_scales],     'lr': lr * 0.5},
            {'params': [self.quaternions],    'lr': lr * 0.3},
            {'params': [self.log_amplitudes], 'lr': lr},
        ])

    def _reset_grad_accum(self):
        """Reset gradient accumulators to match current Gaussian count."""
        K = self.means.shape[0]
        self._grad_accum = torch.zeros(K, device=self.device)
        self._grad_count = torch.zeros(K, device=self.device)

    # ------------------------------------------------------------------
    #  Pruning
    # ------------------------------------------------------------------
    def prune_gaussians(self, step=0):
        """Remove low-intensity Gaussians."""
        with torch.no_grad():
            intensity = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
            keep = intensity > self.prune_intensity_thresh
            n_before = keep.shape[0]
            n_keep = keep.sum().item()

            if n_keep >= n_before or n_keep < self.prune_min_gaussians:
                return 0

            self.means = nn.Parameter(self.means.data[keep].clone())
            self.log_scales = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_amplitudes = nn.Parameter(self.log_amplitudes.data[keep].clone())

            self._rebuild_optimizer()
            self._reset_grad_accum()

            n_pruned = n_before - n_keep
            print(f"  [Prune @ step {step}] {n_before} → {n_keep} "
                  f"(removed {n_pruned}, intensity < {self.prune_intensity_thresh})")
            return n_pruned

    # ------------------------------------------------------------------
    #  Densification: split large / clone small high-gradient Gaussians
    # ------------------------------------------------------------------
    def densify_and_prune(self, step: int = 0):
        """
        Standard 3DGS adaptive density control.

        1. Compute average position-gradient norm per Gaussian.
        2. **Split** Gaussians with high gradient AND large max scale:
           remove parent, create 2 children with scale / 1.6, offset
           sampled from the parent's distribution.
        3. **Clone** Gaussians with high gradient AND small max scale:
           keep original, add one copy at the same position.
        4. **Prune** low-intensity Gaussians.
        5. Rebuild optimizer and reset gradient accumulators.

        Returns (n_split, n_clone).
        """
        with torch.no_grad():
            K = self.means.shape[0]
            avg_grad = self._grad_accum / self._grad_count.clamp(min=1)

            # Diagnostic: show gradient statistics
            n_tracked = int(self._grad_count[0].item())
            tqdm.write(f"  [Densify @ step {step}] grad stats: "
                       f"tracked={n_tracked} steps, "
                       f"avg_norm min={avg_grad.min():.6f} "
                       f"mean={avg_grad.mean():.6f} "
                       f"max={avg_grad.max():.6f} "
                       f"(thresh={self.densify_grad_thresh})")

            # ── masks ────────────────────────────────────────────────
            high_grad = avg_grad > self.densify_grad_thresh

            scales    = torch.exp(self.log_scales).clamp(1e-5, 1e2)  # (K, 3)
            max_scale = scales.max(dim=-1).values                    # (K,)

            split_mask = high_grad & (max_scale > self.densify_scale_thresh)
            clone_mask = high_grad & ~split_mask

            n_split = int(split_mask.sum().item())
            n_clone = int(clone_mask.sum().item())
            n_high  = int(high_grad.sum().item())

            tqdm.write(f"    high_grad={n_high}/{K}  "
                       f"split={n_split}  clone={n_clone}  "
                       f"(scale_thresh={self.densify_scale_thresh})")

            # Nothing to densify — skip rebuild to preserve optimizer state
            if n_split == 0 and n_clone == 0:
                self._reset_grad_accum()
                tqdm.write(f"    → no densification needed, skipping")
                return 0, 0

            # Budget check (split: net +1 each; clone: +1 each)
            if K + n_split + n_clone > self.max_gaussians:
                self._reset_grad_accum()
                tqdm.write(f"    → budget exceeded ({K}+{n_split}+{n_clone}"
                           f"={K+n_split+n_clone} > {self.max_gaussians}), skipping")
                return 0, 0

            # ── generate new Gaussians ───────────────────────────────
            new_m, new_ls, new_q, new_la = [], [], [], []

            if n_split > 0:
                s_m  = self.means.data[split_mask]
                s_ls = self.log_scales.data[split_mask]
                s_q  = self.quaternions.data[split_mask]
                s_la = self.log_amplitudes.data[split_mask]

                reduced_ls = s_ls - math.log(1.6)          # smaller children
                s_scales   = torch.exp(s_ls)               # (N, 3)

                # Offset along the principal axis (largest-scale direction
                # rotated into world space by the quaternion) — critical for
                # elongated neurite Gaussians so children land along the
                # neurite, not along arbitrary world axes.
                q = F.normalize(s_q, p=2, dim=-1)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                # Build rotation matrix columns (local→world)
                col0 = torch.stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)], dim=-1)
                col1 = torch.stack([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)], dim=-1)
                col2 = torch.stack([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)], dim=-1)
                R_cols = torch.stack([col0, col1, col2], dim=-1)   # (N, 3, 3)
                # Pick the column corresponding to each Gaussian's largest scale
                max_axis = s_scales.argmax(dim=-1)                # (N,)
                principal = R_cols[torch.arange(len(max_axis), device=self.device), :, max_axis]  # (N, 3)
                offset = principal * s_scales.max(dim=-1, keepdim=True).values  # (N, 3)

                for sign in (1.0, -1.0):                   # two children
                    new_m .append(s_m + sign * offset)
                    new_ls.append(reduced_ls.clone())
                    new_q .append(s_q.clone())
                    new_la.append(s_la.clone())

            if n_clone > 0:
                new_m .append(self.means.data[clone_mask].clone())
                new_ls.append(self.log_scales.data[clone_mask].clone())
                new_q .append(self.quaternions.data[clone_mask].clone())
                new_la.append(self.log_amplitudes.data[clone_mask].clone())

            # ── assemble: keep non-split originals + new ─────────────
            keep_mask = ~split_mask
            parts_m  = [self.means.data[keep_mask]]          + new_m
            parts_ls = [self.log_scales.data[keep_mask]]     + new_ls
            parts_q  = [self.quaternions.data[keep_mask]]    + new_q
            parts_la = [self.log_amplitudes.data[keep_mask]] + new_la

            all_m  = torch.cat(parts_m,  dim=0)
            all_ls = torch.cat(parts_ls, dim=0)
            all_q  = torch.cat(parts_q,  dim=0)
            all_la = torch.cat(parts_la, dim=0)

            # ── prune low-intensity ──────────────────────────────────
            intensity = torch.exp(all_la.clamp(-10, 6)).clamp(0, 1)
            alive     = intensity > self.prune_intensity_thresh
            if alive.sum() < self.prune_min_gaussians:
                alive = torch.ones(all_m.shape[0], dtype=torch.bool,
                                   device=self.device)
            n_pruned = int((~alive).sum().item())

            self.means          = nn.Parameter(all_m[alive])
            self.log_scales     = nn.Parameter(all_ls[alive])
            self.quaternions    = nn.Parameter(all_q[alive])
            self.log_amplitudes = nn.Parameter(all_la[alive])

            self._rebuild_optimizer()
            self._reset_grad_accum()

            K_new = self.means.shape[0]
            print(f"  [Densify @ step {step}] K: {K} → {K_new}  "
                  f"(split {n_split}, clone {n_clone}, pruned {n_pruned})")
            return n_split, n_clone

    def save_checkpoint(self, path, step):
        """Save learnable parameters."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'means': self.means.data.cpu(),
            'log_scales': self.log_scales.data.cpu(),
            'quaternions': self.quaternions.data.cpu(),
            'log_amplitudes': self.log_amplitudes.data.cpu(),
            'step': step,
        }, path)
