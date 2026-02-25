"""
neuro3dgs/losses.py

Loss functions for fluorescence microscopy 3DGS optimisation.

Design rationale for neurite volumes:
──────────────────────────────────────
  1.  L1 + D-SSIM (photometric)    — perceptual fidelity to MIP ground truth
  2.  Morphology-aware depth loss   — penalise depth inconsistency along SWC edges
  3.  Sparsity / opacity regulariser— prevents floaters in empty regions
  4.  Gaussian size regulariser     — prevents Gaussians collapsing or exploding
  5.  Feature smoothness            — encourages smooth intensity along neurite branches
  6.  SWC proximity loss            — anchors Gaussians near known morphology nodes
  7.  Volume consistency loss       — soft constraint: rendered MIP should match
                                       GT max-intensity projection of the .tif volume
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  SSIM
# ─────────────────────────────────────────────────────────────────────────────

def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) on single-channel images [H, W].
    Returns mean SSIM scalar.
    """
    # Expand to [1, 1, H, W]
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)

    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]

    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_xx = F.conv2d(x * x, kernel, padding=pad)
    mu_yy = F.conv2d(y * y, kernel, padding=pad)
    mu_xy = F.conv2d(x * y, kernel, padding=pad)

    sigma_x  = mu_xx - mu_x**2
    sigma_y  = mu_yy - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Component losses
# ─────────────────────────────────────────────────────────────────────────────

def photometric_loss(
    rendered: torch.Tensor,    # [H, W, C]
    target: torch.Tensor,      # [H, W, C]
    lambda_dssim: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """
    L1 + (1 - SSIM) per channel, averaged.

    For multi-channel fluorescence (C>1) we compute per-channel then mean.
    """
    C = rendered.shape[-1]
    l1  = (rendered - target).abs().mean()

    ssim_vals = []
    for c in range(C):
        ssim_vals.append(_ssim(rendered[..., c], target[..., c]))
    ssim_mean = torch.stack(ssim_vals).mean()

    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_mean)
    return loss, {"l1": l1.item(), "ssim": ssim_mean.item()}


def opacity_sparsity_loss(
    opacity: torch.Tensor,   # [N, 1] sigmoid activations
    target_mean: float = 0.1,
) -> torch.Tensor:
    """
    Entropy-based sparsity:  H(α) = -α log(α) - (1-α) log(1-α)
    Penalises uncertain opacities; encourages each Gaussian to be
    either clearly visible or transparent.

    Also includes a mean-opacity penalty to prevent globally high opacity.
    """
    eps = 1e-6
    op = opacity.clamp(eps, 1 - eps)
    entropy = -(op * op.log() + (1 - op) * (1 - op).log()).mean()

    # Penalty for mean opacity drifting too high (floaters)
    mean_penalty = F.relu(op.mean() - target_mean)

    return entropy * 0.1 + mean_penalty


def scale_regularisation(
    log_scales: torch.Tensor,   # [N, 3]
    min_log_s: float = -8.0,
    max_log_s: float = -2.0,
) -> torch.Tensor:
    """
    Soft barrier loss that keeps Gaussian scales in a meaningful range.
    Too small → numerical instability; too large → blurry reconstructions.
    """
    too_small = F.relu(min_log_s - log_scales).pow(2).mean()
    too_large = F.relu(log_scales - max_log_s).pow(2).mean()
    return too_small + too_large


def swc_proximity_loss(
    means: torch.Tensor,      # [N, 3] Gaussian positions
    swc_xyz: torch.Tensor,    # [M, 3] SWC node positions (normalised)
    k: int = 3,               # check nearest k SWC nodes
    lambda_prox: float = 0.01,
) -> torch.Tensor:
    """
    Encourage Gaussians to stay near known morphology nodes.
    Only acts on the subset of Gaussians that originated from SWC
    (identified by proximity at init; here we apply globally with a weak weight).

    dist_to_nearest_swc: average minimum distance from each Gaussian to SWC tree.
    """
    # Pairwise distances [N, M]
    diff = means.unsqueeze(1) - swc_xyz.unsqueeze(0)    # [N, M, 3]
    dist = diff.norm(dim=-1)                             # [N, M]

    topk_dist, _ = dist.topk(k, dim=-1, largest=False)  # [N, k]
    min_dist = topk_dist[:, 0]                           # [N] nearest SWC node

    return lambda_prox * min_dist.mean()


def volume_mip_consistency_loss(
    rendered_mip: torch.Tensor,   # [H, W, C]  rendered soft-MIP
    gt_mip: torch.Tensor,         # [H, W, C]  ground-truth max-intensity projection
    mask: Optional[torch.Tensor] = None,  # [H, W] foreground mask
) -> torch.Tensor:
    """
    Mean-squared error between rendered MIP and GT MIP from the .tif volume.
    Optional mask focuses the loss on foreground neurite regions.
    """
    diff = (rendered_mip - gt_mip) ** 2
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
    return diff.mean()


def depth_edge_loss(
    depth: torch.Tensor,   # [H, W]  rendered depth map
) -> torch.Tensor:
    """
    TV-like smoothness on depth within connected neurite regions.
    Discourages jagged depth transitions that would indicate floaters.
    """
    dy = (depth[1:, :] - depth[:-1, :]).abs()
    dx = (depth[:, 1:] - depth[:, :-1]).abs()
    return dy.mean() + dx.mean()


def feature_smoothness_loss(
    features: torch.Tensor,    # [N, C]
    means: torch.Tensor,       # [N, 3]
    k_nn: int = 4,
    max_gaussians: int = 4000,
) -> torch.Tensor:
    """
    Encourage similar features for spatially proximate Gaussians.
    Computed on a random subsample to keep O(N^2) tractable.
    """
    if means.shape[0] > max_gaussians:
        idx = torch.randperm(means.shape[0], device=means.device)[:max_gaussians]
        means_s = means[idx]
        feats_s = features[idx]
    else:
        means_s, feats_s = means, features

    diff_xyz = means_s.unsqueeze(1) - means_s.unsqueeze(0)      # [M,M,3]
    dist2    = (diff_xyz ** 2).sum(-1)                           # [M,M]
    # Exclude self (zero distance)
    dist2.fill_diagonal_(1e10)
    nn_idx = dist2.topk(k_nn, dim=-1, largest=False).indices     # [M, k]

    f_center = feats_s.unsqueeze(1).expand(-1, k_nn, -1)        # [M,k,C]
    f_nn     = feats_s[nn_idx]                                   # [M,k,C]
    return (f_center - f_nn).abs().mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class NeuriteReconLoss(nn.Module):
    """
    Composite loss for neurite 3DGS optimisation.

    Weights are tuned for fluorescence microscopy:
      - Photometric is primary
      - Volume MIP consistency is strong early (forces rough shape)
      - Regularisation terms are mild to allow free-form representation
    """

    def __init__(
        self,
        lambda_photometric:    float = 1.0,
        lambda_mip_consist:    float = 0.5,
        lambda_intensity_sparse: float = 0.05,
        lambda_scale_reg:      float = 0.01,
        lambda_depth_smooth:   float = 0.005,
        lambda_feat_smooth:    float = 0.001,
        lambda_swc_prox:       float = 0.01,
        lambda_dssim:          float = 0.2,
    ):
        super().__init__()
        self.w_photo  = lambda_photometric
        self.w_mip    = lambda_mip_consist
        self.w_int    = lambda_intensity_sparse
        self.w_scale  = lambda_scale_reg
        self.w_depth  = lambda_depth_smooth
        self.w_feat   = lambda_feat_smooth
        self.w_swc    = lambda_swc_prox
        self.lambda_dssim = lambda_dssim

    def forward(
        self,
        rendered: torch.Tensor,          # [H, W, C]
        target_view: torch.Tensor,       # [H, W, C] — GT view (MIP slice)
        depth: torch.Tensor,             # [H, W]
        weight: torch.Tensor,            # [H, W]
        gaussians,                        # NeuriteGaussians
        gt_mip: Optional[torch.Tensor] = None,  # [H, W, C] GT max proj
        swc_xyz: Optional[torch.Tensor] = None, # [M, 3]
        iteration: int = 0,
    ) -> Tuple[torch.Tensor, dict]:

        losses = {}

        # 1. Photometric (primary)
        L_photo, photo_dict = photometric_loss(rendered, target_view, self.lambda_dssim)
        losses.update({f"photo_{k}": v for k, v in photo_dict.items()})

        # 2. Volume MIP consistency (anneal weight after warm-up)
        L_mip = torch.tensor(0.0, device=rendered.device)
        if gt_mip is not None:
            L_mip = volume_mip_consistency_loss(rendered, gt_mip)
        mip_w = self.w_mip * max(0.0, 1.0 - iteration / 3000.0)  # decay over 3k iters

        # 3. Intensity sparsity (encourages compact emission)
        L_int = opacity_sparsity_loss(gaussians.intensity)

        # 4. Scale regularisation
        L_scale = scale_regularisation(gaussians._log_scales)

        # 5. Depth smoothness
        L_depth = depth_edge_loss(depth)

        # 6. Feature smoothness (expensive; apply every 10 iters)
        L_feat = torch.tensor(0.0, device=rendered.device)
        if iteration % 10 == 0:
            L_feat = feature_smoothness_loss(gaussians.features, gaussians.means)

        # 7. SWC proximity
        L_swc = torch.tensor(0.0, device=rendered.device)
        if swc_xyz is not None:
            L_swc = swc_proximity_loss(gaussians.means, swc_xyz)

        total = (
            self.w_photo  * L_photo  +
            mip_w         * L_mip    +
            self.w_int    * L_int    +
            self.w_scale  * L_scale  +
            self.w_depth  * L_depth  +
            self.w_feat   * L_feat   +
            self.w_swc    * L_swc
        )

        losses.update({
            "total":    total.item(),
            "photo":    L_photo.item(),
            "mip_cons": L_mip.item(),
            "intensity": L_int.item(),
            "scale":    L_scale.item(),
            "depth":    L_depth.item(),
            "feat":     L_feat.item(),
            "swc_prox": L_swc.item(),
        })
        return total, losses
