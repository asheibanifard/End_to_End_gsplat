"""
neuro3dgs/gaussian_model.py

3D Gaussian representation for dense neurite volumes.
Initialised from:
  1. SWC morphology nodes  (high-confidence structure)
  2. Voxel intensity seeds (fill-in for unlabelled fluorescence signal)

Parameters per Gaussian:
  means3d  : [N, 3]  world position (normalised voxel [0,1]^3)
  quats    : [N, 4]  unit quaternion (w,x,y,z) for rotation
  log_scales: [N, 3]  log of scale in each principal axis
  logit_intensity: [N, C]  emission intensity (sigmoid → [0,1])
  
Note: For fluorescence MIP, there is no opacity/absorption — only emission.
Each Gaussian emits light with intensity = sigmoid(logit_intensity).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class NeuriteGaussians(nn.Module):
    """
    Learnable 3D Gaussian cloud representing a fluorescence microscopy neurite volume.
    """

    # Adaptive density control thresholds
    GRAD_THRESHOLD  = 0.002    # positional gradient magnitude for cloning/splitting (10× higher)
    INTENSITY_PRUNE = 0.01     # intensity below which a Gaussian is pruned
    MAX_SCREEN_SIZE = 0.05     # fraction of image — big Gaussians get split
    DENSIFY_INTERVAL= 500      # iterations between densification steps (5× less frequent)
    PRUNE_INTERVAL  = 500      # iterations between intensity/size pruning
    INTENSITY_RESET_INTERVAL = 3000

    def __init__(
        self,
        means3d: torch.Tensor,           # [N, 3]
        quats: Optional[torch.Tensor]   = None,  # [N, 4]
        log_scales: Optional[torch.Tensor] = None, # [N, 3]
        logit_intensity: Optional[torch.Tensor] = None,  # [N, C] emission intensity
        intensity_dim: int = 1,  # 1 for grayscale fluorescence
    ):
        super().__init__()
        N = means3d.shape[0]
        self.intensity_dim = intensity_dim

        self._means   = nn.Parameter(means3d.float())
        self._quats   = nn.Parameter(
            quats if quats is not None else self._init_quats(N, means3d.device)
        )
        self._log_scales = nn.Parameter(
            log_scales if log_scales is not None else self._init_scales(N, means3d.device)
        )
        # Emission intensity: sigmoid(logit_intensity) → [0, 1]
        # Init to 0 → sigmoid(0) = 0.5 mid-intensity
        self._logit_intensity = nn.Parameter(
            logit_intensity if logit_intensity is not None
            else torch.zeros(N, intensity_dim, device=means3d.device)
        )

        # Buffers for adaptive density control (not optimised)
        self.register_buffer("_grad_accum",   torch.zeros(N, device=means3d.device))
        self.register_buffer("_grad_count",   torch.zeros(N, device=means3d.device))
        self.register_buffer("_max_radii2D",  torch.zeros(N, device=means3d.device))

    # ─────────────────────────────────────────────────────────────────────────
    #  Initialisers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _init_quats(N: int, device) -> torch.Tensor:
        # Identity quaternion [1, 0, 0, 0]
        q = torch.zeros(N, 4, device=device)
        q[:, 0] = 1.0
        return q

    @staticmethod
    def _init_scales(N: int, device, init_size: float = -5.0) -> torch.Tensor:
        # isotropic small Gaussians (exp(-5) ≈ 0.007 in normalised coords)
        return torch.full((N, 3), init_size, device=device)

    @classmethod
    def from_swc_and_volume(
        cls,
        swc_xyz: torch.Tensor,       # [M, 3]  normalised
        swc_radii: torch.Tensor,     # [M]
        vol_seeds: torch.Tensor,     # [K, 3]  normalised
        vol_intensities: torch.Tensor, # [K]
        intensity_dim: int = 1,      # 1 for grayscale fluorescence
    ) -> "NeuriteGaussians":
        """
        Fuse SWC nodes and volume voxel seeds into a single Gaussian cloud.
        
        For fluorescence MIP: each Gaussian has emission intensity (no opacity).
        SWC nodes initialized with high intensity; volume seeds use voxel values.
        """
        device = swc_xyz.device
        M, K = swc_xyz.shape[0], vol_seeds.shape[0]
        N = M + K

        # ── Positions ──────────────────────────────────────────────────────
        means = torch.cat([swc_xyz, vol_seeds], dim=0)          # [N, 3]

        # ── Scales ─────────────────────────────────────────────────────────
        #  SWC: use morphological radius; volume seeds: reasonable size
        #  exp(-2) ≈ 0.135 of unit volume — better coverage for soft-MIP
        min_log_scale = -2.5
        swc_log_s = torch.log(swc_radii.clamp(min=0.05)).unsqueeze(1).expand(-1, 3)
        swc_log_s = swc_log_s.clamp(min=min_log_scale)
        vol_log_s = torch.full((K, 3), -2.5, device=device)  # exp(-2.5) ≈ 0.08
        log_scales = torch.cat([swc_log_s, vol_log_s], dim=0)   # [N, 3]

        # ── Quaternions (identity everywhere) ──────────────────────────────
        quats = cls._init_quats(N, device)

        # ── Emission Intensity ─────────────────────────────────────────────
        # SWC nodes: high intensity (bright neurite structures) → logit(0.9) ≈ 2.2
        # Volume seeds: use actual voxel intensity
        swc_intensity = torch.full((M, intensity_dim), 0.9, device=device)
        vol_intensity = vol_intensities.unsqueeze(1).clamp(0.01, 0.99)
        if intensity_dim > 1:
            vol_intensity = vol_intensity.expand(-1, intensity_dim).clone()
        
        intensity = torch.cat([swc_intensity, vol_intensity], dim=0)  # [N, C]
        logit_intensity = torch.logit(intensity)

        return cls(
            means3d=means,
            quats=quats,
            log_scales=log_scales,
            logit_intensity=logit_intensity,
            intensity_dim=intensity_dim,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Properties (with activations)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def quats(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._quats, dim=-1)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._log_scales)

    @property
    def intensity(self) -> torch.Tensor:
        """Emission intensity in [0, 1] — the primary parameter for fluorescence."""
        return torch.sigmoid(self._logit_intensity)

    # Backward compatibility aliases
    @property
    def opacity(self) -> torch.Tensor:
        """Alias for intensity (for compatibility with renderer)."""
        return self.intensity
    
    @property
    def features(self) -> torch.Tensor:
        """Alias for intensity (for compatibility with renderer)."""
        return self.intensity

    @property
    def N(self) -> int:
        return self._means.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    #  Build 3-D covariance (PyTorch path; CUDA path called externally)
    # ─────────────────────────────────────────────────────────────────────────

    def build_cov3d_torch(self) -> torch.Tensor:
        """
        Σ = R * S * S^T * R^T  → upper triangle [N, 6]
        Used on CPU or when CUDA extension is not available.
        """
        q = self.quats           # [N, 4]
        s = self.scales          # [N, 3]

        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        R00 = 1 - 2*(y**2 + z**2)
        R01 = 2*(x*y - w*z)
        R02 = 2*(x*z + w*y)
        R10 = 2*(x*y + w*z)
        R11 = 1 - 2*(x**2 + z**2)
        R12 = 2*(y*z - w*x)
        R20 = 2*(x*z - w*y)
        R21 = 2*(y*z + w*x)
        R22 = 1 - 2*(x**2 + y**2)

        sx, sy, sz = s[..., 0:1], s[..., 1:2], s[..., 2:3]

        M0 = torch.stack([R00*sx, R01*sy, R02*sz], dim=-1).squeeze(1)  # [N, 3]
        M1 = torch.stack([R10*sx, R11*sy, R12*sz], dim=-1).squeeze(1)
        M2 = torch.stack([R20*sx, R21*sy, R22*sz], dim=-1).squeeze(1)

        cov = torch.stack([
            (M0*M0).sum(-1),   # c00
            (M0*M1).sum(-1),   # c01
            (M0*M2).sum(-1),   # c02
            (M1*M1).sum(-1),   # c11
            (M1*M2).sum(-1),   # c12
            (M2*M2).sum(-1),   # c22
        ], dim=-1)   # [N, 6]
        return cov

    # ─────────────────────────────────────────────────────────────────────────
    #  Adaptive density control
    # ─────────────────────────────────────────────────────────────────────────

    def accumulate_grad(self, grad_mag: torch.Tensor, radii2D: torch.Tensor):
        """Called every iteration to collect view-space gradient statistics."""
        self._grad_accum  += grad_mag.detach()
        self._grad_count  += (grad_mag > 0).float()
        self._max_radii2D  = torch.max(self._max_radii2D, radii2D.detach())

    def densify_and_prune(
        self,
        optimizer: torch.optim.Optimizer,
        img_size: float = 1.0,
    ):
        """
        Adaptive density control step:
          - Clone small Gaussians with large positional gradients
          - Split large Gaussians with large positional gradients
          - Prune transparent / too-large Gaussians
        """
        avg_grad = self._grad_accum / (self._grad_count + 1e-6)
        high_grad = avg_grad > self.GRAD_THRESHOLD
        big = self._max_radii2D > self.MAX_SCREEN_SIZE * img_size

        clone_mask = high_grad & ~big   # small + high grad → clone
        split_mask = high_grad & big    # large + high grad → split

        # Clone
        if clone_mask.any():
            self._clone(clone_mask, optimizer)

        # Split (re-fetch masks after clone changes N)
        # recompute because N changed
        avg_grad = self._grad_accum / (self._grad_count + 1e-6)
        high_grad = avg_grad > self.GRAD_THRESHOLD
        big = self._max_radii2D > self.MAX_SCREEN_SIZE * img_size
        split_mask = high_grad & big
        if split_mask.any():
            self._split(split_mask, optimizer)

        # Prune dim Gaussians (low intensity contributes little to fluorescence)
        # Use mean intensity across channels for multi-channel
        if self.intensity.shape[-1] > 1:
            mean_intensity = self.intensity.mean(dim=-1)
        else:
            mean_intensity = self.intensity.squeeze(-1)
        prune_mask = (mean_intensity < self.INTENSITY_PRUNE)
        # Keep at least some Gaussians to avoid degenerate state
        if prune_mask.sum() < prune_mask.numel():  # don't prune all
            if prune_mask.any():
                self._prune(prune_mask, optimizer)

        # Reset accumulators
        self._reset_grad_stats()

    def _clone(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer):
        new_means   = self._means[mask].detach().clone()
        new_means  += torch.randn_like(new_means) * self.scales[mask].detach().mean(-1, keepdim=True) * 0.1
        self._append_gaussians(mask, new_means, optimizer)

    def _split(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer):
        s = self.scales[mask].detach()
        new_means_a = self._means[mask].detach() + s * 0.5
        new_means_b = self._means[mask].detach() - s * 0.5
        # shrink scales
        new_log_s = self._log_scales[mask].detach() - np.log(1.6)

        idx = mask.nonzero(as_tuple=True)[0]
        # Remove originals, add two children
        self._remove_gaussians(mask, optimizer)
        self._append_raw(
            torch.cat([new_means_a, new_means_b]),
            self.quats[mask].detach().repeat(2, 1),
            new_log_s.repeat(2, 1),
            self._logit_intensity[mask].detach().repeat(2, 1),
            optimizer,
        )

    def _prune(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer):
        self._remove_gaussians(mask, optimizer)

    def reset_intensity(self):
        """Periodically reset intensity to mid-level."""
        with torch.no_grad():
            self._logit_intensity.zero_()  # sigmoid(0) = 0.5

    # ── Low-level add/remove ────────────────────────────────────────────────

    def _cat_param(self, param: nn.Parameter, new_data: torch.Tensor) -> nn.Parameter:
        return nn.Parameter(torch.cat([param.data, new_data], dim=0))

    def _append_gaussians(
        self, mask: torch.Tensor, new_means: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        M = new_means.shape[0]
        d = new_means.device
        self._append_raw(
            new_means,
            self._init_quats(M, d),
            self._log_scales[mask].detach().clone(),
            self._logit_intensity[mask].detach().clone(),
            optimizer,
        )

    def _append_raw(
        self,
        new_means, new_quats, new_log_scales,
        new_logit_intensity,
        optimizer: torch.optim.Optimizer,
    ):
        M = new_means.shape[0]
        d = new_means.device

        for p_name, new_data in [
            ("_means",           new_means),
            ("_quats",           new_quats),
            ("_log_scales",      new_log_scales),
            ("_logit_intensity", new_logit_intensity),
        ]:
            old_p = getattr(self, p_name)
            new_p = nn.Parameter(torch.cat([old_p.data, new_data], dim=0))
            setattr(self, p_name, new_p)
            # Update optimizer state
            for group in optimizer.param_groups:
                for i, p in enumerate(group["params"]):
                    if p is old_p:
                        group["params"][i] = new_p
                        if old_p in optimizer.state:
                            state = optimizer.state.pop(old_p)
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == old_p.shape[0]:
                                    state[k] = torch.cat([v, torch.zeros_like(new_data[:1].expand(M, *v.shape[1:]))], dim=0)
                            optimizer.state[new_p] = state

        # Extend buffers
        N = self._means.shape[0]
        self._grad_accum  = torch.zeros(N, device=d)
        self._grad_count  = torch.zeros(N, device=d)
        self._max_radii2D = torch.zeros(N, device=d)

    def _remove_gaussians(self, mask: torch.Tensor, optimizer: torch.optim.Optimizer):
        keep = ~mask
        for p_name in ["_means", "_quats", "_log_scales", "_logit_intensity"]:
            old_p = getattr(self, p_name)
            new_data = old_p.data[keep]
            new_p = nn.Parameter(new_data)
            setattr(self, p_name, new_p)
            for group in optimizer.param_groups:
                for i, p in enumerate(group["params"]):
                    if p is old_p:
                        group["params"][i] = new_p
                        if old_p in optimizer.state:
                            state = optimizer.state.pop(old_p)
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == old_p.shape[0]:
                                    state[k] = v[keep]
                            optimizer.state[new_p] = state

        N = self._means.data.shape[0]
        d = self._means.device
        self._grad_accum  = torch.zeros(N, device=d)
        self._grad_count  = torch.zeros(N, device=d)
        self._max_radii2D = torch.zeros(N, device=d)

    def _reset_grad_stats(self):
        self._grad_accum.zero_()
        self._grad_count.zero_()
        self._max_radii2D.zero_()

    # ─────────────────────────────────────────────────────────────────────────
    #  State dict helpers
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "means":            self._means.data,
            "quats":            self._quats.data,
            "log_scales":       self._log_scales.data,
            "logit_intensity":  self._logit_intensity.data,
            "intensity_dim":    self.intensity_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device="cpu") -> "NeuriteGaussians":
        d = torch.load(path, map_location=device)
        return cls(
            means3d=d["means"].to(device),
            quats=d["quats"].to(device),
            log_scales=d["log_scales"].to(device),
            logit_intensity=d["logit_intensity"].to(device),
            intensity_dim=d.get("intensity_dim", d["logit_intensity"].shape[1]),
        )
