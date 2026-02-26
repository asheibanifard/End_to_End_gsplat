#!/usr/bin/env python3
"""Debug why rendered images differ from targets"""
import torch
import sys
sys.path.insert(0, 'neurogs/neurogs_v7')
from neurogs_v7 import GaussianMixtureField
from train_end_to_end import NeuRoGStoRenderer, create_multiview_cameras
from renderer import Camera, render

# Load checkpoint
device = 'cuda'
checkpoint = torch.load('../neurogs_v7/gmf_refined_best.pt', map_location=device)
K = checkpoint['means'].shape[0]
gaussian_field = GaussianMixtureField(num_gaussians=K, init_amplitude=0.1)
gaussian_field.load_state_dict(checkpoint)
gaussian_field = gaussian_field.to(device)

# Create adapter
adapter = NeuRoGStoRenderer(gaussian_field)

# Create camera
cameras = create_multiview_cameras(128, 128, num_views=1, device=device)
camera, view_mat = cameras[0]

# Helper to create gaussians object
class TempGaussians:
    def __init__(self, m, cov, inten):
        self.means = m
        self.intensity = inten
        self.N = m.shape[0]
        self.quats = torch.zeros(self.N, 4, device=m.device)
        self.quats[:, 0] = 1.0
        self._log_scales = torch.zeros(self.N, 3, device=m.device)
        self._cov3d = cov
    def build_cov3d_torch(self):
        return self._cov3d

# Render multiple times
print("Rendering same scene 5 times to check for non-determinism:\n")
for i in range(5):
    means, cov3d, intensity = adapter.get_renderer_params()
    gaussians = TempGaussians(means, cov3d, intensity)
    img, weight, depth = render(gaussians, camera, view_mat, use_cuda_cov=False)
    print(f"Render {i}: min={img.min():.6f}, max={img.max():.6f}, mean={img.mean():.6f}, "
          f"sum={img.sum():.2f}")

# Check if parameters are changing
print("\nChecking if adapter parameters change:")
for i in range(3):
    means, cov3d, intensity = adapter.get_renderer_params()
    print(f"Call {i}: means_hash={hash(means.data_ptr())}, "
          f"means_range=[{means.min():.3f}, {means.max():.3f}]")
