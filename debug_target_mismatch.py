#!/usr/bin/env python3
"""Compare target generation vs training rendering"""
import torch
import sys
sys.path.insert(0, 'neurogs/neurogs_v7')
from neurogs_v7 import GaussianMixtureField
from train_end_to_end import NeuRoGStoRenderer, create_multiview_cameras, generate_synthetic_targets
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

# Create cameras
cameras = create_multiview_cameras(128, 128, num_views=2, device=device)

# Generate targets
print("Generating targets...")
targets = generate_synthetic_targets(gaussian_field, cameras, adapter)
print(f"Target 0: min={targets[0].min():.6f}, max={targets[0].max():.6f}")
print(f"Target 1: min={targets[1].min():.6f}, max={targets[1].max():.6f}")

# Now render the same way as training loop
print("\nRendering in training style...")
means, cov3d, intensity = adapter.get_renderer_params()

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

gaussians = TempGaussians(means, cov3d, intensity)

for idx, (camera, view_mat) in enumerate(cameras):
    img, _, _ = render(gaussians, camera, view_mat, use_cuda_cov=False)
    print(f"Render {idx}: min={img.min():.6f}, max={img.max():.6f}")
    print(f"  Difference from target: {(img - targets[idx]).abs().max():.6f}")
