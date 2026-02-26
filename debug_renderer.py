#!/usr/bin/env python3
"""Debug renderer output"""
import torch
import sys
sys.path.insert(0, 'neurogs/neurogs_v7')
from neurogs_v7 import GaussianMixtureField
from train_end_to_end import NeuRoGStoRenderer
from renderer import Camera, render

# Load checkpoint
device = 'cuda'
checkpoint = torch.load('../neurogs_v7/gmf_refined_best.pt', map_location=device)
K = checkpoint['means'].shape[0]
gaussian_field = GaussianMixtureField(num_gaussians=K, init_amplitude=0.1)
gaussian_field.load_state_dict(checkpoint)
gaussian_field = gaussian_field.to(device)

# Get parameters
adapter = NeuRoGStoRenderer(gaussian_field)
means, cov3d, intensity = adapter.get_renderer_params()

print(f'Parameters OK')
print(f'  Means: {means.shape}')
print(f'  Cov3d: {cov3d.shape}')
print(f'  Intensity: {intensity.shape}')

# Create camera
cam = Camera(128, 128, device=device)
view_mat = cam.view_matrix_orthographic(axis=2)

print(f'\nCamera OK')
print(f'  View matrix: {view_mat.shape}')
print(f'  View matrix range: [{view_mat.min():.3f}, {view_mat.max():.3f}]')
print(f'  Has NaN: {torch.isnan(view_mat).any()}')

# Create temp gaussians
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

print(f'\nRendering...')
try:
    img, weight, depth = render(gaussians, cam, view_mat, use_cuda_cov=False)
    print(f'  Image: {img.shape}')
    print(f'  Image range: [{img.min():.3f}, {img.max():.3f}]')
    print(f'  Has NaN: {torch.isnan(img).any()}')
    print(f'  Weight: {weight.shape}')
    print(f'  Weight range: [{weight.min():.3f}, {weight.max():.3f}]')
    print(f'  Depth: {depth.shape}')
    print(f'  Depth range: [{depth.min():.3f}, {depth.max():.3f}]')
except Exception as e:
    print(f'  ERROR: {e}')
    import traceback
    traceback.print_exc()
