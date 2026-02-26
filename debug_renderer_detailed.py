#!/usr/bin/env python3
"""Debug renderer in detail"""
import torch
import sys
import math
sys.path.insert(0, 'neurogs/neurogs_v7')
from neurogs_v7 import GaussianMixtureField
from train_end_to_end import NeuRoGStoRenderer
from renderer import Camera

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

print(f'Inspecting renderer internals...\n')

# Camera setup
H, W = 128, 128
fov_deg = 60.0
cam = Camera(H, W, fov_deg=fov_deg, device=device)
print(f'Camera:')
print(f'  fx={cam.fx:.2f}, fy={cam.fy:.2f}')
print(f'  cx={cam.cx:.2f}, cy={cam.cy:.2f}')

# View matrix
M = cam.view_matrix_orthographic(axis=2)
print(f'\nView matrix:')
print(M)

# Transform points
ones = torch.ones(means.shape[0], 1, device=device)
pts_h = torch.cat([means, ones], dim=1)  # [N, 4]
pc = (M @ pts_h.T).T  # [N, 4]
xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]

print(f'\nTransformed points (camera space):')
print(f'  xc: [{xc.min():.3f}, {xc.max():.3f}]')
print(f'  yc: [{yc.min():.3f}, {yc.max():.3f}]')
print(f'  zc: [{zc.min():.3f}, {zc.max():.3f}]')
print(f'  valid (zc > 0.01): {(zc > 0.01).sum()}/{len(zc)}')

# Projection
inv_z = 1.0 / zc.clamp(min=0.01)
print(f'\ninv_z: [{inv_z.min():.3f}, {inv_z.max():.3f}]')
print(f'  Has inf: {torch.isinf(inv_z).any()}')

u0 = xc * inv_z * cam.fx + cam.cx
v0 = yc * inv_z * cam.fy + cam.cy

print(f'\nProjected 2D centers:')
print(f'  u0: [{u0.min():.3f}, {u0.max():.3f}]')
print(f'  v0: [{v0.min():.3f}, {v0.max():.3f}]')
print(f'  Has inf: {torch.isinf(u0).any()} / {torch.isinf(v0).any()}')

# Check covariance projection
s00 = cov3d[:, 0]; s01 = cov3d[:, 1]; s02 = cov3d[:, 2]
s11 = cov3d[:, 3]; s12 = cov3d[:, 4]; s22 = cov3d[:, 5]

inv_z2 = inv_z ** 2
J00 = cam.fx * inv_z
J02 = -cam.fx * xc * inv_z2
J11 = cam.fy * inv_z
J12 = -cam.fy * yc * inv_z2

print(f'\nJacobian terms:')
print(f'  J00: [{J00.min():.3f}, {J00.max():.3f}]')
print(f'  J02: [{J02.min():.3f}, {J02.max():.3f}]')
print(f'  J11: [{J11.min():.3f}, {J11.max():.3f}]')
print(f'  J12: [{J12.min():.3f}, {J12.max():.3f}]')

a = J00**2 * s00 + 2*J00*J02 * s02 + J02**2 * s22 + 0.3
b = J00*J11 * s01 + J00*J12 * s02 + J02*J11 * s12 + J02*J12 * s22
c = J11**2 * s11 + 2*J11*J12 * s12 + J12**2 * s22 + 0.3

print(f'\n2D covariance elements:')
print(f'  a: [{a.min():.3f}, {a.max():.3f}] inf={torch.isinf(a).any()}')
print(f'  b: [{b.min():.3f}, {b.max():.3f}] inf={torch.isinf(b).any()}')
print(f'  c: [{c.min():.3f}, {c.max():.3f}] inf={torch.isinf(c).any()}')

det = (a * c - b**2).clamp(min=1e-10)
print(f'\nDeterminant:')
print(f'  det: [{det.min():.3f}, {det.max():.3f}] inf={torch.isinf(det).any()}')

inv_det = 1.0 / det
print(f'  inv_det: [{inv_det.min():.3f}, {inv_det.max():.3f}] inf={torch.isinf(inv_det).any()}')
