# End-to-End MIP Projection Training Architecture

## Overview

This implements an end-to-end architecture that optimizes 3D Gaussians by comparing rendered MIP projections with ground truth projections from the input volume.

## Architecture Components

### 1. **3D Gaussian Mixture Field** (`neurogs_v7.py`)
- Represents the volume as: `V(x) = Σ_k a_k · exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))`
- Parameters per Gaussian:
  - `means`: [K, 3] positions
  - `quaternions`: [K, 4] rotations  
  - `log_scales`: [K, 3] anisotropic scales
  - `log_amplitudes`: [K] emission intensities

### 2. **MIP Splatting Renderer** (`renderer.py`)
- Differentiable maximum intensity projection renderer
- Projects 3D Gaussians to 2D using soft-max approximation
- Supports multiple camera views

### 3. **Training Loop** (`train_mip_projection.py`)

```
For each iteration:
  1. Render N MIP projections from different viewpoints
     - 3 orthogonal views (XY, XZ, YZ)
     - M orbit views (optional, for regularization)
  
  2. Compare rendered projections with GT projections
     - L1 photometric loss
     - SSIM structural similarity
     - Scale regularization
  
  3. Backpropagate gradients through:
     - Renderer (MIP splatting)
     - 3D Gaussian parameters
  
  4. Update Gaussians using Adam optimizer
```

## Key Differences from Original Approaches

| Aspect | `neurogs_v7.py` (Volume) | `train.py` (Surface) | **`train_mip_projection.py` (Hybrid)** |
|--------|-------------------------|---------------------|--------------------------------|
| Supervision | Dense 3D volume sampling | Sparse voxel seeds + morphology | **Multi-view MIP projections** |
| Loss | Charbonnier + gradients | Photometric + MIP consistency | **Projection-based (L1 + SSIM)** |
| Memory | O(N · K) volume points | O(N_seeds) surface points | **O(N_views · H · W)** |
| Speed | Slower (dense sampling) | Faster (sparse) | **Medium (projection-based)** |

## Advantages

1. **Memory Efficient**: No need to sample dense 3D points
2. **Differentiable**: End-to-end gradient flow from 2D projections to 3D Gaussians
3. **Multi-View Consistency**: Naturally enforces 3D consistency through multiple projections
4. **Fast Inference**: Direct projection rendering (no volume sampling needed)

## Usage

```bash
# Basic usage with TIF volume
python train_mip_projection.py \
    --tif data/volume.tif \
    --out outputs/run1 \
    --iterations 10000

# With SWC morphology initialization
python train_mip_projection.py \
    --tif data/volume.tif \
    --swc data/morphology.swc \
    --out outputs/run2 \
    --num_gaussians 2000 \
    --iterations 10000

# High-quality with more views
python train_mip_projection.py \
    --tif data/volume.tif \
    --swc data/morphology.swc \
    --out outputs/run3 \
    --num_gaussians 5000 \
    --n_views 12 \
    --H 1024 --W 1024 \
    --iterations 20000
```

## Parameters

### Model Parameters
- `--num_gaussians`: Number of 3D Gaussians (default: 2000)
- `--init_scale`: Initial Gaussian scale (default: 0.05)
- `--init_amplitude`: Initial emission intensity (default: 0.1)

### Rendering Parameters
- `--H`, `--W`: Render resolution (default: 512x512)
- `--fov`: Field of view in degrees (default: 60)
- `--n_views`: Number of training views (default: 8)
  - First 3 are always orthogonal (XY, XZ, YZ)
  - Additional views are orbit cameras

### Training Parameters
- `--iterations`: Training steps (default: 10000)
- `--lr_pos`: Learning rate for positions (default: 1e-3)
- `--lr_rot`: Learning rate for rotations (default: 1e-3)
- `--lr_scale`: Learning rate for scales (default: 5e-3)
- `--lr_amp`: Learning rate for amplitudes (default: 1e-2)

### Loss Weights
- `--lambda_scale`: Scale regularization weight (default: 1e-4)
  - Prevents Gaussians from growing too large

## Output Structure

```
outputs/run_name/
├── tb/                          # TensorBoard logs
├── gt_projections/              # Ground truth MIP projections
│   ├── gt_mip_xy.png
│   ├── gt_mip_xz.png
│   └── gt_mip_yz.png
├── render_xy_001000.png         # Rendered projections at iter 1000
├── render_xz_001000.png
├── render_yz_001000.png
└── field_001000.pt              # Model checkpoint
```

## Checkpoint Format

Checkpoints contain:
- `iteration`: Current training iteration
- `model_state_dict`: GaussianMixtureField parameters
- `optimizer_state_dict`: Optimizer state

Load checkpoint:
```python
checkpoint = torch.load("field_010000.pt")
field.load_state_dict(checkpoint['model_state_dict'])
```

## Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir outputs/run_name/tb --port 6006
```

Metrics logged:
- `loss/total`: Total weighted loss
- `loss/l1`: L1 photometric error
- `loss/ssim`: SSIM structural error
- `loss/scale_reg`: Scale regularization
- `gaussians/N`: Number of Gaussians

## Tips for Best Results

1. **Start with morphology**: If available, use SWC initialization (`--swc`)
2. **Tune number of Gaussians**: 
   - Small volumes: 1000-2000
   - Large volumes: 5000-10000
3. **Use multiple views**: At least 8 views for good 3D coverage
4. **Monitor loss curves**: L1 should decrease steadily; if plateau, try:
   - Increase learning rates
   - Add more views
   - Adjust scale regularization

## Future Extensions

- [ ] Adaptive densification (clone/split Gaussians during training)
- [ ] Depth supervision from multi-plane acquisitions
- [ ] Time-series support for dynamic volumes
- [ ] Export to neuroglancer/VAST format
