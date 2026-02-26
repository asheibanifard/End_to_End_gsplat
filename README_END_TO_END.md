# End-to-End Training: NeuRoGS v7 → MIP Renderer

This module implements an end-to-end differentiable pipeline that optimizes 3D Gaussian parameters based on multi-view maximum intensity projection (MIP) rendering supervision.

## Architecture

```
┌─────────────────┐
│ 3D Gaussians    │  Parameters:
│ (NeuRoGS v7)    │  • means: [K, 3]
│                 │  • log_scales: [K, 3]
│                 │  • quaternions: [K, 4]
│                 │  • log_amplitudes: [K]
└────────┬────────┘
         │
         │ Extract & Normalize
         ↓
┌─────────────────┐
│ Renderer Input  │  • means ∈ [0,1]³
│                 │  • cov3d (upper triangle)
│                 │  • intensity (emission)
└────────┬────────┘
         │
         │ Multi-view
         ↓
┌─────────────────┐
│ MIP Renderer    │  For each view:
│                 │  1. Transform to camera space
│                 │  2. Project 3D → 2D
│                 │  3. Splat Gaussians
│                 │  4. Soft-MIP blending
└────────┬────────┘
         │
         │ M views
         ↓
┌─────────────────┐
│ Loss Function   │  L = λ_MSE·MSE + λ_SSIM·(1-SSIM)
│                 │
│  MSE:  Pixel-wise intensity error
│  SSIM: Structural similarity
└────────┬────────┘
         │
         │ Backprop
         ↓
┌─────────────────┐
│ Optimizer       │  Adam with gradient clipping
│                 │  • means: lr
│                 │  • scales: lr * 0.1
│                 │  • quaternions: lr * 0.1
│                 │  • amplitudes: lr
└─────────────────┘
```

## Key Components

### 1. NeuRoGStoRenderer Adapter
Converts NeuRoGS v7 Gaussian parameters to renderer format:
- Normalizes positions to [0, 1]³ cube
- Constructs 3D covariance matrices from scales + quaternions
- Extracts emission intensities from log-amplitudes

### 2. MIP Renderer
Differentiable maximum intensity projection:
- Emission-based soft-MIP (no depth bias)
- Anisotropic 3D Gaussians
- PyTorch fallback when CUDA unavailable

### 3. Loss Functions

**MSE (Mean Squared Error)**
```
L_MSE = (1/N) Σ (I_rendered - I_target)²
```

**SSIM (Structural Similarity)**
```
SSIM = [(2μ_xμ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]
L_SSIM = 1 - SSIM
```

Total loss:
```
L = λ_MSE · L_MSE + λ_SSIM · L_SSIM
```

## Usage

### Basic Training

```bash
python train_end_to_end.py \
    --checkpoint ../neurogs_v7/gmf_refined_best.pt \
    --iterations 1000 \
    --lr 1e-3 \
    --lambda_mse 1.0 \
    --lambda_ssim 0.1 \
    --num_views 3 \
    --resolution 256 256 \
    --device cuda
```

### Using the Run Script

```bash
bash run_end_to_end.sh
```

### Initialize from Scratch

```bash
python train_end_to_end.py \
    --num_gaussians 5000 \
    --iterations 2000 \
    --device cuda
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | None | Path to NeuRoGS v7 checkpoint |
| `--num_gaussians` | 2000 | Number of Gaussians (if no checkpoint) |
| `--iterations` | 1000 | Training iterations |
| `--lr` | 1e-3 | Learning rate for means and amplitudes |
| `--lambda_mse` | 1.0 | Weight for MSE loss |
| `--lambda_ssim` | 0.1 | Weight for SSIM loss |
| `--num_views` | 3 | Number of projection views |
| `--resolution` | 256 256 | Image resolution [H W] |
| `--device` | cuda | Device (cuda/cpu) |

## Output

- **Checkpoints**: Saved every 500 iterations as `checkpoint_iter{N}.pt`
- **Final Model**: `neurogs_renderer_optimized.pt`

## Coordinate System

- **NeuRoGS v7**: Arbitrary scale, centered around 0
- **Renderer**: Normalized to [0, 1]³ cube
- **Adapter**: Automatically normalizes positions and scales covariances

## Numerical Stability

The pipeline includes several numerical stability measures:
1. **Covariance regularization**: `Σ_2D += 0.3·I` (low-pass filter)
2. **Soft-MIP beta**: Reduced to 0.01 to prevent overflow
3. **Gradient clipping**: Max norm of 1.0
4. **Parameter clamping**:
   - log_scales ∈ [-6, 0]
   - log_amplitudes ∈ [-10, 6]
   - means within AABB

## Example Results

Starting from a pretrained NeuRoGS v7 model with 12,145 Gaussians:
- Iteration 0: loss=0.045, mse=0.002, ssim=0.43
- Iteration 50: loss=0.038, mse=0.001, ssim=0.37
- Training speed: ~30 it/s on GPU (128×128 resolution)

## Requirements

```
torch >= 2.0
numpy
tqdm
pyyaml
```

## Notes

- The renderer uses PyTorch fallback (CPU-speed rendering) when CUDA extension is unavailable
- For production use, compile the CUDA extension for 10-100× speedup
- Multi-view supervision prevents mode collapse and ensures 3D consistency
- SSIM loss preserves structural details better than MSE alone
