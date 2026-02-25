# NeuroSGM — 3D Gaussian Splatting for Dense Neurite Volumes

**NCCA / NeuroSGM Research Pipeline**  
*Armin — Marie Skłodowska-Curie Actions Fellow, Bournemouth University*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT                                                              │
│   ├─  neurite.tif   [D × H × W]  fluorescence microscopy volume    │
│   └─  neurite.swc   [N nodes]    morphology coordinates + radii    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   data_io.py        │
          │  • TIF normalise    │
          │  • SWC parse        │
          │  • Voxel seed extr. │
          └──────────┬──────────┘
                     │ [M SWC nodes] + [K voxel seeds]
          ┌──────────▼──────────┐
          │  gaussian_model.py  │
          │  NeuriteGaussians   │
          │  • means  [N,3]     │
          │  • quats  [N,4]     │
          │  • scales [N,3]     │
          │  • opacity[N,1]     │
          │  • features[N,C]    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐        ┌───────────────────────┐
          │   renderer.py       │        │  mip_splat_kernel.cu  │
          │   MIPSplatFunction  │──CUDA─▶│  • build_cov3d        │
          │   (autograd)        │◀──────│  • forward kernel     │
          │                     │        │  • backward kernel    │
          │  Σ = R·S·Sᵀ·Rᵀ      │        │  • grad_magnitude     │
          │  2D proj + EWA      │        └───────────────────────┘
          │  soft-MIP splat     │
          └──────────┬──────────┘
                     │ rendered [H,W,C]
          ┌──────────▼──────────┐
          │   losses.py         │
          │  NeuriteReconLoss   │
          │  (see below)        │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Adaptive Density  │
          │   Control (Adam)    │
          │  • clone (small+∇)  │
          │  • split (large+∇)  │
          │  • prune (α < ε)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   visualise.py      │
          │  • MIP trio renders  │
          │  • PLY export        │
          │  • Orbit animation   │
          └─────────────────────┘
```

---

## Loss Function Design

The total loss is a weighted sum of seven complementary terms:

```
L = w_photo · L_photo
  + w_mip(t)· L_mip
  + w_op    · L_opacity
  + w_scale · L_scale
  + w_depth · L_depth
  + w_feat  · L_feat
  + w_swc   · L_swc
```

### 1. Photometric Loss  `L_photo`  (primary, w=1.0)
```
L_photo = (1 - λ_ssim) · L1(I_render, I_gt)
        + λ_ssim        · (1 - SSIM(I_render, I_gt))
```
- L1 handles gross intensity matching
- D-SSIM (λ=0.2) enforces structural similarity, critical for thin neurite processes
- Applied per-channel and averaged

### 2. Volume MIP Consistency  `L_mip`  (annealed, w: 0.5 → 0)
```
L_mip = MSE(I_render, MaxProj_z(Volume))
w_mip(t) = 0.5 · max(0, 1 - t/3000)
```
- Forces early-stage Gaussians to match the actual fluorescence signal shape
- Annealed to zero so later iterations can refine fine structure freely
- Computed against the true max-intensity projection of the .tif stack

### 3. Opacity Sparsity  `L_opacity`  (w=0.05)
```
L_opacity = -E[α·log(α) + (1-α)·log(1-α)]    ← entropy
           + ReLU(mean(α) - α_target)           ← mean penalty
```
- Entropy regularisation pushes each Gaussian to be either on or off
- Suppresses the "floaters" that are common with dense initialisation from voxels
- Mean penalty (α_target=0.1) prevents overall scene from becoming opaque

### 4. Scale Regularisation  `L_scale`  (w=0.01)
```
L_scale = ||ReLU(s_min - log_s)||² + ||ReLU(log_s - s_max)||²
s_min = -8,  s_max = -2  (in normalised voxel space)
```
- Soft barrier prevents Gaussians collapsing to δ-functions (numerical instability)
- Upper bound prevents a few Gaussians growing to fill the entire volume

### 5. Depth Smoothness  `L_depth`  (w=0.005)
```
L_depth = TV(depth_map) = ||∂D/∂x|| + ||∂D/∂y||
```
- Penalises jagged depth gradients across the rendered MIP
- Encourages consistent depth ordering along neurite branches
- Helps prevent z-fighting between nearby Gaussians

### 6. Feature Smoothness  `L_feat`  (w=0.001, every 10 iters)
```
L_feat = (1/N) Σ_i  (1/k) Σ_{j∈NN_k(i)} ||f_i - f_j||₁
```
- Applied on a random subsample (4000 Gaussians) for tractability
- Encourages intensity continuity along neurite processes
- k=4 nearest neighbours in 3D position space

### 7. SWC Proximity  `L_swc`  (w=0.01)
```
L_swc = mean_i( min_j ||μ_i - p_j||₂ )
```
- Soft anchor: pulls each Gaussian toward the nearest known SWC morphology node
- Prevents Gaussians drifting into empty space during early optimisation
- Weakened (λ=0.01) to allow Gaussians to represent fluorescence signal not in SWC

---

## Quick Start

### 1. Build CUDA extension
```bash
cd neuro3dgs
pip install -r requirements.txt
pip install -e .          # builds mip_splat_kernel.cu
```

### 2. Train
```bash
python train.py \
    --tif  data/neurite.tif \
    --swc  data/neurite.swc \
    --out  outputs/run_01  \
    --H 512 --W 512        \
    --iterations 30000     \
    --voxel-size 0.2 0.1 0.1
```

### 3. Visualise
```bash
python visualise.py \
    --ckpt outputs/run_01/gaussians_030000.pt \
    --mode all
```

---

## File Structure

```
neuro3dgs/
├── cuda/
│   └── mip_splat_kernel.cu     CUDA kernels (forward + backward + cov3d builder)
├── data_io.py                  TIF + SWC loaders, voxel seed extraction
├── gaussian_model.py           NeuriteGaussians: parameters + adaptive density ctrl
├── renderer.py                 Differentiable MIP splatting (CUDA or PyTorch fallback)
├── losses.py                   NeuriteReconLoss (7 terms)
├── train.py                    End-to-end training pipeline with CLI
├── visualise.py                MIP renders, PLY export, orbit animation
├── setup.py                    CUDA extension build
└── requirements.txt
```

---

## CUDA Kernels Summary

| Kernel | Purpose | Threads |
|--------|---------|---------|
| `build_cov3d_kernel` | Quat+scale → 3×3 covariance matrix (upper tri) | 1 thread / Gaussian |
| `mip_splat_forward_kernel` | Project all Gaussians onto each pixel, compute soft-MIP | 1 thread / pixel |
| `mip_splat_backward_kernel` | Analytic gradients w.r.t. μ, Σ, α, f | 1 thread / Gaussian |
| `compute_grad_magnitude_kernel` | Per-Gaussian view-space ∇ magnitude for ADC | 1 thread / Gaussian |

---

## Expected Performance

| Configuration | Gaussians | Resolution | Speed |
|--------------|-----------|------------|-------|
| RTX 3090 + CUDA ext | 200k | 512×512 | ~15 it/s |
| RTX 3090 + PyTorch fallback | 10k | 256×256 | ~2 it/s |
| A100 + CUDA ext | 500k | 1024×1024 | ~8 it/s |
