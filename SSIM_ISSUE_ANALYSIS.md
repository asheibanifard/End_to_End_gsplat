# SSIM Decreasing Issue - Analysis and Solution

## Problem

During training, SSIM similarity decreases from ~0.90 to ~0.46, indicating the optimization is degrading structural quality.

## Root CauseAnalysis

### ✓ Verified Working:
1. **SSIM function is correct** - tested with identical/different images
2. **SSIM gradient is correct** - tested gradient descent decreases loss
3. **Rendering is deterministic** - same inputs produce same outputs
4. **No coordinate drift** - normalization bounds fixed at initialization

### ✗ Issue Identified:
The rendered images at **iteration 0** already differ from targets (SSIM loss = 0.1, not 0), even though they should be identical. This suggests there may be a subtle difference in the rendering path or gradient computation.

## Current Status

The end-to-end pipeline is **functionally complete**. The architecture works for:
- ✅ Loading NeuRoGS v7 checkpoints  
- ✅ Multi-view MIP rendering
- ✅ Differentiable gradient flow
- ✅ Parameter optimization with constraints
- ⚠️ SSIM degradation during optimization (under investigation)

## Improvements Made

1. **Fixed normalization bounds** - prevents coordinate drift during training
2. **Added SSIM warmup** - gradually increases SSIM weight
3. **Added MSE fadeout** - can reduce MSE influence over time  
4. **Better logging** - shows SSIM similarity (not just loss) and loss weights
5. **Disabled image normalization in SSIM** - was causing instability
6. **Set SSIM as primary loss** - default λ_MSE=0, λ_SSIM=1

## Recommended Usage

**Current best configuration:**

```bash
python train_end_to_end.py \
    --checkpoint ../neurogs_v7/gmf_refined_best.pt \
    --iterations 500 \
    --lambda_mse 0.1 \
    --lambda_ssim 0.5 \
    --resolution 192 192 \
    --num_views 3 \
    --device cuda
```

**Alternative (MSE-only for stability):**

```bash
python train_end_to_end.py \
    --checkpoint ../neurogs_v7/gmf_refined_best.pt \
    --iterations 500 \
    --lambda_mse 1.0 \
    --lambda_ssim 0.0 \
    --resolution 192 192 \
    --num_views 3 \
    --device cuda
```

## Key Takeaways

- **MSE + SSIM conflict**: Optimizing for pixel-wise error can harm structural similarity
- **Solution**: Reduce or disable MSE, emphasize SSIM
- **Warmup helps**: Gradual introduction of SSIM loss provides stability
- **Fixed normalization critical**: Prevents coordinate system drift

## Future Improvements

1. Investigate why initial renders differ from targets
2. Consider using pre-computed SSIM targets  
3. Try perceptual loss (VGG features) as alternative to SSIM  
4. Implement two-stage training: MSE → SSIM
5. Add validation metrics to monitor structural quality
