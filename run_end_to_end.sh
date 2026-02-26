#!/bin/bash

# End-to-End Training Script
# NeuRoGS v7 → MIP Renderer with MSE + SSIM loss

# Activate conda environment
source /opt/miniforge3/bin/activate neurogs

# Run training with pretrained checkpoint
python train_end_to_end.py \
    --checkpoint ../neurogs_v7/gmf_refined_best.pt \
    --iterations 2000 \
    --lr 1e-3 \
    --lambda_mse 1.0 \
    --lambda_ssim 0.1 \
    --num_views 3 \
    --resolution 256 256 \
    --device cuda

echo ""
echo "Training complete!"
echo "Output: neurogs_renderer_optimized.pt"
