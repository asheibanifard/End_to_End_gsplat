#!/bin/bash
# End-to-End MIP Projection Training

# Activate conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate neurogs

# Example: Train with TIF volume and optional SWC
python train_mip_projection.py \
    --tif dataset/your_volume.tif \
    --swc dataset/your_morphology.swc \
    --out outputs/mip_projection_run \
    --num_gaussians 2000 \
    --init_scale 0.05 \
    --init_amplitude 0.1 \
    --H 512 --W 512 \
    --n_views 8 \
    --iterations 10000 \
    --lr_pos 1e-3 \
    --lr_rot 1e-3 \
    --lr_scale 5e-3 \
    --lr_amp 1e-2 \
    --lambda_scale 1e-4 \
    --log_interval 100 \
    --save_interval 1000
