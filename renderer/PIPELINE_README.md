# End-to-End MIP Splatting Pipeline

Complete pipeline from raw volume data to trained 3D Gaussian representation and interactive visualization.

## Pipeline Overview

```
Volume Data (TIFF)
    ↓
Random Gaussian Initialization
    ↓
Ground Truth MIP Generation
    ↓
Training (Optimize Gaussians)
    ↓
Interactive 3D Viewer
```

## Quick Start

### Basic Usage

Train on a volume file with default settings:

```bash
cd /workspace/hisnegs/src/renderer

python train_and_view_pipeline.py \
    --volume ../../dataset/10-2900-control-cell-05_cropped_corrected.tif \
    --num_gaussians 10000 \
    --steps 5000 \
    --num_views 50 \
    --launch_viewer
```

### Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--volume` | Required | Path to input TIFF volume |
| `--num_gaussians` | 10000 | Number of 3D Gaussians to fit |
| `--steps` | 5000 | Training iterations |
| `--num_views` | 50 | GT projection views for training |
| `--batch_size` | 4 | Views per training step |
| `--lr` | 0.001 | Learning rate |
| `--beta` | 50.0 | MIP soft-max temperature |
| `--output_dir` | ./checkpoints/pipeline | Checkpoint directory |
| `--save_every` | 500 | Checkpoint frequency |
| `--launch_viewer` | False | Auto-launch viewer after training |
| `--viewer_port` | 8090 | Viewer port |

### Examples

**Fast test (5 minutes):**
```bash
python train_and_view_pipeline.py \
    --volume dataset/volume.tif \
    --num_gaussians 5000 \
    --steps 1000 \
    --num_views 20 \
    --launch_viewer
```

**High quality (1-2 hours):**
```bash
python train_and_view_pipeline.py \
    --volume dataset/volume.tif \
    --num_gaussians 20000 \
    --steps 10000 \
    --num_views 100 \
    --batch_size 8 \
    --lr 0.002 \
    --output_dir ./trained_models/high_quality
```

**View trained model later:**
```bash
python interactive_mip_viewer.py \
    --ckpt checkpoints/pipeline/final_model.pt \
    --port 8090 \
    --beta 50.0
```

## Pipeline Stages

### 1. Volume Loading
- Loads TIFF file
- Normalizes to [0, 1]
- Computes aspect correction scales

### 2. Gaussian Initialization
- Uniform random positions in [-1, 1]³
- Isotropic scales (default: 0.05)
- Identity rotations
- Moderate intensities (~0.1)

### 3. Ground Truth Generation
- Ray marching through volume
- Multi-view camera poses (orbiting)
- Soft-MIP projection
- Aspect-corrected sampling

### 4. Training
- Loss: MSE + Weighted MSE + SSIM
- Optimizer: Adam with cosine annealing
- Gradient clipping (max_norm=1.0)
- Checkpointing every N steps

### 5. Visualization
- Interactive 3D viewer (viser + nerfview)
- Real-time rendering with CUDA
- Beta slider for MIP sharpness
- Public URL via cloudflared tunnel

## Training Tips

### Number of Gaussians
- **Small volumes (<100³)**: 5K-10K Gaussians
- **Medium volumes (256³)**: 10K-20K Gaussians
- **Large volumes (512³+)**: 20K-50K Gaussians

### Training Steps
- **Initial fit**: 2K-5K steps
- **Good quality**: 5K-10K steps
- **High quality**: 10K-20K steps

### Convergence Monitoring
Watch for:
- Loss decreasing steadily
- Visible Gaussians staying high (>80%)
- SSIM loss approaching 0.9+

### Troubleshooting

**Loss not decreasing:**
- Increase learning rate (try 0.002-0.005)
- Reduce beta (try 20-30 for softer MIP)
- Check if Gaussians are visible

**Out of memory:**
- Reduce `--num_gaussians`
- Reduce `--batch_size`
- Reduce camera resolution (edit Camera.from_fov width/height)

**Training too slow:**
- Ensure CUDA is available
- Reduce `--num_views`
- Increase `--batch_size` if memory allows

## Output Files

After training, you'll find:

```
checkpoints/pipeline/
├── checkpoint_step0.pt      # Initial state
├── checkpoint_step500.pt    # Intermediate checkpoints
├── checkpoint_step1000.pt
├── ...
├── checkpoint_step5000.pt   # Final step
└── final_model.pt          # Best model (use for viewer)
```

Each checkpoint contains:
- `means`: Gaussian positions (K, 3)
- `log_scales`: Log scales (K, 3)
- `quaternions`: Rotations (K, 4)
- `log_intensities`: Intensities (K,)
- `optimizer`: Optimizer state
- `loss`: Training loss
- `vol_shape`: Volume dimensions

## Advanced Usage

### Custom Loss Weights

Edit `train_step()` in the script:
```python
loss_weights={'mse': 1.0, 'wmse': 0.5, 'ssim': 0.1}
```

### Different Initialization

Edit `initialize_gaussians_random()`:
- Change bounds for larger/smaller region
- Use different init_scale for Gaussian sizes
- Initialize from point cloud (add your code)

### Custom Camera Poses

Edit `generate_dataset()`:
- Use different elevation/azimuth ranges
- Change orbit radius
- Add custom pose generation

## Comparison with Pretrained Model

**Pipeline (Random Init):**
- ✅ Full control over initialization
- ✅ Train from scratch on any volume
- ✅ Understand complete pipeline
- ⏱️ Requires training time (5min - 2hrs)

**Pretrained Model (interactive_mip_viewer.py):**
- ✅ Instant visualization
- ✅ Already optimized
- ✅ No training needed
- ❌ Fixed to one volume/checkpoint

## Next Steps

After training:
1. **Analyze results**: Check loss convergence, visual quality
2. **Fine-tune**: Adjust hyperparameters and retrain
3. **Share**: Use cloudflared to create public viewer URL
4. **Export**: Save renderings for papers/presentations
5. **Compare**: Train with different initializations/hyperparameters
