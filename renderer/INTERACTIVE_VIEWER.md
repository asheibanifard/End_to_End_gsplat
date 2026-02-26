# Interactive MIP Splatting Viewer

Real-time 3D interactive viewer for MIP splatting models using Viser and Nerfview.

## Installation

First, install the required dependencies:

```bash
pip install viser nerfview
```

## Usage

### Command Line (Recommended)

```bash
cd /workspace/hisnegs/src/renderer
python interactive_mip_viewer.py --ckpt <path_to_checkpoint.pt> --port 8080
```

**Arguments:**
- `--ckpt`: Path to your MIP splatting checkpoint (.pt file) [required]
- `--port`: Port for the web viewer (default: 8080)
- `--resolution`: Render resolution (default: 512)
- `--beta`: MIP softmax temperature (default: 50.0, higher = sharper)
- `--fov`: Field of view in degrees (default: 50.0)

**Example:**
```bash
python interactive_mip_viewer.py \
    --ckpt ../../neurogs_v7/splat_clean_best.pt \
    --port 8080 \
    --resolution 512 \
    --beta 50.0
```

### From Jupyter Notebook

See the code cells in `splat_experiments.ipynb` for launching from within a notebook.

## How It Works

1. **Loads YourCheckpoint**: Loads means, covariances, and intensities from your trained MIP splatting model
2. **Renders with CUDA Kernel**: Uses your custom MIP projection CUDA kernel for fast rendering
3. **Interactive Navigation**: 
   - **Click and drag** to orbit around the object
   - **Scroll** to zoom in/out
   - **Right-click and drag** to pan
4. **GUI Controls**: Adjust beta (soft-MIP temperature) in real-time

## Access the Viewer

Once running, open your browser to:
```
http://localhost:8080
```

The viewer will display:
- Real-time MIP projection rendering
- Number of Gaussians
- Interactive camera controls
- Adjustable soft-MIP temperature (beta)

## Controls

- **Left Mouse Button + Drag**: Rotate camera (orbit)
- **Right Mouse Button + Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Beta Slider**: Adjust soft-MIP sharpness

## Stopping the Viewer

Press `Ctrl+C` in the terminal where the viewer is running.

## Technical Details

- **Rendering Backend**: Custom CUDA MIP projection kernel
- **Viewer Framework**: Viser + Nerfview (used by gsplat, nerfstudio)
- **Camera Model**: Pinhole with configurable FOV
- **Aspect Correction**: Automatically applied based on volume dimensions

## Troubleshooting

**Port already in use:**
```bash
python interactive_mip_viewer.py --ckpt <ckpt> --port 8081
```

**CUDA out of memory:**
- Reduce `--resolution` (try 256 or 128)
- Close other GPU processes

**Viewer connection refused:**
- Check firewall settings
- Try accessing from `http://127.0.0.1:<port>` instead of `localhost`

**ImportError for viser/nerfview:**
```bash
pip install viser nerfview
```
