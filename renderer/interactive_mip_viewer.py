#!/usr/bin/env python3
"""
Interactive MIP Splatting Viewer using Viser

Real-time camera rotation around the object using the custom
MIP projection CUDA kernel. Navigate by dragging in the browser.
"""
import argparse
import math
import sys
import os
from typing import Tuple
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import custom renderer
from renderer.rendering import (
    Camera, 
    GaussianParameters,
    render_mip_projection,
    _orbit_pose,
    compute_aspect_scales
)

try:
    import viser
    import nerfview
except ImportError:
    print("Error: viser and nerfview are required for the interactive viewer.")
    print("Install with: pip install viser nerfview")
    sys.exit(1)


def load_mip_checkpoint(ckpt_path: str, device: torch.device) -> GaussianParameters:
    """Load MIP splatting checkpoint and build Gaussians."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    means = ckpt["means"].to(device)
    log_scales = ckpt["log_scales"].to(device)
    quaternions = ckpt["quaternions"].to(device)
    
    # Handle different naming conventions
    if "log_intensities" in ckpt:
        log_intensities = ckpt["log_intensities"].to(device)
    elif "log_amplitudes" in ckpt:
        log_intensities = ckpt["log_amplitudes"].to(device)
    else:
        raise KeyError("Checkpoint must contain 'log_intensities' or 'log_amplitudes'")
    
    K = means.shape[0]
    step = ckpt.get("step", "unknown")
    print(f"  Loaded {K} Gaussians (step {step})")
    
    # Build covariances from scales and rotations
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = torch.nn.functional.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Rotation matrix from quaternion
    R = torch.zeros(K, 3, 3, device=device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    S2 = torch.diag_embed(scales ** 2)
    covariances = R @ S2 @ R.transpose(-2, -1)
    intensities = torch.sigmoid(log_intensities)
    
    # Apply aspect correction to match physical volume dimensions
    # Volume: Z=100, Y=647, X=813 (Z is ~8x thinner than X)
    means_range = (means.min().item(), means.max().item())
    print(f"  Means range before correction: [{means_range[0]:.3f}, {means_range[1]:.3f}]")
    
    aspect_scales = compute_aspect_scales((100, 647, 813)).to(device)
    print(f"  Aspect scales (X,Y,Z): {aspect_scales.tolist()}")
    
    # Apply to both means and covariances
    means_corrected = means * aspect_scales
    S_aspect = torch.diag(aspect_scales)
    covariances_corrected = S_aspect.unsqueeze(0) @ covariances @ S_aspect.unsqueeze(0).transpose(-2, -1)
    
    means_range_after = (means_corrected.min().item(), means_corrected.max().item())
    print(f"  Means range after correction: [{means_range_after[0]:.3f}, {means_range_after[1]:.3f}]")
    
    return GaussianParameters(
        means=means_corrected,
        covariances=covariances_corrected,
        intensities=intensities
    ), K


def main():
    parser = argparse.ArgumentParser(description="Interactive MIP Splatting Viewer")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (.pt file)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server (default: 8080)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=813,
        help="Render width in pixels (default: 813, original training resolution)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=647,
        help="Render height in pixels (default: 647, original training resolution)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=50.0,
        help="MIP softmax temperature (default: 50.0)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=50.0,
        help="Field of view in degrees (default: 50.0)"
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    gaussians, K = load_mip_checkpoint(args.ckpt, device)
    
    # Create rendering function
    def render_fn(camera_state: nerfview.CameraState, img_wh):
        """Render function called by the viewer for each frame."""
        # Handle both old and new API
        if isinstance(img_wh, tuple):
            W, H = img_wh
        else:
            # New API passes render_tab_state as second argument
            W, H = args.width, args.height
        
        # Create camera
        camera = Camera.from_fov(
            fov_x_deg=args.fov,
            width=W,
            height=H,
            near=0.01,
            far=1000.0  # Increased for larger scenes
        )
        
        # Extract camera pose from viewer state
        # nerfview provides c2w (camera-to-world) matrix
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        
        # Convert to our convention: R and T
        # Our renderer expects: world points transformed to camera frame
        # c2w[:3, :3] is camera axes in world frame (columns are camera x, y, z)
        # c2w[:3, 3] is camera position in world frame
        
        # World-to-camera rotation: inverse of camera-to-world
        R = c2w[:3, :3].T  # Transpose for inverse rotation
        
        # Camera position in camera frame: -R @ camera_world_position
        T = -R @ c2w[:3, 3]
        
        # Render using MIP projection
        with torch.no_grad():
            img, n_vis = render_mip_projection(
                gaussians,
                camera,
                R,
                T,
                beta=args.beta
            )
        
        # Debug: Print visibility occasionally
        if hasattr(render_fn, '_frame_count'):
            render_fn._frame_count += 1
            if render_fn._frame_count % 30 == 0:  # Every 30 frames
                print(f"[Frame {render_fn._frame_count}] Visible Gaussians: {n_vis}/{K}, Mean intensity: {img.mean():.4f}")
        else:
            render_fn._frame_count = 1
            print(f"[First render] Visible Gaussians: {n_vis}/{K}")
        
        # Convert to RGB (expand grayscale to 3 channels)
        img_rgb = img.unsqueeze(-1).expand(-1, -1, 3)
        
        # Return as numpy array [H, W, 3] in range [0, 1]
        return img_rgb.cpu().numpy()
    
    # Create viser server (binds to all interfaces by default)
    print(f"\nStarting interactive viewer on port {args.port}...")
    server = viser.ViserServer(port=args.port, verbose=False, host="0.0.0.0")
    
    # Add some GUI controls
    with server.gui.add_folder("MIP Splatting"):
        beta_slider = server.gui.add_slider(
            "Beta (Temperature)",
            min=1.0,
            max=100.0,
            step=1.0,
            initial_value=args.beta,
            hint="Soft-MIP temperature (higher = sharper)",
        )
        
        gaussians_count = server.gui.add_number(
            "Total Gaussians",
            initial_value=K,
            disabled=True,
            hint="Total number of Gaussians in the scene",
        )
    
    @beta_slider.on_update
    def _(_) -> None:
        args.beta = beta_slider.value
        viewer.rerender(None)
    
    # Compute object center and bounding box for camera setup
    import numpy as np
    means_np = gaussians.means.cpu().numpy()
    object_center = means_np.mean(axis=0)
    bbox_min = means_np.min(axis=0)
    bbox_max = means_np.max(axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    print(f"\nObject stats:")
    print(f"  Center: [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}]")
    print(f"  Bbox size: {bbox_size:.3f}")
    
    # Set camera distance to ~2x the bounding box diagonal
    camera_distance = bbox_size * 2.0
    initial_position = object_center + np.array([0.0, 0.0, camera_distance])
    
    print(f"  Initial camera distance: {camera_distance:.3f}")
    
    # Create viewer with camera looking at object center
    viewer = nerfview.Viewer(
        server=server,
        render_fn=render_fn,
        mode="rendering",
    )
    
    # Get server IP for sharing
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        server_ip = s.getsockname()[0]
        s.close()
    except:
        server_ip = "unknown"
    
    print("\n" + "="*60)
    print(f"Viewer running:")
    print(f"  Local:   http://localhost:{args.port}")
    if server_ip != "unknown":
        print(f"  Network: http://{server_ip}:{args.port}")
    print(f"\nGaussians: {K}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Beta: {args.beta}")
    print("="*60)
    print("\nNavigate by clicking and dragging in the browser.")
    print("Share the Network URL with others to view remotely.")
    print("Press Ctrl+C to exit.\n")
    
    # Keep server running
    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    main()
