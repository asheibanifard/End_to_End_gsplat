"""
Complete training example using CUDA-accelerated Gaussian fields.

This script demonstrates:
1. Loading/creating training data
2. Training with CUDA-accelerated model
3. Performance monitoring
4. Saving/loading trained models

Run with:
    python train_with_cuda.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.insert(0, '/workspace/end_to_end')

from gaussian_field_ops import CUDALearnableGaussianField, CUDA_AVAILABLE

# Check CUDA availability
if not CUDA_AVAILABLE:
    print("ERROR: CUDA extension not available.")
    print("Install with: python setup_gaussian_field.py install")
    sys.exit(1)

print("✅ CUDA extension loaded")
print("=" * 80)


def create_synthetic_data(num_voxels=5000, volume_size=10.0, device='cuda'):
    """Create synthetic training data with known Gaussian distribution."""
    
    print(f"Creating synthetic training data...")
    print(f"  Volume size: {volume_size}")
    print(f"  Num voxels: {num_voxels}")
    
    # Random voxel coordinates
    coords = torch.rand(num_voxels, 3, device=device) * volume_size
    
    # Ground truth: 3 Gaussians
    gt_means = torch.tensor([
        [2.0, 2.0, 2.0],
        [5.0, 5.0, 5.0],
        [8.0, 8.0, 8.0]
    ], device=device)
    
    gt_scales = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.8, 0.8, 0.8],
        [0.6, 0.6, 0.6]
    ], device=device)
    
    gt_weights = torch.tensor([1.0, 0.8, 0.6], device=device)
    
    # Generate target values
    values = torch.zeros(num_voxels, device=device)
    for i in range(3):
        diff = coords - gt_means[i]
        inv_var = 1.0 / (gt_scales[i] ** 2)
        mahal = (diff ** 2 * inv_var).sum(dim=-1)
        values += gt_weights[i] * torch.exp(-0.5 * mahal)
    
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  ✅ Data created\n")
    
    return coords, values


def train_cuda_model(
    model,
    train_coords,
    train_values,
    num_iterations=500,
    learning_rate=0.01,
    batch_size=512,
    log_every=50
):
    """Train CUDA-accelerated Gaussian field."""
    
    print(f"Training Configuration:")
    print(f"  Model: CUDA-accelerated LearnableGaussianField")
    print(f"  Num Gaussians: {model.num_gaussians}")
    print(f"  Num iterations: {num_iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print("=" * 80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    loss_history = []
    time_history = []
    
    M = train_coords.shape[0]
    
    # Training loop
    print(f"{'Iter':<6} {'Loss':<12} {'Time (ms)':<12} {'LR':<10}")
    print("-" * 80)
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Forward + backward pass
        optimizer.zero_grad()
        
        total_loss = 0.0
        for i in range(0, M, batch_size):
            batch_coords = train_coords[i:i+batch_size]
            batch_values = train_values[i:i+batch_size]
            
            predictions = model(batch_coords)
            batch_loss = F.mse_loss(predictions, batch_values)
            
            if i == 0:
                total_loss = batch_loss
            else:
                total_loss = total_loss + batch_loss * (len(batch_coords) / M)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Record metrics
        iter_time = (time.time() - iter_start) * 1000  # ms
        loss_history.append(total_loss.item())
        time_history.append(iter_time)
        
        # Logging
        if (iteration + 1) % log_every == 0 or iteration == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{iteration+1:<6d} {total_loss.item():<12.6f} {iter_time:<12.2f} {current_lr:<10.2e}")
    
    print("=" * 80)
    print(f"Training complete!")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    print(f"  Avg iteration time: {np.mean(time_history):.2f} ms")
    print(f"  Total training time: {sum(time_history)/1000:.1f} seconds")
    
    return loss_history, time_history


def evaluate_model(model, test_coords, test_values):
    """Evaluate model on test data."""
    
    print(f"\n📊 Evaluation:")
    print("=" * 80)
    
    with torch.no_grad():
        predictions = model(test_coords)
        mse = F.mse_loss(predictions, test_values)
        mae = (predictions - test_values).abs().mean()
        
        # Correlation
        pred_mean = predictions.mean()
        test_mean = test_values.mean()
        numerator = ((predictions - pred_mean) * (test_values - test_mean)).sum()
        denom = torch.sqrt(((predictions - pred_mean)**2).sum() * 
                          ((test_values - test_mean)**2).sum())
        correlation = numerator / (denom + 1e-8)
    
    print(f"  MSE:  {mse.item():.6f}")
    print(f"  MAE:  {mae.item():.6f}")
    print(f"  Correlation: {correlation.item():.4f}")
    print("=" * 80)
    
    return mse.item(), mae.item(), correlation.item()


def save_model(model, filepath="trained_gaussian_field.pt"):
    """Save trained model."""
    torch.save({
        'num_gaussians': model.num_gaussians,
        'volume_size': model.volume_size,
        'use_full_cov': model.use_full_cov,
        'state_dict': model.state_dict()
    }, filepath)
    print(f"\n✅ Model saved to {filepath}")


def load_model(filepath="trained_gaussian_field.pt", device='cuda'):
    """Load trained model."""
    checkpoint = torch.load(filepath)
    model = CUDALearnableGaussianField(
        num_gaussians=checkpoint['num_gaussians'],
        volume_size=checkpoint['volume_size'],
        use_full_cov=checkpoint['use_full_cov'],
        device=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    print(f"✅ Model loaded from {filepath}")
    return model


def main():
    """Main training script."""
    
    print("\n🚀 CUDA-Accelerated Gaussian Field Training")
    print("=" * 80)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gaussians = 20
    volume_size = 10.0
    num_train = 5000
    num_test = 1000
    num_iterations = 500
    
    print(f"Device: {device}")
    print(f"Num Gaussians: {num_gaussians}")
    print("\n")
    
    # Create data
    train_coords, train_values = create_synthetic_data(
        num_voxels=num_train,
        volume_size=volume_size,
        device=device
    )
    
    test_coords, test_values = create_synthetic_data(
        num_voxels=num_test,
        volume_size=volume_size,
        device=device
    )
    
    # Create model
    model = CUDALearnableGaussianField(
        num_gaussians=num_gaussians,
        volume_size=volume_size,
        use_full_cov=True,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Train
    loss_history, time_history = train_cuda_model(
        model=model,
        train_coords=train_coords,
        train_values=train_values,
        num_iterations=num_iterations,
        learning_rate=0.01,
        batch_size=512,
        log_every=100
    )
    
    # Evaluate
    evaluate_model(model, test_coords, test_values)
    
    # Save model
    save_model(model, "trained_cuda_gaussian_field.pt")
    
    print(f"\n✅ Training complete! Model saved and ready to use.")
    print(f"   Load with: model = load_model('trained_cuda_gaussian_field.pt')")


if __name__ == "__main__":
    main()
