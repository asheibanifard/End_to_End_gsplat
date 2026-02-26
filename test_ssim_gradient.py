#!/usr/bin/env python3
"""Test if SSIM gradient points in the correct direction"""
import torch
from train_end_to_end import ssim_loss

device = 'cuda'

# Create target
target = torch.rand(128, 128, 1, device=device) * 0.5 +0.25  # [0.25, 0.75]

# Create prediction DIFFERENT from target
pred = torch.rand(128, 128, 1, device=device) * 0.5  # [0, 0.5]
pred.requires_grad = True

# Compute initial loss
loss_initial = ssim_loss(pred, target)
print(f"Initial SSIM loss: {loss_initial.item():.6f}")
print(f"Initial SSIM similarity: {1 - loss_initial.item():.6f}")

# Compute gradient
loss_initial.backward()
grad = pred.grad.clone()

# Take a small step in the NEGATIVE gradient direction (gradient descent)
step_size = 0.001
pred_improved = (pred - step_size * grad).detach()
pred_improved.requires_grad = False

# Compute new loss
loss_after = ssim_loss(pred_improved, target)
print(f"\nAfter gradient descent step:")
print(f"SSIM loss: {loss_after.item():.6f}")
print(f"SSIM similarity: {1 - loss_after.item():.6f}")
print(f"Loss change: {loss_after.item() - loss_initial.item():.6f} (should be negative)")

if loss_after < loss_initial:
    print("\n✓ Gradient is correct - loss decreased")
else:
    print("\n✗ Gradient is WRONG - loss increased!")
