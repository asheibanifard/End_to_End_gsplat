#!/usr/bin/env python3
"""Test SSIM computation"""
import torch
import sys
sys.path.insert(0, '.')
from train_end_to_end import ssim_loss

device = 'cuda'

# Test 1: Identical images should have SSIM loss ~ 0
print("Test 1: Identical images")
img1 = torch.rand(128, 128, 1, device=device)
img2 = img1.clone()
loss = ssim_loss(img1, img2)
print(f"  SSIM loss: {loss.item():.6f} (expect ~0)")
print(f"  SSIM similarity: {1 - loss.item():.6f} (expect ~1)")

# Test 2: Slightly different images
print("\nTest 2: Slightly different images")
img1 = torch.rand(128, 128, 1, device=device)
img2 = img1 + torch.randn_like(img1) * 0.01
loss = ssim_loss(img1, img2)
print(f"  SSIM loss: {loss.item():.6f}")
print(f"  SSIM similarity: {1 - loss.item():.6f}")

# Test 3: Very different images
print("\nTest 3: Very different images")
img1 = torch.rand(128, 128, 1, device=device)
img2 = torch.rand(128, 128, 1, device=device)
loss = ssim_loss(img1, img2)
print(f"  SSIM loss: {loss.item():.6f}")
print(f"  SSIM similarity: {1 - loss.item():.6f}")

# Test 4: Check if normalization is causing issues
print("\nTest 4: With different scales but same structure")
img1 = torch.rand(128, 128, 1, device=device) * 0.1
img2 = img1 * 10.0  # Same structure, different scale
loss = ssim_loss(img1, img2)
print(f"  SSIM loss: {loss.item():.6f}")
print(f"  SSIM similarity: {1 - loss.item():.6f} (should be high due to same structure)")
