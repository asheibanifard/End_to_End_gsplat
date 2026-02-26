import torch
import sys
sys.path.insert(0, '/workspace/end_to_end')
import gaussian_field_cuda

# Create simple test case
B, N = 2, 1
device = 'cuda'

points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
means = torch.tensor([[0.0, 0.0, 0.0]], device=device)

# Create simple identity-like Cholesky factor
L = torch.zeros(1, 3, 3, device=device)
L[0, 0, 0] = 1.0
L[0, 1, 1] = 1.0
L[0, 2, 2] = 1.0

print("Points shape:", points.shape)
print("Points:\n", points)
print("\nMeans shape:", means.shape)
print("Means:\n", means)
print("\nL shape:", L.shape)
print("L:\n", L)
print("\nL contiguous:", L.is_contiguous())
print("L strides:", L.stride())
print("L data (flat):", L.view(-1))

# Call CUDA kernel
try:
    mahal_cuda = gaussian_field_cuda.mahalanobis_distance_forward(
        points.contiguous(),
        means.contiguous(),
        L.contiguous()
    )
    print("\nCUDA result:")
    print(mahal_cuda)
except Exception as e:
    print("\nCUDA Error:", e)

# PyTorch reference
diff = points - means  # [2, 3]
print("\nDiff (points - means):")
print(diff)

# Expected mahalanobis: ||diff||^2 for identity covariance
mahal_expected = (diff ** 2).sum(dim=1)
print("\nExpected mahalanobis (||diff||^2):")
print(mahal_expected)

# Manual calculation
# For point [1, 2, 3] and mean [0, 0, 0] with I covariance:
# diff = [1, 2, 3]
# L @ v = diff -> v = diff (since L = I)
# mahal = v^T v = 1 + 4 + 9 = 14
print("\nManual calculation for first point: 1^2 + 2^2 + 3^2 =", 1+4+9)
print("Manual calculation for second point: 4^2 + 5^2 + 6^2 =", 16+25+36)
