#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for transforming 3D Gaussians to camera space
// P_cam = R @ P_world + T
// Σ_cam = R @ Σ_world @ R^T
__global__ void transform_gaussians_kernel(
    const float* __restrict__ means_3d,      // [N, 3]
    const float* __restrict__ covs_3d,       // [N, 3, 3]
    const float* __restrict__ R,             // [3, 3]
    const float* __restrict__ T,             // [3]
    float* __restrict__ means_cam,           // [N, 3] output
    float* __restrict__ covs_cam,            // [N, 3, 3] output
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Transform mean: means_cam = R @ means_3d + T
    float mean_out[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mean_out[i] += R[i * 3 + j] * means_3d[idx * 3 + j];
        }
        mean_out[i] += T[i];
        means_cam[idx * 3 + i] = mean_out[i];
    }
    
    // Transform covariance: covs_cam = R @ covs_3d @ R^T
    // First: temp = R @ covs_3d
    float temp[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                temp[i * 3 + j] += R[i * 3 + k] * covs_3d[idx * 9 + k * 3 + j];
            }
        }
    }
    
    // Second: covs_cam = temp @ R^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += temp[i * 3 + k] * R[j * 3 + k];  // R^T so indices swapped
            }
            covs_cam[idx * 9 + i * 3 + j] = val;
        }
    }
}

// CUDA kernel for 2D projection with Jacobian
__global__ void project_to_2d_kernel(
    const float* __restrict__ means_cam,     // [N, 3]
    const float* __restrict__ covs_cam,      // [N, 3, 3]
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    float* __restrict__ means_2d,            // [N, 2] output
    float* __restrict__ covs_2d,             // [N, 2, 2] output
    float* __restrict__ depths,              // [N] output
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const float x = means_cam[idx * 3 + 0];
    const float y = means_cam[idx * 3 + 1];
    const float z = means_cam[idx * 3 + 2];
    
    // Store depth
    depths[idx] = z;
    
    // Avoid division by zero
    if (z < 1e-6f) {
        means_2d[idx * 2 + 0] = 0.0f;
        means_2d[idx * 2 + 1] = 0.0f;
        covs_2d[idx * 4 + 0] = 1e6f;
        covs_2d[idx * 4 + 1] = 0.0f;
        covs_2d[idx * 4 + 2] = 0.0f;
        covs_2d[idx * 4 + 3] = 1e6f;
        return;
    }
    
    const float z_inv = 1.0f / z;
    const float z_inv2 = z_inv * z_inv;
    
    // Project to 2D
    means_2d[idx * 2 + 0] = fx * x * z_inv + cx;
    means_2d[idx * 2 + 1] = fy * y * z_inv + cy;
    
    // Jacobian matrix J = [fx/z, 0, -fx*x/z^2]
    //                     [0, fy/z, -fy*y/z^2]
    const float J[6] = {
        fx * z_inv, 0.0f, -fx * x * z_inv2,
        0.0f, fy * z_inv, -fy * y * z_inv2
    };
    
    // Load 3D covariance
    float cov3d[9];
    for (int i = 0; i < 9; i++) {
        cov3d[i] = covs_cam[idx * 9 + i];
    }
    
    // Compute Σ_2D = J @ Σ_3D @ J^T
    // First: temp = J @ Σ_3D (2x3 @ 3x3 = 2x3)
    float temp[6] = {0.0f};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                temp[i * 3 + j] += J[i * 3 + k] * cov3d[k * 3 + j];
            }
        }
    }
    
    // Second: Σ_2D = temp @ J^T (2x3 @ 3x2 = 2x2)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += temp[i * 3 + k] * J[j * 3 + k];  // J^T
            }
            covs_2d[idx * 4 + i * 2 + j] = val;
        }
    }
    
    // Add small regularization for numerical stability
    covs_2d[idx * 4 + 0] += 1e-6f;  // Σ[0,0]
    covs_2d[idx * 4 + 3] += 1e-6f;  // Σ[1,1]
}

// CUDA kernel for rendering 2D Gaussians to pixels
__global__ void render_gaussians_2d_kernel(
    const float* __restrict__ pixel_coords,  // [H*W, 2]
    const float* __restrict__ means_2d,      // [N, 2]
    const float* __restrict__ covs_2d,       // [N, 2, 2]
    const float* __restrict__ weights,       // [N]
    float* __restrict__ output,              // [H*W] output
    const int num_pixels,
    const int num_gaussians
) {
    const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;
    
    const float px = pixel_coords[pixel_idx * 2 + 0];
    const float py = pixel_coords[pixel_idx * 2 + 1];
    
    float sum = 0.0f;
    
    // Accumulate contributions from all Gaussians
    for (int g = 0; g < num_gaussians; g++) {
        const float mx = means_2d[g * 2 + 0];
        const float my = means_2d[g * 2 + 1];
        
        const float dx = px - mx;
        const float dy = py - my;
        
        // Load 2D covariance
        const float cov00 = covs_2d[g * 4 + 0];
        const float cov01 = covs_2d[g * 4 + 1];
        const float cov11 = covs_2d[g * 4 + 3];
        
        // Compute determinant
        const float det = cov00 * cov11 - cov01 * cov01;
        
        if (det < 1e-6f) continue;  // Skip degenerate Gaussians
        
        // Inverse covariance
        const float inv_det = 1.0f / det;
        const float inv_cov00 = cov11 * inv_det;
        const float inv_cov01 = -cov01 * inv_det;
        const float inv_cov11 = cov00 * inv_det;
        
        // Mahalanobis distance: (dx dy) @ inv_cov @ (dx dy)^T
        const float mahal = dx * (inv_cov00 * dx + inv_cov01 * dy) + 
                           dy * (inv_cov01 * dx + inv_cov11 * dy);
        
        // Gaussian evaluation with cutoff for efficiency
        if (mahal < 16.0f) {  // 4 standard deviations
            sum += weights[g] * expf(-0.5f * mahal);
        }
    }
    
    output[pixel_idx] = sum;
}

// Host functions
std::vector<torch::Tensor> transform_gaussians_cuda(
    torch::Tensor means_3d,
    torch::Tensor covs_3d,
    torch::Tensor R,
    torch::Tensor T
) {
    const int N = means_3d.size(0);
    
    auto means_cam = torch::empty_like(means_3d);
    auto covs_cam = torch::empty_like(covs_3d);
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    transform_gaussians_kernel<<<blocks, threads>>>(
        means_3d.data_ptr<float>(),
        covs_3d.data_ptr<float>(),
        R.data_ptr<float>(),
        T.data_ptr<float>(),
        means_cam.data_ptr<float>(),
        covs_cam.data_ptr<float>(),
        N
    );
    
    return {means_cam, covs_cam};
}

std::vector<torch::Tensor> project_to_2d_cuda(
    torch::Tensor means_cam,
    torch::Tensor covs_cam,
    float fx,
    float fy,
    float cx,
    float cy
) {
    const int N = means_cam.size(0);
    
    auto means_2d = torch::empty({N, 2}, means_cam.options());
    auto covs_2d = torch::empty({N, 2, 2}, covs_cam.options());
    auto depths = torch::empty({N}, means_cam.options());
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    project_to_2d_kernel<<<blocks, threads>>>(
        means_cam.data_ptr<float>(),
        covs_cam.data_ptr<float>(),
        fx, fy, cx, cy,
        means_2d.data_ptr<float>(),
        covs_2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        N
    );
    
    return {means_2d, covs_2d, depths};
}

torch::Tensor render_gaussians_2d_cuda(
    torch::Tensor pixel_coords,
    torch::Tensor means_2d,
    torch::Tensor covs_2d,
    torch::Tensor weights
) {
    const int num_pixels = pixel_coords.size(0);
    const int num_gaussians = means_2d.size(0);
    
    auto output = torch::zeros({num_pixels}, pixel_coords.options());
    
    const int threads = 256;
    const int blocks = (num_pixels + threads - 1) / threads;
    
    render_gaussians_2d_kernel<<<blocks, threads>>>(
        pixel_coords.data_ptr<float>(),
        means_2d.data_ptr<float>(),
        covs_2d.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        num_pixels,
        num_gaussians
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transform_gaussians", &transform_gaussians_cuda, "Transform 3D Gaussians to camera space (CUDA)");
    m.def("project_to_2d", &project_to_2d_cuda, "Project 3D Gaussians to 2D (CUDA)");
    m.def("render_gaussians_2d", &render_gaussians_2d_cuda, "Render 2D Gaussians to pixels (CUDA)");
}
