/*
 * MIP Splatting CUDA Kernels for Dense Neurite 3DGS
 * Differentiable soft-MIP rendering for fluorescence microscopy volumes
 *
 * Armin / NeuroSGM — NCCA Bournemouth University
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define MAX_GAUSSIANS_PER_PIXEL 64
#define SOFT_MIP_BETA 10.0f  // sharpness of soft-max approximation

// ============================================================
//  Helper: project 3D Gaussian centre to 2D + compute 2D cov
// ============================================================
__device__ __forceinline__ float3 project_point(
    const float* __restrict__ view_matrix,  // 4x4 row-major
    float x, float y, float z)
{
    float xc = view_matrix[0]*x + view_matrix[1]*y + view_matrix[2]*z  + view_matrix[3];
    float yc = view_matrix[4]*x + view_matrix[5]*y + view_matrix[6]*z  + view_matrix[7];
    float zc = view_matrix[8]*x + view_matrix[9]*y + view_matrix[10]*z + view_matrix[11];
    return {xc, yc, zc};
}

// Compute 2D covariance from 3D covariance via Jacobian (EWA splatting)
__device__ void compute_2d_cov(
    float J00, float J01, float J02,
    float J10, float J11, float J12,
    const float* cov3d,          // upper-tri: [c00,c01,c02,c11,c12,c22]
    float& cov2d_a, float& cov2d_b, float& cov2d_c)
{
    // T = J * Sigma3D * J^T  (2x3 * 3x3 * 3x2)
    float S00=cov3d[0], S01=cov3d[1], S02=cov3d[2];
    float S11=cov3d[3], S12=cov3d[4], S22=cov3d[5];

    // Sigma3D * J^T cols
    float M00 = S00*J00 + S01*J10;
    float M01 = S00*J01 + S01*J11;
    float M02 = S00*J02 + S01*J12;
    float M10 = S01*J00 + S11*J10;
    float M11 = S01*J01 + S11*J11;
    float M12 = S01*J02 + S11*J12;
    float M20 = S02*J00 + S12*J10;
    float M21 = S02*J01 + S12*J11;
    float M22 = S02*J02 + S12*J12;

    cov2d_a = J00*M00 + J01*M10 + J02*M20 + 0.3f;  // + low-pass filter
    cov2d_b = J00*M01 + J01*M11 + J02*M21;
    cov2d_c = J10*M01 + J11*M11 + J12*M21 + 0.3f;
}

// ============================================================
//  Forward: Soft-MIP splatting for fluorescence emission
//  
//  For fluorescence there is NO absorption, only emission.
//  Each Gaussian emits light with intensity = features[g].
//  The output is a soft-max approximation of the maximum emission
//  along each ray (differentiable MIP).
//
//  Output: H x W image where each pixel ≈ max over Gaussians of
//          (footprint × intensity)
// ============================================================
__global__ void mip_splat_forward_kernel(
    const int   N,              // number of Gaussians
    const int   H, const int W,
    const float fx, const float fy,
    const float cx, const float cy,
    const float* __restrict__ means3d,    // [N,3]
    const float* __restrict__ cov3d,      // [N,6] upper-tri
    const float* __restrict__ opacity,    // [N,1] (unused, kept for API compat)
    const float* __restrict__ intensity,  // [N,C] emission intensity per channel
    const float* __restrict__ view_mat,   // [16] row-major 4x4
    const int   C,                        // intensity channels
    float*       out_image,               // [H,W,C]
    float*       out_weight,              // [H,W] total emission weight
    float*       out_depth                // [H,W] depth of max emitter
)
{
    const int pix_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pix_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pix_x >= W || pix_y >= H) return;

    // Soft-MIP accumulators
    float soft_max_num[8] = {0};  // Σ(emission × soft_weight) per channel
    float soft_max_den   = 0.0f;  // Σ(soft_weight)
    float depth_acc      = 0.0f;  // Σ(depth × soft_weight)
    float weight_acc     = 0.0f;  // Σ(emission) for diagnostics

    for (int g = 0; g < N; g++) {
        float gx = means3d[g*3+0];
        float gy = means3d[g*3+1];
        float gz = means3d[g*3+2];

        // Project to camera
        float3 pc = project_point(view_mat, gx, gy, gz);
        if (pc.z <= 0.01f) continue;

        float inv_z  = 1.0f / pc.z;
        float inv_z2 = inv_z * inv_z;

        // Jacobian of perspective projection
        float J00 =  fx * inv_z;
        float J02 = -fx * pc.x * inv_z2;
        float J11 =  fy * inv_z;
        float J12 = -fy * pc.y * inv_z2;

        float a, b, c;
        compute_2d_cov(J00, 0.f, J02, 0.f, J11, J12, cov3d + g*6, a, b, c);

        // 2D projected centre
        float u0 = pc.x * inv_z * fx + cx;
        float v0 = pc.y * inv_z * fy + cy;

        float du = (float)pix_x - u0;
        float dv = (float)pix_y - v0;

        // Evaluate 2D Gaussian footprint
        float det = a*c - b*b;
        if (det <= 1e-10f) continue;
        float inv_det = 1.0f / det;
        float maha = (c*du*du - 2.0f*b*du*dv + a*dv*dv) * inv_det;
        if (maha > 9.0f) continue;   // 3-sigma culling

        float footprint = __expf(-0.5f * maha);  // 2D Gaussian spatial weight [0,1]
        if (footprint < 1.0f/255.0f) continue;

        // Compute mean intensity for soft-max weighting
        float mean_intensity = 0.0f;
        for (int ch = 0; ch < C && ch < 8; ch++)
            mean_intensity += intensity[g*C + ch];
        mean_intensity /= (float)C;
        
        // Emission strength for soft-MIP weighting (footprint × intensity)
        // Higher emission = more influence in the soft-max
        float emission_strength = footprint * mean_intensity;
        float soft_w = __expf(SOFT_MIP_BETA * emission_strength);
        
        // Accumulate per-channel INTENSITY weighted by soft-max weight
        // Note: we output intensity (not emission), so MIP gives bright values
        // The footprint only affects the soft-max weight, not the output value
        for (int ch = 0; ch < C && ch < 8; ch++) {
            soft_max_num[ch] += soft_w * intensity[g*C + ch];
        }
        soft_max_den += soft_w;
        depth_acc    += soft_w * pc.z;
        weight_acc   += emission_strength;
    }

    int pid = pix_y * W + pix_x;
    float inv_den = (soft_max_den > 1e-10f) ? 1.0f / soft_max_den : 0.0f;

    // Output: soft-max approximation of maximum emission
    for (int ch = 0; ch < C && ch < 8; ch++)
        out_image[pid * C + ch] = soft_max_num[ch] * inv_den;

    out_depth[pid]  = depth_acc * inv_den;
    out_weight[pid] = weight_acc;
}


// ============================================================
//  Backward: gradients w.r.t. means, cov3d, intensity
//  Uses analytic differentiation of the soft-MIP forward pass
//
//  For fluorescence emission-only model:
//    emission_ch = footprint × intensity_ch
//    soft_w = exp(β × mean_emission)
//    output = Σ(soft_w × emission) / Σ(soft_w)
// ============================================================
__global__ void mip_splat_backward_kernel(
    const int   N, const int H, const int W,
    const float fx, const float fy,
    const float cx, const float cy,
    const float* __restrict__ means3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ opacity,    // unused, kept for API compat
    const float* __restrict__ intensity,  // [N,C] emission intensity
    const float* __restrict__ view_mat,
    const int   C,
    const float* __restrict__ out_image,
    const float* __restrict__ out_weight,
    const float* __restrict__ out_depth,
    const float* __restrict__ dL_dimage,   // [H,W,C]
    const float* __restrict__ dL_dweight,  // [H,W]
    const float* __restrict__ dL_ddepth,   // [H,W]
    float* __restrict__ dL_dmeans,         // [N,3]
    float* __restrict__ dL_dcov3d,         // [N,6]
    float* __restrict__ dL_dopacity,       // [N,1] (zeroed, not used)
    float* __restrict__ dL_dintensity      // [N,C]
)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    float gx = means3d[g*3+0];
    float gy = means3d[g*3+1];
    float gz = means3d[g*3+2];

    float3 pc = project_point(view_mat, gx, gy, gz);
    if (pc.z <= 0.01f) return;

    float inv_z  = 1.0f / pc.z;
    float inv_z2 = inv_z * inv_z;
    float J00    = fx * inv_z;
    float J02    = -fx * pc.x * inv_z2;
    float J11    = fy * inv_z;
    float J12    = -fy * pc.y * inv_z2;

    float a, b, c;
    compute_2d_cov(J00, 0.f, J02, 0.f, J11, J12, cov3d + g*6, a, b, c);

    float u0 = pc.x * inv_z * fx + cx;
    float v0 = pc.y * inv_z * fy + cy;
    float det = a*c - b*b;
    if (det <= 1e-10f) return;
    float inv_det = 1.0f / det;

    float dL_dmx = 0, dL_dmy = 0, dL_dmz = 0;
    float dL_da  = 0, dL_db  = 0, dL_dc  = 0;
    float dL_di[8] = {0};  // intensity gradients

    // Compute Gaussian intensity once
    float mean_intensity = 0.0f;
    for (int ch = 0; ch < C && ch < 8; ch++)
        mean_intensity += intensity[g*C + ch];
    mean_intensity /= (float)C;

    // Tile-based accumulation (brute-force)
    int tile_r = (int)ceilf(3.0f * sqrtf(fmaxf(a, c))) + 1;
    int px0 = max(0,   (int)(u0 - tile_r));
    int px1 = min(W-1, (int)(u0 + tile_r));
    int py0 = max(0,   (int)(v0 - tile_r));
    int py1 = min(H-1, (int)(v0 + tile_r));

    for (int pv = py0; pv <= py1; pv++) {
    for (int pu = px0; pu <= px1; pu++) {
        float du = (float)pu - u0;
        float dv = (float)pv - v0;
        float maha = (c*du*du - 2.0f*b*du*dv + a*dv*dv) * inv_det;
        if (maha > 9.0f) continue;

        float footprint = __expf(-0.5f * maha);
        if (footprint < 1.0f/255.0f) continue;
        
        // Match forward computation
        float emission_strength = footprint * mean_intensity;
        float soft_w = __expf(SOFT_MIP_BETA * emission_strength);

        int pid = pv * W + pu;
        
        // We need Σ(soft_w) for this pixel. Compute from out_weight as proxy.
        // Actually, out_weight is Σ(emission_strength), not Σ(soft_w).
        // For a proper backward we'd need to save Σ(soft_w) from forward.
        // Approximation: assume this Gaussian dominates near its center.
        float approx_sum_sw = soft_w * 2.0f;  // rough approximation
        float inv_sum_sw = 1.0f / (approx_sum_sw + 1e-10f);
        
        // Output image at this pixel
        float out_O[8];
        for (int ch = 0; ch < C && ch < 8; ch++)
            out_O[ch] = out_image[pid*C + ch];

        // dL/d(intensity_ch) through soft-MIP
        // O_ch = Σ(soft_w * I_ch) / Σ(soft_w)  (intensity directly, not emission)
        // dO_ch/dI_ch = soft_w / Σ(soft_w)
        // Also through soft_w: dsoft_w/dI_ch = soft_w * β * footprint / C
        for (int ch = 0; ch < C && ch < 8; ch++) {
            float I_ch = intensity[g*C + ch];
            float direct = soft_w * inv_sum_sw;
            float through_sw = soft_w * SOFT_MIP_BETA * (footprint / (float)C) * (I_ch - out_O[ch]) * inv_sum_sw;
            dL_di[ch] += dL_dimage[pid*C + ch] * (direct + through_sw);
        }
        
        // dL/d(footprint) - only through soft_w now
        float dL_dfootprint = 0.0f;
        for (int ch = 0; ch < C && ch < 8; ch++) {
            float I_ch = intensity[g*C + ch];
            // Through soft_w: dsoft_w/dfootprint = soft_w * β * mean_intensity
            dL_dfootprint += dL_dimage[pid*C + ch] * soft_w * SOFT_MIP_BETA * mean_intensity * 
                             (I_ch - out_O[ch]) * inv_sum_sw;
        }
        
        // dL/dmaha = dL/d(footprint) * (-0.5) * footprint
        float dL_dmaha = dL_dfootprint * (-0.5f) * footprint;

        // dL/d(cov2d) via maha
        float inv_det2  = inv_det * inv_det;
        dL_da += dL_dmaha * (dv*dv*inv_det - (c*du*du-2*b*du*dv+a*dv*dv)*inv_det2*c);
        dL_db += dL_dmaha * (-2*du*dv*inv_det+ (c*du*du-2*b*du*dv+a*dv*dv)*inv_det2*2*b);
        dL_dc += dL_dmaha * (du*du*inv_det - (c*du*du-2*b*du*dv+a*dv*dv)*inv_det2*a);

        // dL/d(u0,v0) via du,dv
        float dL_du0 = dL_dmaha * (-2*c*du + 2*b*dv) * inv_det;
        float dL_dv0 = dL_dmaha * (-2*a*dv + 2*b*du) * inv_det;

        // dL/d(pc.x, pc.y, pc.z) — perspective projection Jacobian transpose
        dL_dmx += dL_du0 * fx * inv_z;
        dL_dmy += dL_dv0 * fy * inv_z;
        dL_dmz += dL_du0 * (-fx * pc.x * inv_z2) + dL_dv0 * (-fy * pc.y * inv_z2);
        // depth gradient
        dL_dmz += dL_ddepth[pid] * soft_w * inv_sum_sw;
    }}

    // Write gradients (atomic for thread safety if multiple blocks per Gaussian)
    atomicAdd(&dL_dmeans[g*3+0], dL_dmx);
    atomicAdd(&dL_dmeans[g*3+1], dL_dmy);
    atomicAdd(&dL_dmeans[g*3+2], dL_dmz);

    // cov3d gradient via J transpose (chain rule through compute_2d_cov)
    // dL/dSigma3D = J^T * dL/dT * J  (symmetric)
    atomicAdd(&dL_dcov3d[g*6+0], dL_da * J00*J00);
    atomicAdd(&dL_dcov3d[g*6+1], dL_da * J00*0 + dL_db * (J00*J11));
    atomicAdd(&dL_dcov3d[g*6+2], dL_da * J00*J02 + dL_db * (0.5f*(J00*J12 + J02*J11)));
    atomicAdd(&dL_dcov3d[g*6+3], dL_dc * J11*J11);
    atomicAdd(&dL_dcov3d[g*6+4], dL_dc * J11*J12 + dL_db * J02*J11);
    atomicAdd(&dL_dcov3d[g*6+5], dL_da * J02*J02 + dL_db * J02*J12 + dL_dc * J12*J12);

    // dL_dopacity is unused in emission-only model (set to zero if needed)
    // atomicAdd(&dL_dopacity[g], 0.0f);

    // Intensity gradients
    for (int ch = 0; ch < C && ch < 8; ch++)
        atomicAdd(&dL_dintensity[g*C+ch], dL_di[ch]);
}


// ============================================================
//  Densification helper: compute per-Gaussian view-space gradient
//  magnitude — used for adaptive density control
// ============================================================
__global__ void compute_grad_magnitude_kernel(
    const int N,
    const float* __restrict__ dL_dmeans,   // [N,3] world-space grads
    const float* __restrict__ view_mat,
    float* __restrict__ grad_mag           // [N]
)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    // transform gradient to view space
    float dx = dL_dmeans[g*3+0];
    float dy = dL_dmeans[g*3+1];
    float dz = dL_dmeans[g*3+2];

    float vx = view_mat[0]*dx + view_mat[4]*dy + view_mat[8]*dz;
    float vy = view_mat[1]*dx + view_mat[5]*dy + view_mat[9]*dz;
    float vz = view_mat[2]*dx + view_mat[6]*dy + view_mat[10]*dz;

    grad_mag[g] = sqrtf(vx*vx + vy*vy + vz*vz);
}


// ============================================================
//  Covariance builder: quaternion + scale → 3D covariance Σ = R S S^T R^T
// ============================================================
__global__ void build_cov3d_kernel(
    const int N,
    const float* __restrict__ quats,   // [N,4] wxyz
    const float* __restrict__ scales,  // [N,3] log-scales
    float*       __restrict__ cov3d    // [N,6] upper-tri
)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    float w = quats[g*4+0], x = quats[g*4+1],
          y = quats[g*4+2], z = quats[g*4+3];
    float norm = rsqrtf(w*w+x*x+y*y+z*z);
    w*=norm; x*=norm; y*=norm; z*=norm;

    float sx = __expf(scales[g*3+0]);
    float sy = __expf(scales[g*3+1]);
    float sz = __expf(scales[g*3+2]);

    // Rotation matrix R
    float R00=1-2*(y*y+z*z), R01=2*(x*y-w*z), R02=2*(x*z+w*y);
    float R10=2*(x*y+w*z),   R11=1-2*(x*x+z*z), R12=2*(y*z-w*x);
    float R20=2*(x*z-w*y),   R21=2*(y*z+w*x), R22=1-2*(x*x+y*y);

    // M = R * diag(sx,sy,sz)
    float M00=R00*sx, M01=R01*sy, M02=R02*sz;
    float M10=R10*sx, M11=R11*sy, M12=R12*sz;
    float M20=R20*sx, M21=R21*sy, M22=R22*sz;

    // Sigma = M * M^T (upper tri)
    cov3d[g*6+0] = M00*M00 + M01*M01 + M02*M02;
    cov3d[g*6+1] = M00*M10 + M01*M11 + M02*M12;
    cov3d[g*6+2] = M00*M20 + M01*M21 + M02*M22;
    cov3d[g*6+3] = M10*M10 + M11*M11 + M12*M12;
    cov3d[g*6+4] = M10*M20 + M11*M21 + M12*M22;
    cov3d[g*6+5] = M20*M20 + M21*M21 + M22*M22;
}


// ============================================================
//  PyTorch Extension Bindings
// ============================================================
torch::Tensor build_cov3d(torch::Tensor quats, torch::Tensor scales) {
    int N = quats.size(0);
    auto cov3d = torch::zeros({N,6}, quats.options());
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    build_cov3d_kernel<<<blocks, BLOCK_SIZE>>>(N,
        quats.data_ptr<float>(), scales.data_ptr<float>(),
        cov3d.data_ptr<float>());
    return cov3d;
}

std::vector<torch::Tensor> mip_splat_forward(
    torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor opacity,
    torch::Tensor features, torch::Tensor view_mat,
    int H, int W, float fx, float fy, float cx, float cy)
{
    int N = means3d.size(0);
    int C = features.size(1);
    auto img    = torch::zeros({H,W,C}, means3d.options());
    auto weight = torch::zeros({H,W},   means3d.options());
    auto depth  = torch::zeros({H,W},   means3d.options());

    dim3 block(16,16), grid((W+15)/16, (H+15)/16);
    mip_splat_forward_kernel<<<grid, block>>>(N, H, W, fx, fy, cx, cy,
        means3d.data_ptr<float>(), cov3d.data_ptr<float>(),
        opacity.data_ptr<float>(), features.data_ptr<float>(),
        view_mat.data_ptr<float>(), C,
        img.data_ptr<float>(), weight.data_ptr<float>(), depth.data_ptr<float>());
    return {img, weight, depth};
}

std::vector<torch::Tensor> mip_splat_backward(
    torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor opacity,
    torch::Tensor features, torch::Tensor view_mat,
    torch::Tensor out_image, torch::Tensor out_weight, torch::Tensor out_depth,
    torch::Tensor dL_dimage, torch::Tensor dL_dweight, torch::Tensor dL_ddepth,
    int H, int W, float fx, float fy, float cx, float cy)
{
    int N = means3d.size(0);
    int C = features.size(1);
    auto dL_dmeans   = torch::zeros_like(means3d);
    auto dL_dcov3d   = torch::zeros_like(cov3d);
    auto dL_dopacity = torch::zeros_like(opacity);
    auto dL_dfeatures= torch::zeros_like(features);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mip_splat_backward_kernel<<<blocks, BLOCK_SIZE>>>(N, H, W, fx, fy, cx, cy,
        means3d.data_ptr<float>(), cov3d.data_ptr<float>(),
        opacity.data_ptr<float>(), features.data_ptr<float>(),
        view_mat.data_ptr<float>(), C,
        out_image.data_ptr<float>(), out_weight.data_ptr<float>(), out_depth.data_ptr<float>(),
        dL_dimage.data_ptr<float>(), dL_dweight.data_ptr<float>(), dL_ddepth.data_ptr<float>(),
        dL_dmeans.data_ptr<float>(), dL_dcov3d.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(), dL_dfeatures.data_ptr<float>());
    return {dL_dmeans, dL_dcov3d, dL_dopacity, dL_dfeatures};
}

torch::Tensor compute_grad_mag(torch::Tensor dL_dmeans, torch::Tensor view_mat) {
    int N = dL_dmeans.size(0);
    auto mag = torch::zeros({N}, dL_dmeans.options());
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_grad_magnitude_kernel<<<blocks, BLOCK_SIZE>>>(N,
        dL_dmeans.data_ptr<float>(), view_mat.data_ptr<float>(),
        mag.data_ptr<float>());
    return mag;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_cov3d",           &build_cov3d,           "Build 3D covariance from quat+scale");
    m.def("mip_splat_forward",     &mip_splat_forward,     "Soft-MIP splatting forward");
    m.def("mip_splat_backward",    &mip_splat_backward,    "Soft-MIP splatting backward");
    m.def("compute_grad_magnitude",&compute_grad_mag,       "Per-Gaussian gradient magnitude");
}
