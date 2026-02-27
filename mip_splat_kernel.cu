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
#define SOFT_MIP_BETA 2.0f  // sharpness of soft-max approximation

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
// J is the 2x3 perspective projection Jacobian:
//   row 0: [fx/z,  0,   -fx*x/z²]
//   row 1: [0,    fy/z, -fy*y/z²]
// T = J * Sigma3D * J^T
__device__ void compute_2d_cov(
    float J00, float J01, float J02,
    float J10, float J11, float J12,
    const float* cov3d,          // upper-tri: [c00,c01,c02,c11,c12,c22]
    float& cov2d_a, float& cov2d_b, float& cov2d_c)
{
    float S00=cov3d[0], S01=cov3d[1], S02=cov3d[2];
    float S11=cov3d[3], S12=cov3d[4], S22=cov3d[5];

    // M = Sigma3D * J^T  (3x3 * 3x2, stored column-by-column as two 3-vectors)
    // M[:,0] = Sigma * J[0,:] = Sigma * [J00, J10, 0]  (note J01=J10=0 for pinhole)
    // M[:,1] = Sigma * J[1,:] = Sigma * [0,   J11, J12]
    // Full form kept for generality:
    float M00 = S00*J00 + S01*J10;
    float M01 = S00*J01 + S01*J11;
    float M02 = S00*J02 + S01*J12;
    float M10 = S01*J00 + S11*J10;
    float M11 = S01*J01 + S11*J11;
    float M12 = S01*J02 + S11*J12;
    float M20 = S02*J00 + S12*J10;
    float M21 = S02*J01 + S12*J11;
    float M22 = S02*J02 + S12*J12;

    // T = J * M  — only upper-tri of the symmetric 2x2 result
    cov2d_a = J00*M00 + J01*M10 + J02*M20 + 0.3f;  // T[0,0] + low-pass filter
    cov2d_b = J00*M01 + J01*M11 + J02*M21;           // T[0,1] = T[1,0]
    cov2d_c = J10*M01 + J11*M11 + J12*M21 + 0.3f;  // T[1,1] + low-pass filter
}

// ============================================================
//  Forward: Soft-MIP splatting for fluorescence emission
//
//  For fluorescence there is NO absorption, only emission.
//  Output is a soft-max approximation of the maximum emission
//  along each ray (differentiable MIP).
//
//  FIX: Added out_sum_sw output buffer to save Σ(soft_w) per
//  pixel for use in the backward pass. Without this, the backward
//  had no access to the true denominator and used a wrong
//  approximation (soft_w * 2.0f), producing incorrect gradients.
// ============================================================
__global__ void mip_splat_forward_kernel(
    const int   N,
    const int   H, const int W,
    const float fx, const float fy,
    const float cx, const float cy,
    const float* __restrict__ means3d,    // [N,3]
    const float* __restrict__ cov3d,      // [N,6] upper-tri
    const float* __restrict__ opacity,    // [N,1] (unused, kept for API compat)
    const float* __restrict__ intensity,  // [N,C]
    const float* __restrict__ view_mat,   // [16] row-major 4x4
    const int   C,
    float*       out_image,               // [H,W,C]
    float*       out_weight,              // [H,W] Σ(emission_strength) — diagnostic
    float*       out_depth,               // [H,W] depth of max emitter
    float*       out_sum_sw               // [H,W] Σ(soft_w) — saved for backward (FIX)
)
{
    const int pix_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pix_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pix_x >= W || pix_y >= H) return;

    float soft_max_num[8] = {0};
    float soft_max_den   = 0.0f;
    float depth_acc      = 0.0f;
    float weight_acc     = 0.0f;

    for (int g = 0; g < N; g++) {
        float gx = means3d[g*3+0];
        float gy = means3d[g*3+1];
        float gz = means3d[g*3+2];

        float3 pc = project_point(view_mat, gx, gy, gz);
        if (pc.z <= 0.01f) continue;

        float inv_z  = 1.0f / pc.z;
        float inv_z2 = inv_z * inv_z;

        float J00 =  fx * inv_z;
        float J02 = -fx * pc.x * inv_z2;
        float J11 =  fy * inv_z;
        float J12 = -fy * pc.y * inv_z2;

        float a, b, c;
        compute_2d_cov(J00, 0.f, J02, 0.f, J11, J12, cov3d + g*6, a, b, c);

        float u0 = pc.x * inv_z * fx + cx;
        float v0 = pc.y * inv_z * fy + cy;

        float du = (float)pix_x - u0;
        float dv = (float)pix_y - v0;

        float det = a*c - b*b;
        if (det <= 1e-10f) continue;
        float inv_det = 1.0f / det;
        float maha = (c*du*du - 2.0f*b*du*dv + a*dv*dv) * inv_det;
        if (maha > 9.0f) continue;

        float footprint = __expf(-0.5f * maha);
        if (footprint < 1.0f/255.0f) continue;

        float mean_intensity = 0.0f;
        for (int ch = 0; ch < C && ch < 8; ch++)
            mean_intensity += intensity[g*C + ch];
        mean_intensity /= (float)C;

        float emission_strength = footprint * mean_intensity;
        float soft_w = __expf(SOFT_MIP_BETA * emission_strength);

        for (int ch = 0; ch < C && ch < 8; ch++)
            soft_max_num[ch] += soft_w * intensity[g*C + ch];

        soft_max_den += soft_w;
        depth_acc    += soft_w * pc.z;
        weight_acc   += emission_strength;
    }

    int pid = pix_y * W + pix_x;
    float inv_den = (soft_max_den > 1e-10f) ? 1.0f / soft_max_den : 0.0f;

    for (int ch = 0; ch < C && ch < 8; ch++)
        out_image[pid * C + ch] = soft_max_num[ch] * inv_den;

    out_depth[pid]  = depth_acc * inv_den;
    out_weight[pid] = weight_acc;
    out_sum_sw[pid] = soft_max_den;  // FIX: save exact Σ(soft_w) for backward
}


// ============================================================
//  Backward: gradients w.r.t. means, cov3d, intensity
//
//  FIX 1: Uses saved out_sum_sw (exact Σ(soft_w) from forward)
//  instead of the broken `approx_sum_sw = soft_w * 2.0f`.
//  The old approximation produced arbitrarily wrong gradients
//  whenever more than one Gaussian overlapped a pixel.
//
//  FIX 2: dL/dS01 (cov3d[1]) gradient now correct.
//  Old code had `dL_da * J00*0` which is always zero, silently
//  dropping the contribution. Full Jacobian chain-rule applied.
// ============================================================
__global__ void mip_splat_backward_kernel(
    const int   N, const int H, const int W,
    const float fx, const float fy,
    const float cx, const float cy,
    const float* __restrict__ means3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ opacity,
    const float* __restrict__ intensity,  // [N,C]
    const float* __restrict__ view_mat,
    const int   C,
    const float* __restrict__ out_image,
    const float* __restrict__ out_weight,
    const float* __restrict__ out_depth,
    const float* __restrict__ out_sum_sw,  // FIX 1: exact Σ(soft_w) from forward
    const float* __restrict__ dL_dimage,
    const float* __restrict__ dL_dweight,
    const float* __restrict__ dL_ddepth,
    float* __restrict__ dL_dmeans,
    float* __restrict__ dL_dcov3d,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dintensity
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
    float J00    =  fx * inv_z;
    float J02    = -fx * pc.x * inv_z2;
    float J11    =  fy * inv_z;
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
    float dL_di[8] = {0};

    float mean_intensity = 0.0f;
    for (int ch = 0; ch < C && ch < 8; ch++)
        mean_intensity += intensity[g*C + ch];
    mean_intensity /= (float)C;

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

        float emission_strength = footprint * mean_intensity;
        float soft_w = __expf(SOFT_MIP_BETA * emission_strength);

        int pid = pv * W + pu;

        // FIX 1: Use exact Σ(soft_w) saved from forward pass.
        // Old code used `approx_sum_sw = soft_w * 2.0f` — completely wrong
        // when multiple Gaussians overlap the pixel.
        float sum_sw    = out_sum_sw[pid];
        float inv_sum_sw = 1.0f / (sum_sw + 1e-10f);

        float out_O[8];
        for (int ch = 0; ch < C && ch < 8; ch++)
            out_O[ch] = out_image[pid*C + ch];

        // dL/d(I_ch):
        //   O_ch = Σ_g(soft_w_g * I_{g,ch}) / Σ_g(soft_w_g)
        //   dO_ch/dI_{g,ch} = soft_w / Σ(soft_w)                         [direct]
        //   dsoft_w/dI_{g,ch} = soft_w * β * footprint / C                [through soft_w]
        //   => dO_ch/dI_{g,ch} += soft_w * β * (footprint/C) * (I_ch - O_ch) / Σ(soft_w)
        for (int ch = 0; ch < C && ch < 8; ch++) {
            float I_ch = intensity[g*C + ch];
            float direct    = soft_w * inv_sum_sw;
            float thru_sw   = soft_w * SOFT_MIP_BETA * (footprint / (float)C)
                              * (I_ch - out_O[ch]) * inv_sum_sw;
            dL_di[ch] += dL_dimage[pid*C + ch] * (direct + thru_sw);
        }

        // dL/d(footprint) — only via soft_w path:
        //   dsoft_w/dfootprint = soft_w * β * mean_intensity
        float dL_dfootprint = 0.0f;
        for (int ch = 0; ch < C && ch < 8; ch++) {
            float I_ch = intensity[g*C + ch];
            dL_dfootprint += dL_dimage[pid*C + ch]
                           * soft_w * SOFT_MIP_BETA * mean_intensity
                           * (I_ch - out_O[ch]) * inv_sum_sw;
        }

        // dL/dmaha = dL/dfootprint * d(exp(-0.5*maha))/dmaha
        float dL_dmaha = dL_dfootprint * (-0.5f) * footprint;

        // dL/d(cov2d {a,b,c}) via maha = (c*du² - 2b*du*dv + a*dv²) / det
        float inv_det2 = inv_det * inv_det;
        float maha_num = c*du*du - 2.0f*b*du*dv + a*dv*dv;  // = maha * det
        dL_da += dL_dmaha * ( dv*dv*inv_det - maha_num*inv_det2*c );
        dL_db += dL_dmaha * (-2.0f*du*dv*inv_det + maha_num*inv_det2*2.0f*b);
        dL_dc += dL_dmaha * ( du*du*inv_det - maha_num*inv_det2*a );

        // dL/d(u0,v0) via du = pu-u0, dv = pv-v0
        float dL_du0 = dL_dmaha * (-2.0f*c*du + 2.0f*b*dv) * inv_det;
        float dL_dv0 = dL_dmaha * (-2.0f*a*dv + 2.0f*b*du) * inv_det;

        // dL/d(pc.x, pc.y, pc.z) via perspective projection
        dL_dmx += dL_du0 * fx * inv_z;
        dL_dmy += dL_dv0 * fy * inv_z;
        dL_dmz += dL_du0 * (-fx * pc.x * inv_z2)
                + dL_dv0 * (-fy * pc.y * inv_z2);

        // depth term
        dL_dmz += dL_ddepth[pid] * soft_w * inv_sum_sw;
    }}

    atomicAdd(&dL_dmeans[g*3+0], dL_dmx);
    atomicAdd(&dL_dmeans[g*3+1], dL_dmy);
    atomicAdd(&dL_dmeans[g*3+2], dL_dmz);

    // FIX 2: Correct dL/dSigma3D chain rule through T = J * Sigma3D * J^T.
    // J = [[J00, 0, J02], [0, J11, J12]]
    //
    // dL/dS00 = dL/dT00 * J00^2
    atomicAdd(&dL_dcov3d[g*6+0], dL_da * J00*J00);
    //
    // dL/dS01: T00 += 2*J00*0*S01=0; T01 += J00*J11*S01 (S01 appears once, not twice,
    //          because S is symmetric and T01 = J[0]*Sigma*J[1]^T).
    // Old code had `dL_da * J00*0` which is always zero — the correct term is just:
    atomicAdd(&dL_dcov3d[g*6+1], dL_db * J00*J11);
    //
    // dL/dS02: T00 += 2*J00*J02*S02 (from sym); T01 += J00*J12*S02 + J02*J11*S02
    atomicAdd(&dL_dcov3d[g*6+2], dL_da * 2.0f*J00*J02
                                + dL_db * (J00*J12 + J02*J11));
    //
    // dL/dS11 = dL/dT11 * J11^2
    atomicAdd(&dL_dcov3d[g*6+3], dL_dc * J11*J11);
    //
    // dL/dS12: T11 += 2*J11*J12*S12; T01 += J02*J11*S12
    atomicAdd(&dL_dcov3d[g*6+4], dL_dc * 2.0f*J11*J12
                                + dL_db * J02*J11);
    //
    // dL/dS22: T00 += J02^2; T01 += J02*J12; T11 += J12^2
    atomicAdd(&dL_dcov3d[g*6+5], dL_da * J02*J02
                                + dL_db * J02*J12
                                + dL_dc * J12*J12);

    for (int ch = 0; ch < C && ch < 8; ch++)
        atomicAdd(&dL_dintensity[g*C+ch], dL_di[ch]);
}


// ============================================================
//  Densification helper: per-Gaussian view-space gradient magnitude
// ============================================================
__global__ void compute_grad_magnitude_kernel(
    const int N,
    const float* __restrict__ dL_dmeans,
    const float* __restrict__ view_mat,
    float* __restrict__ grad_mag
)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;

    float dx = dL_dmeans[g*3+0];
    float dy = dL_dmeans[g*3+1];
    float dz = dL_dmeans[g*3+2];

    // Rotate gradient into view space using the upper-left 3x3 of the view matrix
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

    float R00=1-2*(y*y+z*z), R01=2*(x*y-w*z), R02=2*(x*z+w*y);
    float R10=2*(x*y+w*z),   R11=1-2*(x*x+z*z), R12=2*(y*z-w*x);
    float R20=2*(x*z-w*y),   R21=2*(y*z+w*x), R22=1-2*(x*x+y*y);

    float M00=R00*sx, M01=R01*sy, M02=R02*sz;
    float M10=R10*sx, M11=R11*sy, M12=R12*sz;
    float M20=R20*sx, M21=R21*sy, M22=R22*sz;

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

// FIX: mip_splat_forward now returns 4 tensors: {img, weight, depth, sum_sw}
// sum_sw = Σ(soft_w) per pixel, required by the backward pass.
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
    auto sum_sw = torch::zeros({H,W},   means3d.options());  // FIX: new output

    dim3 block(16,16), grid((W+15)/16, (H+15)/16);
    mip_splat_forward_kernel<<<grid, block>>>(N, H, W, fx, fy, cx, cy,
        means3d.data_ptr<float>(), cov3d.data_ptr<float>(),
        opacity.data_ptr<float>(), features.data_ptr<float>(),
        view_mat.data_ptr<float>(), C,
        img.data_ptr<float>(), weight.data_ptr<float>(),
        depth.data_ptr<float>(), sum_sw.data_ptr<float>());
    return {img, weight, depth, sum_sw};
}

// FIX: mip_splat_backward now accepts out_sum_sw as an additional argument.
std::vector<torch::Tensor> mip_splat_backward(
    torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor opacity,
    torch::Tensor features, torch::Tensor view_mat,
    torch::Tensor out_image, torch::Tensor out_weight, torch::Tensor out_depth,
    torch::Tensor out_sum_sw,   // FIX: was missing; caused wrong gradients
    torch::Tensor dL_dimage, torch::Tensor dL_dweight, torch::Tensor dL_ddepth,
    int H, int W, float fx, float fy, float cx, float cy)
{
    int N = means3d.size(0);
    int C = features.size(1);
    auto dL_dmeans    = torch::zeros_like(means3d);
    auto dL_dcov3d    = torch::zeros_like(cov3d);
    auto dL_dopacity  = torch::zeros_like(opacity);
    auto dL_dfeatures = torch::zeros_like(features);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mip_splat_backward_kernel<<<blocks, BLOCK_SIZE>>>(N, H, W, fx, fy, cx, cy,
        means3d.data_ptr<float>(), cov3d.data_ptr<float>(),
        opacity.data_ptr<float>(), features.data_ptr<float>(),
        view_mat.data_ptr<float>(), C,
        out_image.data_ptr<float>(), out_weight.data_ptr<float>(),
        out_depth.data_ptr<float>(), out_sum_sw.data_ptr<float>(),
        dL_dimage.data_ptr<float>(), dL_dweight.data_ptr<float>(),
        dL_ddepth.data_ptr<float>(),
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
    m.def("build_cov3d",            &build_cov3d,           "Build 3D covariance from quat+scale");
    m.def("mip_splat_forward",      &mip_splat_forward,     "Soft-MIP splatting forward");
    m.def("mip_splat_backward",     &mip_splat_backward,    "Soft-MIP splatting backward");
    m.def("compute_grad_magnitude", &compute_grad_mag,      "Per-Gaussian gradient magnitude");
}