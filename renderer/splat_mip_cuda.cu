/*
 * splat_mip_cuda.cu — CUDA kernel for MIP (Maximum Intensity Projection) splatting
 *
 * MIP formula per pixel:
 *     g_k   = intensity_k * exp(-0.5 * mahalanobis_k(pixel))
 *     out   = sum_k [ softmax(beta * g)_k * g_k ]
 *
 * This replaces the Python chunk-loop in splat_mip_grid() which materialises
 * large (chunk_size, K) intermediate tensors.  Each CUDA thread handles one
 * pixel, streaming over K Gaussians with O(1) memory using the online
 * softmax trick.
 *
 * Backward: analytic gradient  d(out)/d(g_k) = sm_k * (1 + beta*(g_k - out))
 * chained through the Gaussian evaluation to means_2d, cov_inv, intensities.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define BLOCK_SIZE 256
#define MAHAL_CUTOFF 16.0f   // exp(-8) ≈ 3.4e-4


// ============================================================================
//  Forward kernel — one thread per pixel, online softmax
// ============================================================================
__global__ void splat_mip_forward_kernel(
    const float* __restrict__ means_2d,      // (K, 2)
    const float* __restrict__ cov_inv,       // (K, 3) packed [a, b, d]
    const float* __restrict__ intensities,   // (K,)
    const float* __restrict__ pixels,        // (N, 2)
    float*       __restrict__ rendered,      // (N,)  output image
    float*       __restrict__ out_max_bg,    // (N,)  saved max(beta*g)  — for backward
    float*       __restrict__ out_sum_exp,   // (N,)  saved sum_exp       — for backward
    const int N,
    const int K,
    const float beta
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    /* Online numerically-stable softmax accumulation:
     *   m   = running max of beta*g_k
     *   S   = sum of exp(beta*g_k - m)
     *   W   = sum of exp(beta*g_k - m) * g_k          */
    float m = -1e30f;
    float S = 0.0f;
    float W = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float ia = cov_inv[k * 3 + 0];
        const float ib = cov_inv[k * 3 + 1];
        const float id = cov_inv[k * 3 + 2];

        const float mahal = ia * dx * dx + 2.0f * ib * dx * dy + id * dy * dy;
        if (mahal > MAHAL_CUTOFF) continue;

        const float g  = intensities[k] * expf(-0.5f * mahal);
        const float bg = beta * g;

        if (bg > m) {
            const float corr = expf(m - bg);
            S = S * corr + 1.0f;
            W = W * corr + g;
            m = bg;
        } else {
            const float e = expf(bg - m);
            S += e;
            W += e * g;
        }
    }

    float out_val = 0.0f;
    if (S > 0.0f) {
        out_val = W / S;
    }

    rendered[n]    = out_val;
    out_max_bg[n]  = m;
    out_sum_exp[n] = S;
}


// ============================================================================
//  Backward kernel — one thread per pixel, atomicAdd to per-Gaussian grads
// ============================================================================
//
//  out  = Σ_k sm_k * g_k     where  sm_k = exp(β g_k) / Σ_j exp(β g_j)
//
//  ∂out/∂g_k = sm_k · [1 + β (g_k − out)]
//
//  g_k = intensity_k · exp(−½ mahal_k)
//    ∂g_k/∂intensity_k = exp(−½ mahal_k)
//    ∂g_k/∂mahal_k     = g_k · (−½)
//
//  mahal = ia·dx² + 2·ib·dx·dy + id·dy²
//    ∂mahal/∂(ia)     =  dx²
//    ∂mahal/∂(ib)     =  2 dx dy
//    ∂mahal/∂(id)     =  dy²
//    ∂mahal/∂(mean_x) = −(2 ia dx + 2 ib dy)
//    ∂mahal/∂(mean_y) = −(2 ib dx + 2 id dy)
//
__global__ void splat_mip_backward_kernel(
    const float* __restrict__ dL_drendered,  // (N,)
    const float* __restrict__ means_2d,      // (K, 2)
    const float* __restrict__ cov_inv,       // (K, 3)
    const float* __restrict__ intensities,   // (K,)
    const float* __restrict__ pixels,        // (N, 2)
    const float* __restrict__ rendered,      // (N,)  — forward output
    const float* __restrict__ fwd_max_bg,    // (N,)  — saved from forward
    const float* __restrict__ fwd_sum_exp,   // (N,)  — saved from forward
    float*       __restrict__ grad_means_2d, // (K, 2)
    float*       __restrict__ grad_cov_inv,  // (K, 3)
    float*       __restrict__ grad_intensities, // (K,)
    const int N,
    const int K,
    const float beta
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float dL_dout = dL_drendered[n];
    if (fabsf(dL_dout) < 1e-12f) return;

    const float px      = pixels[n * 2 + 0];
    const float py      = pixels[n * 2 + 1];
    const float out_val = rendered[n];
    const float m       = fwd_max_bg[n];
    const float S       = fwd_sum_exp[n];

    if (S <= 0.0f) return;

    // log-sum-exp = m + log(S)
    const float lse = m + logf(S);

    for (int k = 0; k < K; ++k) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float ia = cov_inv[k * 3 + 0];
        const float ib = cov_inv[k * 3 + 1];
        const float id = cov_inv[k * 3 + 2];

        const float mahal = ia * dx * dx + 2.0f * ib * dx * dy + id * dy * dy;
        if (mahal > MAHAL_CUTOFF) continue;

        const float gauss_exp = expf(-0.5f * mahal);
        const float g  = intensities[k] * gauss_exp;
        const float bg = beta * g;

        // softmax weight: sm_k = exp(β g_k − lse)
        const float sm = expf(bg - lse);

        // ∂out/∂g_k = sm_k * (1 + β (g_k − out))
        const float dout_dg = sm * (1.0f + beta * (g - out_val));
        const float dL_dg   = dL_dout * dout_dg;

        // ∂g / ∂intensity = exp(−½ mahal)
        atomicAdd(&grad_intensities[k], dL_dg * gauss_exp);

        // ∂g / ∂mahal = g * (−½)
        const float dL_dmahal = dL_dg * g * (-0.5f);

        // ∂mahal / ∂(cov_inv components)
        atomicAdd(&grad_cov_inv[k * 3 + 0], dL_dmahal * dx * dx);
        atomicAdd(&grad_cov_inv[k * 3 + 1], dL_dmahal * 2.0f * dx * dy);
        atomicAdd(&grad_cov_inv[k * 3 + 2], dL_dmahal * dy * dy);

        // ∂mahal / ∂(mean)  (note the negative: dx = px − mean_x)
        const float dmahal_ddx = 2.0f * ia * dx + 2.0f * ib * dy;
        const float dmahal_ddy = 2.0f * ib * dx + 2.0f * id * dy;
        atomicAdd(&grad_means_2d[k * 2 + 0], -dL_dmahal * dmahal_ddx);
        atomicAdd(&grad_means_2d[k * 2 + 1], -dL_dmahal * dmahal_ddy);
    }
}


// ============================================================================
//  C++ interface
// ============================================================================

std::vector<torch::Tensor> splat_mip_forward(
    torch::Tensor means_2d,      // (K, 2)
    torch::Tensor cov_inv,       // (K, 3) packed [a, b, d]
    torch::Tensor intensities,   // (K,)
    torch::Tensor pixels,        // (N, 2)
    float beta
) {
    means_2d    = means_2d.contiguous();
    cov_inv     = cov_inv.contiguous();
    intensities = intensities.contiguous();
    pixels      = pixels.contiguous();

    const int K = means_2d.size(0);
    const int N = pixels.size(0);

    auto opts     = pixels.options();                        // float32, same device
    auto rendered = torch::zeros({N}, opts);
    auto max_bg   = torch::full({N}, -1e30f, opts);
    auto sum_exp  = torch::zeros({N}, opts);

    if (K > 0 && N > 0) {
        const int threads = BLOCK_SIZE;
        const int blocks  = (N + threads - 1) / threads;
        splat_mip_forward_kernel<<<blocks, threads>>>(
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            intensities.data_ptr<float>(),
            pixels.data_ptr<float>(),
            rendered.data_ptr<float>(),
            max_bg.data_ptr<float>(),
            sum_exp.data_ptr<float>(),
            N, K, beta
        );
    }

    return {rendered, max_bg, sum_exp};
}


std::vector<torch::Tensor> splat_mip_backward(
    torch::Tensor dL_drendered,  // (N,)
    torch::Tensor means_2d,      // (K, 2)
    torch::Tensor cov_inv,       // (K, 3)
    torch::Tensor intensities,   // (K,)
    torch::Tensor pixels,        // (N, 2)
    torch::Tensor rendered,      // (N,)
    torch::Tensor fwd_max_bg,    // (N,)
    torch::Tensor fwd_sum_exp,   // (N,)
    float beta
) {
    dL_drendered = dL_drendered.contiguous();
    means_2d     = means_2d.contiguous();
    cov_inv      = cov_inv.contiguous();
    intensities  = intensities.contiguous();
    pixels       = pixels.contiguous();
    rendered     = rendered.contiguous();
    fwd_max_bg   = fwd_max_bg.contiguous();
    fwd_sum_exp  = fwd_sum_exp.contiguous();

    const int K = means_2d.size(0);
    const int N = pixels.size(0);

    auto grad_means_2d    = torch::zeros({K, 2}, means_2d.options());
    auto grad_cov_inv     = torch::zeros({K, 3}, cov_inv.options());
    auto grad_intensities = torch::zeros({K},    intensities.options());

    if (K > 0 && N > 0) {
        const int threads = BLOCK_SIZE;
        const int blocks  = (N + threads - 1) / threads;
        splat_mip_backward_kernel<<<blocks, threads>>>(
            dL_drendered.data_ptr<float>(),
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            intensities.data_ptr<float>(),
            pixels.data_ptr<float>(),
            rendered.data_ptr<float>(),
            fwd_max_bg.data_ptr<float>(),
            fwd_sum_exp.data_ptr<float>(),
            grad_means_2d.data_ptr<float>(),
            grad_cov_inv.data_ptr<float>(),
            grad_intensities.data_ptr<float>(),
            N, K, beta
        );
    }

    return {grad_means_2d, grad_cov_inv, grad_intensities};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &splat_mip_forward,  "MIP splatting forward  (CUDA)");
    m.def("backward", &splat_mip_backward, "MIP splatting backward (CUDA)");
}
