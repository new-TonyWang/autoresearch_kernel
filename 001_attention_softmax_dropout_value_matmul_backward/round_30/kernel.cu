#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int N_HEADS = 80;
constexpr int N_KV_HEADS = 8;
constexpr int N_GROUPS = 10;
constexpr int H_DIM = 128;
constexpr int WARP_SZ = 32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

__global__ void __launch_bounds__(256, 8) attn_bw_v30(
    __nv_bfloat16* gs, float* gv, const __nv_bfloat16* go,
    const __nv_bfloat16* aw, const __nv_bfloat16* ad, const __nv_bfloat16* vs,
    const bool* dm, int B, int sq, int sv, float ds) {
    
    const int b = blockIdx.x, h = blockIdx.y, q = blockIdx.z;
    if (b >= B || h >= N_HEADS || q >= sq) return;
    
    const int kh = h / N_GROUPS;
    const int tid = threadIdx.x, lane = tid % WARP_SZ, warp = tid / WARP_SZ;
    const int goff = ((b * sq + q) * N_HEADS + h) * H_DIM;
    const int woff = ((b * N_HEADS + h) * sq + q) * sv;
    const int voff = ((b * N_KV_HEADS + kh) * sv) * H_DIM;
    const int aoff = ((b * N_KV_HEADS + kh) * sv) * H_DIM;
    
    __shared__ float sg[H_DIM], ss;
    
    #pragma unroll 4
    for (int d = tid; d < H_DIM; d += 256) sg[d] = __bfloat162float(go[goff + d]);
    if (!tid) ss = 0.0f;
    __syncthreads();
    
    const int nw = 8; // 256/32
    const int per_w = (sv + nw - 1) / nw;
    const int s0 = warp * per_w;
    const int s1 = min(s0 + per_w, sv);
    
    float lsum = 0.0f;
    
    // First pass
    for (int s = s0 + lane; s < s1; s += WARP_SZ) {
        float d = 0.0f;
        #pragma unroll 32
        for (int i = 0; i < H_DIM; ++i) d += sg[i] * __bfloat162float(vs[voff + s * H_DIM + i]);
        bool m = dm[woff + s];
        float g = m ? d * ds : 0.0f;
        lsum += g * __bfloat162float(aw[woff + s]);
        float a = __bfloat162float(ad[woff + s]);
        #pragma unroll 32
        for (int i = 0; i < H_DIM; ++i) atomicAdd(&gv[aoff + s * H_DIM + i], a * sg[i]);
    }
    
    // Reduce
    lsum = warp_sum(lsum);
    __shared__ float sp[8];
    if (!lane) sp[warp] = lsum;
    __syncthreads();
    if (!warp) {
        lsum = (lane < 8) ? sp[lane] : 0.0f;
        lsum = warp_sum(lsum);
        if (!lane) ss = lsum;
    }
    __syncthreads();
    float ts = ss;
    
    // Second pass
    for (int s = s0 + lane; s < s1; s += WARP_SZ) {
        float d = 0.0f;
        #pragma unroll 32
        for (int i = 0; i < H_DIM; ++i) d += sg[i] * __bfloat162float(vs[voff + s * H_DIM + i]);
        bool m = dm[woff + s];
        float g = m ? d * ds : 0.0f;
        float av = __bfloat162float(aw[woff + s]);
        gs[woff + s] = __float2bfloat16(av * (g - ts));
    }
}

__global__ void __launch_bounds__(256, 4) cv2bf16(__nv_bfloat16* o, const float* i, int n) {
    int x = blockIdx.x * 256 + threadIdx.x;
    if (x < n) o[x] = __float2bfloat16(i[x]);
}

void attention_backward_launcher(torch::Tensor& gs, torch::Tensor& gv, const torch::Tensor& go,
    const torch::Tensor& aw, const torch::Tensor& ad, const torch::Tensor& vs,
    const torch::Tensor& dm, float do_, cudaStream_t s) {
    const int B = go.size(0), sq = go.size(1), sv = vs.size(2);
    const float ds = do_ > 0.0f ? 1.0f / (1.0f - do_) : 1.0f;
    
    auto ga = torch::zeros({B, N_KV_HEADS, sv, H_DIM},
        torch::TensorOptions().dtype(torch::kFloat32).device(gv.device()));
    
    attn_bw_v30<<<dim3(B, N_HEADS, sq), 256, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(gs.data_ptr<at::BFloat16>()),
        ga.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(go.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(aw.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ad.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        dm.data_ptr<bool>(), B, sq, sv, ds);
    
    cv2bf16<<<(gv.numel() + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(gv.data_ptr<at::BFloat16>()),
        ga.data_ptr<float>(), gv.numel());
}
