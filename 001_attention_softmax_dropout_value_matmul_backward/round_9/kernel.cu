#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Round 9: Warp coalescing - each warp processes contiguous seq_kv elements
__global__ void __launch_bounds__(256, 4) attn_bw_v9(
    __nv_bfloat16* __restrict__ gs, float* __restrict__ gv,
    const __nv_bfloat16* __restrict__ go, const __nv_bfloat16* __restrict__ aw,
    const __nv_bfloat16* __restrict__ ad, const __nv_bfloat16* __restrict__ vs,
    const bool* __restrict__ dm, const int B, const int seq_q, const int seq_kv,
    const float ds) {
    
    const int b = blockIdx.x, h = blockIdx.y, q = blockIdx.z;
    if (b >= B || h >= NUM_HEADS || q >= seq_q) return;
    
    const int kh = h / NUM_GROUPS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int goff = ((b * seq_q + q) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_q + q) * seq_kv;
    const int voff = ((b * NUM_KV_HEADS + kh) * seq_kv) * HEAD_DIM;
    
    __shared__ float sg[HEAD_DIM], ss;
    #pragma unroll 4 for (int d = tid; d < HEAD_DIM; d += 256) sg[d] = __bfloat162float(go[goff + d]);
    if (!tid) ss = 0.0f;
    __syncthreads();
    
    // Warp coalescing: each warp processes contiguous elements
    const int warps = 256 / WARP_SIZE;
    const int per_warp = (seq_kv + warps - 1) / warps;
    const int s0 = warp * per_warp;
    const int s1 = min(s0 + per_warp, seq_kv);
    
    float ls = 0.0f;
    
    // Each warp processes its contiguous range
    for (int sk = s0 + lane; sk < s1; sk += WARP_SIZE) {
        float d = 0.0f;
        #pragma unroll 16
        for (int i = 0; i < HEAD_DIM; ++i) d += sg[i] * __bfloat162float(vs[voff + sk * HEAD_DIM + i]);
        
        bool m = dm[woff + sk];
        float g = m ? d * ds : 0.0f;
        float a = __bfloat162float(aw[woff + sk]);
        ls += g * a;
        
        float av = __bfloat162float(ad[woff + sk]);
        #pragma unroll 16
        for (int i = 0; i < HEAD_DIM; ++i) atomicAdd(&gv[voff + sk * HEAD_DIM + i], av * sg[i]);
    }
    
    // Block reduction
    ls = warp_sum(ls);
    __shared__ float sp[8];
    if (!lane) sp[warp] = ls;
    __syncthreads();
    
    if (!warp) {
        ls = (lane < 8) ? sp[lane] : 0.0f;
        ls = warp_sum(ls);
        if (!lane) ss = ls;
    }
    __syncthreads();
    
    float ts = ss;
    
    // Second pass
    for (int sk = s0 + lane; sk < s1; sk += WARP_SIZE) {
        float d = 0.0f;
        #pragma unroll 16
        for (int i = 0; i < HEAD_DIM; ++i) d += sg[i] * __bfloat162float(vs[voff + sk * HEAD_DIM + i]);
        
        bool m = dm[woff + sk];
        float g = m ? d * ds : 0.0f;
        float a = __bfloat162float(aw[woff + sk]);
        gs[woff + sk] = __float2bfloat16(a * (g - ts));
    }
}

void attention_backward_launcher(torch::Tensor& gs, torch::Tensor& gv_f32, const torch::Tensor& go,
    const torch::Tensor& aw, const torch::Tensor& ad, const torch::Tensor& vs,
    const torch::Tensor& dm, float do_, cudaStream_t stream) {
    const int B = go.size(0), seq_q = go.size(1), seq_kv = vs.size(2);
    gv_f32.zero_();
    float ds = do_ > 0.0f ? 1.0f / (1.0f - do_) : 1.0f;
    attn_bw_v9<<<dim3(B, NUM_HEADS, seq_q), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(gs.data_ptr<at::BFloat16>()),
        gv_f32.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(go.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(aw.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ad.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        dm.data_ptr<bool>(), B, seq_q, seq_kv, ds);
}
