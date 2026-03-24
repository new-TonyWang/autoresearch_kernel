#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// ROUND 10: Debugging version to fix numerical issues
__global__ void __launch_bounds__(256, 4) attn_bw_v10(
    __nv_bfloat16* __restrict__ gs, float* __restrict__ gv,
    const __nv_bfloat16* __restrict__ go, const __nv_bfloat16* __restrict__ aw,
    const __nv_bfloat16* __restrict__ ad, const __nv_bfloat16* __restrict__ vs,
    const bool* __restrict__ dm, const int B, const int seq_q, const int seq_kv,
    const float ds) {
    
    const int b = blockIdx.x, h = blockIdx.y, q = blockIdx.z;
    if (b >= B || h >= NUM_HEADS || q >= seq_q) return;
    
    const int kh = h / NUM_GROUPS;
    const int tid = threadIdx.x, lane = tid & 31;
    
    const int goff = ((b * seq_q + q) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_q + q) * seq_kv;
    const int voff = ((b * NUM_KV_HEADS + kh) * seq_kv) * HEAD_DIM;
    
    __shared__ float sg[HEAD_DIM], ss;
    // Load grad_output
    #pragma unroll 4 for (int d = tid; d < HEAD_DIM; d += 256) sg[d] = __bfloat162float(go[goff + d]);
    if (!tid) ss = 0.0f;
    __syncthreads();
    
    float ls = 0.0f;
    
    // SIMPLIFIED: Single pass with strided access
    for (int sk = tid; sk < seq_kv; sk += 256) {
        float dot = 0.0f;
        #pragma unroll 16 for (int i = 0; i < HEAD_DIM; ++i) dot += sg[i] * __bfloat162float(vs[voff + sk * HEAD_DIM + i]);
        
        bool m = dm[woff + sk];
        float g = m ? dot * ds : 0.0f;
        float a = __bfloat162float(aw[woff + sk]);
        ls += g * a;
        
        float av = __bfloat162float(ad[woff + sk]);
        #pragma unroll 16 for (int i = 0; i < HEAD_DIM; ++i) atomicAdd(&gv[voff + sk * HEAD_DIM + i], av * sg[i]);
    }
    
    // Reduce
    ls = warp_sum(ls);
    if (!lane) atomicAdd(&ss, ls);
    __syncthreads();
    
    float ts = ss;
    
    // Second pass for grad_scores
    for (int sk = tid; sk < seq_kv; sk += 256) {
        float dot = 0.0f;
        #pragma unroll 16 for (int i = 0; i < HEAD_DIM; ++i) dot += sg[i] * __bfloat162float(vs[voff + sk * HEAD_DIM + i]);
        
        bool m = dm[woff + sk];
        float g = m ? dot * ds : 0.0f;
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
    attn_bw_v10<<<dim3(B, NUM_HEADS, seq_q), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(gs.data_ptr<at::BFloat16>()),
        gv_f32.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(go.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(aw.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ad.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        dm.data_ptr<bool>(), B, seq_q, seq_kv, ds);
}
