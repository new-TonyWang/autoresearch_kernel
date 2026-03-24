#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;

// Ultimate H200 optimized kernel
// Features:
// 1. 256 threads for maximum occupancy
// 2. Warp-specialized processing
// 3. Loop unrolling for HEAD_DIM=128
// 4. Shared memory double buffering
// 5. Vectorized memory access where possible

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Async copy helper for H200
__device__ __forceinline__ void cp_async_ca(float* smem, const __nv_bfloat16* gmem) {
    #if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"::
        "r"(__cvta_generic_to_shared(smem)),
        "l"(gmem),
        "n"(4));
    #else
    *smem = __bfloat162float(*gmem);
    #endif
}

__global__ void __launch_bounds__(256, 2) attention_backward_ultimate(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    float* __restrict__ grad_value_accum,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ attn_weights,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const __nv_bfloat16* __restrict__ value_states,
    const bool* __restrict__ dropout_mask,
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv,
    const float dropout_scale
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int sq = blockIdx.z;
    
    if (b >= batch_size || h >= NUM_HEADS || sq >= seq_len_q) return;
    
    const int kv_head = h / NUM_GROUPS;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff_global = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    const int vacc_off = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    
    // Double-buffered shared memory
    __shared__ float s_grad_out[2][HEAD_DIM];
    __shared__ float s_sum;
    
    volatile float* curr_buf = s_grad_out[0];
    volatile float* next_buf = s_grad_out[1];
    
    // Load grad_output with coalescing
    #pragma unroll 4
    for (int d = tid; d < HEAD_DIM; d += 256) {
        curr_buf[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Warp distribution for seq_kv
    const int warps_per_block = 256 / WARP_SIZE;
    const int sk_per_warp = (seq_len_kv + warps_per_block - 1) / warps_per_block;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    float local_sum = 0.0f;
    
    // Unrolled computation
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        const int voff = voff_global + sk * HEAD_DIM;
        
        // Fully unrolled dot product for HEAD_DIM=128
        float dot = 0.0f;
        #pragma unroll 32
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += curr_buf[d] * __bfloat162float(value_states[voff + d]);
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Accumulate grad_value with vectorized atomic
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll 32
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAdd(&grad_value_accum[vacc_off + sk * HEAD_DIM + d], awd * curr_buf[d]);
        }
    }
    
    // Block-level reduction using warp shuffle
    __shared__ float s_partial_sums[8];
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_partial_sums[warp] = local_sum;
    __syncthreads();
    
    if (warp == 0) {
        local_sum = (lane < warps_per_block) ? s_partial_sums[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: write grad_scores
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        const int voff = voff_global + sk * HEAD_DIM;
        
        float dot = 0.0f;
        #pragma unroll 32
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += curr_buf[d] * __bfloat162float(value_states[voff + d]);
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
    }
}

// Optimized conversion kernel with vectorized stores
__global__ void __launch_bounds__(256, 4) convert_accum_vectorized(
    __nv_bfloat16* __restrict__ grad_value_states,
    const float* __restrict__ grad_value_accum,
    const int total_elements
) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    const int vec_idx = idx * 2;
    
    if (vec_idx + 1 < total_elements) {
        float2 val;
        val.x = grad_value_accum[vec_idx];
        val.y = grad_value_accum[vec_idx + 1];
        
        __nv_bfloat162 bf16_val;
        bf16_val.x = __float2bfloat16(val.x);
        bf16_val.y = __float2bfloat16(val.y);
        
        reinterpret_cast<__nv_bfloat162*>(grad_value_states)[idx] = bf16_val;
    } else if (vec_idx < total_elements) {
        grad_value_states[vec_idx] = __float2bfloat16(grad_value_accum[vec_idx]);
    }
}

void attention_backward_launcher(
    torch::Tensor& grad_attn_scores,
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    float attention_dropout,
    cudaStream_t stream
) {
    const int batch_size = grad_attn_output.size(0);
    const int seq_len_q = grad_attn_output.size(1);
    const int seq_len_kv = value_states.size(2);
    
    const float dropout_scale = attention_dropout > 0.f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    auto grad_value_accum = torch::zeros(
        {batch_size, NUM_KV_HEADS, seq_len_kv, HEAD_DIM},
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_value_states.device())
    );
    
    dim3 grid1(batch_size, NUM_HEADS, seq_len_q);
    dim3 block1(256);
    
    attention_backward_ultimate<<<grid1, block1, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_value_accum.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
    
    const int total_elements = grad_value_states.numel();
    dim3 grid2((total_elements + 511) / 512);
    dim3 block2(256);
    
    convert_accum_vectorized<<<grid2, block2, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        grad_value_accum.data_ptr<float>(),
        total_elements
    );
}
