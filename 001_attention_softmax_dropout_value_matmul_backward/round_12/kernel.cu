#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized kernel v12 - focus on correctness and speed
// Key: Use float accumulation in registers, minimize shared memory
__global__ void __launch_bounds__(128, 8) attention_backward_v12(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    __nv_bfloat16* __restrict__ grad_value_states,
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
    const int lane = tid & 31;
    const int warp = tid >> 5;
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    
    // Load grad_output into shared memory (single load)
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_sum;
    
    #pragma unroll 4
    for (int d = tid; d < HEAD_DIM; d += 128) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each warp handles a subset of seq_len_kv
    const int warps_per_block = 4;  // 128/32
    const int sk_per_warp = (seq_len_kv + warps_per_block - 1) / warps_per_block;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    float local_sum = 0.0f;
    
    // First pass: compute dot products
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }
    
    // Reduce sum across warps
    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_partial_sums[4];
    if (lane == 0) s_partial_sums[warp] = local_sum;
    __syncthreads();
    
    if (tid < 4) {
        local_sum = s_partial_sums[tid];
    } else {
        local_sum = 0.0f;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (tid == 0) s_sum = local_sum;
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: write outputs
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Recompute dot
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
        
        // Accumulate grad_value
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            // Use 32-bit atomic add with address alignment check
            // This assumes value_states has 4-byte alignment
            float* ptr = reinterpret_cast<float*>(&grad_value_states[voff + sk * HEAD_DIM + d]);
            atomicAdd(ptr, val);
        }
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
    
    grad_value_states.zero_();
    
    const float dropout_scale = attention_dropout > 0.0f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    dim3 grid(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(128);
    
    attention_backward_v12<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
}
