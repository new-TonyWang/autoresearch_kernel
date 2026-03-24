#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

// Block-level reduce sum using warp shuffles
__device__ __forceinline__ float block_reduce_sum(float val) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    
    __shared__ float s_warp_sums[32];
    if (lane_id == 0) s_warp_sums[warp_id] = val;
    __syncthreads();
    
    // Final reduce
    val = (lane_id < blockDim.x >> 5) ? s_warp_sums[lane_id] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Highly optimized fused kernel with template parameters
template<int BLOCK_SIZE, int WARPS_PER_BLOCK>
__global__ void attention_backward_v15(
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
    
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_sum;
    
    // Vectorized load (coalesced)
    #pragma unroll
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Warp distribution
    const int sk_per_warp = (seq_len_kv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    // First pass: compute dot products and accumulate
    float local_sum = 0.0f;
    float cache[8];  // Cache dot products for reuse
    int cache_idx = 0;
    
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 16
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        }
        cache[cache_idx++ & 7] = dot;
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        local_sum += gw * __bfloat162float(attn_weights[woff + sk]);
    }
    
    // Block-reduce sum
    local_sum = block_reduce_sum(local_sum);
    if (tid == 0) s_sum = local_sum;
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: write outputs
    cache_idx = 0;
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        float dot = cache[cache_idx++ & 7];
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
        
        // Grad value - accumulate with CAS atomic (optimized for bfloat16)
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            // Use 16-bit atomicCAS
            unsigned short* ptr = reinterpret_cast<unsigned short*>(&grad_value_states[voff + sk * HEAD_DIM + d]);
            unsigned short old = *ptr;
            unsigned short assumed;
            do {
                assumed = old;
                __nv_bfloat16 old_bf16 = __ushort_as_bfloat16(old);
                float old_f = __bfloat162float(old_bf16);
                float new_f = old_f + val;
                __nv_bfloat16 new_bf16 = __float2bfloat16(new_f);
                unsigned short new_u = __bfloat16_as_ushort(new_bf16);
                old = atomicCAS(ptr, assumed, new_u);
            } while (assumed != old);
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
    
    attention_backward_v15<128, 4><<<grid, block, 0, stream>>>(
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
