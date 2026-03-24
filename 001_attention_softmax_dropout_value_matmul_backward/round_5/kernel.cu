#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

// Use __forceinline__ for performance
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized attention backward kernel for H200
// Block structure: (batch, head, seq_q) with 128 threads
// Strategy: maximize memory throughput and minimize register pressure
__global__ void __launch_bounds__(128, 8) attention_backward_optimized_v5(
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
    
    // Base pointers
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    
    // Shared memory - minimal usage
    __shared__ float s_grad_out[128];  // HEAD_DIM = 128
    __shared__ float s_sum;
    
    // Load grad_output vectorized
    const float* gptr = reinterpret_cast<const float*>(grad_attn_output + goff);
    float* sptr = s_grad_out;
    
    #pragma unroll
    for (int i = tid; i < HEAD_DIM >> 1; i += 64) {
        float2 g = reinterpret_cast<const float2*>(gptr)[i];
        int base = i << 1;
        sptr[base] = g.x;
        sptr[base + 1] = g.y;
    }
    __syncthreads();
    
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // First pass: compute dot products across seq_len_kv
    float local_sum = 0.0f;
    float grad_w_cache[4];  // Cache for second pass
    int cache_idx = 0;
    
    // Each warp processes consecutive elements for coalescing
    const int elems_per_warp = (seq_len_kv + 3) >> 2;
    const int warp_start = warp * elems_per_warp;
    const int warp_end = min(warp_start + elems_per_warp, seq_len_kv);
    
    for (int sk = warp_start + lane; sk < warp_end; sk += 32) {
        // Dot product with vectorization
        float dot = 0.0f;
        const __nv_bfloat16* vptr = value_states + voff + (sk * HEAD_DIM);
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 4) {
            float4 g;
            g.x = s_grad_out[d];
            g.y = s_grad_out[d + 1];
            g.z = s_grad_out[d + 2];
            g.w = s_grad_out[d + 3];
            
            float v0 = __bfloat162float(vptr[d]);
            float v1 = __bfloat162float(vptr[d + 1]);
            float v2 = __bfloat162float(vptr[d + 2]);
            float v3 = __bfloat162float(vptr[d + 3]);
            
            dot += g.x * v0 + g.y * v1 + g.z * v2 + g.w * v3;
        }
        
        // Apply dropout and accumulate
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Cache for second pass
        grad_w_cache[cache_idx & 3] = gw;
        cache_idx++;
    }
    
    // Reduce partial sums
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: write gradients
    cache_idx = 0;
    
    for (int sk = warp_start + lane; sk < warp_end; sk += 32) {
        float gw = grad_w_cache[cache_idx & 3];
        cache_idx++;
        
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
        
        // Update grad_value with atomicAdd
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        __nv_bfloat16* dvptr = grad_value_states + voff + (sk * HEAD_DIM);
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            atomicAdd(reinterpret_cast<float*>(dvptr + d), val);
        }
    }
}

// Version with improved grad_value accumulation
// Use separate kernel or shared memory reduction
__global__ void __launch_bounds__(256, 4) attention_backward_v5_improved(
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
    
    // Larger shared memory for grad_value accumulation
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_grad_val[HEAD_DIM];  // Local accumulation
    __shared__ float s_sum;
    
    // Load grad_output
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
        s_grad_val[d] = 0.0f;  // Initialize
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Process seq_len_kv in chunks
    // Each thread handles multiple sk positions
    float local_sum = 0.0f;
    
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        // Compute dot product
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        // Apply dropout
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // For softmax grad
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Accumulate grad_value (shared memory first)
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            atomicAdd(&s_grad_val[d], val);
        }
    }
    
    // Reduce sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Re-process for grad_attn_scores
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
    }
    
    __syncthreads();
    
    // Write grad_value_states
    for (int d = tid; d < HEAD_DIM; d += 256) {
        atomicAdd(reinterpret_cast<float*>(grad_value_states + voff + d), s_grad_val[d]);
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
    
    // Choose kernel based on workload characteristics
    if (seq_len_kv <= 512) {
        dim3 block(128);
        attention_backward_optimized_v5<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
            dropout_mask.data_ptr<bool>(),
            batch_size, seq_len_q, seq_len_kv, dropout_scale
        );
    } else {
        dim3 block(256);
        attention_backward_v5_improved<<<grid, block, 0, stream>>>(
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
}
