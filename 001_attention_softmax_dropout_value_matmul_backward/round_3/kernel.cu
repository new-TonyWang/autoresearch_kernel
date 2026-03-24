#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

// Optimized kernel: process multiple query positions per block for better SM utilization
// Grid: (batch, num_kv_heads, tiles)
// Each block handles: all query heads for a group, multiple seq_q positions
__global__ void __launch_bounds__(512, 2) attention_backward_kernel_optimized(
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
    const float dropout_scale,
    const int q_per_block  // Number of query positions processed per block
) {
    const int b = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tile = blockIdx.z;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int num_warps = 512 / 32;  // 16 warps
    
    const int sq_start = tile * q_per_block;
    const int sq_end = min(sq_start + q_per_block, seq_len_q);
    
    // Shared memory for grad_output - store for all heads in this group
    __shared__ float s_grad_out[16][HEAD_DIM];  // [q_per_block][HEAD_DIM]
    __shared__ float s_accum_value[HEAD_DIM];   // For grad_value accumulation
    
    // Zero out accumulation buffer
    for (int d = tid; d < HEAD_DIM; d += 512) {
        s_accum_value[d] = 0.0f;
    }
    __syncthreads();
    
    const int voff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    
    // Process each query position in the tile
    for (int sq = sq_start; sq < sq_end; ++sq) {
        // Each warp handles one head in this group
        const int group_idx = warp % NUM_GROUPS;
        const int h = kv_head * NUM_GROUPS + group_idx;
        
        if (h >= NUM_HEADS) continue;
        
        const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
        const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
        const int moff = woff;
        
        // Load grad_output into shared memory (cooperative)
        if (warp == 0) {
            for (int d = lane; d < HEAD_DIM; d += 32) {
                int idx = ((sq - sq_start) * HEAD_DIM + d);
                if (idx < 16 * HEAD_DIM) {
                    s_grad_out[sq - sq_start][d] = __bfloat162float(grad_attn_output[goff + d]);
                }
            }
        }
        __syncthreads();
        
        // First pass: compute grad_weights and reduce sum
        float local_sum = 0.0f;
        
        // Each thread in warp handles multiple seq_kv positions
        const int sk_per_thread = (seq_len_kv + 31) / 32;
        
        for (int i = 0; i < sk_per_thread; ++i) {
            int sk = lane + i * 32;
            if (sk < seq_len_kv) {
                // Dot product
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float g = s_grad_out[sq - sq_start][d];
                    float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
                    dot += g * v;
                }
                
                // Apply dropout
                bool mask = dropout_mask[moff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                
                // Accumulate for softmax grad
                float aw = __bfloat162float(attn_weights[woff + sk]);
                local_sum += gw * aw;
                
                // Store for second pass
                // Use registers to store - we'll recompute or store in shared
            }
        }
        
        // Reduce sum within warp
        float warp_sum = warp_reduce_sum(local_sum);
        
        __shared__ float s_warp_sums[16];
        if (lane == 0) {
            s_warp_sums[warp] = warp_sum;
        }
        __syncthreads();
        
        // Final reduction
        float total_sum = 0.0f;
        if (warp == 0) {
            for (int i = lane; i < num_warps; i += 32) {
                total_sum += s_warp_sums[i];
            }
            total_sum = warp_reduce_sum(total_sum);
        }
        __syncthreads();
        
        // Broadcast total_sum to all warps
        __shared__ float s_total;
        if (warp == 0 && lane == 0) {
            s_total = total_sum;
        }
        __syncthreads();
        total_sum = s_total;
        
        // Second pass: compute gradients and accumulate
        for (int i = 0; i < sk_per_thread; ++i) {
            int sk = lane + i * 32;
            if (sk < seq_len_kv) {
                // Recompute dot
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float g = s_grad_out[sq - sq_start][d];
                    float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
                    dot += g * v;
                }
                
                // Apply dropout
                bool mask = dropout_mask[moff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                
                // Softmax gradient
                float aw = __bfloat162float(attn_weights[woff + sk]);
                float grad_score = aw * (gw - total_sum);
                grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
                
                // Accumulate grad_value (atomically later)
                float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float val = awd * s_grad_out[sq - sq_start][d];
                    atomicAdd(&s_accum_value[d], val);
                }
            }
        }
        __syncthreads();
    }
    
    // Write accumulated grad_value to global memory
    for (int sk = tid / HEAD_DIM; sk < seq_len_kv; sk += 512 / HEAD_DIM) {
        int d = tid % HEAD_DIM;
        if (d < HEAD_DIM) {
            int idx = voff + sk * HEAD_DIM + d;
            float val = s_accum_value[d];
            atomicAdd(reinterpret_cast<float*>(&grad_value_states[idx]), val);
        }
    }
}

// Simpler but efficient kernel - focus on memory bandwidth
__global__ void __launch_bounds__(256, 4) attention_backward_kernel_simple(
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
    const int lane = tid % 32;
    const int warp = tid / 32;
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int moff = woff;
    
    // Load grad_output to shared memory
    __shared__ float s_grad_out[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    __syncthreads();
    
    // Compute grad_weights for all seq_kv
    __shared__ float s_grad_weights[1024];  // Assumes seq_len_kv <= 4096
    __shared__ float s_sum;
    
    float local_sum = 0.0f;
    
    // Each thread computes multiple positions
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        s_grad_weights[sk] = dot;
        
        // Apply dropout and accumulate
        bool mask = dropout_mask[moff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }
    
    // Reduce sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Compute output gradients
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = s_grad_weights[sk];
        bool mask = dropout_mask[moff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
        
        // Compute grad_value
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            int idx = voff + sk * HEAD_DIM + d;
            atomicAdd(reinterpret_cast<float*>(&grad_value_states[idx]), val);
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
    
    float dropout_scale = 1.0f;
    if (attention_dropout > 0.0f) {
        dropout_scale = 1.0f / (1.0f - attention_dropout);
    }
    
    // Use simple kernel for all cases - easier to tune
    dim3 grid(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(256);
    
    attention_backward_kernel_simple<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size,
        seq_len_q,
        seq_len_kv,
        dropout_scale
    );
}
