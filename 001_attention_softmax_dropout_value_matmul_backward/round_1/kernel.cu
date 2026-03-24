#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for the kernel
constexpr int NUM_ATTENTION_HEADS = 80;
constexpr int NUM_KEY_VALUE_HEADS = 8;
constexpr int NUM_KEY_VALUE_GROUPS = 10;  // 80 / 8
constexpr int HEAD_DIM = 128;

// Warp-level reduction for sum
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Vectorized load/store helpers
struct BFloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

__device__ inline float2 bfloat162_to_float2(__nv_bfloat162 v) {
    float x = __bfloat162float(v.x);
    float y = __bfloat162float(v.y);
    return {x, y};
}

// Optimized kernel for attention backward
// Each thread block processes one (batch, head, seq_q) triplet
__global__ void __launch_bounds__(256, 2) attention_backward_kernel(
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
    const float attention_dropout,
    const float scale_dropout
) {
    const int b = blockIdx.x;  // Batch
    const int h = blockIdx.y;  // Head
    const int sq = blockIdx.z; // Query sequence
    
    if (b >= batch_size || h >= NUM_ATTENTION_HEADS || sq >= seq_len_q) return;
    
    const int kv_head = h / NUM_KEY_VALUE_GROUPS;
    const int tid = threadIdx.x;
    const int wg = threadIdx.x / 32;  // Warp group
    const int lid = threadIdx.x % 32; // Lane id
    
    // Shared memory
    __shared__ float s_grad_weights[256];
    __shared__ float s_sum[32];
    __shared__ float s_grad_value[HEAD_DIM];
    
    // Pointers with offsets
    const int grad_attn_output_offset = ((b * seq_len_q + sq) * NUM_ATTENTION_HEADS + h) * HEAD_DIM;
    const int attn_weights_offset = ((b * NUM_ATTENTION_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int value_states_offset = (b * NUM_KEY_VALUE_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int dropout_mask_offset = attn_weights_offset;
    
    // Step 1: Compute grad_value_states = attn_weights_dropped^T @ grad_attn_output
    // Each thread processes part of head_dim
    // We use cooperative loading to reduce memory traffic
    
    float grad_value_acc[4] = {0.0f};
    
    // Compute grad_attn_weights_dropped and grad_value_states simultaneously
    __shared__ float s_grad_attn_weights_dropped[256];
    
    // Process seq_len_kv in chunks
    for (int sk_start = 0; sk_start < seq_len_kv; sk_start += 256) {
        int sk = sk_start + tid;
        float grad_weight_dropped = 0.0f;
        
        if (sk < seq_len_kv) {
            // Load attn_weights_dropped
            __nv_bfloat16 awd = attn_weights_dropped[attn_weights_offset + sk];
            
            // Load dropout_mask
            bool mask = dropout_mask[dropout_mask_offset + sk];
            
            // Compute grad_attn_weights_dropped = grad_attn_output @ value_states^T
            // For this position: sum over head_dim
            float sum = 0.0f;
            
            // Load grad_attn_output for this seq_q
            float grad_out[4];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int did = lid * 4 + d;
                if (did < HEAD_DIM) {
                    grad_out[d] = __bfloat162float(grad_attn_output[grad_attn_output_offset + did]);
                }
            }
            
            // Load value_states and compute dot product
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int did = lid * 4 + d;
                if (did < HEAD_DIM) {
                    int v_offset = value_states_offset + sk * HEAD_DIM + did;
                    __nv_bfloat16 v_val = value_states[v_offset];
                    sum += grad_out[d] * __bfloat162float(v_val);
                }
            }
            
            sum = warp_reduce_sum(sum);
            
            // Store for gradient through dropout
            if (lid == 0) {
                s_grad_attn_weights_dropped[wg] = sum;
            }
            
            // Compute grad_value contribution for this sk
            // grad_value[sk, d] += attn_weights_dropped[sk] * grad_output[d]
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int did = lid * 4 + d;
                if (did < HEAD_DIM) {
                    grad_value_acc[d] += __bfloat162float(awd) * grad_out[d];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Accumulate grad_value across warps
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        int did = lid * 4 + d;
        if (did < HEAD_DIM) {
            float val = grad_value_acc[d];
            // Reduce across all threads
            val = warp_reduce_sum(val);
            if (tid == 0) {
                atomicAdd(&grad_value_states[(b * NUM_KEY_VALUE_HEADS + kv_head) * seq_len_kv * HEAD_DIM + did], 
                         __float2bfloat16(val));
            }
        }
    }
    
    // Step 2: Gradient through softmax
    // Process each seq_kv position
    for (int sk_start = 0; sk_start < seq_len_kv; sk_start += 256) {
        int sk = sk_start + tid;
        
        if (sk < seq_len_kv) {
            // Load values
            float grad_wd = s_grad_attn_weights_dropped[wg];
            bool mask = dropout_mask[dropout_mask_offset + sk];
            float aw = __bfloat162float(attn_weights[attn_weights_offset + sk]);
            
            // Gradient through dropout
            float grad_w = mask ? grad_wd * scale_dropout : 0.0f;
            if (attention_dropout == 0.0f) {
                grad_w = grad_wd;
            }
            
            // Gradient through softmax
            // grad_input = softmax * (grad_output - sum(grad_output * softmax))
            float weighted_grad = grad_w * aw;
            
            // Warp reduction to compute sum
            float sum_weighted = warp_reduce_sum(weighted_grad);
            if (lid == 0) {
                s_sum[wg] = sum_weighted;
            }
            __syncthreads();
            
            // Compute final gradient
            if (tid < 32) {
                sum_weighted = s_sum[tid];
            }
            sum_weighted = warp_reduce_sum(sum_weighted);
            
            float grad_score = aw * (grad_w - sum_weighted);
            
            // Store result
            grad_attn_scores[attn_weights_offset + sk] = __float2bfloat16(grad_score);
        }
        
        __syncthreads();
    }
}

// Simpler, optimized kernel focusing on fusion
__global__ void __launch_bounds__(128, 4) attention_backward_v2_kernel(
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
    const float attention_dropout,
    const float scale_dropout
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int sq = blockIdx.z;
    
    if (b >= batch_size || h >= NUM_ATTENTION_HEADS || sq >= seq_len_q) return;
    
    const int kv_head = h / NUM_KEY_VALUE_GROUPS;
    const int tid = threadIdx.x;
    
    const int goffset = ((b * seq_len_q + sq) * NUM_ATTENTION_HEADS + h) * HEAD_DIM;
    const int woffset = ((b * NUM_ATTENTION_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voffset = (b * NUM_KEY_VALUE_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    
    // Load grad_attn_output into registers (vectorized)
    float grad_out[HEAD_DIM];
    #pragma unroll
    for (int d = tid; d < HEAD_DIM; d += 128) {
        grad_out[d] = __bfloat162float(grad_attn_output[goffset + d]);
    }
    
    // Process seq_len_kv positions
    __shared__ float s_grad_weights[256];
    __shared__ float s_sum;
    
    // First pass: compute grad_weights_dropped and accumulate sum
    float local_sum = 0.0f;
    
    for (int sk = tid; sk < seq_len_kv; sk += 128) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            __nv_bfloat16 v = value_states[voffset + sk * HEAD_DIM + d];
            dot += grad_out[d] * __bfloat162float(v);
        }
        s_grad_weights[sk] = dot;
        
        // Gradient through dropout
        bool mask = dropout_mask[woffset + sk];
        float gw = mask ? dot * scale_dropout : 0.0f;
        if (attention_dropout == 0.0f) gw = dot;
        
        // Accumulate for softmax grad
        float aw = __bfloat162float(attn_weights[woffset + sk]);
        local_sum += gw * aw;
    }
    
    // Reduce sum across threads
    local_sum = warp_reduce_sum(local_sum);
    if (tid % 32 == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: compute softmax gradient
    for (int sk = tid; sk < seq_len_kv; sk += 128) {
        float gw = s_grad_weights[sk];
        
        // Gradient through dropout
        bool mask = dropout_mask[woffset + sk];
        if (attention_dropout > 0.0f) {
            gw = mask ? gw * scale_dropout : 0.0f;
        }
        
        float aw = __bfloat162float(attn_weights[woffset + sk]);
        float grad_score = aw * (gw - total_sum);
        
        grad_attn_scores[woffset + sk] = __float2bfloat16(grad_score);
    }
    
    // Compute grad_value_states
    // grad_value[kv_head, sk, d] += attn_weights_dropped[h, sq, sk] * grad_output[d]
    for (int sk = tid; sk < seq_len_kv; sk += 128) {
        float awd = __bfloat162float(attn_weights_dropped[woffset + sk]);
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * grad_out[d];
            int gvs_idx = voffset + sk * HEAD_DIM + d;
            atomicAdd(reinterpret_cast<float*>(&grad_value_states[gvs_idx]), val);
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
    // Get dimensions
    const int batch_size = grad_attn_output.size(0);
    const int seq_len_q = grad_attn_output.size(1);
    const int seq_len_kv = value_states.size(2);
    
    const float scale_dropout = attention_dropout > 0.0f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    // Zero grad_value_states
    grad_value_states.zero_();
    
    // Launch configuration
    dim3 grid(batch_size, NUM_ATTENTION_HEADS, seq_len_q);
    dim3 block(128);
    
    attention_backward_v2_kernel<<<grid, block, 0, stream>>>(
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
        attention_dropout,
        scale_dropout
    );
}
