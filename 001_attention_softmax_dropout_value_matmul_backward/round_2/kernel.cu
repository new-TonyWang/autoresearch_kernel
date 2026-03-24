#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Architecture specific constants for H200/B200
constexpr int WARP_SIZE = 32;
constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

// Vectorized bfloat16 load (2 elements)
__device__ __forceinline__ void load_bfloat16x2(const __nv_bfloat16* ptr, float& o1, float& o2) {
    __nv_bfloat16 v[2];
    v[0] = ptr[0];
    v[1] = ptr[1];
    o1 = __bfloat162float(v[0]);
    o2 = __bfloat162float(v[1]);
}

// Main kernel - highly optimized for attention backward
// Strategy: Process (batch, head, seq_q) per block with 128 threads
// Each thread processes multiple seq_kv elements sequentially
__global__ void __launch_bounds__(128, 8) attention_backward_kernel_v2(
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
    
    // Compute offsets
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int dvoff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int moff = woff;
    
    // Step 1: Load grad_attn_output into registers (vectorized)
    // Each thread loads HEAD_DIM/128 elements
    float grad_out[HEAD_DIM / 128 + 1];
    const int elems_per_thread = (HEAD_DIM + 127) / 128;
    
    #pragma unroll
    for (int i = 0; i < elems_per_thread; ++i) {
        int d = tid + i * 128;
        if (d < HEAD_DIM) {
            grad_out[i] = __bfloat162float(grad_attn_output[goff + d]);
        }
    }
    
    // Shared memory for cooperative reductions
    __shared__ float s_grad_weights[128];
    __shared__ float s_sum_softmax[4];  // One per warp
    __shared__ float s_sum_total;
    
    // Step 2: Process seq_len_kv elements
    // Each thread handles seq_len_kv/128 elements
    // Compute both grad_weights and accumulate grad_value
    
    const int sk_per_thread = (seq_len_kv + 127) / 128;
    
    // First compute grad_weights_dropped and reduce for softmax grad
    float local_sum = 0.0f;
    
    for (int sk = tid; sk < seq_len_kv; sk += 128) {
        // Compute dot product: grad_output @ value_states[sk]
        float dot = 0.0f;
        
        // Vectorized dot product
        #pragma unroll
        for (int d0 = 0; d0 < HEAD_DIM; d0 += 2) {
            int d = d0 + (lane % 2);  // Distribute across lanes
            if (d < HEAD_DIM) {
                float g = grad_out[d / 128];
                float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
                dot += g * v;
            }
        }
        
        // Full warp reduction for dot product
        dot = warp_reduce_sum(dot);
        
        // Store for later use
        if (lane == 0) {
            s_grad_weights[warp] = dot;
        }
        __syncthreads();
        
        // Load gradient weight and apply dropout
        float gw = s_grad_weights[warp];
        bool mask = dropout_scale > 1.0f ? dropout_mask[moff + sk] : true;
        gw = mask ? gw * dropout_scale : 0.0f;
        
        // Accumulate for softmax gradient: sum(gw * softmax)
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        __syncthreads();
    }
    
    // Reduce local_sum across all threads
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        s_sum_softmax[warp] = local_sum;
    }
    __syncthreads();
    
    // Final reduction of sum
    if (tid < 4) {
        local_sum = s_sum_softmax[tid];
    } else {
        local_sum = 0.0f;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (tid == 0) {
        s_sum_total = local_sum;
    }
    __syncthreads();
    
    float sum_total = s_sum_total;
    
    // Second pass: compute softmax gradient and grad_value
    // Process in chunks to optimize register usage
    for (int sk_base = 0; sk_base < seq_len_kv; sk_base += 128) {
        int sk = sk_base + tid;
        
        if (sk < seq_len_kv) {
            // Recompute grad weight
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float g = grad_out[d / 128];
                float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
                dot += g * v;
            }
            
            // Apply dropout
            bool mask = dropout_scale > 1.0f ? dropout_mask[moff + sk] : true;
            float gw = mask ? dot * dropout_scale : 0.0f;
            
            // Softmax gradient: grad = softmax * (gw - sum(gw * softmax))
            float aw = __bfloat162float(attn_weights[woff + sk]);
            float grad_score = aw * (gw - sum_total);
            grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
            
            // Accumulate grad_value: grad_value[sk] += attn_weights_dropped[sk] * grad_output
            // Use atomicAdd for accumulation across different h values
            float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float gval = grad_out[d / 128];
                float val = awd * gval;
                int idx = dvoff + sk * HEAD_DIM + d;
                atomicAdd(reinterpret_cast<float*>(&grad_value_states[idx]), val);
            }
        }
    }
}

// Optimized kernel v3 - uses warp-level parallelism better
// Each warp handles a subset of seq_kv, improving memory coalescing
__global__ void __launch_bounds__(256, 4) attention_backward_kernel_v3(
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
    const int num_warps = 8;  // 256/32
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int moff = woff;
    
    // Load grad_output into shared memory (cooperative load)
    __shared__ float s_grad_out[HEAD_DIM];
    
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    __syncthreads();
    
    // Each warp handles seq_len_kv/num_warps positions
    const int sk_per_warp = (seq_len_kv + num_warps - 1) / num_warps;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    // First pass: compute grad_weights and accumulate sum for softmax
    float warp_sum = 0.0f;
    
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Dot product with value_states
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        // Apply dropout
        bool mask = dropout_scale > 1.0f ? dropout_mask[moff + sk] : true;
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // Save for later
        float aw = __bfloat162float(attn_weights[woff + sk]);
        warp_sum += gw * aw;
        
        // Store in registers or shared mem for second pass
        // For now, recompute in second pass
    }
    
    // Reduce sum within warp
    warp_sum = warp_reduce_sum(warp_sum);
    
    __shared__ float s_warp_sums[8];
    if (lane == 0) {
        s_warp_sums[warp] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction
    float total_sum = 0.0f;
    if (tid < 8) {
        total_sum = s_warp_sums[tid];
    }
    total_sum = warp_reduce_sum(total_sum);
    
    __shared__ float s_total;
    if (tid == 0) {
        s_total = total_sum;
    }
    __syncthreads();
    
    total_sum = s_total;
    
    // Second pass: compute final gradients
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Recompute dot product
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        // Apply dropout
        bool mask = dropout_scale > 1.0f ? dropout_mask[moff + sk] : true;
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // Softmax gradient
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
        
        // Grad value accumulation
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
    
    const float dropout_scale = attention_dropout > 0.0f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    // Zero grad_value_states
    grad_value_states.zero_();
    
    // Choose kernel based on sequence length
    dim3 grid(batch_size, NUM_HEADS, seq_len_q);
    
    if (seq_len_kv <= 256) {
        // Use v3 with 256 threads for better parallelism on small seq_lens
        dim3 block(256);
        attention_backward_kernel_v3<<<grid, block, 0, stream>>>(
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
    } else {
        // Use v2 for larger sequence lengths
        dim3 block(128);
        attention_backward_kernel_v2<<<grid, block, 0, stream>>>(
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
}
