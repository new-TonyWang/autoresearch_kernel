#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

// Get cublas handle from PyTorch
static cublasHandle_t get_cublas_handle() {
    return at::cuda::getCurrentCUDABlasHandle();
}

// Warp reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

// Kernel for first part: grad_attn_weights_dropped = grad_output @ V^T
// This uses shared memory matrix multiply
__global__ void __launch_bounds__(256, 4) compute_grad_weights_kernel(
    __nv_bfloat16* __restrict__ grad_attn_weights,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ value_states,
    const bool* __restrict__ dropout_mask,
    const __nv_bfloat16* __restrict__ attn_weights,
    __nv_bfloat16* __restrict__ grad_attn_scores,
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
    const int voff_base = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;
    const int moff = woff;
    
    // Shared memory tiles
    __shared__ float s_grad_out[2][128];   // 2 rows of 128 floats
    __shared__ float s_value[128];         // 1 row of value
    __shared__ float s_result[1024];       // Results for seq_len_kv
    __shared__ float s_sum;
    
    // Load grad_output to shared memory
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[0][d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    __syncthreads();
    
    // Compute matmul: grad_weights[sk] = sum_d(grad_out[d] * value[sk, d])
    // Process in tiles
    float local_sum = 0.0f;
    
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 4) {
            // Prefetch value data
            float4 g = *reinterpret_cast<float4*>(&s_grad_out[0][d]);
            
            __nv_bfloat16 v01 = value_states[voff_base + sk * HEAD_DIM + d];
            __nv_bfloat16 v23 = value_states[voff_base + sk * HEAD_DIM + d + 2];
            
            dot += g.x * __bfloat162float(v01.x);
            dot += g.y * __bfloat162float(v01.y);
            dot += g.z * __bfloat162float(v23.x);
            dot += g.w * __bfloat162float(v23.y);
        }
        
        // Apply dropout mask
        bool mask = dropout_mask[moff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        s_result[sk] = gw;
        
        // Accumulate for softmax grad
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }
    
    // Reduce sum across block
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Compute softmax gradient and write output
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float gw = s_result[sk];
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
    }
}

// Kernel for grad_value computation
__global__ void __launch_bounds__(256, 4) compute_grad_value_kernel(
    __nv_bfloat16* __restrict__ grad_value_states,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv
) {
    const int b = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int sk = blockIdx.z;  // Each block handles one sk
    
    if (b >= batch_size || kv_head >= NUM_KV_HEADS || sk >= seq_len_kv) return;
    
    const int tid = threadIdx.x;
    
    const int voff = (b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM + sk * HEAD_DIM;
    
    // Each thread accumulates gradients from all heads in the group
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process all query heads belonging to this kv_head
    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int h = kv_head * NUM_GROUPS + g;
        
        for (int sq = 0; sq < seq_len_q; ++sq) {
            int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
            int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            
            float awd = __bfloat162float(attn_weights_dropped[woff]);
            
            // Process HEAD_DIM elements per thread
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = tid + i * 64;  // 64 threads * 4 = 256, but HEAD_DIM=128
                if (d < HEAD_DIM) {
                    float g = __bfloat162float(grad_attn_output[goff + d]);
                    accum[i] += awd * g;
                }
            }
        }
    }
    
    // Write results with atomic add
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = tid + i * 64;
        if (d < HEAD_DIM && accum[i] != 0.0f) {
            atomicAdd(reinterpret_cast<float*>(&grad_value_states[voff + d]), accum[i]);
        }
    }
}

// Fused kernel - most efficient
__global__ void __launch_bounds__(256, 4) attention_backward_fused(
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
    
    // Shared memory
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_sum;
    
    // Load grad_output
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    
    // First pass: compute grad_weights and accumulate sum
    // Each warp handles a subset of seq_kv for better memory coalescing
    const int sk_per_warp = (seq_len_kv + 7) / 8;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Dot product with value
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        // Apply dropout
        bool mask = dropout_mask[moff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // Accumulate for softmax
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Store grad in register for second pass (recompute to save registers)
    }
    
    // Reduce sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: write outputs
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Recompute dot
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        bool mask = dropout_mask[moff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
        
        // Accumulate grad_value
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float val = awd * s_grad_out[d];
            atomicAdd(reinterpret_cast<float*>(&grad_value_states[voff + sk * HEAD_DIM + d]), val);
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
    
    float dropout_scale = attention_dropout > 0.0f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    // Use fused kernel
    dim3 grid(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(256);
    
    attention_backward_fused<<<grid, block, 0, stream>>>(
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
