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

__device__ __forceinline__ float4 load_float4(const void* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Ultra-optimized kernel v6
// Key optimizations:
// 1. Float4 (128-bit) loads for maximum bandwidth
// 2. Two-stage grad_value accumulation (shared -> global)
// 3. Better register allocation
__global__ void __launch_bounds__(128, 10) attention_backward_v6(
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
    
    // Compute offsets
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) << 7;  // * HEAD_DIM
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) << 7;
    
    // Shared memory layout
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_grad_val[HEAD_DIM];
    __shared__ float s_sum;
    
    // Load grad_output using float4 (4 floats = 128 bits)
    // Each thread loads 4 consecutive elements
    const float4* gptr_f4 = reinterpret_cast<const float4*>(grad_attn_output + goff);
    
    #pragma unroll
    for (int i = tid; i < HEAD_DIM / 4; i += 32) {
        float4 g4 = gptr_f4[i];
        int base = i << 2;
        s_grad_out[base] = g4.x;
        s_grad_out[base + 1] = g4.y;
        s_grad_out[base + 2] = g4.z;
        s_grad_out[base + 3] = g4.w;
    }
    __syncthreads();
    
    // Initialize grad_val accumulator
    #pragma unroll
    for (int d = tid; d < HEAD_DIM; d += 32) {
        s_grad_val[d] = 0.0f;
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Warp-based processing
    const int warps_per_block = 4;  // 128/32
    const int sk_per_warp = (seq_len_kv + warps_per_block - 1) / warps_per_block;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    float local_sum = 0.0f;
    
    // First pass: compute dot products and accumulate grad_value
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Load value row with float4
        const int vrow = voff + (sk << 7);
        
        float dot = 0.0f;
        
        #pragma unroll
        for (int d4 = 0; d4 < HEAD_DIM; d4 += 4) {
            float4 g4;
            g4.x = s_grad_out[d4];
            g4.y = s_grad_out[d4 + 1];
            g4.z = s_grad_out[d4 + 2];
            g4.w = s_grad_out[d4 + 3];
            
            // Manual unroll of bfloat16 loads
            float v0 = __bfloat162float(value_states[vrow + d4]);
            float v1 = __bfloat162float(value_states[vrow + d4 + 1]);
            float v2 = __bfloat162float(value_states[vrow + d4 + 2]);
            float v3 = __bfloat162float(value_states[vrow + d4 + 3]);
            
            dot += g4.x * v0 + g4.y * v1 + g4.z * v2 + g4.w * v3;
        }
        
        // Apply dropout
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // For softmax grad
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Accumulate grad_value
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        
        #pragma unroll
        for (int d4 = 0; d4 < HEAD_DIM; d4 += 4) {
            float4 g4;
            g4.x = s_grad_out[d4];
            g4.y = s_grad_out[d4 + 1];
            g4.z = s_grad_out[d4 + 2];
            g4.w = s_grad_out[d4 + 3];
            
            atomicAdd(&s_grad_val[d4], awd * g4.x);
            atomicAdd(&s_grad_val[d4 + 1], awd * g4.y);
            atomicAdd(&s_grad_val[d4 + 2], awd * g4.z);
            atomicAdd(&s_grad_val[d4 + 3], awd * g4.w);
        }
    }
    
    // Reduce local_sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Second pass: compute and write grad_attn_scores
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        // Recompute dot
        float dot = 0.0f;
        const int vrow = voff + (sk << 7);
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[vrow + d]);
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        float grad_score = aw * (gw - total_sum);
        
        grad_attn_scores[woff + sk] = __float2bfloat16(grad_score);
    }
    
    __syncthreads();
    
    // Write grad_value to global with atomicAdd
    for (int d = tid; d < HEAD_DIM; d += 128) {
        if (s_grad_val[d] != 0.0f) {
            atomicAdd(reinterpret_cast<float*>(grad_value_states + voff + d), s_grad_val[d]);
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
    
    attention_backward_v6<<<grid, block, 0, stream>>>(
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
