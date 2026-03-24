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

// Device-side atomicAdd for __nv_bfloat16
__device__ __forceinline__ void atomicAdd_bfloat16(__nv_bfloat16* ptr, float val) {
    unsigned int* ptr_as_uint = reinterpret_cast<unsigned int*>(ptr);
    unsigned int old = *ptr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        __nv_bfloat16 old_val = *reinterpret_cast<__nv_bfloat16*>(&assumed);
        float new_val = __bfloat162float(old_val) + val;
        __nv_bfloat16 new_bf16 = __float2bfloat16(new_val);
        old = atomicCAS(ptr_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_bf16));
    } while (assumed != old);
}

// Simpler: use separate accumulation in shared memory then write
template<int BLOCK_SIZE>
__global__ void attention_backward_v8(
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
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_grad_val[HEAD_DIM];
    __shared__ float s_sum;
    
    // Load grad_output
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Local accumulation for grad_value
    float local_grad_val[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        local_grad_val[d] = 0.0f;
    }
    
    float local_sum = 0.0f;
    
    // Process all seq_kv positions
    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float dot = 0.0f;
        
        #pragma unroll 8
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
        
        // Accumulate grad_value locally
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            local_grad_val[d] += awd * s_grad_out[d];
        }
    }
    
    // Reduce local_sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Re-process for grad_attn_scores
    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float dot = 0.0f;
        #pragma unroll 8
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
    
    // Write grad_value - use float shared memory first
    for (int d = 0; d < HEAD_DIM; ++d) {
        // Reduce grad_val across block
        float val = local_grad_val[d];
        val = warp_reduce_sum(val);
        
        if (lane == 0) {
            atomicAdd(&s_grad_val[d], val);
        }
    }
    __syncthreads();
    
    // Now write to global memory with proper casting
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        if (s_grad_val[d] != 0.0f) {
            // Use atomicCAS-based atomicAdd for bfloat16
            __nv_bfloat16* ptr = &grad_value_states[voff + d];
            unsigned int* ptr_as_uint = reinterpret_cast<unsigned int*>(ptr);
            unsigned int old = *ptr_as_uint;
            unsigned int assumed;
            do {
                assumed = old;
                __nv_bfloat16 old_val = *reinterpret_cast<__nv_bfloat16*>(&assumed);
                float new_val_f = __bfloat162float(old_val) + s_grad_val[d];
                __nv_bfloat16 new_bf16 = __float2bfloat16(new_val_f);
                old = atomicCAS(ptr_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_bf16));
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
    dim3 block(256);
    
    attention_backward_v8<256><<<grid, block, 0, stream>>>(
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
