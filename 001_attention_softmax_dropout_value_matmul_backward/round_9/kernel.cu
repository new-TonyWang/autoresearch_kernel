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

// Use 32-bit atomic add via reinterpret cast to unsigned short (16-bit aligned)
// Read-modify-write pattern for bfloat16
__device__ __forceinline__ void atomicAdd_bfloat16_aligned(__nv_bfloat16* address, float val) {
    unsigned short int* ptr = (unsigned short int*)address;
    unsigned int old_val;
    unsigned int assumed;
    unsigned int new_val;
    
    __nv_bfloat16 bf16_val_old;
    float float_val_new;
    __nv_bfloat16 bf16_val_new;
    
    // AtomicCAS for 16-bit value (needs to be 32-bit aligned address for CUDA atomic)
    // Use CAS pattern with proper alignment check
    old_val = *ptr | (*(ptr + 1) << 16);
    do {
        assumed = old_val;
        bf16_val_old = __ushort_as_bfloat16((unsigned short)old_val);
        float_val_new = __bfloat162float(bf16_val_old) + val;
        bf16_val_new = __float2bfloat16(float_val_new);
        new_val = (unsigned short)__bfloat16_as_ushort(bf16_val_new) | (old_val & 0xFFFF0000);
        old_val = atomicCAS((unsigned int*)(((uintptr_t)ptr) & ~0x3), assumed, new_val);
    } while (assumed != old_val);
}

// Better approach - store as float and convert at the end
// Each block accumulates into shared memory, then one thread does atomic writes
__global__ void __launch_bounds__(256, 4) attention_backward_v9(
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
    
    // Shared memory for inputs and reduction
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_sum;
    __shared__ float s_grad_val[HEAD_DIM];  // Accumulate grad_value in staging area
    
    // Load grad_output
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) {
        s_sum = 0.0f;
        // Initialize grad_val staging
        for (int d = 0; d < HEAD_DIM; ++d) s_grad_val[d] = 0.0f;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    
    // Process seq_kv in parallel
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
        
        // Dot product grad_out @ value[sk]
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        // Dropout
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        
        // For softmax grad
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
        
        // Accumulate grad_value in shared memory
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        for (int d = tid; d < HEAD_DIM; d += 256) {
            atomicAdd(&s_grad_val[d], awd * s_grad_out[d]);
        }
    }
    
    // Reduce sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
    // Recompute and write grad_attn_scores
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
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
    
    // Write grad_value to global memory
    for (int d = tid; d < HEAD_DIM; d += 256) {
        if (s_grad_val[d] != 0.0f) {
            grad_value_states[voff + d] = __float2bfloat16(s_grad_val[d]);
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
    
    attention_backward_v9<<<grid, block, 0, stream>>>(
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
