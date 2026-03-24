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

// Kernel for grad_attn_scores only
__global__ void __launch_bounds__(256, 4) compute_grad_scores(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ attn_weights,
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
    __shared__ float s_sum;
    
    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            float v = __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
            dot += s_grad_out[d] * v;
        }
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }
    
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float total_sum = s_sum;
    
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
}

// Kernel for grad_value using float accumulation array
// Store intermediate in float shared memory, then convert to bfloat16
__global__ void __launch_bounds__(256, 4) compute_grad_value(
    __nv_bfloat16* __restrict__ grad_value_states,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv
) {
    const int b = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int sk = blockIdx.z;
    
    if (b >= batch_size || kv_head >= NUM_KV_HEADS || sk >= seq_len_kv) return;
    
    const int tid = threadIdx.x;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk) * HEAD_DIM;
    
    // Each thread handles part of HEAD_DIM
    float grad_val[HEAD_DIM / 256 + 1];
    for (int i = 0; i < HEAD_DIM / 256 + 1; ++i) grad_val[i] = 0.0f;
    
    // Accumulate across all query heads and seq_q positions
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int h = kv_head * NUM_GROUPS + g;
        
        for (int sq = 0; sq < seq_len_q; ++sq) {
            int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
            int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            
            float awd = __bfloat162float(attn_weights_dropped[woff]);
            
            for (int d = tid; d < HEAD_DIM; d += 256) {
                float g = __bfloat162float(grad_attn_output[goff + d]);
                grad_val[d / 256] += awd * g;
            }
        }
    }
    
    // Write results using CAS-based atomic add for bfloat16
    for (int d = tid; d < HEAD_DIM; d += 256) {
        float val = grad_val[d / 256];
        if (val != 0.0f) {
            // Proper atomicCAS implementation for bfloat16
            unsigned short* ptr = reinterpret_cast<unsigned short*>(&grad_value_states[voff + d]);
            unsigned short old = *ptr;
            unsigned short assumed;
            do {
                assumed = old;
                __nv_bfloat16 old_bf16 = __ushort_as_bfloat16(old);
                float old_f = __bfloat162float(old_bf16);
                float new_f = old_f + val;
                __nv_bfloat16 new_bf16 = __float2bfloat16(new_f);
                unsigned short new_val = __bfloat16_as_ushort(new_bf16);
                old = atomicCAS(ptr, assumed, new_val);
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
    
    // Kernel 1: grad_attn_scores
    dim3 grid1(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(256);
    
    compute_grad_scores<<<grid1, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
    
    // Kernel 2: grad_value_states
    dim3 grid2(batch_size, NUM_KV_HEADS, seq_len_kv);
    
    compute_grad_value<<<grid2, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
}
