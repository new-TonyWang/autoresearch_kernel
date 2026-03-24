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

// Kernel 1: grad_attn_scores only
__global__ void __launch_bounds__(128, 8) compute_grad_scores_v14(
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
    const int warp = tid >> 5;
    
    const int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;
    
    __shared__ float s_grad_out[HEAD_DIM];
    __shared__ float s_sum;
    
    for (int d = tid; d < HEAD_DIM; d += 128) {
        s_grad_out[d] = __bfloat162float(grad_attn_output[goff + d]);
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    const int warps_per_block = 4;
    const int sk_per_warp = (seq_len_kv + warps_per_block - 1) / warps_per_block;
    const int sk_start = warp * sk_per_warp;
    const int sk_end = min(sk_start + sk_per_warp, seq_len_kv);
    
    float local_sum = 0.0f;
    
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        }
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        local_sum += gw * __bfloat162float(attn_weights[woff + sk]);
    }
    
    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_partial[4];
    if (lane == 0) s_partial[warp] = local_sum;
    __syncthreads();
    
    if (warp == 0) {
        local_sum = (lane < 4) ? s_partial[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();
    
    float total_sum = s_sum;
    
    for (int sk = sk_start + lane; sk < sk_end; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; ++d) {
            dot += s_grad_out[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        }
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
    }
}

// Kernel 2: grad_value_states with float accumulation
__global__ void __launch_bounds__(128, 8) compute_grad_value_v14(
    float* __restrict__ grad_value_states_float,
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
    
    // Each thread accumulates partial results
    float grad_val[HEAD_DIM / 128 + 1] = {0.0f};
    
    for (int g = 0; g < NUM_GROUPS; ++g) {
        int h = kv_head * NUM_GROUPS + g;
        for (int sq = 0; sq < seq_len_q; ++sq) {
            int woff = ((b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
            int goff = ((b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            
            float awd = __bfloat162float(attn_weights_dropped[woff]);
            
            for (int d = tid; d < HEAD_DIM; d += 128) {
                float g = __bfloat162float(grad_attn_output[goff + d]);
                grad_val[d / 128] += awd * g;
            }
        }
    }
    
    // Write to float accumulation buffer
    for (int d = tid; d < HEAD_DIM; d += 128) {
        int idx = d / 128;
        if (grad_val[idx] != 0.0f) {
            atomicAdd(&grad_value_states_float[voff + d], grad_val[idx]);
        }
    }
}

// Kernel 3: Convert float buffer to bfloat16
__global__ void __launch_bounds__(256, 4) convert_grad_value_v14(
    __nv_bfloat16* __restrict__ grad_value_states,
    const float* __restrict__ grad_value_states_float,
    const int total_elements
) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < total_elements) {
        grad_value_states[idx] = __float2bfloat16(grad_value_states_float[idx]);
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
    
    // Create temporary float buffer
    auto grad_value_float = torch::zeros_like(grad_value_states, torch::dtype(torch::kFloat32));
    
    // Kernel 1: grad_attn_scores
    dim3 grid1(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(128);
    
    compute_grad_scores_v14<<<grid1, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
    
    // Kernel 2: grad_value_states accumulation (float)
    dim3 grid2(batch_size, NUM_KV_HEADS, seq_len_kv);
    
    compute_grad_value_v14<<<grid2, block, 0, stream>>>(
        grad_value_float.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
    
    // Kernel 3: Convert to bfloat16
    const int total_elements = grad_value_states.numel();
    dim3 grid3((total_elements + 255) / 256);
    dim3 block3(256);
    
    convert_grad_value_v14<<<grid3, block3, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        grad_value_float.data_ptr<float>(),
        total_elements
    );
}
