#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void __launch_bounds__(256, 4) attn_backward_v27(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    float* __restrict__ grad_val_accum,
    const __nv_bfloat16* __restrict__ grad_attn_out,
    const __nv_bfloat16* __restrict__ attn_weights,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const __nv_bfloat16* __restrict__ value_states,
    const bool* __restrict__ dropout_mask,
    const int B, const int seq_q, const int seq_kv, const float dropout_scale
) {
    const int b = blockIdx.x, h = blockIdx.y, sq = blockIdx.z;
    if (b >= B || h >= NUM_HEADS || sq >= seq_q) return;
    
    const int kv_head = h / NUM_GROUPS;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    
    const int goff = ((b * seq_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const int woff = ((b * NUM_HEADS + h) * seq_q + sq) * seq_kv;
    const int voff = ((b * NUM_KV_HEADS + kv_head) * seq_kv) * HEAD_DIM;
    const int vacc = ((b * NUM_KV_HEADS + kv_head) * seq_kv) * HEAD_DIM;
    
    __shared__ float s_grad[HEAD_DIM], s_sum;
    
    for (int d = tid; d < HEAD_DIM; d += 256) s_grad[d] = __bfloat162float(grad_attn_out[goff + d]);
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    
    const int warps = 8;
    const int sk_per_warp = (seq_kv + warps - 1) / warps;
    const int sk_s = warp * sk_per_warp;
    const int sk_e = min(sk_s + sk_per_warp, seq_kv);
    
    float local_sum = 0.0f;
    
    for (int sk = sk_s + lane; sk < sk_e; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 16
        for (int d = 0; d < HEAD_DIM; ++d)
            dot += s_grad[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        local_sum += gw * __bfloat162float(attn_weights[woff + sk]);
        
        float awd = __bfloat162float(attn_weights_dropped[woff + sk]);
        #pragma unroll 16
        for (int d = 0; d < HEAD_DIM; ++d)
            atomicAdd(&grad_val_accum[vacc + sk * HEAD_DIM + d], awd * s_grad[d]);
    }
    
    local_sum = warp_reduce_sum(local_sum);
    __shared__ float s_ps[8];
    if (lane == 0) s_ps[warp] = local_sum;
    __syncthreads();
    
    if (warp == 0) {
        local_sum = (lane < 8) ? s_ps[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();
    
    float total = s_sum;
    
    for (int sk = sk_s + lane; sk < sk_e; sk += 32) {
        float dot = 0.0f;
        #pragma unroll 16
        for (int d = 0; d < HEAD_DIM; ++d)
            dot += s_grad[d] * __bfloat162float(value_states[voff + sk * HEAD_DIM + d]);
        bool m = dropout_mask[woff + sk];
        float gw = m ? dot * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total));
    }
}

__global__ void __launch_bounds__(256) convert_bf16(__nv_bfloat16* out, const float* in, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

void attention_backward_launcher(
    torch::Tensor& grad_attn_scores, torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output, const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped, const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask, float attention_dropout, cudaStream_t stream
) {
    const int B = grad_attn_output.size(0), seq_q = grad_attn_output.size(1), seq_kv = value_states.size(2);
    float dropout_scale = attention_dropout > 0.0f ? 1.0f / (1.0f - attention_dropout) : 1.0f;
    
    auto grad_val_acc = torch::zeros({B, NUM_KV_HEADS, seq_kv, HEAD_DIM},
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_value_states.device()));
    
    attn_backward_v27<<<dim3(B, NUM_HEADS, seq_q), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_val_acc.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(), B, seq_q, seq_kv, dropout_scale);
    
    convert_bf16<<<(grad_value_states.numel() + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        grad_val_acc.data_ptr<float>(), grad_value_states.numel());
}
