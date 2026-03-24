#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Fused dropout backward + softmax backward.
// Grid: (batch * num_heads, seq_q)
// Block: 256 threads, each handles ceil(seq_kv / 256) elements
//
// For each (b, h, sq) row:
//   Phase 1: gw[sk] = grad_aw_dropped[sk] * mask[sk] * scale   (dropout backward)
//            sum_term += gw[sk] * aw[sk]                        (accumulate)
//   Phase 2: reduce sum_term across block
//   Phase 3: grad_scores[sk] = aw[sk] * (gw[sk] - sum_term)    (softmax backward)
//
// gw and aw values stored in registers between phases (no second pass needed).
__global__ void __launch_bounds__(256, 8)
fused_dropout_softmax_backward_kernel(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    const float* __restrict__ grad_aw_dropped,
    const __nv_bfloat16* __restrict__ attn_weights,
    const bool* __restrict__ dropout_mask,
    const float dropout_scale,
    const int seq_len_kv
) {
    // Row index: (b * num_heads + h) * seq_q + sq
    const long long row = (long long)blockIdx.x * gridDim.y + blockIdx.y;
    const long long row_off = row * seq_len_kv;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_warp_sum[NUM_WARPS];

    // Phase 1: Compute gw, accumulate sum_term, store in registers
    float gw_regs[16];  // max ceil(4096/256) = 16
    float aw_regs[16];
    float local_sum = 0.0f;
    int n = 0;

    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float raw = grad_aw_dropped[row_off + sk];
        bool mask = dropout_mask[row_off + sk];
        float gw = mask ? raw * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[row_off + sk]);

        gw_regs[n] = gw;
        aw_regs[n] = aw;
        local_sum += gw * aw;
        n++;
    }

    // Phase 2: Block-wide reduction
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_warp_sum[warp_id] = local_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (tid < NUM_WARPS) total_sum = s_warp_sum[tid];
    if (tid < 32) {
        total_sum = warp_reduce_sum(total_sum);
        if (tid == 0) s_warp_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_warp_sum[0];

    // Phase 3: Compute and write grad_scores (coalesced writes)
    n = 0;
    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float grad_score = aw_regs[n] * (gw_regs[n] - total_sum);
        grad_attn_scores[row_off + sk] = __float2bfloat16(grad_score);
        n++;
    }
}

void fused_dropout_softmax_backward_launcher(
    torch::Tensor& grad_attn_scores,
    const torch::Tensor& grad_aw_dropped,
    const torch::Tensor& attn_weights,
    const torch::Tensor& dropout_mask,
    float dropout_scale,
    int batch_size, int num_heads, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
) {
    dim3 grid(batch_size * num_heads, seq_len_q);
    dim3 block(BLOCK_SIZE);

    fused_dropout_softmax_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_aw_dropped.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        dropout_scale,
        seq_len_kv
    );
}
