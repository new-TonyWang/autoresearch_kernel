#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;  // 8

// Warp-level sum reduction (all lanes get the result via xor)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-cooperative 128-dim dot product with coalesced memory access.
// All 32 lanes cooperate: each lane reads 4 consecutive bf16 elements.
// Lane k reads elements [k*4, k*4+3] -> perfectly coalesced across the warp.
// Returns the dot product sum in ALL lanes (via xor reduce).
__device__ __forceinline__ float warp_dot_128(
    const float* __restrict__ s_vec,
    const __nv_bfloat16* __restrict__ g_vec,
    int lane
) {
    int base = lane * 4;
    __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&g_vec[base]);
    __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&g_vec[base + 2]);
    float2 f01 = __bfloat1622float2(v01);
    float2 f23 = __bfloat1622float2(v23);

    float partial = s_vec[base]     * f01.x + s_vec[base + 1] * f01.y +
                    s_vec[base + 2] * f23.x + s_vec[base + 3] * f23.y;

    return warp_reduce_sum(partial);
}

// Three-phase fused kernel for grad_attn_scores:
//   Phase 1: Warp-cooperative dot products (coalesced value_states reads)
//            -> store results in shared memory
//   Phase 2: Apply dropout mask + accumulate softmax sum_term
//            -> all 256 threads participate (coalesced mask/weight reads)
//   Phase 3: Compute softmax backward + write grad_attn_scores
//            -> all 256 threads participate (coalesced writes)
//
// NO atomicAdd anywhere. grad_value is computed separately via torch::matmul.
__global__ void __launch_bounds__(256, 4)
compute_grad_attn_scores_kernel(
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
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Use long long to avoid int32 overflow for large workloads
    const long long goff = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const long long voff = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;

    // Dynamic shared memory layout:
    //   [0, HEAD_DIM)                          : s_grad_out  (128 floats)
    //   [HEAD_DIM, HEAD_DIM + seq_len_kv)      : s_dot       (seq_len_kv floats)
    //   [HEAD_DIM + seq_len_kv, ... + NUM_WARPS): s_warp_sum  (8 floats)
    extern __shared__ float s_data[];
    float* s_grad_out = s_data;
    float* s_dot      = s_data + HEAD_DIM;
    float* s_warp_sum = s_data + HEAD_DIM + seq_len_kv;

    // Load grad_output[b, sq, h, :] into shared memory (128 values, coalesced)
    if (tid < HEAD_DIM) {
        s_grad_out[tid] = __bfloat162float(grad_attn_output[goff + tid]);
    }
    __syncthreads();

    // ==================== Phase 1 ====================
    // Warp-cooperative dot products: each warp handles one sk at a time.
    // 32 lanes read 4 consecutive bf16 values each -> coalesced 256-byte access.
    for (int sk = warp_id; sk < seq_len_kv; sk += NUM_WARPS) {
        float dot = warp_dot_128(s_grad_out, &value_states[voff + (long long)sk * HEAD_DIM], lane);
        if (lane == 0) {
            s_dot[sk] = dot;
        }
    }
    __syncthreads();

    // ==================== Phase 2 ====================
    // Apply dropout + accumulate sum_term for softmax backward.
    // All 256 threads participate -> coalesced reads of dropout_mask and attn_weights.
    float local_sum = 0.0f;
    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float dot = s_dot[sk];
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        s_dot[sk] = gw;  // Store gw for Phase 3
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }

    // Block-wide reduction of local_sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        s_warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Inter-warp reduce in first warp
    float total_sum = 0.0f;
    if (tid < NUM_WARPS) total_sum = s_warp_sum[tid];
    if (tid < 32) {
        total_sum = warp_reduce_sum(total_sum);
        if (tid == 0) s_warp_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_warp_sum[0];

    // ==================== Phase 3 ====================
    // Compute softmax backward + write grad_attn_scores.
    // All 256 threads participate -> coalesced writes.
    for (int sk = tid; sk < seq_len_kv; sk += BLOCK_SIZE) {
        float gw = s_dot[sk];
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
    }
}

void compute_grad_attn_scores_launcher(
    torch::Tensor& grad_attn_scores,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    int batch_size, int seq_len_q, int seq_len_kv,
    float dropout_scale,
    cudaStream_t stream
) {
    dim3 grid(batch_size, NUM_HEADS, seq_len_q);
    dim3 block(BLOCK_SIZE);

    int smem_size = (HEAD_DIM + seq_len_kv + NUM_WARPS) * sizeof(float);

    cudaFuncSetAttribute(
        compute_grad_attn_scores_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    compute_grad_attn_scores_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
}
