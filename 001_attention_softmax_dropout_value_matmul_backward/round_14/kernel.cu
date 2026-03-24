#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;  // 8
constexpr int V_TILE = 128;  // value_states rows per shared memory tile

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-cooperative 128-dim dot product from shared memory.
// s_vec: float[128] in shared memory (grad_out)
// s_val: __nv_bfloat16[128] in shared memory (value tile row)
// Returns dot product in ALL lanes.
__device__ __forceinline__ float warp_dot_smem(
    const float* __restrict__ s_vec,
    const __nv_bfloat16* __restrict__ s_val,
    int lane
) {
    int base = lane * 4;
    __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&s_val[base]);
    __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&s_val[base + 2]);
    float2 f01 = __bfloat1622float2(v01);
    float2 f23 = __bfloat1622float2(v23);

    float partial = s_vec[base]     * f01.x + s_vec[base + 1] * f01.y +
                    s_vec[base + 2] * f23.x + s_vec[base + 3] * f23.y;

    return warp_reduce_sum(partial);
}

// HEAD_TILE + shared memory tiling kernel.
//
// Grid: (batch * kv_heads, seq_q)
// Block: 256 threads
//
// Each block processes ONE (b, kv_head, sq) position for ALL 10 heads in the group.
// Value states are tiled in shared memory (V_TILE rows at a time) and reused across heads.
//
// Two-pass approach:
//   Pass 1: For each V tile, for each head → dot products + dropout + accumulate sum_term
//   Pass 2: For each V tile, for each head → recompute dots + write grad_scores
//
// This reduces L2 traffic by ~5x compared to round_12 by reusing value tiles across heads.
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
    // Decode grid indices
    const int bkv = blockIdx.x;  // batch * kv_heads combined
    const int sq  = blockIdx.y;

    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv % NUM_KV_HEADS;

    if (b >= batch_size || sq >= seq_len_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Value states offset for this (b, kv_head)
    const long long voff = ((long long)b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;

    // Dynamic shared memory layout:
    //   s_value_tile: V_TILE * HEAD_DIM __nv_bfloat16  (V_TILE * 256 bytes)
    //   s_grad_out:   HEAD_DIM floats                   (512 bytes)
    //   s_dot:        V_TILE floats                     (V_TILE * 4 bytes)
    //   s_head_sums:  NUM_GROUPS floats                 (40 bytes)
    //   s_warp_sum:   NUM_WARPS floats                  (32 bytes)
    extern __shared__ char s_raw[];
    __nv_bfloat16* s_value_tile = reinterpret_cast<__nv_bfloat16*>(s_raw);
    float* s_grad_out = reinterpret_cast<float*>(s_raw + V_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
    float* s_dot      = s_grad_out + HEAD_DIM;
    float* s_head_sums = s_dot + V_TILE;
    float* s_warp_sum  = s_head_sums + NUM_GROUPS;

    // Initialize head sums
    if (tid < NUM_GROUPS) s_head_sums[tid] = 0.0f;
    __syncthreads();

    // ==================== PASS 1: Accumulate sum_term for each head ====================
    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += V_TILE) {
        const int tile_size = min(V_TILE, seq_len_kv - tile_start);

        // Cooperatively load V tile into shared memory (coalesced from L2)
        const int tile_elems = tile_size * HEAD_DIM;
        for (int idx = tid; idx < tile_elems; idx += BLOCK_SIZE) {
            s_value_tile[idx] = value_states[voff + (long long)(tile_start * HEAD_DIM) + idx];
        }
        __syncthreads();

        // Process each head in the KV group
        for (int h_local = 0; h_local < NUM_GROUPS; ++h_local) {
            const int h = kv_head * NUM_GROUPS + h_local;
            const long long goff = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            const long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;

            // Load this head's grad_out
            if (tid < HEAD_DIM) {
                s_grad_out[tid] = __bfloat162float(grad_attn_output[goff + tid]);
            }
            __syncthreads();

            // Phase 1a: Warp-cooperative dot products → store in s_dot
            for (int sk_local = warp_id; sk_local < tile_size; sk_local += NUM_WARPS) {
                float dot = warp_dot_smem(s_grad_out, &s_value_tile[sk_local * HEAD_DIM], lane);
                if (lane == 0) {
                    s_dot[sk_local] = dot;
                }
            }
            __syncthreads();

            // Phase 1b: All threads apply dropout + accumulate (coalesced reads)
            float local_sum = 0.0f;
            for (int sk_local = tid; sk_local < tile_size; sk_local += BLOCK_SIZE) {
                const int sk = tile_start + sk_local;
                float dot = s_dot[sk_local];
                bool mask = dropout_mask[woff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + sk]);
                local_sum += gw * aw;
            }

            // Reduce across block
            local_sum = warp_reduce_sum(local_sum);
            if (lane == 0) s_warp_sum[warp_id] = local_sum;
            __syncthreads();

            float tile_sum = 0.0f;
            if (tid < NUM_WARPS) tile_sum = s_warp_sum[tid];
            if (tid < 32) {
                tile_sum = warp_reduce_sum(tile_sum);
                if (tid == 0) s_head_sums[h_local] += tile_sum;
            }
            __syncthreads();
        }
        // syncthreads before next tile load
    }

    // ==================== PASS 2: Write grad_attn_scores ====================
    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += V_TILE) {
        const int tile_size = min(V_TILE, seq_len_kv - tile_start);

        // Reload V tile into shared memory
        const int tile_elems = tile_size * HEAD_DIM;
        for (int idx = tid; idx < tile_elems; idx += BLOCK_SIZE) {
            s_value_tile[idx] = value_states[voff + (long long)(tile_start * HEAD_DIM) + idx];
        }
        __syncthreads();

        for (int h_local = 0; h_local < NUM_GROUPS; ++h_local) {
            const int h = kv_head * NUM_GROUPS + h_local;
            const long long goff = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            const long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;

            // Reload grad_out
            if (tid < HEAD_DIM) {
                s_grad_out[tid] = __bfloat162float(grad_attn_output[goff + tid]);
            }
            __syncthreads();

            // Phase 2a: Warp-cooperative dot products (recompute)
            for (int sk_local = warp_id; sk_local < tile_size; sk_local += NUM_WARPS) {
                float dot = warp_dot_smem(s_grad_out, &s_value_tile[sk_local * HEAD_DIM], lane);
                if (lane == 0) {
                    s_dot[sk_local] = dot;
                }
            }
            __syncthreads();

            // Phase 2b: All threads compute + write grad_scores (coalesced writes)
            const float total_sum = s_head_sums[h_local];
            for (int sk_local = tid; sk_local < tile_size; sk_local += BLOCK_SIZE) {
                const int sk = tile_start + sk_local;
                float dot = s_dot[sk_local];
                bool mask = dropout_mask[woff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + sk]);
                grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
            }
            __syncthreads();
        }
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
    dim3 grid(batch_size * NUM_KV_HEADS, seq_len_q);
    dim3 block(BLOCK_SIZE);

    // Shared memory: V_TILE*HEAD_DIM bf16 + HEAD_DIM float + V_TILE float + GROUPS float + WARPS float
    int smem_size = V_TILE * HEAD_DIM * sizeof(__nv_bfloat16)
                  + HEAD_DIM * sizeof(float)
                  + V_TILE * sizeof(float)
                  + NUM_GROUPS * sizeof(float)
                  + NUM_WARPS * sizeof(float);

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
