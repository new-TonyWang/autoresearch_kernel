#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Warp-cooperative 128-dim dot product from shared memory bf16 tile.
__device__ __forceinline__ float warp_dot_smem(
    const float* __restrict__ s_go,
    const __nv_bfloat16* __restrict__ s_v,
    int lane
) {
    int base = lane * 4;
    __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&s_v[base]);
    __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&s_v[base + 2]);
    float2 f01 = __bfloat1622float2(v01);
    float2 f23 = __bfloat1622float2(v23);
    float p = s_go[base]*f01.x + s_go[base+1]*f01.y + s_go[base+2]*f23.x + s_go[base+3]*f23.y;
    return warp_reduce_sum(p);
}

// =====================================================================
// KERNEL 1: Fused grad_attn_scores with SQ_TILE
//
// Processes SQ_TILE query positions per block with shared memory tiling
// of value_states. V tiles are loaded once and reused across SQ_TILE queries.
//
// Grid: (batch * NUM_HEADS, ceil(seq_q / SQ_TILE))
// Block: 256 threads
//
// Two-pass:
//   Pass 1: For each V tile, for each sq → dot products + accumulate sum_term
//   Pass 2: For each V tile, for each sq → recompute dots + write output
// =====================================================================

constexpr int K1_BLOCK = 256;
constexpr int K1_WARPS = K1_BLOCK / 32;
constexpr int SQ_TILE = 4;
constexpr int V_TILE = 128;  // rows of V per shared memory tile

__global__ void __launch_bounds__(256, 2)
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
    const int bh = blockIdx.x;
    const int sq_tile_idx = blockIdx.y;
    const int b = bh / NUM_HEADS;
    const int h = bh % NUM_HEADS;
    if (b >= batch_size) return;

    const int sq_start = sq_tile_idx * SQ_TILE;
    const int kv_head = h / NUM_GROUPS;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const long long voff = ((long long)b * NUM_KV_HEADS + kv_head) * seq_len_kv * HEAD_DIM;

    // Dynamic shared memory layout:
    //   s_vtile:    V_TILE * HEAD_DIM bf16        (V_TILE * 256 bytes)
    //   s_grad_out: SQ_TILE * HEAD_DIM float      (SQ_TILE * 512 bytes)
    //   s_dot:      SQ_TILE * V_TILE float         (SQ_TILE * V_TILE * 4 bytes)
    //   s_sums:     SQ_TILE float                  (SQ_TILE * 4 bytes)
    //   s_warp_sum: K1_WARPS float                 (K1_WARPS * 4 bytes)
    extern __shared__ char s_raw[];

    __nv_bfloat16* s_vtile = reinterpret_cast<__nv_bfloat16*>(s_raw);
    const int vtile_bytes = V_TILE * HEAD_DIM * sizeof(__nv_bfloat16);

    float* s_grad_out = reinterpret_cast<float*>(s_raw + vtile_bytes);
    float* s_dot      = s_grad_out + SQ_TILE * HEAD_DIM;
    float* s_sums     = s_dot + SQ_TILE * V_TILE;
    float* s_warp_sum = s_sums + SQ_TILE;

    // Load SQ_TILE grad_out vectors (loaded once, reused across V tiles)
    for (int sq_l = 0; sq_l < SQ_TILE; ++sq_l) {
        int sq = sq_start + sq_l;
        if (sq < seq_len_q && tid < HEAD_DIM) {
            long long goff = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
            s_grad_out[sq_l * HEAD_DIM + tid] = __bfloat162float(grad_attn_output[goff + tid]);
        }
    }
    // Init sums
    if (tid < SQ_TILE) s_sums[tid] = 0.0f;
    __syncthreads();

    // ==================== PASS 1: accumulate sum_term ====================
    for (int vt_start = 0; vt_start < seq_len_kv; vt_start += V_TILE) {
        int vt_size = min(V_TILE, seq_len_kv - vt_start);

        // Load V tile into shared memory (coalesced)
        for (int idx = tid; idx < vt_size * HEAD_DIM; idx += K1_BLOCK) {
            s_vtile[idx] = value_states[voff + (long long)(vt_start * HEAD_DIM) + idx];
        }
        __syncthreads();

        for (int sq_l = 0; sq_l < SQ_TILE; ++sq_l) {
            int sq = sq_start + sq_l;
            if (sq >= seq_len_q) continue;

            long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;

            // Warp-cooperative dot products
            for (int sk_l = warp_id; sk_l < vt_size; sk_l += K1_WARPS) {
                float dot = warp_dot_smem(&s_grad_out[sq_l * HEAD_DIM],
                                          &s_vtile[sk_l * HEAD_DIM], lane);
                if (lane == 0) s_dot[sq_l * V_TILE + sk_l] = dot;
            }
            __syncthreads();

            // All threads: dropout + accumulate
            float local_sum = 0.0f;
            for (int sk_l = tid; sk_l < vt_size; sk_l += K1_BLOCK) {
                int sk = vt_start + sk_l;
                float dot = s_dot[sq_l * V_TILE + sk_l];
                bool mask = dropout_mask[woff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + sk]);
                local_sum += gw * aw;
            }
            local_sum = warp_reduce_sum(local_sum);
            if (lane == 0) atomicAdd(&s_sums[sq_l], local_sum);
            __syncthreads();
        }
    }

    // ==================== PASS 2: write grad_scores ====================
    for (int vt_start = 0; vt_start < seq_len_kv; vt_start += V_TILE) {
        int vt_size = min(V_TILE, seq_len_kv - vt_start);

        // Reload V tile
        for (int idx = tid; idx < vt_size * HEAD_DIM; idx += K1_BLOCK) {
            s_vtile[idx] = value_states[voff + (long long)(vt_start * HEAD_DIM) + idx];
        }
        __syncthreads();

        for (int sq_l = 0; sq_l < SQ_TILE; ++sq_l) {
            int sq = sq_start + sq_l;
            if (sq >= seq_len_q) continue;

            long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
            float total_sum = s_sums[sq_l];

            // Recompute dots
            for (int sk_l = warp_id; sk_l < vt_size; sk_l += K1_WARPS) {
                float dot = warp_dot_smem(&s_grad_out[sq_l * HEAD_DIM],
                                          &s_vtile[sk_l * HEAD_DIM], lane);
                if (lane == 0) s_dot[sq_l * V_TILE + sk_l] = dot;
            }
            __syncthreads();

            // Write output
            for (int sk_l = tid; sk_l < vt_size; sk_l += K1_BLOCK) {
                int sk = vt_start + sk_l;
                float dot = s_dot[sq_l * V_TILE + sk_l];
                bool mask = dropout_mask[woff + sk];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + sk]);
                grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
            }
            __syncthreads();
        }
    }
}

// =====================================================================
// KERNEL 2: Grad value states (same as round 19)
// =====================================================================

constexpr int K2_BLOCK = 256;
constexpr int SK_TILE2 = 32;
constexpr int SQ_CHUNK = 8;
constexpr int K2_OUT_PER_THREAD = (SK_TILE2 * HEAD_DIM) / K2_BLOCK;  // 16

__global__ void __launch_bounds__(256, 2)
compute_grad_value_kernel(
    __nv_bfloat16* __restrict__ grad_value_states,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv
) {
    const int bkv = blockIdx.x;
    const int sk_tile_idx = blockIdx.y;
    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv % NUM_KV_HEADS;
    if (b >= batch_size) return;

    const int sk_start = sk_tile_idx * SK_TILE2;
    if (sk_start >= seq_len_kv) return;
    const int tid = threadIdx.x;

    __shared__ float s_awd[SQ_CHUNK * SK_TILE2];
    __shared__ float s_go[SQ_CHUNK * HEAD_DIM];

    float acc[K2_OUT_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < K2_OUT_PER_THREAD; ++i) acc[i] = 0.0f;

    const int total_iters = NUM_GROUPS * seq_len_q;

    for (int chunk_start = 0; chunk_start < total_iters; chunk_start += SQ_CHUNK) {
        int chunk_size = min(SQ_CHUNK, total_iters - chunk_start);

        // Load awd chunk
        int awd_elems = chunk_size * SK_TILE2;
        for (int idx = tid; idx < awd_elems; idx += K2_BLOCK) {
            int sq_l = idx / SK_TILE2;
            int sk_l = idx % SK_TILE2;
            int iter = chunk_start + sq_l;
            int h_l = iter / seq_len_q;
            int sq = iter % seq_len_q;
            int h = kv_head * NUM_GROUPS + h_l;
            int sk = sk_start + sk_l;
            float val = 0.0f;
            if (sk < seq_len_kv && h_l < NUM_GROUPS) {
                long long off = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
                val = __bfloat162float(attn_weights_dropped[off]);
            }
            s_awd[sq_l * SK_TILE2 + sk_l] = val;
        }

        // Load go chunk
        int go_elems = chunk_size * HEAD_DIM;
        for (int idx = tid; idx < go_elems; idx += K2_BLOCK) {
            int sq_l = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int iter = chunk_start + sq_l;
            int h_l = iter / seq_len_q;
            int sq = iter % seq_len_q;
            int h = kv_head * NUM_GROUPS + h_l;
            float val = 0.0f;
            if (h_l < NUM_GROUPS) {
                long long off = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM + d;
                val = __bfloat162float(grad_attn_output[off]);
            }
            s_go[sq_l * HEAD_DIM + d] = val;
        }
        __syncthreads();

        for (int sq_l = 0; sq_l < chunk_size; ++sq_l) {
            #pragma unroll
            for (int k = 0; k < K2_OUT_PER_THREAD; ++k) {
                int out_idx = tid + k * K2_BLOCK;
                int sk_l = out_idx / HEAD_DIM;
                int d = out_idx % HEAD_DIM;
                acc[k] += s_awd[sq_l * SK_TILE2 + sk_l] * s_go[sq_l * HEAD_DIM + d];
            }
        }
        __syncthreads();
    }

    // Write output
    long long out_base = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk_start) * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < K2_OUT_PER_THREAD; ++k) {
        int out_idx = tid + k * K2_BLOCK;
        int sk_l = out_idx / HEAD_DIM;
        if (sk_start + sk_l < seq_len_kv) {
            int d = out_idx % HEAD_DIM;
            grad_value_states[out_base + (long long)sk_l * HEAD_DIM + d] = __float2bfloat16(acc[k]);
        }
    }
}

// =====================================================================
// Launchers
// =====================================================================

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
    int sq_tiles = (seq_len_q + SQ_TILE - 1) / SQ_TILE;
    dim3 grid(batch_size * NUM_HEADS, sq_tiles);
    dim3 block(K1_BLOCK);

    // Shared memory: V_TILE*HEAD_DIM bf16 + SQ_TILE*HEAD_DIM float + SQ_TILE*V_TILE float + SQ_TILE float + K1_WARPS float
    int smem = V_TILE * HEAD_DIM * sizeof(__nv_bfloat16)
             + SQ_TILE * HEAD_DIM * sizeof(float)
             + SQ_TILE * V_TILE * sizeof(float)
             + SQ_TILE * sizeof(float)
             + K1_WARPS * sizeof(float);

    cudaFuncSetAttribute(compute_grad_attn_scores_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    compute_grad_attn_scores_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale
    );
}

void compute_grad_value_launcher(
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights_dropped,
    int batch_size, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
) {
    int sk_tiles = (seq_len_kv + SK_TILE2 - 1) / SK_TILE2;
    dim3 grid(batch_size * NUM_KV_HEADS, sk_tiles);
    dim3 block(K2_BLOCK);

    compute_grad_value_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
}
