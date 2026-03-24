#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

// =====================================================================
// Common utilities
// =====================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-cooperative 128-dim dot product with coalesced global memory reads.
// 32 lanes each read 4 consecutive bf16 elements → 256-byte coalesced access.
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
    float partial = s_vec[base] * f01.x + s_vec[base+1] * f01.y +
                    s_vec[base+2] * f23.x + s_vec[base+3] * f23.y;
    return warp_reduce_sum(partial);
}

// =====================================================================
// KERNEL 1: Fused grad_attn_scores
//
// Computes: dot(grad_out, V) → dropout backward → softmax backward
// Grid: (batch * NUM_HEADS, seq_q)
// Block: 256 threads
//
// Three-phase design:
//   Phase 1: Warp-cooperative dot products → s_dot (shared memory)
//   Phase 2: All threads apply dropout + accumulate sum_term (coalesced)
//   Phase 3: All threads write grad_scores (coalesced)
// =====================================================================

constexpr int K1_BLOCK = 256;
constexpr int K1_WARPS = K1_BLOCK / 32;

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
    const int bh = blockIdx.x;  // batch * NUM_HEADS
    const int sq = blockIdx.y;

    const int b = bh / NUM_HEADS;
    const int h = bh % NUM_HEADS;
    if (b >= batch_size || sq >= seq_len_q) return;

    const int kv_head = h / NUM_GROUPS;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const long long goff = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM;
    const long long woff = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv;
    const long long voff = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv) * HEAD_DIM;

    // Dynamic shared memory: s_grad_out[128] + s_dot[seq_kv] + s_warp_sum[8]
    extern __shared__ float s_data[];
    float* s_grad_out = s_data;
    float* s_dot      = s_data + HEAD_DIM;
    float* s_warp_sum = s_data + HEAD_DIM + seq_len_kv;

    // Load grad_output
    if (tid < HEAD_DIM) {
        s_grad_out[tid] = __bfloat162float(grad_attn_output[goff + tid]);
    }
    __syncthreads();

    // Phase 1: Warp-cooperative dot products
    for (int sk = warp_id; sk < seq_len_kv; sk += K1_WARPS) {
        float dot = warp_dot_128(s_grad_out, &value_states[voff + (long long)sk * HEAD_DIM], lane);
        if (lane == 0) s_dot[sk] = dot;
    }
    __syncthreads();

    // Phase 2: Dropout + accumulate sum_term
    float local_sum = 0.0f;
    for (int sk = tid; sk < seq_len_kv; sk += K1_BLOCK) {
        float dot = s_dot[sk];
        bool mask = dropout_mask[woff + sk];
        float gw = mask ? dot * dropout_scale : 0.0f;
        s_dot[sk] = gw;
        float aw = __bfloat162float(attn_weights[woff + sk]);
        local_sum += gw * aw;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_warp_sum[warp_id] = local_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (tid < K1_WARPS) total_sum = s_warp_sum[tid];
    if (tid < 32) {
        total_sum = warp_reduce_sum(total_sum);
        if (tid == 0) s_warp_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_warp_sum[0];

    // Phase 3: Write grad_scores
    for (int sk = tid; sk < seq_len_kv; sk += K1_BLOCK) {
        float gw = s_dot[sk];
        float aw = __bfloat162float(attn_weights[woff + sk]);
        grad_attn_scores[woff + sk] = __float2bfloat16(aw * (gw - total_sum));
    }
}

// =====================================================================
// KERNEL 2: Grad value states
//
// Computes: grad_v[b,kv_h,sk,d] = sum_{h in group} sum_sq awd[b,h,sq,sk] * go[b,sq,h,d]
// GQA aggregation (sum over 10 heads) is built into the kernel.
//
// Grid: (batch * NUM_KV_HEADS, ceil(seq_kv / SK_TILE))
// Block: 256 threads
//
// Each block computes SK_TILE × HEAD_DIM output elements.
// Inner loop processes SQ_CHUNK iterations at a time with shared memory.
// Coalesced access: awd loaded along sk (contiguous), go along d (contiguous).
// =====================================================================

constexpr int K2_BLOCK = 256;
constexpr int SK_TILE = 32;
constexpr int SQ_CHUNK = 8;  // Process 8 (h,sq) iterations per shared memory load

__global__ void __launch_bounds__(256, 2)
compute_grad_value_kernel(
    __nv_bfloat16* __restrict__ grad_value_states,  // (B, 8, sk, 128) bf16
    const __nv_bfloat16* __restrict__ grad_attn_output,  // (B, sq, 80, 128) bf16
    const __nv_bfloat16* __restrict__ attn_weights_dropped,  // (B, 80, sq, sk) bf16
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv
) {
    const int bkv = blockIdx.x;  // batch * NUM_KV_HEADS
    const int sk_tile_idx = blockIdx.y;

    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv % NUM_KV_HEADS;
    if (b >= batch_size) return;

    const int sk_start = sk_tile_idx * SK_TILE;
    if (sk_start >= seq_len_kv) return;

    const int tid = threadIdx.x;
    const int actual_sk_tile = min(SK_TILE, seq_len_kv - sk_start);

    // Output elements per block: SK_TILE * HEAD_DIM = 32 * 128 = 4096
    // Threads: 256 → 16 elements per thread
    constexpr int OUTPUTS_PER_THREAD = (SK_TILE * HEAD_DIM) / K2_BLOCK;  // 16

    // Shared memory for chunked loading
    // s_awd: SQ_CHUNK * SK_TILE floats
    // s_go:  SQ_CHUNK * HEAD_DIM floats
    __shared__ float s_awd[SQ_CHUNK * SK_TILE];    // 8 * 32 = 256 floats = 1 KB
    __shared__ float s_go[SQ_CHUNK * HEAD_DIM];     // 8 * 128 = 1024 floats = 4 KB

    // Per-thread accumulators
    float acc[OUTPUTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; ++i) acc[i] = 0.0f;

    // Total iterations: NUM_GROUPS * seq_len_q
    const int total_iters = NUM_GROUPS * seq_len_q;

    for (int chunk_start = 0; chunk_start < total_iters; chunk_start += SQ_CHUNK) {
        const int chunk_size = min(SQ_CHUNK, total_iters - chunk_start);

        // Load awd chunk: chunk_size * SK_TILE values
        const int awd_elems = chunk_size * SK_TILE;
        for (int idx = tid; idx < awd_elems; idx += K2_BLOCK) {
            int sq_local = idx / SK_TILE;
            int sk_l = idx % SK_TILE;
            int iter = chunk_start + sq_local;
            int h_local = iter / seq_len_q;
            int sq = iter % seq_len_q;
            int h = kv_head * NUM_GROUPS + h_local;
            int sk = sk_start + sk_l;

            float val = 0.0f;
            if (sk < seq_len_kv && h_local < NUM_GROUPS) {
                long long awd_off = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
                val = __bfloat162float(attn_weights_dropped[awd_off]);
            }
            s_awd[sq_local * SK_TILE + sk_l] = val;
        }

        // Load go chunk: chunk_size * HEAD_DIM values
        const int go_elems = chunk_size * HEAD_DIM;
        for (int idx = tid; idx < go_elems; idx += K2_BLOCK) {
            int sq_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int iter = chunk_start + sq_local;
            int h_local = iter / seq_len_q;
            int sq = iter % seq_len_q;
            int h = kv_head * NUM_GROUPS + h_local;

            float val = 0.0f;
            if (h_local < NUM_GROUPS) {
                long long go_off = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM + d;
                val = __bfloat162float(grad_attn_output[go_off]);
            }
            s_go[sq_local * HEAD_DIM + d] = val;
        }
        __syncthreads();

        // Compute outer products
        for (int sq_local = 0; sq_local < chunk_size; ++sq_local) {
            #pragma unroll
            for (int k = 0; k < OUTPUTS_PER_THREAD; ++k) {
                int out_idx = tid + k * K2_BLOCK;
                int sk_l = out_idx / HEAD_DIM;
                int d = out_idx % HEAD_DIM;
                acc[k] += s_awd[sq_local * SK_TILE + sk_l] * s_go[sq_local * HEAD_DIM + d];
            }
        }
        __syncthreads();
    }

    // Write output
    const long long out_base = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk_start) * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < OUTPUTS_PER_THREAD; ++k) {
        int out_idx = tid + k * K2_BLOCK;
        int sk_l = out_idx / HEAD_DIM;
        int d = out_idx % HEAD_DIM;
        if (sk_start + sk_l < seq_len_kv) {
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
    dim3 grid(batch_size * NUM_HEADS, seq_len_q);
    dim3 block(K1_BLOCK);
    int smem = (HEAD_DIM + seq_len_kv + K1_WARPS) * sizeof(float);

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
    int sk_tiles = (seq_len_kv + SK_TILE - 1) / SK_TILE;
    dim3 grid(batch_size * NUM_KV_HEADS, sk_tiles);
    dim3 block(K2_BLOCK);

    compute_grad_value_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
}
