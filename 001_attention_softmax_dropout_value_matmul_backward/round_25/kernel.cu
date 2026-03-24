#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

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

// =====================================================================
// KERNEL 1: FUSED TC GEMM + dropout + softmax backward
//
// Computes grad_attn_scores WITHOUT materializing the 5GB f32 intermediate.
// Each block handles SQ_TILE=16 query positions × ALL seq_kv positions.
// Value states are processed in SK_TILE=128 chunks (8 WMMA tiles per chunk).
// Two-pass: pass 1 accumulates sum_term, pass 2 writes grad_scores.
//
// Grid: (batch * heads, ceil(seq_q / SQ_TILE))
// Block: 256 threads (8 warps)
// =====================================================================

constexpr int SQ_TILE = 16;
constexpr int SK_TILE = 128;  // 8 warps × 16 WMMA tile each
constexpr int BLOCK_K1 = 256;
constexpr int K_STEP = 16;
constexpr int V_PAD = 8;
constexpr int V_STRIDE = K_STEP + V_PAD;  // 24
constexpr int GO_PAD = 8;
constexpr int GO_STRIDE = HEAD_DIM + GO_PAD;  // 136

__global__ void __launch_bounds__(256, 4)
fused_grad_scores_kernel(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    const __nv_bfloat16* __restrict__ go,
    const __nv_bfloat16* __restrict__ attn_weights,
    const __nv_bfloat16* __restrict__ value_states,
    const bool* __restrict__ dropout_mask,
    const int batch_size, const int seq_q, const int seq_kv,
    const float dropout_scale
) {
    const int bh = blockIdx.x;
    const int sq_tile = blockIdx.y;
    const int b = bh / NUM_HEADS, h = bh % NUM_HEADS;
    if (b >= batch_size) return;
    const int kv_head = h / NUM_GROUPS;
    const int sq_start = sq_tile * SQ_TILE;
    const int warp_id = threadIdx.x / 32;
    const int tid = threadIdx.x;

    // Shared memory layout (using extern dynamic):
    // s_go:  SQ_TILE × GO_STRIDE bf16 = 16 × 136 × 2 = 4352 bytes (loaded once)
    // s_V:   SK_TILE × V_STRIDE bf16  = 128 × 24 × 2 = 6144 bytes (per K-step of V)
    // --- after K-loop, reuse s_V area for: ---
    // s_C:   SQ_TILE × SK_TILE float  = 16 × 128 × 4 = 8192 bytes (WMMA output)
    // s_sum: SQ_TILE float             = 64 bytes
    extern __shared__ char smem[];
    __nv_bfloat16* s_go = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_V  = reinterpret_cast<__nv_bfloat16*>(smem + SQ_TILE * GO_STRIDE * sizeof(__nv_bfloat16));
    float* s_C   = reinterpret_cast<float*>(smem + SQ_TILE * GO_STRIDE * sizeof(__nv_bfloat16));
    float* s_sum = s_C + SQ_TILE * SK_TILE;

    // Load grad_output for SQ_TILE query positions (ONCE, reused for all V-tiles)
    for (int idx = tid; idx < SQ_TILE * HEAD_DIM; idx += BLOCK_K1) {
        int sq_l = idx / HEAD_DIM, d = idx % HEAD_DIM;
        int sq = sq_start + sq_l;
        __nv_bfloat16 val = {};
        if (sq < seq_q) {
            long long off = ((long long)(b * seq_q + sq) * NUM_HEADS + h) * HEAD_DIM + d;
            val = go[off];
        }
        s_go[sq_l * GO_STRIDE + d] = val;
    }
    if (tid < SQ_TILE) s_sum[tid] = 0.0f;
    __syncthreads();

    // Base offset for V
    const long long v_base = ((long long)b * NUM_KV_HEADS + kv_head) * seq_kv * HEAD_DIM;

    // ==================== PASS 1: Compute dot products + accumulate sum_term ====================
    for (int sk_start = 0; sk_start < seq_kv; sk_start += SK_TILE) {
        int sk_tile_size = min(SK_TILE, seq_kv - sk_start);

        // WMMA accumulators: each warp owns one 16×16 tile
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // K-loop over HEAD_DIM in steps of 16
        for (int ks = 0; ks < HEAD_DIM / K_STEP; ++ks) {
            int k0 = ks * K_STEP;

            // Load V chunk: SK_TILE × K_STEP bf16
            for (int idx = tid; idx < SK_TILE * K_STEP; idx += BLOCK_K1) {
                int sk_l = idx / K_STEP, k = idx % K_STEP;
                int sk = sk_start + sk_l;
                __nv_bfloat16 val = {};
                if (sk < seq_kv)
                    val = value_states[v_base + (long long)sk * HEAD_DIM + k0 + k];
                s_V[sk_l * V_STRIDE + k] = val;
            }
            __syncthreads();

            // WMMA: each warp computes its 16×16 tile
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

            // A from s_go: rows 0..15 (sq), cols k0..k0+15
            wmma::load_matrix_sync(a_frag, &s_go[k0], GO_STRIDE);
            // B from s_V: rows warp_id*16..warp_id*16+15 (sk), cols 0..15 (k)
            wmma::load_matrix_sync(b_frag, &s_V[warp_id * 16 * V_STRIDE], V_STRIDE);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }

        // Store WMMA output to shared memory
        wmma::store_matrix_sync(&s_C[warp_id * 16], c_frag, SK_TILE, wmma::mem_row_major);
        __syncthreads();

        // Apply dropout + accumulate sum_term
        // 16 × sk_tile_size elements; 256 threads handle them
        long long woff = ((long long)(b * NUM_HEADS + h) * seq_q + sq_start) * seq_kv + sk_start;
        for (int idx = tid; idx < SQ_TILE * sk_tile_size; idx += BLOCK_K1) {
            int sq_l = idx / sk_tile_size, sk_l = idx % sk_tile_size;
            int sq = sq_start + sq_l;
            if (sq < seq_q) {
                float dot = s_C[sq_l * SK_TILE + sk_l];
                long long pos = ((long long)sq_l) * seq_kv + sk_l;
                bool mask = dropout_mask[woff + pos];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + pos]);
                atomicAdd(&s_sum[sq_l], gw * aw);
            }
        }
        __syncthreads();
    }

    // ==================== PASS 2: Recompute + write grad_scores ====================
    for (int sk_start = 0; sk_start < seq_kv; sk_start += SK_TILE) {
        int sk_tile_size = min(SK_TILE, seq_kv - sk_start);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int ks = 0; ks < HEAD_DIM / K_STEP; ++ks) {
            int k0 = ks * K_STEP;
            for (int idx = tid; idx < SK_TILE * K_STEP; idx += BLOCK_K1) {
                int sk_l = idx / K_STEP, k = idx % K_STEP;
                int sk = sk_start + sk_l;
                __nv_bfloat16 val = {};
                if (sk < seq_kv)
                    val = value_states[v_base + (long long)sk * HEAD_DIM + k0 + k];
                s_V[sk_l * V_STRIDE + k] = val;
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &s_go[k0], GO_STRIDE);
            wmma::load_matrix_sync(b_frag, &s_V[warp_id * 16 * V_STRIDE], V_STRIDE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }

        wmma::store_matrix_sync(&s_C[warp_id * 16], c_frag, SK_TILE, wmma::mem_row_major);
        __syncthreads();

        // Compute softmax backward + write output
        long long woff = ((long long)(b * NUM_HEADS + h) * seq_q + sq_start) * seq_kv + sk_start;
        for (int idx = tid; idx < SQ_TILE * sk_tile_size; idx += BLOCK_K1) {
            int sq_l = idx / sk_tile_size, sk_l = idx % sk_tile_size;
            int sq = sq_start + sq_l;
            if (sq < seq_q) {
                float dot = s_C[sq_l * SK_TILE + sk_l];
                long long pos = ((long long)sq_l) * seq_kv + sk_l;
                bool mask = dropout_mask[woff + pos];
                float gw = mask ? dot * dropout_scale : 0.0f;
                float aw = __bfloat162float(attn_weights[woff + pos]);
                float grad = aw * (gw - s_sum[sq_l]);
                grad_attn_scores[woff + pos] = __float2bfloat16(grad);
            }
        }
        __syncthreads();
    }
}

// =====================================================================
// KERNEL 2: Grad value with WMMA (same as round 24)
// =====================================================================
constexpr int GV_BM = 64, GV_BN = 128, GV_KT = 16, GV_BLOCK = 256;
constexpr int GV_APAD = 8, GV_BPAD = 8;
constexpr int GV_AS = GV_BM + GV_APAD, GV_BS = GV_BN + GV_BPAD;

__global__ void __launch_bounds__(256, 2)
grad_value_wmma_kernel(
    __nv_bfloat16* __restrict__ grad_value,
    const __nv_bfloat16* __restrict__ go,
    const __nv_bfloat16* __restrict__ awd,
    const int batch_size, const int seq_q, const int seq_kv
) {
    const int bkv = blockIdx.x, sk_tile = blockIdx.y;
    const int b = bkv / NUM_KV_HEADS, kv_head = bkv % NUM_KV_HEADS;
    if (b >= batch_size) return;
    const int sk_start = sk_tile * GV_BM;
    if (sk_start >= seq_kv) return;
    const int warp_id = threadIdx.x / 32;
    const int wm = warp_id / 2, wn = warp_id % 2;

    extern __shared__ char smem[];
    __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_B = s_A + GV_KT * GV_AS;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4];
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(c_frag[i], 0.0f);

    for (int h_local = 0; h_local < NUM_GROUPS; ++h_local) {
        const int h = kv_head * NUM_GROUPS + h_local;
        const long long awd_hbase = ((long long)(b * NUM_HEADS + h) * seq_q) * seq_kv;

        for (int sq_start = 0; sq_start < seq_q; sq_start += GV_KT) {
            const int kt = min(GV_KT, seq_q - sq_start);
            for (int idx = threadIdx.x; idx < GV_KT * GV_BM; idx += GV_BLOCK) {
                int kl = idx / GV_BM, sk_l = idx % GV_BM;
                int sk = sk_start + sk_l;
                __nv_bfloat16 val = {};
                if (kl < kt && sk < seq_kv)
                    val = awd[awd_hbase + (long long)(sq_start + kl) * seq_kv + sk];
                s_A[kl * GV_AS + sk_l] = val;
            }
            for (int idx = threadIdx.x; idx < GV_KT * GV_BN; idx += GV_BLOCK) {
                int kl = idx / GV_BN, d = idx % GV_BN;
                __nv_bfloat16 val = {};
                if (kl < kt) {
                    long long off = ((long long)(b*seq_q+sq_start+kl)*NUM_HEADS+h)*HEAD_DIM+d;
                    val = go[off];
                }
                s_B[kl * GV_BS + d] = val;
            }
            __syncthreads();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[4];
            wmma::load_matrix_sync(a_frag, &s_A[wm * 16], GV_AS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::load_matrix_sync(b_frag[ni], &s_B[wn*64+ni*16], GV_BS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::mma_sync(c_frag[ni], a_frag, b_frag[ni], c_frag[ni]);
            __syncthreads();
        }
    }

    float* s_C = reinterpret_cast<float*>(smem);
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni)
        wmma::store_matrix_sync(&s_C[(wm*16)*GV_BN+wn*64+ni*16], c_frag[ni], GV_BN, wmma::mem_row_major);
    __syncthreads();
    const long long out_base = ((long long)(b*NUM_KV_HEADS+kv_head)*seq_kv+sk_start)*HEAD_DIM;
    for (int idx = threadIdx.x; idx < GV_BM * GV_BN; idx += GV_BLOCK) {
        int m = idx / GV_BN, d = idx % GV_BN;
        if (sk_start + m < seq_kv)
            grad_value[out_base + (long long)m*HEAD_DIM+d] = __float2bfloat16(s_C[m*GV_BN+d]);
    }
}

// =====================================================================
// Launchers
// =====================================================================
void fused_grad_scores_launcher(
    torch::Tensor& grad_attn_scores,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    int batch_size, int seq_len_q, int seq_len_kv,
    float dropout_scale, cudaStream_t stream
) {
    int sq_tiles = (seq_len_q + SQ_TILE - 1) / SQ_TILE;
    dim3 grid(batch_size * NUM_HEADS, sq_tiles);
    dim3 block(BLOCK_K1);

    // smem: s_go + max(s_V for K-loop, s_C + s_sum for dropout)
    int smem_go = SQ_TILE * GO_STRIDE * sizeof(__nv_bfloat16);
    int smem_v  = SK_TILE * V_STRIDE * sizeof(__nv_bfloat16);
    int smem_c  = SQ_TILE * SK_TILE * sizeof(float) + SQ_TILE * sizeof(float);
    int smem = smem_go + max(smem_v, smem_c);

    cudaFuncSetAttribute(fused_grad_scores_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fused_grad_scores_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        batch_size, seq_len_q, seq_len_kv, dropout_scale);
}

void compute_grad_value_launcher(
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights_dropped,
    int batch_size, int seq_len_q, int seq_len_kv, cudaStream_t stream
) {
    int skt = (seq_len_kv + GV_BM - 1) / GV_BM;
    dim3 grid(batch_size * NUM_KV_HEADS, skt);
    int smem = max(GV_KT*GV_AS*2 + GV_KT*GV_BS*2, GV_BM*GV_BN*4);
    cudaFuncSetAttribute(grad_value_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    grad_value_wmma_kernel<<<grid, GV_BLOCK, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv);
}
