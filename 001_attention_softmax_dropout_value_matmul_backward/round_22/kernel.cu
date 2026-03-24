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
// KERNEL 1: Tensor Core GEMM for go @ V^T
//
// Uses WMMA (16×16×16 bf16→f32) tensor core operations.
// BM=64, BN=64, K=128 processed in 8 steps of 16.
// 256 threads = 8 warps in 4×2 layout.
// Each warp computes 16×32 output (2 WMMA tiles).
//
// Grid: (batch*heads, ceil(sq/64), ceil(sk/64))
// =====================================================================

constexpr int G_BM = 64, G_BN = 64, G_BLOCK = 256;
constexpr int G_PAD = 8;
constexpr int G_STRIDE = 16 + G_PAD;  // padded K-tile stride = 24

__global__ void __launch_bounds__(256, 2)
gemm_tc_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ go,
    const __nv_bfloat16* __restrict__ V,
    const int batch_size, const int seq_q, const int seq_kv
) {
    const int bh = blockIdx.x;
    const int sq_tile = blockIdx.y;
    const int sk_tile = blockIdx.z;
    const int b = bh / NUM_HEADS;
    const int h = bh % NUM_HEADS;
    if (b >= batch_size) return;
    const int kv_head = h / NUM_GROUPS;
    const int sq_start = sq_tile * G_BM;
    const int sk_start = sk_tile * G_BN;

    const int warp_id = threadIdx.x / 32;
    const int wm = warp_id / 2;  // 0..3 (M dim)
    const int wn = warp_id % 2;  // 0..1 (N dim)

    // Dynamic shared memory: A tile + B tile (K step), then reused for output
    extern __shared__ char smem[];
    __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem + G_BM * G_STRIDE * sizeof(__nv_bfloat16));

    // Accumulators: 2 WMMA tiles per warp (16×16 each, covering 16×32)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    // K loop: 8 steps of 16
    for (int k_step = 0; k_step < HEAD_DIM / 16; ++k_step) {
        const int k_start = k_step * 16;

        // Load A tile: BM rows × 16 cols from go
        for (int idx = threadIdx.x; idx < G_BM * 16; idx += G_BLOCK) {
            int m = idx / 16, k = idx % 16;
            int sq = sq_start + m;
            __nv_bfloat16 val = {};
            if (sq < seq_q) {
                long long off = ((long long)(b * seq_q + sq) * NUM_HEADS + h) * HEAD_DIM + k_start + k;
                val = go[off];
            }
            s_A[m * G_STRIDE + k] = val;
        }

        // Load B tile: BN rows × 16 cols from V
        for (int idx = threadIdx.x; idx < G_BN * 16; idx += G_BLOCK) {
            int n = idx / 16, k = idx % 16;
            int sk = sk_start + n;
            __nv_bfloat16 val = {};
            if (sk < seq_kv) {
                long long off = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_kv + sk) * HEAD_DIM + k_start + k;
                val = V[off];
            }
            s_B[n * G_STRIDE + k] = val;
        }
        __syncthreads();

        // WMMA: each warp loads its A and B fragments, then MMA
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0_frag, b1_frag;

        wmma::load_matrix_sync(a_frag, &s_A[wm * 16 * G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b0_frag, &s_B[(wn * 32) * G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b1_frag, &s_B[(wn * 32 + 16) * G_STRIDE], G_STRIDE);

        wmma::mma_sync(c0, a_frag, b0_frag, c0);
        wmma::mma_sync(c1, a_frag, b1_frag, c1);

        __syncthreads();
    }

    // Store output via shared memory (handles boundary correctly)
    float* s_C = reinterpret_cast<float*>(smem);  // reuse smem

    wmma::store_matrix_sync(&s_C[(wm * 16) * G_BN + wn * 32], c0, G_BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_C[(wm * 16) * G_BN + wn * 32 + 16], c1, G_BN, wmma::mem_row_major);
    __syncthreads();

    // Write to global with boundary checks (coalesced)
    for (int idx = threadIdx.x; idx < G_BM * G_BN; idx += G_BLOCK) {
        int m = idx / G_BN, n = idx % G_BN;
        int sq = sq_start + m, sk = sk_start + n;
        if (sq < seq_q && sk < seq_kv) {
            long long off = ((long long)(b * NUM_HEADS + h) * seq_q + sq) * seq_kv + sk;
            output[off] = s_C[m * G_BN + n];
        }
    }
}

// =====================================================================
// KERNEL 2: Fused dropout + softmax backward (proven from round 18)
// =====================================================================

__global__ void __launch_bounds__(256, 4)
fused_dropout_softmax_backward_kernel(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    const float* __restrict__ grad_aw_dropped,
    const __nv_bfloat16* __restrict__ attn_weights,
    const bool* __restrict__ dropout_mask,
    const float dropout_scale,
    const int seq_len_kv
) {
    const long long row = (long long)blockIdx.x * gridDim.y + blockIdx.y;
    const long long row_off = row * seq_len_kv;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    __shared__ float s_warp_sum[8];
    float gw_regs[16], aw_regs[16];
    float local_sum = 0.0f;
    int n = 0;

    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float raw = grad_aw_dropped[row_off + sk];
        bool mask = dropout_mask[row_off + sk];
        float gw = mask ? raw * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[row_off + sk]);
        gw_regs[n] = gw; aw_regs[n] = aw;
        local_sum += gw * aw;
        n++;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_warp_sum[warp_id] = local_sum;
    __syncthreads();
    float total_sum = 0.0f;
    if (tid < 8) total_sum = s_warp_sum[tid];
    if (tid < 32) { total_sum = warp_reduce_sum(total_sum); if (tid == 0) s_warp_sum[0] = total_sum; }
    __syncthreads();
    total_sum = s_warp_sum[0];

    n = 0;
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        grad_attn_scores[row_off + sk] = __float2bfloat16(aw_regs[n] * (gw_regs[n] - total_sum));
        n++;
    }
}

// =====================================================================
// KERNEL 3: Grad value (tiled, from round 19/21)
// =====================================================================

constexpr int SK_TILE = 32, SQ_CHUNK = 8;
constexpr int K3_OPT = (SK_TILE * HEAD_DIM) / 256;  // 16

__global__ void __launch_bounds__(256, 2)
compute_grad_value_kernel(
    __nv_bfloat16* __restrict__ grad_value_states,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ attn_weights_dropped,
    const int batch_size, const int seq_len_q, const int seq_len_kv
) {
    const int bkv = blockIdx.x;
    const int sk_tile_idx = blockIdx.y;
    const int b = bkv / NUM_KV_HEADS;
    const int kv_head = bkv % NUM_KV_HEADS;
    if (b >= batch_size) return;
    const int sk_start = sk_tile_idx * SK_TILE;
    if (sk_start >= seq_len_kv) return;
    const int tid = threadIdx.x;

    __shared__ float s_awd[SQ_CHUNK * SK_TILE];
    __shared__ float s_go[SQ_CHUNK * HEAD_DIM];

    float acc[K3_OPT];
    #pragma unroll
    for (int i = 0; i < K3_OPT; ++i) acc[i] = 0.0f;

    const int total_iters = NUM_GROUPS * seq_len_q;
    for (int cs = 0; cs < total_iters; cs += SQ_CHUNK) {
        int chunk_size = min(SQ_CHUNK, total_iters - cs);

        for (int idx = tid; idx < chunk_size * SK_TILE; idx += 256) {
            int sq_l = idx / SK_TILE, sk_l = idx % SK_TILE;
            int iter = cs + sq_l, h_l = iter / seq_len_q, sq = iter % seq_len_q;
            int h = kv_head * NUM_GROUPS + h_l, sk = sk_start + sk_l;
            float val = 0.0f;
            if (sk < seq_len_kv && h_l < NUM_GROUPS) {
                long long off = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq) * seq_len_kv + sk;
                val = __bfloat162float(attn_weights_dropped[off]);
            }
            s_awd[sq_l * SK_TILE + sk_l] = val;
        }
        for (int idx = tid; idx < chunk_size * HEAD_DIM; idx += 256) {
            int sq_l = idx / HEAD_DIM, d = idx % HEAD_DIM;
            int iter = cs + sq_l, h_l = iter / seq_len_q, sq = iter % seq_len_q;
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
            for (int k = 0; k < K3_OPT; ++k) {
                int out_idx = tid + k * 256;
                acc[k] += s_awd[sq_l * SK_TILE + out_idx / HEAD_DIM] * s_go[sq_l * HEAD_DIM + out_idx % HEAD_DIM];
            }
        }
        __syncthreads();
    }

    long long out_base = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk_start) * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < K3_OPT; ++k) {
        int out_idx = tid + k * 256;
        int sk_l = out_idx / HEAD_DIM;
        if (sk_start + sk_l < seq_len_kv)
            grad_value_states[out_base + (long long)sk_l * HEAD_DIM + out_idx % HEAD_DIM] = __float2bfloat16(acc[k]);
    }
}

// =====================================================================
// Launchers
// =====================================================================

void gemm_tc_launcher(
    torch::Tensor& output,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& value_states,
    int batch_size, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
) {
    int sq_tiles = (seq_len_q + G_BM - 1) / G_BM;
    int sk_tiles = (seq_len_kv + G_BN - 1) / G_BN;
    dim3 grid(batch_size * NUM_HEADS, sq_tiles, sk_tiles);
    dim3 block(G_BLOCK);

    // smem: max(A+B tiles for K loop, C tile for output)
    int smem_kb = G_BM * G_STRIDE * 2 + G_BN * G_STRIDE * 2;  // A+B tiles in bf16
    int smem_out = G_BM * G_BN * 4;  // output tile in f32
    int smem = max(smem_kb, smem_out);

    cudaFuncSetAttribute(gemm_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gemm_tc_kernel<<<grid, block, smem, stream>>>(
        output.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
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
    fused_dropout_softmax_backward_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_aw_dropped.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(), dropout_scale, seq_len_kv
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
    compute_grad_value_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
}
