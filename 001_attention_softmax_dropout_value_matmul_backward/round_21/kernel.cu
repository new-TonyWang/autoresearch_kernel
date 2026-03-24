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

// =====================================================================
// KERNEL 1: Tiled GEMM for go @ V^T
//
// C[sq, sk] = sum_d A[sq, d] * B[sk, d]
// A = grad_out (bf16), B = value_states (bf16), C = output (f32)
//
// BM=64, BN=64, K=128 (no K tiling needed).
// 256 threads = 16x16. Each thread computes 4x4 = 16 output elements.
// Shared memory with row padding to avoid bank conflicts.
//
// Grid: (batch * heads, ceil(sq/BM), ceil(sk/BN))
// Block: 256 threads
// =====================================================================

constexpr int G_BM = 64;
constexpr int G_BN = 64;
constexpr int G_TM = 4;  // thread tile M
constexpr int G_TN = 4;  // thread tile N
constexpr int G_BLOCK = 256;
constexpr int G_PAD = 4;  // padding to avoid bank conflicts
constexpr int G_K_STRIDE = HEAD_DIM + G_PAD;  // 132

__global__ void __launch_bounds__(256, 2)
gemm_go_vt_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ grad_attn_output,
    const __nv_bfloat16* __restrict__ value_states,
    const int batch_size,
    const int seq_len_q,
    const int seq_len_kv
) {
    const int bh = blockIdx.x;
    const int sq_tile = blockIdx.y;
    const int sk_tile = blockIdx.z;

    const int b = bh / NUM_HEADS;
    const int h = bh % NUM_HEADS;
    if (b >= batch_size) return;

    const int kv_head = h / NUM_GROUPS;
    const int tid = threadIdx.x;
    const int tm = tid / 16;  // 0..15 (M dimension)
    const int tn = tid % 16;  // 0..15 (N dimension)

    const int sq_start = sq_tile * G_BM;
    const int sk_start = sk_tile * G_BN;

    // Shared memory with padding
    __shared__ __nv_bfloat16 s_A[G_BM * G_K_STRIDE];  // go tile: 64 × 132
    __shared__ __nv_bfloat16 s_B[G_BN * G_K_STRIDE];  // V tile: 64 × 132

    // Cooperatively load A tile (grad_attn_output)
    // go layout: (B, sq, heads, head_dim)
    for (int idx = tid; idx < G_BM * HEAD_DIM; idx += G_BLOCK) {
        int m = idx / HEAD_DIM;
        int k = idx % HEAD_DIM;
        int sq = sq_start + m;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (sq < seq_len_q) {
            long long off = ((long long)(b * seq_len_q + sq) * NUM_HEADS + h) * HEAD_DIM + k;
            val = grad_attn_output[off];
        }
        s_A[m * G_K_STRIDE + k] = val;
    }

    // Cooperatively load B tile (value_states)
    // V layout: (B, kv_heads, sk, head_dim)
    for (int idx = tid; idx < G_BN * HEAD_DIM; idx += G_BLOCK) {
        int n = idx / HEAD_DIM;
        int k = idx % HEAD_DIM;
        int sk = sk_start + n;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (sk < seq_len_kv) {
            long long off = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk) * HEAD_DIM + k;
            val = value_states[off];
        }
        s_B[n * G_K_STRIDE + k] = val;
    }
    __syncthreads();

    // Compute 4x4 thread tile
    float c[G_TM][G_TN] = {{0}};

    int m_base = tm * G_TM;
    int n_base = tn * G_TN;

    #pragma unroll 4
    for (int k = 0; k < HEAD_DIM; ++k) {
        float a[G_TM];
        float b_vals[G_TN];

        #pragma unroll
        for (int i = 0; i < G_TM; ++i)
            a[i] = __bfloat162float(s_A[(m_base + i) * G_K_STRIDE + k]);
        #pragma unroll
        for (int j = 0; j < G_TN; ++j)
            b_vals[j] = __bfloat162float(s_B[(n_base + j) * G_K_STRIDE + k]);

        #pragma unroll
        for (int i = 0; i < G_TM; ++i)
            #pragma unroll
            for (int j = 0; j < G_TN; ++j)
                c[i][j] += a[i] * b_vals[j];
    }

    // Write output: (B, heads, sq, sk)
    long long out_base = ((long long)(b * NUM_HEADS + h) * seq_len_q + sq_start) * seq_len_kv + sk_start;
    #pragma unroll
    for (int i = 0; i < G_TM; ++i) {
        int sq = sq_start + m_base + i;
        if (sq < seq_len_q) {
            #pragma unroll
            for (int j = 0; j < G_TN; ++j) {
                int sk = sk_start + n_base + j;
                if (sk < seq_len_kv) {
                    output[out_base + (long long)(m_base + i) * seq_len_kv + (n_base + j)] = c[i][j];
                }
            }
        }
    }
}

// =====================================================================
// KERNEL 2: Fused dropout + softmax backward (from round 18)
// =====================================================================

constexpr int K2_BLOCK = 256;
constexpr int K2_WARPS = K2_BLOCK / 32;

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

    __shared__ float s_warp_sum[K2_WARPS];

    float gw_regs[16];
    float aw_regs[16];
    float local_sum = 0.0f;
    int n = 0;

    for (int sk = tid; sk < seq_len_kv; sk += K2_BLOCK) {
        float raw = grad_aw_dropped[row_off + sk];
        bool mask = dropout_mask[row_off + sk];
        float gw = mask ? raw * dropout_scale : 0.0f;
        float aw = __bfloat162float(attn_weights[row_off + sk]);
        gw_regs[n] = gw;
        aw_regs[n] = aw;
        local_sum += gw * aw;
        n++;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_warp_sum[warp_id] = local_sum;
    __syncthreads();

    float total_sum = 0.0f;
    if (tid < K2_WARPS) total_sum = s_warp_sum[tid];
    if (tid < 32) {
        total_sum = warp_reduce_sum(total_sum);
        if (tid == 0) s_warp_sum[0] = total_sum;
    }
    __syncthreads();
    total_sum = s_warp_sum[0];

    n = 0;
    for (int sk = tid; sk < seq_len_kv; sk += K2_BLOCK) {
        float grad_score = aw_regs[n] * (gw_regs[n] - total_sum);
        grad_attn_scores[row_off + sk] = __float2bfloat16(grad_score);
        n++;
    }
}

// =====================================================================
// KERNEL 3: Grad value (from round 19)
// =====================================================================

constexpr int K3_BLOCK = 256;
constexpr int SK_TILE = 32;
constexpr int SQ_CHUNK = 8;
constexpr int K3_OPT = (SK_TILE * HEAD_DIM) / K3_BLOCK;  // 16

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

    const int sk_start = sk_tile_idx * SK_TILE;
    if (sk_start >= seq_len_kv) return;
    const int tid = threadIdx.x;

    __shared__ float s_awd[SQ_CHUNK * SK_TILE];
    __shared__ float s_go[SQ_CHUNK * HEAD_DIM];

    float acc[K3_OPT];
    #pragma unroll
    for (int i = 0; i < K3_OPT; ++i) acc[i] = 0.0f;

    const int total_iters = NUM_GROUPS * seq_len_q;

    for (int chunk_start = 0; chunk_start < total_iters; chunk_start += SQ_CHUNK) {
        int chunk_size = min(SQ_CHUNK, total_iters - chunk_start);

        for (int idx = tid; idx < chunk_size * SK_TILE; idx += K3_BLOCK) {
            int sq_l = idx / SK_TILE;
            int sk_l = idx % SK_TILE;
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
            s_awd[sq_l * SK_TILE + sk_l] = val;
        }

        for (int idx = tid; idx < chunk_size * HEAD_DIM; idx += K3_BLOCK) {
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
            for (int k = 0; k < K3_OPT; ++k) {
                int out_idx = tid + k * K3_BLOCK;
                int sk_l = out_idx / HEAD_DIM;
                int d = out_idx % HEAD_DIM;
                acc[k] += s_awd[sq_l * SK_TILE + sk_l] * s_go[sq_l * HEAD_DIM + d];
            }
        }
        __syncthreads();
    }

    long long out_base = ((long long)(b * NUM_KV_HEADS + kv_head) * seq_len_kv + sk_start) * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < K3_OPT; ++k) {
        int out_idx = tid + k * K3_BLOCK;
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

void gemm_go_vt_launcher(
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

    gemm_go_vt_kernel<<<grid, block, 0, stream>>>(
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
    dim3 block(K2_BLOCK);
    fused_dropout_softmax_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_aw_dropped.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(),
        dropout_scale, seq_len_kv
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
    dim3 block(K3_BLOCK);
    compute_grad_value_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv
    );
}
