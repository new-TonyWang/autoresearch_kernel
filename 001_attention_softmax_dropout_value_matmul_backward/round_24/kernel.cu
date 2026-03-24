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
// KERNEL 1: Tensor Core GEMM for go @ V^T (same as round 22/23)
// =====================================================================
constexpr int G_BM = 64, G_BN = 64, G_BLOCK = 256;
constexpr int G_PAD = 8, G_STRIDE = 16 + G_PAD;

__global__ void __launch_bounds__(256, 2)
gemm_tc_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ go,
    const __nv_bfloat16* __restrict__ V,
    const int batch_size, const int seq_q, const int seq_kv
) {
    const int bh = blockIdx.x, sq_tile = blockIdx.y, sk_tile = blockIdx.z;
    const int b = bh / NUM_HEADS, h = bh % NUM_HEADS;
    if (b >= batch_size) return;
    const int kv_head = h / NUM_GROUPS;
    const int sq_start = sq_tile * G_BM, sk_start = sk_tile * G_BN;
    const int warp_id = threadIdx.x / 32;
    const int wm = warp_id / 2, wn = warp_id % 2;

    extern __shared__ char smem[];
    __nv_bfloat16* s_A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_B = s_A + G_BM * G_STRIDE;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.0f); wmma::fill_fragment(c1, 0.0f);

    for (int ks = 0; ks < HEAD_DIM / 16; ++ks) {
        int k0 = ks * 16;
        for (int idx = threadIdx.x; idx < G_BM * 16; idx += G_BLOCK) {
            int m = idx / 16, k = idx % 16, sq = sq_start + m;
            __nv_bfloat16 v = {};
            if (sq < seq_q) { long long o = ((long long)(b*seq_q+sq)*NUM_HEADS+h)*HEAD_DIM+k0+k; v = go[o]; }
            s_A[m * G_STRIDE + k] = v;
        }
        for (int idx = threadIdx.x; idx < G_BN * 16; idx += G_BLOCK) {
            int n = idx / 16, k = idx % 16, sk = sk_start + n;
            __nv_bfloat16 v = {};
            if (sk < seq_kv) { long long o = ((long long)(b*NUM_KV_HEADS+kv_head)*seq_kv+sk)*HEAD_DIM+k0+k; v = V[o]; }
            s_B[n * G_STRIDE + k] = v;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0f, b1f;
        wmma::load_matrix_sync(af, &s_A[wm*16*G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b0f, &s_B[(wn*32)*G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b1f, &s_B[(wn*32+16)*G_STRIDE], G_STRIDE);
        wmma::mma_sync(c0, af, b0f, c0);
        wmma::mma_sync(c1, af, b1f, c1);
        __syncthreads();
    }

    float* s_C = reinterpret_cast<float*>(smem);
    wmma::store_matrix_sync(&s_C[(wm*16)*G_BN + wn*32], c0, G_BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_C[(wm*16)*G_BN + wn*32+16], c1, G_BN, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < G_BM * G_BN; idx += G_BLOCK) {
        int m = idx / G_BN, n = idx % G_BN;
        int sq = sq_start + m, sk = sk_start + n;
        if (sq < seq_q && sk < seq_kv)
            output[((long long)(b*NUM_HEADS+h)*seq_q+sq)*seq_kv+sk] = s_C[m * G_BN + n];
    }
}

// =====================================================================
// KERNEL 2: Fused dropout + softmax backward (same as round 22/23)
// =====================================================================
__global__ void __launch_bounds__(256, 4)
fused_dropout_softmax_backward_kernel(
    __nv_bfloat16* __restrict__ grad_attn_scores,
    const float* __restrict__ grad_aw_dropped,
    const __nv_bfloat16* __restrict__ attn_weights,
    const bool* __restrict__ dropout_mask,
    const float dropout_scale, const int seq_len_kv
) {
    const long long row = (long long)blockIdx.x * gridDim.y + blockIdx.y;
    const long long off = row * seq_len_kv;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    __shared__ float ws[8];
    float gw_r[16], aw_r[16]; float ls = 0; int n = 0;
    for (int sk = tid; sk < seq_len_kv; sk += 256) {
        float r = grad_aw_dropped[off+sk]; bool m = dropout_mask[off+sk];
        float gw = m ? r*dropout_scale : 0.f;
        float aw = __bfloat162float(attn_weights[off+sk]);
        gw_r[n]=gw; aw_r[n]=aw; ls += gw*aw; n++;
    }
    ls = warp_reduce_sum(ls); if(lane==0) ws[wid]=ls; __syncthreads();
    float ts=0; if(tid<8) ts=ws[tid];
    if(tid<32){ts=warp_reduce_sum(ts); if(tid==0) ws[0]=ts;} __syncthreads();
    ts=ws[0]; n=0;
    for (int sk = tid; sk < seq_len_kv; sk += 256)
        { grad_attn_scores[off+sk] = __float2bfloat16(aw_r[n]*(gw_r[n]-ts)); n++; }
}

// =====================================================================
// KERNEL 3: Tensor Core grad_value (OPTIMIZED: nested loops, no int div)
//
// Restructured inner loop: iterate over (h_local, sq_chunk) explicitly
// instead of flattened k with k/seq_q and k%seq_q integer divisions.
// This eliminates ~32-cycle integer division per iteration.
//
// Grid: (batch * kv_heads, ceil(seq_kv / BM))
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

    // Nested loop: iterate over heads and sq chunks (NO integer division)
    for (int h_local = 0; h_local < NUM_GROUPS; ++h_local) {
        const int h = kv_head * NUM_GROUPS + h_local;
        // Base offsets for this head
        const long long awd_head_base = ((long long)(b * NUM_HEADS + h) * seq_q) * seq_kv;
        const long long go_head_base_b_h = (long long)b * seq_q * NUM_HEADS + h;

        for (int sq_start = 0; sq_start < seq_q; sq_start += GV_KT) {
            const int kt_size = min(GV_KT, seq_q - sq_start);

            // Load s_A: awd[b, h, sq_start:sq_start+kt, sk_start:sk_start+BM]
            // Layout: awd is (B, 80, seq_q, seq_kv), last dim contiguous
            for (int idx = threadIdx.x; idx < GV_KT * GV_BM; idx += GV_BLOCK) {
                int kl = idx / GV_BM, sk_l = idx % GV_BM;
                int sq = sq_start + kl;
                int sk = sk_start + sk_l;
                __nv_bfloat16 val = {};
                if (kl < kt_size && sk < seq_kv) {
                    val = awd[awd_head_base + (long long)sq * seq_kv + sk];
                }
                s_A[kl * GV_AS + sk_l] = val;
            }

            // Load s_B: go[b, sq_start:sq_start+kt, h, 0:128]
            // Layout: go is (B, seq_q, 80, 128), last dim contiguous
            for (int idx = threadIdx.x; idx < GV_KT * GV_BN; idx += GV_BLOCK) {
                int kl = idx / GV_BN, d = idx % GV_BN;
                int sq = sq_start + kl;
                __nv_bfloat16 val = {};
                if (kl < kt_size) {
                    long long off = ((long long)(b * seq_q + sq) * NUM_HEADS + h) * HEAD_DIM + d;
                    val = go[off];
                }
                s_B[kl * GV_BS + d] = val;
            }
            __syncthreads();

            // WMMA: C += A^T @ B (col_major A for transpose)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[4];

            wmma::load_matrix_sync(a_frag, &s_A[wm * 16], GV_AS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::load_matrix_sync(b_frag[ni], &s_B[wn * 64 + ni * 16], GV_BS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::mma_sync(c_frag[ni], a_frag, b_frag[ni], c_frag[ni]);

            __syncthreads();
        }
    }

    // Store output via shared memory
    float* s_C = reinterpret_cast<float*>(smem);
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni)
        wmma::store_matrix_sync(&s_C[(wm*16)*GV_BN + wn*64 + ni*16], c_frag[ni], GV_BN, wmma::mem_row_major);
    __syncthreads();

    const long long out_base = ((long long)(b*NUM_KV_HEADS+kv_head)*seq_kv+sk_start)*HEAD_DIM;
    for (int idx = threadIdx.x; idx < GV_BM * GV_BN; idx += GV_BLOCK) {
        int m = idx / GV_BN, d = idx % GV_BN;
        if (sk_start + m < seq_kv)
            grad_value[out_base + (long long)m * HEAD_DIM + d] = __float2bfloat16(s_C[m * GV_BN + d]);
    }
}

// =====================================================================
// Launchers
// =====================================================================

void gemm_tc_launcher(
    torch::Tensor& output, const torch::Tensor& grad_attn_output,
    const torch::Tensor& value_states,
    int batch_size, int seq_len_q, int seq_len_kv, cudaStream_t stream
) {
    int sqt = (seq_len_q+G_BM-1)/G_BM, skt = (seq_len_kv+G_BN-1)/G_BN;
    dim3 grid(batch_size*NUM_HEADS, sqt, skt);
    int smem = max((G_BM+G_BN)*(int)G_STRIDE*2, G_BM*G_BN*4);
    cudaFuncSetAttribute(gemm_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gemm_tc_kernel<<<grid, G_BLOCK, smem, stream>>>(
        output.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_states.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv);
}

void fused_dropout_softmax_backward_launcher(
    torch::Tensor& grad_attn_scores, const torch::Tensor& grad_aw_dropped,
    const torch::Tensor& attn_weights, const torch::Tensor& dropout_mask,
    float dropout_scale, int batch_size, int num_heads, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
) {
    fused_dropout_softmax_backward_kernel<<<dim3(batch_size*num_heads, seq_len_q), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_attn_scores.data_ptr<at::BFloat16>()),
        grad_aw_dropped.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        dropout_mask.data_ptr<bool>(), dropout_scale, seq_len_kv);
}

void compute_grad_value_launcher(
    torch::Tensor& grad_value_states, const torch::Tensor& grad_attn_output,
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
