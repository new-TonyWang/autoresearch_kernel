#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
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
// KERNEL 1: TC GEMM with cp.async TMA double-buffered K-tiles
//
// Double-buffers the K-tile loads: while WMMA computes with buffer[i],
// cp.async loads buffer[i+1] from global memory via TMA unit.
// This overlaps memory latency with tensor core compute.
//
// Grid: (batch*heads, ceil(sq/64), ceil(sk/64))
// =====================================================================

constexpr int G_BM = 64, G_BN = 64, G_BLOCK = 256;
constexpr int G_PAD = 8, G_STRIDE = 16 + G_PAD;  // 24
constexpr int K_STEPS = HEAD_DIM / 16;  // 8

// Helper: async load one K-tile into shared memory via cp.async
__device__ __forceinline__ void gemm_async_load_tile(
    __nv_bfloat16* s_A, __nv_bfloat16* s_B,
    const __nv_bfloat16* go, const __nv_bfloat16* V,
    int b, int h, int kv_head, int sq_start, int sk_start,
    int k0, int seq_q, int seq_kv, int tid
) {
    // A tile: 64 rows × 16 cols = 128 copies of 16 bytes (8 bf16 each)
    // B tile: 64 rows × 16 cols = 128 copies of 16 bytes
    // Total: 256 copies, 256 threads → 1 copy per thread
    if (tid < 128) {
        int row = tid >> 1;      // 0..63
        int chunk = tid & 1;     // 0 or 1 (two 8-element chunks per row)
        int sq = sq_start + row;
        __nv_bfloat16* dst = &s_A[row * G_STRIDE + chunk * 8];
        if (sq < seq_q) {
            const __nv_bfloat16* src = &go[((long long)(b*seq_q+sq)*NUM_HEADS+h)*HEAD_DIM + k0 + chunk*8];
            __pipeline_memcpy_async(dst, src, 16);
        }
    } else {
        int idx = tid - 128;
        int row = idx >> 1;
        int chunk = idx & 1;
        int sk = sk_start + row;
        __nv_bfloat16* dst = &s_B[row * G_STRIDE + chunk * 8];
        if (sk < seq_kv) {
            const __nv_bfloat16* src = &V[((long long)(b*NUM_KV_HEADS+kv_head)*seq_kv+sk)*HEAD_DIM + k0 + chunk*8];
            __pipeline_memcpy_async(dst, src, 16);
        }
    }
}

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
    const int tid = threadIdx.x;

    // Double buffer: s_A[2][BM*STRIDE], s_B[2][BN*STRIDE]
    extern __shared__ char smem[];
    const int ab_size = G_BM * G_STRIDE;  // per-buffer A size (in bf16 elements)
    const int bb_size = G_BN * G_STRIDE;
    __nv_bfloat16* s_buf = reinterpret_cast<__nv_bfloat16*>(smem);
    // buf 0: s_buf[0 .. ab_size+bb_size-1]
    // buf 1: s_buf[ab_size+bb_size .. 2*(ab_size+bb_size)-1]

    #define S_A(buf) (&s_buf[(buf) * (ab_size + bb_size)])
    #define S_B(buf) (&s_buf[(buf) * (ab_size + bb_size) + ab_size])

    // Pre-zero both buffers for boundary handling
    for (int i = tid; i < 2 * (ab_size + bb_size); i += G_BLOCK)
        s_buf[i] = __nv_bfloat16();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    // Preload K-step 0 into buffer 0
    gemm_async_load_tile(S_A(0), S_B(0), go, V, b, h, kv_head,
                         sq_start, sk_start, 0, seq_q, seq_kv, tid);
    __pipeline_commit();

    for (int ks = 0; ks < K_STEPS; ++ks) {
        int cur = ks & 1;
        int nxt = 1 - cur;

        // Wait for current buffer's async load to complete
        __pipeline_wait_prior(0);
        __syncthreads();

        // Start async load of next K-step into other buffer
        if (ks + 1 < K_STEPS) {
            gemm_async_load_tile(S_A(nxt), S_B(nxt), go, V, b, h, kv_head,
                                 sq_start, sk_start, (ks+1)*16, seq_q, seq_kv, tid);
            __pipeline_commit();
        }

        // WMMA compute with current buffer (overlaps with next load)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0f, b1f;

        wmma::load_matrix_sync(af, &S_A(cur)[wm * 16 * G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b0f, &S_B(cur)[(wn*32) * G_STRIDE], G_STRIDE);
        wmma::load_matrix_sync(b1f, &S_B(cur)[(wn*32+16) * G_STRIDE], G_STRIDE);

        wmma::mma_sync(c0, af, b0f, c0);
        wmma::mma_sync(c1, af, b1f, c1);
    }
    __syncthreads();

    #undef S_A
    #undef S_B

    // Store output via shared memory (reuse smem)
    float* s_C = reinterpret_cast<float*>(smem);
    wmma::store_matrix_sync(&s_C[(wm*16)*G_BN + wn*32], c0, G_BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_C[(wm*16)*G_BN + wn*32+16], c1, G_BN, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < G_BM * G_BN; idx += G_BLOCK) {
        int m = idx / G_BN, n = idx % G_BN;
        int sq = sq_start + m, sk = sk_start + n;
        if (sq < seq_q && sk < seq_kv)
            output[((long long)(b*NUM_HEADS+h)*seq_q+sq)*seq_kv+sk] = s_C[m*G_BN+n];
    }
}

// =====================================================================
// KERNEL 2: Fused dropout + softmax backward (unchanged from round 24)
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
// KERNEL 3: TC grad_value with cp.async TMA double-buffered loads
// =====================================================================

constexpr int GV_BM = 64, GV_BN = 128, GV_KT = 16, GV_BLOCK = 256;
constexpr int GV_APAD = 8, GV_BPAD = 8;
constexpr int GV_AS = GV_BM + GV_APAD, GV_BS = GV_BN + GV_BPAD;

// Helper: load awd (regular, may be misaligned) + go (cp.async, always aligned)
__device__ __forceinline__ void gv_load_tile(
    __nv_bfloat16* s_A, __nv_bfloat16* s_B,
    const __nv_bfloat16* awd, const __nv_bfloat16* go,
    int b, int h, int kv_head, int sk_start, int sq_off, int seq_q, int seq_kv,
    int tid
) {
    // A (awd): regular loads (seq_kv stride may not be 16-byte aligned)
    for (int idx = tid; idx < GV_KT * GV_BM; idx += GV_BLOCK) {
        int kl = idx / GV_BM, sk_l = idx % GV_BM;
        int sq = sq_off + kl, sk = sk_start + sk_l;
        __nv_bfloat16 val = {};
        if (kl < min(GV_KT, seq_q - sq_off) && sk < seq_kv) {
            long long off = ((long long)(b*NUM_HEADS+h)*seq_q+sq)*seq_kv+sk;
            val = awd[off];
        }
        s_A[kl * GV_AS + sk_l] = val;
    }

    // B (go): cp.async 16-byte copies (HEAD_DIM stride always 16-byte aligned)
    // 16 rows × 16 chunks of 8 = 256 copies, 256 threads → 1 per thread
    {
        int row = tid / 16;
        int chunk = tid % 16;
        int sq = sq_off + row;
        __nv_bfloat16* dst = &s_B[row * GV_BS + chunk * 8];
        if (row < GV_KT && sq < seq_q) {
            long long src_off = ((long long)(b*seq_q+sq)*NUM_HEADS+h)*HEAD_DIM + chunk*8;
            __pipeline_memcpy_async(dst, &go[src_off], 16);
        }
    }
}

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
    const int tid = threadIdx.x;

    // Double-buffered shared memory
    extern __shared__ char smem[];
    const int a_buf_size = GV_KT * GV_AS;
    const int b_buf_size = GV_KT * GV_BS;
    const int buf_stride = a_buf_size + b_buf_size;
    __nv_bfloat16* s_buf = reinterpret_cast<__nv_bfloat16*>(smem);

    #define SA(buf) (&s_buf[(buf) * buf_stride])
    #define SB(buf) (&s_buf[(buf) * buf_stride + a_buf_size])

    // Pre-zero both buffers
    for (int i = tid; i < 2 * buf_stride; i += GV_BLOCK)
        s_buf[i] = __nv_bfloat16();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4];
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(c_frag[i], 0.0f);

    for (int h_local = 0; h_local < NUM_GROUPS; ++h_local) {
        const int h = kv_head * NUM_GROUPS + h_local;

        for (int sq_start = 0; sq_start < seq_q; sq_start += GV_KT) {
            // Load: awd (regular) + go (cp.async)
            gv_load_tile(SA(0), SB(0), awd, go, b, h, kv_head, sk_start,
                         sq_start, seq_q, seq_kv, tid);
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();

            // WMMA compute
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[4];

            wmma::load_matrix_sync(a_frag, &SA(0)[wm * 16], GV_AS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::load_matrix_sync(b_frag[ni], &SB(0)[wn*64+ni*16], GV_BS);
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                wmma::mma_sync(c_frag[ni], a_frag, b_frag[ni], c_frag[ni]);
            __syncthreads();
        }
    }

    #undef SA
    #undef SB

    // Store output via shared memory
    float* s_C = reinterpret_cast<float*>(smem);
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni)
        wmma::store_matrix_sync(&s_C[(wm*16)*GV_BN+wn*64+ni*16], c_frag[ni], GV_BN, wmma::mem_row_major);
    __syncthreads();

    const long long out_base = ((long long)(b*NUM_KV_HEADS+kv_head)*seq_kv+sk_start)*HEAD_DIM;
    for (int idx = tid; idx < GV_BM * GV_BN; idx += GV_BLOCK) {
        int m = idx / GV_BN, d = idx % GV_BN;
        if (sk_start + m < seq_kv)
            grad_value[out_base + (long long)m*HEAD_DIM+d] = __float2bfloat16(s_C[m*GV_BN+d]);
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
    // Double buffer: 2 * (A + B) tiles, or C output
    int smem_db = 2 * (G_BM + G_BN) * G_STRIDE * (int)sizeof(__nv_bfloat16);
    int smem_c = G_BM * G_BN * (int)sizeof(float);
    int smem = max(smem_db, smem_c);
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
    int buf_elems = GV_KT * GV_AS + GV_KT * GV_BS;
    int smem_db = 2 * buf_elems * (int)sizeof(__nv_bfloat16);
    int smem_c = GV_BM * GV_BN * (int)sizeof(float);
    int smem = max(smem_db, smem_c);
    cudaFuncSetAttribute(grad_value_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    grad_value_wmma_kernel<<<grid, GV_BLOCK, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_value_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_attn_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_weights_dropped.data_ptr<at::BFloat16>()),
        batch_size, seq_len_q, seq_len_kv);
}
