#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int NUM_HEADS = 80;
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_GROUPS = 10;
constexpr int HEAD_DIM = 128;

__device__ inline float warp_sum_f(float val) {
    #pragma unroll for (int m = 16; m > 0; m >>= 1) val += __shfl_xor_sync(0xffffffff, val, m);
    return val;
}

__device__ inline float2 load_float2_bfloat16(const __nv_bfloat16* ptr) {
    __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    return make_float2(__bfloat162float(v2.x), __bfloat162float(v2.y));
}

__global__ void __launch_bounds__(256, 4) attn_bw_v29(__nv_bfloat16* gs, float* gv,
    const __nv_bfloat16* go, const __nv_bfloat16* aw, const __nv_bfloat16* ad,
    const __nv_bfloat16* vs, const bool* dm, int B, int sq, int sv, float ds) {
    int b = blockIdx.x, h = blockIdx.y, q = blockIdx.z;
    if (b >= B || h >= 80 || q >= sq) return;
    int kh = h / 10, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int goff = ((b * sq + q) * 80 + h) * 128, woff = ((b * 80 + h) * sq + q) * sv;
    int voff = ((b * 8 + kh) * sv) * 128, aoff = ((b * 8 + kh) * sv) * 128;
    
    __shared__ float sg[128], ss;
    #pragma unroll for (int d = tid; d < 128; d += 256) sg[d] = __bfloat162float(go[goff + d]);
    if (!tid) ss = 0; __syncthreads();
    
    int wp = 8, sp = (sv + wp - 1) / wp, ss2 = warp * sp, se = min(ss2 + sp, sv);
    float ls = 0;
    
    for (int s = ss2 + lane; s < se; s += 32) {
        float d = 0;
        #pragma unroll for (int i = 0; i < 128; i += 2) {
            float2 v = load_float2_bfloat16(&vs[voff + s * 128 + i]);
            d += sg[i] * v.x + sg[i + 1] * v.y;
        }
        bool m = dm[woff + s];
        float g = m ? d * ds : 0;
        ls += g * __bfloat162float(aw[woff + s]);
        float a = __bfloat162float(ad[woff + s]);
        #pragma unroll for (int i = 0; i < 128; ++i) atomicAdd(&gv[aoff + s * 128 + i], a * sg[i]);
    }
    
    ls = warp_sum_f(ls);
    __shared__ float spw[8];
    if (!lane) spw[warp] = ls;
    __syncthreads();
    if (!warp) { ls = (lane < 8) ? spw[lane] : 0; ls = warp_sum_f(ls); if (!lane) ss = ls; }
    __syncthreads();
    float ts = ss;
    
    for (int s = ss2 + lane; s < se; s += 32) {
        float d = 0;
        #pragma unroll for (int i = 0; i < 128; i += 2) {
            float2 v = load_float2_bfloat16(&vs[voff + s * 128 + i]);
            d += sg[i] * v.x + sg[i + 1] * v.y;
        }
        bool m = dm[woff + s];
        float g = m ? d * ds : 0, av = __bfloat162float(aw[woff + s]);
        gs[woff + s] = __float2bfloat16(av * (g - ts));
    }
}

__global__ void cv_bf16(__nv_bfloat16* o, const float* i, int n) {
    int x = blockIdx.x * 256 + threadIdx.x;
    if (x < n) o[x] = __float2bfloat16(i[x]);
}

void attention_backward_launcher(torch::Tensor& gs, torch::Tensor& gv, const torch::Tensor& go,
    const torch::Tensor& aw, const torch::Tensor& ad, const torch::Tensor& vs,
    const torch::Tensor& dm, float do_, cudaStream_t s) {
    int B = go.size(0), sq = go.size(1), sv = vs.size(2);
    float ds = do_ > 0 ? 1.0f / (1.0f - do_) : 1.0f;
    auto ga = torch::zeros({B, NUM_KV_HEADS, sv, HEAD_DIM}, torch::TensorOptions().dtype(torch::kFloat32).device(gv.device()));
    attn_bw_v29<<<dim3(B, NUM_HEADS, sq), 256, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(gs.data_ptr<at::BFloat16>()), ga.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(go.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(aw.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ad.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        dm.data_ptr<bool>(), B, sq, sv, ds);
    cv_bf16<<<(gv.numel() + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(gv.data_ptr<at::BFloat16>()), ga.data_ptr<float>(), gv.numel());
}
