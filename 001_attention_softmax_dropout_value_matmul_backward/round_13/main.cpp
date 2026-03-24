#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

std::vector<torch::Tensor> run(
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    double attention_dropout
) {
    TORCH_CHECK(grad_attn_output.is_cuda(), "CUDA required");
    TORCH_CHECK(grad_attn_output.dtype() == torch::kBFloat16, "bfloat16 required");

    const int64_t batch_size = grad_attn_output.size(0);
    const int64_t seq_len_q  = grad_attn_output.size(1);
    const int64_t seq_len_kv = value_states.size(2);
    const int64_t num_heads      = 80;
    const int64_t num_kv_heads   = 8;
    const int64_t num_groups     = 10;
    const int64_t head_dim       = 128;

    const float dropout_scale = attention_dropout > 0.0 ?
        1.0f / (1.0f - static_cast<float>(attention_dropout)) : 1.0f;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ============================================================
    // Output 1: grad_attn_scores  (fused custom CUDA kernel)
    // ============================================================
    auto grad_attn_scores = torch::empty_like(attn_weights);

    compute_grad_attn_scores_launcher(
        grad_attn_scores,
        grad_attn_output, attn_weights, value_states, dropout_mask,
        static_cast<int>(batch_size),
        static_cast<int>(seq_len_q),
        static_cast<int>(seq_len_kv),
        dropout_scale, stream
    );

    // ============================================================
    // Output 2: grad_value_states  (bf16 matmul on tensor cores)
    // Avoids expensive .to(float32) of the huge attention tensors.
    // bf16 tensor cores accumulate in fp32 internally.
    // ============================================================

    // Transpose without copy (just views)
    auto grad_out_t = grad_attn_output.transpose(1, 2);  // (B, 80, sq, 128) bf16
    auto awd_t = attn_weights_dropped.transpose(-2, -1); // (B, 80, sk, sq) bf16

    // bf16 batched matmul: (B, 80, sk, sq) @ (B, 80, sq, 128) -> (B, 80, sk, 128)
    // Uses tensor cores with internal fp32 accumulation
    auto grad_v_expanded = torch::matmul(awd_t, grad_out_t);

    // GQA aggregation: sum groups in float32 for precision
    // (B, 80, sk, 128) -> (B, 8, 10, sk, 128) -> sum dim 2 -> (B, 8, sk, 128)
    grad_v_expanded = grad_v_expanded.reshape(
        {batch_size, num_kv_heads, num_groups, seq_len_kv, head_dim});
    auto grad_value_states = grad_v_expanded.to(torch::kFloat32)
                             .sum(2)
                             .to(torch::kBFloat16);

    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward v13 - bf16 matmul for grad_value");
}
