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
    const int64_t num_heads      = 80;
    const int64_t num_kv_heads   = 8;
    const int64_t num_groups     = 10;
    const int64_t head_dim       = 128;
    const int64_t seq_len_kv = value_states.size(2);

    const float dropout_scale = attention_dropout > 0.0 ?
        1.0f / (1.0f - static_cast<float>(attention_dropout)) : 1.0f;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ============================================================
    // Step 1: Prepare f32 inputs
    // ============================================================
    // Transpose grad_out: (B, sq, 80, 128) -> (B, 80, sq, 128) in f32
    auto go_f32 = grad_attn_output.transpose(1, 2).to(torch::kFloat32);

    // Expand value states for GQA: (B, 8, sk, 128) -> (B, 80, sk, 128) in f32
    auto v_exp = value_states.unsqueeze(2)
                    .expand({batch_size, num_kv_heads, num_groups, seq_len_kv, head_dim})
                    .reshape({batch_size, num_heads, seq_len_kv, head_dim})
                    .to(torch::kFloat32);

    // ============================================================
    // Step 2: First matmul via cuBLAS (TF32 tensor cores)
    // grad_aw_dropped = go @ V^T : (B,80,sq,128) @ (B,80,128,sk) -> (B,80,sq,sk) f32
    // ============================================================
    auto grad_aw_dropped = torch::matmul(go_f32, v_exp.transpose(-2, -1));

    // ============================================================
    // Step 3: Fused dropout backward + softmax backward (custom kernel)
    // Replaces 5 separate operations in the reference.
    // ============================================================
    auto grad_attn_scores = torch::empty(
        {batch_size, num_heads, seq_len_q, seq_len_kv},
        attn_weights.options());  // bf16

    fused_dropout_softmax_backward_launcher(
        grad_attn_scores, grad_aw_dropped, attn_weights, dropout_mask,
        dropout_scale,
        static_cast<int>(batch_size), static_cast<int>(num_heads),
        static_cast<int>(seq_len_q), static_cast<int>(seq_len_kv),
        stream
    );

    // ============================================================
    // Step 4: Second matmul via cuBLAS for grad_value
    // grad_v_expanded = awd^T @ go : (B,80,sk,sq) @ (B,80,sq,128) -> (B,80,sk,128) f32
    // ============================================================
    auto awd_f32 = attn_weights_dropped.to(torch::kFloat32);
    auto grad_v_expanded = torch::matmul(awd_f32.transpose(-2, -1), go_f32);

    // GQA aggregation: (B, 80, sk, 128) -> (B, 8, 10, sk, 128) -> sum -> (B, 8, sk, 128)
    grad_v_expanded = grad_v_expanded.reshape(
        {batch_size, num_kv_heads, num_groups, seq_len_kv, head_dim});
    auto grad_value_states = grad_v_expanded.sum(2).to(torch::kBFloat16);

    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward v15 - cuBLAS matmuls + fused dropout/softmax");
}
