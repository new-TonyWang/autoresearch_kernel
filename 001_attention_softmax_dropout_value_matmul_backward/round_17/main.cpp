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
    // Prepare f32 inputs (reused across steps)
    // ============================================================
    auto go_f32 = grad_attn_output.transpose(1, 2).to(torch::kFloat32);
    // (B, 80, sq, 128)

    auto v_exp = value_states.unsqueeze(2)
                    .expand({batch_size, num_kv_heads, num_groups, seq_len_kv, head_dim})
                    .reshape({batch_size, num_heads, seq_len_kv, head_dim})
                    .to(torch::kFloat32);
    // (B, 80, sk, 128)

    // ============================================================
    // Allocate ONE shared buffer for the large (B, 80, sq, sk) f32 tensor.
    // Used first as grad_aw_dropped output, then reused as awd_f32.
    // Saves ~5 GB of peak memory allocation.
    // ============================================================
    auto buffer = torch::empty(
        {batch_size, num_heads, seq_len_q, seq_len_kv},
        go_f32.options());

    // ============================================================
    // Step 1: First matmul via cuBLAS (writes into buffer)
    // buffer = go @ V^T : (B,80,sq,128) @ (B,80,128,sk) -> (B,80,sq,sk)
    // ============================================================
    torch::matmul_out(buffer, go_f32, v_exp.transpose(-2, -1));

    // ============================================================
    // Step 2: Fused dropout backward + softmax backward (reads buffer)
    // ============================================================
    auto grad_attn_scores = torch::empty(
        {batch_size, num_heads, seq_len_q, seq_len_kv},
        attn_weights.options());

    fused_dropout_softmax_backward_launcher(
        grad_attn_scores, buffer, attn_weights, dropout_mask,
        dropout_scale,
        static_cast<int>(batch_size), static_cast<int>(num_heads),
        static_cast<int>(seq_len_q), static_cast<int>(seq_len_kv),
        stream
    );

    // ============================================================
    // Step 3: Reuse buffer for awd_f32 (bf16 -> f32 in-place conversion)
    // buffer was grad_aw_dropped, now overwrite with attn_weights_dropped as f32
    // ============================================================
    buffer.copy_(attn_weights_dropped);

    // ============================================================
    // Step 4: Second matmul via cuBLAS
    // grad_v = buffer^T @ go : (B,80,sk,sq) @ (B,80,sq,128) -> (B,80,sk,128)
    // ============================================================
    auto grad_v_expanded = torch::matmul(buffer.transpose(-2, -1), go_f32);

    // GQA aggregation
    grad_v_expanded = grad_v_expanded.reshape(
        {batch_size, num_kv_heads, num_groups, seq_len_kv, head_dim});
    auto grad_value_states = grad_v_expanded.sum(2).to(torch::kBFloat16);

    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward v17 - buffer reuse + fused dropout/softmax");
}
