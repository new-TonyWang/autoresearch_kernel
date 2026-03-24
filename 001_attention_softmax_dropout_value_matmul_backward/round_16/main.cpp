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
    // Step 1: Prepare f32 grad_attn_output (reused by both matmuls)
    // ============================================================
    // (B, sq, 80, 128) -> (B, 80, sq, 128) f32
    auto go_f32 = grad_attn_output.transpose(1, 2).to(torch::kFloat32);

    // ============================================================
    // Step 2: First matmul - grad_aw_dropped = go @ V^T
    // Avoid expanding V to 80 heads: compute per KV-group instead
    // 8 groups of 10 heads each, reusing the same V data
    // ============================================================
    // V in f32: (B, 8, sk, 128)
    auto v_f32 = value_states.to(torch::kFloat32);
    // V transposed: (B, 8, 128, sk)
    auto v_t = v_f32.transpose(-2, -1);

    // Allocate output: (B, 80, sq, sk) f32
    auto grad_aw_dropped = torch::empty(
        {batch_size, num_heads, seq_len_q, seq_len_kv},
        go_f32.options());

    // Process each KV group: 10 query heads share the same V
    for (int64_t kv = 0; kv < num_kv_heads; ++kv) {
        // go slice: (B, 10, sq, 128) for heads [kv*10 .. kv*10+9]
        auto go_group = go_f32.narrow(1, kv * num_groups, num_groups);
        // V slice: (B, 1, 128, sk) -> broadcast to (B, 10, 128, sk)
        auto v_kv = v_t.narrow(1, kv, 1);

        // matmul with broadcast: (B, 10, sq, 128) @ (B, 1, 128, sk) -> (B, 10, sq, sk)
        auto result = torch::matmul(go_group, v_kv);

        // Copy into output
        grad_aw_dropped.narrow(1, kv * num_groups, num_groups).copy_(result);
    }

    // ============================================================
    // Step 3: Fused dropout backward + softmax backward
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
    // Step 4: Second matmul - grad_value = awd^T @ go
    // Also per KV-group to avoid expanding awd to f32 (saves ~5GB)
    // ============================================================
    // Output: (B, 8, sk, 128) f32
    auto grad_value_f32 = torch::zeros(
        {batch_size, num_kv_heads, seq_len_kv, head_dim},
        go_f32.options());

    // Convert awd per-group to avoid 5GB peak f32 allocation
    for (int64_t kv = 0; kv < num_kv_heads; ++kv) {
        // Convert only this group's 10 heads to f32 (1/8 of full tensor)
        auto awd_group_f32 = attn_weights_dropped.narrow(1, kv * num_groups, num_groups)
                                .to(torch::kFloat32);
        // Transpose: (B, 10, sq, sk) -> (B, 10, sk, sq)
        auto awd_t = awd_group_f32.transpose(-2, -1);
        // go slice: (B, 10, sq, 128)
        auto go_group = go_f32.narrow(1, kv * num_groups, num_groups);

        // (B, 10, sk, sq) @ (B, 10, sq, 128) -> (B, 10, sk, 128)
        auto result = torch::matmul(awd_t, go_group);

        // Sum over 10 heads -> (B, 1, sk, 128) and add to output
        grad_value_f32.narrow(1, kv, 1).add_(result.sum(1, /*keepdim=*/true));
    }

    auto grad_value_states = grad_value_f32.to(torch::kBFloat16);

    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward v16 - per-group matmuls + fused dropout/softmax");
}
