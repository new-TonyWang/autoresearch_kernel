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

    const int batch_size = static_cast<int>(grad_attn_output.size(0));
    const int seq_len_q  = static_cast<int>(grad_attn_output.size(1));
    const int seq_len_kv = static_cast<int>(value_states.size(2));

    const float dropout_scale = attention_dropout > 0.0 ?
        1.0f / (1.0f - static_cast<float>(attention_dropout)) : 1.0f;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // f32 intermediate for GEMM output
    auto grad_aw_dropped = torch::empty(
        {batch_size, 80, seq_len_q, seq_len_kv},
        grad_attn_output.options().dtype(torch::kFloat32));

    // Step 1: Tensor Core GEMM: go @ V^T → f32 intermediate
    gemm_tc_launcher(
        grad_aw_dropped, grad_attn_output, value_states,
        batch_size, seq_len_q, seq_len_kv, stream);

    // Step 2: Fused dropout + softmax backward → bf16 output
    auto grad_attn_scores = torch::empty(
        {batch_size, 80, seq_len_q, seq_len_kv},
        attn_weights.options());

    fused_dropout_softmax_backward_launcher(
        grad_attn_scores, grad_aw_dropped, attn_weights, dropout_mask,
        dropout_scale, batch_size, 80, seq_len_q, seq_len_kv, stream);

    // Step 3: Grad value (custom kernel with GQA)
    auto grad_value_states = torch::empty(
        {batch_size, 8, seq_len_kv, 128},
        value_states.options());

    compute_grad_value_launcher(
        grad_value_states, grad_attn_output, attn_weights_dropped,
        batch_size, seq_len_q, seq_len_kv, stream);

    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward v22 - Tensor Core GEMM + fused ops");
}
