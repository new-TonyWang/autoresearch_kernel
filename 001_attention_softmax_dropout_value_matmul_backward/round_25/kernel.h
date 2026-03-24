#pragma once
#include <torch/extension.h>

// Fused: TC GEMM (go@V^T) + dropout backward + softmax backward
// Eliminates the 5GB f32 intermediate entirely
void fused_grad_scores_launcher(
    torch::Tensor& grad_attn_scores,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    int batch_size, int seq_len_q, int seq_len_kv,
    float dropout_scale, cudaStream_t stream
);

// Grad value with WMMA tensor cores + GQA built-in
void compute_grad_value_launcher(
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights_dropped,
    int batch_size, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
);
