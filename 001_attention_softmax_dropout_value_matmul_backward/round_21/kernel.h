#pragma once
#include <torch/extension.h>

// Tiled GEMM: C = A @ B^T  where A is grad_out, B is value_states
void gemm_go_vt_launcher(
    torch::Tensor& output,              // (B, 80, sq, sk) f32
    const torch::Tensor& grad_attn_output,  // (B, sq, 80, 128) bf16
    const torch::Tensor& value_states,      // (B, 8, sk, 128) bf16
    int batch_size, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
);

// Fused dropout backward + softmax backward
void fused_dropout_softmax_backward_launcher(
    torch::Tensor& grad_attn_scores,       // (B, 80, sq, sk) bf16
    const torch::Tensor& grad_aw_dropped,  // (B, 80, sq, sk) f32
    const torch::Tensor& attn_weights,     // (B, 80, sq, sk) bf16
    const torch::Tensor& dropout_mask,     // (B, 80, sq, sk) bool
    float dropout_scale,
    int batch_size, int num_heads, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
);

// Grad value with built-in GQA
void compute_grad_value_launcher(
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights_dropped,
    int batch_size, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
);
