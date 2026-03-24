#pragma once

#ifndef ATTENTION_BACKWARD_KERNEL_H
#define ATTENTION_BACKWARD_KERNEL_H

#include <torch/extension.h>

// Fused dropout backward + softmax backward kernel.
// Replaces 5 separate reference operations:
//   1. grad_aw_dropped * mask / (1-p)
//   2. attn_weights.to(float32)
//   3. (gw * aw).sum(dim=-1)
//   4. aw * (gw - sum_term)
//   5. .to(bfloat16)
void fused_dropout_softmax_backward_launcher(
    torch::Tensor& grad_attn_scores,       // output: (B, 80, sq, sk) bf16
    const torch::Tensor& grad_aw_dropped,  // input:  (B, 80, sq, sk) f32
    const torch::Tensor& attn_weights,     // input:  (B, 80, sq, sk) bf16
    const torch::Tensor& dropout_mask,     // input:  (B, 80, sq, sk) bool
    float dropout_scale,
    int batch_size, int num_heads, int seq_len_q, int seq_len_kv,
    cudaStream_t stream
);

#endif
