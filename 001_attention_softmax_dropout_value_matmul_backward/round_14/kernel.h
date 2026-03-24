#pragma once

#ifndef ATTENTION_BACKWARD_KERNEL_H
#define ATTENTION_BACKWARD_KERNEL_H

#include <torch/extension.h>

void compute_grad_attn_scores_launcher(
    torch::Tensor& grad_attn_scores,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    int batch_size, int seq_len_q, int seq_len_kv,
    float dropout_scale,
    cudaStream_t stream
);

#endif
