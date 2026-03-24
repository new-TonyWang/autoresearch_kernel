#pragma once

#ifndef ATTENTION_BACKWARD_KERNEL_H
#define ATTENTION_BACKWARD_KERNEL_H

#include <torch/extension.h>

/**
 * @brief Host-side launcher for the attention backward CUDA kernel.
 *
 * Computes gradients through:
 * 1. Transpose gradient
 * 2. Batched matmul gradients
 * 3. Dropout gradient
 * 4. Softmax gradient
 * 5. GQA gradient aggregation
 */
void attention_backward_launcher(
    torch::Tensor& grad_attn_scores,
    torch::Tensor& grad_value_states,
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    float attention_dropout,
    cudaStream_t stream
);

#endif // ATTENTION_BACKWARD_KERNEL_H
