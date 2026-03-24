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
    // Validate inputs
    TORCH_CHECK(grad_attn_output.is_cuda(), "grad_attn_output must be CUDA");
    TORCH_CHECK(grad_attn_output.dtype() == torch::kBFloat16, "grad_attn_output must be bfloat16");
    TORCH_CHECK(grad_attn_output.dim() == 4, "grad_attn_output must be 4D");
    
    int batch_size = grad_attn_output.size(0);
    int seq_len_q = grad_attn_output.size(1);
    int seq_len_kv = value_states.size(2);
    
    auto grad_attn_scores = torch::empty_like(attn_weights);
    auto grad_value_states = torch::zeros_like(value_states);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    attention_backward_launcher(
        grad_attn_scores,
        grad_value_states,
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        static_cast<float>(attention_dropout),
        stream
    );
    
    return {grad_attn_scores, grad_value_states};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Attention backward CUDA");
}
