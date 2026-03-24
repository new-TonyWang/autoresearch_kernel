#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// Helper function to check tensor properties
void check_tensor(const torch::Tensor& tensor, const std::string& name, 
                  c10::ScalarType dtype, int ndim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == dtype, name, " must have ", dtype, " dtype, but got ", tensor.dtype());
    TORCH_CHECK(tensor.dim() == ndim, name, " must be ", ndim, "D, but got ", tensor.dim(), " dimensions");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> run(
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    double attention_dropout
) {
    // --- Input Validation ---
    check_tensor(grad_attn_output, "grad_attn_output", torch::kBFloat16, 4);
    check_tensor(attn_weights, "attn_weights", torch::kBFloat16, 4);
    check_tensor(attn_weights_dropped, "attn_weights_dropped", torch::kBFloat16, 4);
    check_tensor(value_states, "value_states", torch::kBFloat16, 4);
    check_tensor(dropout_mask, "dropout_mask", torch::kBool, 4);
    
    const int batch_size = grad_attn_output.size(0);
    const int seq_len_q = grad_attn_output.size(1);
    const int seq_len_kv = value_states.size(2);
    const int head_dim = grad_attn_output.size(3);
    
    TORCH_CHECK(grad_attn_output.size(2) == 80, "grad_attn_output must have 80 heads");
    TORCH_CHECK(grad_attn_output.size(3) == 128, "grad_attn_output must have head_dim=128");
    TORCH_CHECK(attn_weights.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(attn_weights.size(1) == 80, "attn_weights must have 80 heads");
    TORCH_CHECK(attn_weights.size(2) == seq_len_q, "seq_len_q mismatch");
    TORCH_CHECK(attn_weights.size(3) == seq_len_kv, "seq_len_kv mismatch");
    TORCH_CHECK(value_states.size(1) == 8, "value_states must have 8 kv_heads");
    TORCH_CHECK(value_states.size(3) == 128, "value_states must have head_dim=128");
    
    // --- Allocate Output Tensors ---
    auto grad_attn_scores = torch::empty_like(attn_weights);
    auto grad_value_states = torch::empty_like(value_states);
    
    // --- Launch Kernel ---
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
    m.def("run", &run, "Attention backward pass CUDA kernel");
}
