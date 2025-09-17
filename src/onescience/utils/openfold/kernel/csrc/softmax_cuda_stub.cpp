// modified from fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp

#include <torch/extension.h>

void attn_softmax_inplace_forward_(
    at::Tensor input, 
    long long rows, int cols
)
{
    throw std::runtime_error("attn_softmax_inplace_forward_ not implemented on CPU");
};
void attn_softmax_inplace_backward_(
    at::Tensor output, 
    at::Tensor d_ov,
    at::Tensor values,
    long long rows, 
    int cols_output,
    int cols_values
)
{
    throw std::runtime_error("attn_softmax_inplace_backward_ not implemented on CPU");
};