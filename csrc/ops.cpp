#include <torch/extension.h>

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logp", &fused_logp_forward, "Fused logp forward");
}
