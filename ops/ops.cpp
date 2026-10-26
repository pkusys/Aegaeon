#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swap", &swap, "Aegaeon KV cache swap kernel.");
}