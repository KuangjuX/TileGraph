#pragma once
#include <memory>

#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"

namespace tilegraph::kernel {
    class KernelEmiter {
       public:
        std::shared_ptr<graph::GNode> op;
        // The input tensor
        std::shared_ptr<Tensor> inputs;
        // The output tensor
        std::shared_ptr<Tensor> outputs;
        // The allocated tensor
        std::shared_ptr<Tensor> allocated_tensor;
    };
}  // namespace tilegraph::kernels
