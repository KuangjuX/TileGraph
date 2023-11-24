#pragma once
#include <memory>

#include "core/graph/gnode.hpp"
namespace tilegraph::kernels {
    class KernelEmiter {
       public:
        std::shared_ptr<graph::GNode> op;
    };
}  // namespace tilegraph::kernels
