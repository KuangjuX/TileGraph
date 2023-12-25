#pragma once
#include "kernels/cuda/cuda_var.hpp"
#include "kernels/iteration.hpp"
#include <variant>

namespace tilegraph::kernel::cuda {
    class CudaIteration : public Iteration<CudaVar> {
       public:
        CudaIteration(std::unique_ptr<CudaVar> iter_var,
                      std::variant<int, std::shared_ptr<CudaVar>> step,
                      std::variant<int, std::shared_ptr<CudaVar>> start,
                      std::variant<int, std::shared_ptr<CudaVar>> end);
    };

}  // namespace tilegraph::kernel::cuda