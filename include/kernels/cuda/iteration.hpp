#pragma once
#include "kernels/cuda/vars.hpp"
#include <variant>

namespace tilegraph::kernel::cuda {
    class Iteration {
       public:
        std::unique_ptr<CudaVar> iter_var;
        // The step of the iteration.
        std::variant<int, std::shared_ptr<CudaVar>> step;
        // The start and end of the iteration.
        std::variant<int, std::shared_ptr<CudaVar>> start;
        std::variant<int, std::shared_ptr<CudaVar>> end;

       public:
        Iteration(std::unique_ptr<CudaVar> iter_var,
                  std::variant<int, std::shared_ptr<CudaVar>> step,
                  std::variant<int, std::shared_ptr<CudaVar>> start,
                  std::variant<int, std::shared_ptr<CudaVar>> end);

        std::string genIter(int indient);
        std::string getIterVar();
    };
}  // namespace tilegraph::kernel::cuda