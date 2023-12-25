#include "kernels/cuda/cuda_iteration.hpp"

namespace tilegraph::kernel::cuda {
    CudaIteration::CudaIteration(
        std::unique_ptr<CudaVar> iter_var,
        std::variant<int, std::shared_ptr<CudaVar>> step,
        std::variant<int, std::shared_ptr<CudaVar>> start,
        std::variant<int, std::shared_ptr<CudaVar>> end)
        : Iteration<CudaVar>(std::move(iter_var), step, start, end) {}
}  // namespace tilegraph::kernel::cuda
