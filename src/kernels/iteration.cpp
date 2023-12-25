#include "kernels/iteration.hpp"
#include "kernels/cuda/cuda_var.hpp"
#include <fmt/core.h>

namespace tilegraph::kernel {
    template <>
    Iteration<cuda::CudaVar>::Iteration(
        std::unique_ptr<cuda::CudaVar> iter_var,
        std::variant<int, std::shared_ptr<cuda::CudaVar>> step,
        std::variant<int, std::shared_ptr<cuda::CudaVar>> start,
        std::variant<int, std::shared_ptr<cuda::CudaVar>> end)
        : iter_var(std::move(iter_var)), step(step), start(start), end(end) {}

    template <>
    std::string Iteration<cuda::CudaVar>::genIter(int indient) {
        std::string iter;
        for (int i = 0; i < indient; i++) {
            iter += " ";
        }
        auto start_var =
            std::get_if<int>(&start) == nullptr
                ? std::get<std::shared_ptr<cuda::CudaVar>>(start)->name
                : std::to_string(std::get<int>(start));
        auto end_var = std::get_if<int>(&end) == nullptr
                           ? std::get<std::shared_ptr<cuda::CudaVar>>(end)->name
                           : std::to_string(std::get<int>(end));
        auto step_var =
            std::get_if<int>(&step) == nullptr
                ? std::get<std::shared_ptr<cuda::CudaVar>>(step)->name
                : std::to_string(std::get<int>(step));
        iter += fmt::format("for (int {} = {}; {} < {}; {} += {}) {{\n",
                            iter_var->name, start_var, iter_var->name, end_var,
                            iter_var->name, step_var);
        return iter;
    }

    template <>
    std::string Iteration<cuda::CudaVar>::getIterVar() {
        return iter_var->name;
    }

}  // namespace tilegraph::kernel