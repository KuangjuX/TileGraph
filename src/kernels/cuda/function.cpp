#include "kernels/cuda/function.hpp"
#include <fmt/core.h>

namespace tilegraph::kernel::cuda {
    std::string CudaFunctionKernelUnit::declareGlobal(
        std::string name,
        std::vector<std::pair<std::string, std::string>> arguments) {
        std::string function = fmt::format("__global__ void {} (", name);
        for (auto arg : arguments) {
            function += fmt::format("{} {}", arg.first, arg.second);
        }
        function += ")";
        return function;
    }

}  // namespace tilegraph::kernel::cuda