#pragma once
#include <string>
#include <vector>

namespace tilegraph::kernel::cuda {
    class CudaFunctionKernelUnit {
       public:
        std::string declareGlobal(
            std::string name,
            std::vector<std::pair<std::string, std::string>> arguments);
        std::string declareDevice(
            std::string name,
            std::vector<std::pair<std::string, std::string>> arguments);
    };

    class CudaFunction {
       public:
        std::string name;
        std::vector<std::string> arguments;
    };
}  // namespace tilegraph::kernel::cuda