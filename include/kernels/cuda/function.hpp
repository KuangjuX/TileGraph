#pragma once
#include "core/type.hpp"
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
        FuncType func_type;
        std::vector<std::pair<DataType, std::string>> arguments;
        DataType return_type;

        CudaFunction(std::string name, FuncType func_type,
                     DataType return_type);

        std::string declareFunction();
    };
}  // namespace tilegraph::kernel::cuda