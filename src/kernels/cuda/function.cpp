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

    CudaFunction::CudaFunction(std::string name, FuncType func_type,
                               DataType return_type)
        : name(name), func_type(func_type), return_type(return_type) {}

    std::string CudaFunction::declareFunction() {
        std::string func_type_str;
        std::string return_type_str;
        switch (func_type) {
            case FuncType::Global:
                func_type_str = "__global__";
                break;
            case FuncType::Device:
                func_type_str = "__device__";
                break;
            case FuncType::Host:
                func_type_str = "__host__";
                break;
            default:
                fmt::println(
                    "[CudaFunction::declareFunction()] Invalid func_type");
                func_type_str = "__host__";
        }

        switch (return_type) {
            case DataType::Void:
                return_type_str = "void";
                break;
            default:
                // fmt::print("Invalid return type: {}\n", return_type);
                fmt::println(
                    "[CudaFunction::declareFunction()] Invalid return type");
        }

        std::string function =
            fmt::format("{} {} {}(", func_type_str, return_type_str, name);

        for (auto arg : arguments) {
            switch (arg.first) {
                case DataType::Float:
                    function += fmt::format("float* {} ", arg.second);
                    break;
                case DataType::Half:
                    function += fmt::format("half* {} ", arg.second);
                    break;
                default:
                    fmt::println(
                        "[CudaFunction::declareFunction()] Invalid argument "
                        "type");
                    function += fmt::format("int* {} ", arg.second);
            }
        }
        function += ")";
        function += ";\n";
        return function;
    }

}  // namespace tilegraph::kernel::cuda