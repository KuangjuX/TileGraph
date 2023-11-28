#include "core/type.hpp"

namespace tilegraph {
    std::string toString(OperatorType op) {
        switch (op) {
            case OperatorType::ADD:
                return "Add";
            case OperatorType::SUB:
                return "Sub";
            case OperatorType::GEMM:
                return "Gemm";
            case OperatorType::RELU:
                return "Relu";
            case OperatorType::GEMM_RELU:
                return "GemmRelu";
            default:
                return "Unknown";
        }
    }
}  // namespace tilegraph