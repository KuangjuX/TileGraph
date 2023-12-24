#pragma once
#include "core/type.hpp"
#include "kernels/cuda/tensor_core.hpp"
#include <string>

namespace tilegraph::kernel::cuda {
    class CudaVar {
       public:
        MemoryType memory_level;
        // TensorDatatype data_type
        DataType data_type;
        uint32_t len;
        std::string name;

        CudaVar(MemoryType memory_level, DataType data_type, uint32_t len,
                std::string name);

        std::string declareVar(int indient);
        std::string initVar(int indient);
        std::string getVarIndex(uint32_t index);
        std::string getVarIndexByVar(std::string index);
    };
}  // namespace tilegraph::kernel::cuda