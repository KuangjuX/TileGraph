#pragma once
#include "core/type.hpp"
#include "kernels/cuda/memory.hpp"
#include <string>

namespace tilegraph::kernel::cuda {
    class CudaVar {
       public:
        MemoryType memory_level;
        TensorDatatype data_type;
        uint32_t len;
        std::string name;

        CudaVar(MemoryType memory_level, TensorDatatype data_type, uint32_t len,
                std::string name)
            : memory_level(memory_level),
              data_type(data_type),
              len(len),
              name(name) {}

        std::string declareVar(int indient) {
            auto var =
                declareMemory(indient, name, memory_level, data_type, len);
            return var;
        }

        std::string initVar(int indient) { return ""; }
    };
}  // namespace tilegraph::kernel::cuda