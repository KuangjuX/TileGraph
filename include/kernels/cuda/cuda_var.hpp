#pragma once
#include "kernels/var.hpp"
#include "kernels/cuda/tensor_core.hpp"
#include <string>

namespace tilegraph::kernel::cuda {
    class CudaVar : public Var {
       public:
        CudaVar(MemoryType memory_level, DataType data_type, uint32_t len,
                std::string name);
        ~CudaVar() = default;

        std::string declareVar(int indient) override;
        std::string initVar(int indient) override;
        std::string getVarIndex(uint32_t index) override;
        std::string getVarIndexByVar(std::string index) override;
    };
}  // namespace tilegraph::kernel::cuda