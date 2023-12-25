#include "kernels/cuda/cuda_var.hpp"
#include "kernels/cuda/memory.hpp"

namespace tilegraph::kernel::cuda {
    CudaVar::CudaVar(MemoryType memory_level, DataType data_type, uint32_t len,
                     std::string name)
        : Var(memory_level, data_type, len, name) {}

    std::string CudaVar::declareVar(int indient) {
        std::string var;
        if (memory_level != MemoryType::Warp) {
            var = declareMemory(indient, name, memory_level, data_type, len);
        } else {
        }
        return var;
    }

    std::string CudaVar::initVar(int indient) {
        std::string var;
        for (int i = 0; i < indient; i++) {
            var += " ";
        }
        return var;
    }

    std::string CudaVar::getVarIndex(uint32_t index) {
        return fmt::format("{}[{}]", name, index);
    }

    std::string CudaVar::getVarIndexByVar(std::string index) {
        return fmt::format("{}[{}]", name, index);
    }
}  // namespace tilegraph::kernel::cuda