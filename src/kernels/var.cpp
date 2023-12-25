#include "kernels/var.hpp"

namespace tilegraph::kernel {
    Var::Var(MemoryType memory_level, DataType data_type, uint32_t len,
             std::string name)
        : memory_level(memory_level),
          data_type(data_type),
          len(len),
          name(name) {}
}  // namespace tilegraph::kernel