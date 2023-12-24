#pragma once
#include "core/type.hpp"
#include <fmt/core.h>
#include <string>
namespace tilegraph::kernel::cuda {
    std::string declareMemory(int indient, std::string name,
                              MemoryType mem_type, DataType data_type,
                              uint32_t len) {
        std::string memory_declare;

        for (int i = 0; i < indient; i++) {
            memory_declare += " ";
        }

        switch (mem_type) {
            case MemoryType::Global:
                memory_declare += "";
                break;
            case MemoryType::Shared:
                memory_declare += "__shared__";
                break;
            case MemoryType::Warp:
                break;
            case MemoryType::Register:
                memory_declare += "register";
                break;
            default:
                memory_declare += "";
                break;
        }
        memory_declare += " ";

        switch (data_type) {
            case DataType::Float:
                memory_declare += "float";
                break;
            case DataType::Half:
                memory_declare += "half";
                break;
            default:
                fmt::println("[declareMemory] Invalid data type.");
        }
        memory_declare += " ";
        memory_declare += name;
        memory_declare += fmt::format("[{}];\n", len);
        return memory_declare;
    }
}  // namespace tilegraph::kernel::cuda
