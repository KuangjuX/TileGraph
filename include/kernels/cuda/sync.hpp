#pragma once
#include "core/type.hpp"
#include <fmt/core.h>
#include <string>

namespace tilegraph::kernel::cuda {
    std::string insertSyncnorize(int indient, MemoryType memory_level) {
        std::string sync;
        for (int i = 0; i < indient; i++) {
            sync += " ";
        }
        switch (memory_level) {
            case MemoryType::Global:
                sync += "cudaDeviceSynchronize();\n";
                break;
            case MemoryType::Shared:
                sync += "__syncthreads();\n";
                break;
            case MemoryType::Register:
                fmt::println("Register level not supported");
                break;
            default:
                fmt::println("Failed to get memory level");
        }

        return sync;
    }
}  // namespace tilegraph::kernel::cuda