#pragma once
#include "core/type.hpp"

namespace tilegraph::kernel {
    class GEMMKernel {
       public:
        // GEMM parametes.
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t ShardedM;
        uint32_t ShardedN;
        uint32_t ShardedK;
        uint32_t WarpM;
        uint32_t WarpN;
        uint32_t WarpK;
        uint32_t WmmaM;
        uint32_t WmmaN;
        uint32_t WmmaK;

        bool transpose_a;
        bool transpose_b;

        MemoryType memory_level;
    };
}  // namespace tilegraph::kernel