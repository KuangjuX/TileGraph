#pragma once
#include "kernels/cuda/function.hpp"
#include "kernels/cuda/vars.hpp"
#include <memory>

namespace tilegraph::kernel::cuda {
    class CudaGEMMKernel {
       public:
        // GEMM parametes.
        uint32_t ShardedM;
        uint32_t ShardedN;
        uint32_t ShardedK;
        uint32_t WarpM;
        uint32_t WarpN;
        uint32_t WarpK;

        bool transpose_a;
        bool transpose_b;

        // Variables
        std::vector<CudaVar> vars;
        // Functions
        std::vector<CudaFunction> functions;
        // Function Unit
        std::unique_ptr<CudaFunctionKernelUnit> function_unit;

        CudaGEMMKernel(uint32_t ShardedM, uint32_t ShardedN, uint32_t ShardedK,
                       uint32_t WarpM, uint32_t WarpN, uint32_t WarpK,
                       bool tramspose_a, bool transpose_b);
        std::string genTCGEMM(std::string name);
    };
}  // namespace tilegraph::kernel::cuda