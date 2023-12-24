#pragma once
#include "kernels/cuda/function.hpp"
#include "kernels/cuda/vars.hpp"
#include "kernels/cuda/iteration.hpp"
#include <memory>
#include <set>

namespace tilegraph::kernel::cuda {
    class CudaGEMMKernel {
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

        // Variables
        std::set<std::shared_ptr<CudaVar>> vars;
        // Functions
        std::set<std::shared_ptr<CudaFunction>> functions;
        std::set<std::unique_ptr<Iteration>> iterations;
        // Function Unit
        std::unique_ptr<CudaFunctionKernelUnit> function_unit;

        CudaGEMMKernel(uint32_t M, uint32_t N, uint32_t K, uint32_t ShardedM,
                       uint32_t ShardedN, uint32_t ShardedK, uint32_t WarpM,
                       uint32_t WarpN, uint32_t WarpK, uint32_t WmmaM,
                       uint32_t WmmaN, uint32_t WmmaK, bool tramspose_a,
                       bool transpose_b);
        std::string genTCGEMM(std::string name);
    };
}  // namespace tilegraph::kernel::cuda