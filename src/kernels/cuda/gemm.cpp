#include "kernels/cuda/gemm.hpp"
#include "kernels/cuda/sync.hpp"
#include "kernels/cuda/tensor_core.hpp"
#include "kernels/kernel_unit.hpp"

namespace tilegraph::kernel::cuda {
    CudaGEMMKernel::CudaGEMMKernel(uint32_t ShardedM, uint32_t ShardedN,
                                   uint32_t ShardedK, uint32_t WarpM,
                                   uint32_t WarpN, uint32_t WarpK,
                                   bool transpose_a, bool transpose_b)
        : ShardedM(ShardedM),
          ShardedN(ShardedN),
          ShardedK(ShardedK),
          WarpM(WarpM),
          WarpN(WarpN),
          WarpK(WarpK),
          transpose_a(transpose_a),
          transpose_b(transpose_b) {}

    std::string CudaGEMMKernel::genTCGEMM(std::string name) {
        uint16_t indient = 0;
        std::vector<std::pair<std::string, std::string>> arguments;
        arguments.push_back(std::make_pair("half*", "A"));
        auto function = function_unit->declareGlobal(name, arguments);
        function += "{\n";
        indient += 4;

        // Generate tensor core gemm cuda implementation.

        // Decalre Sharded Memory.
        auto smem_a = CudaVar(MemoryType::Shared, TensorDatatype::HALF,
                              ShardedM * ShardedK, "SA");
        auto smem_b = CudaVar(MemoryType::Shared, TensorDatatype::HALF,
                              ShardedK * ShardedN, "SB");
        auto smem_c = CudaVar(MemoryType::Shared, TensorDatatype::FLOAT,
                              ShardedM * ShardedN, "SC");

        // Declare Warp variable.
        auto frag_a =
            CudaVar(MemoryType::Warp, TensorDatatype::HALF, 0, "FragA");
        auto frag_b =
            CudaVar(MemoryType::Warp, TensorDatatype::HALF, 0, "FragB");
        auto accum =
            CudaVar(MemoryType::Warp, TensorDatatype::HALF, 0, "Accum");

        vars.push_back(smem_a);
        vars.push_back(smem_b);
        vars.push_back(smem_c);
        vars.push_back(frag_a);
        vars.push_back(frag_b);
        vars.push_back(accum);

        for (auto var : vars) {
            function += var.declareVar(indient);
        }

        // accum.initVar(indient);

        function += insertIndient(indient);
        function +=
            fmt::format("for (int ko = 0; ko < K / {}; ko += 1){{\n", ShardedK);
        indient += 4;
        // TODO: load sharded memory
        function += insertSyncnorize(indient, MemoryType::Shared);
        uint32_t mtile_iter = ShardedM / WarpM;
        uint32_t ntile_iter = ShardedN / WarpN;
        function += insertIndient(indient);
        function += fmt::format("for (int mii = 0; mii < {}; mii += 1) {{\n",
                                mtile_iter);
        indient += 4;
        function += insertIndient(indient);
        function += fmt::format("for (int nii = 0; nii < {}; nii += 1) {{\n",
                                ntile_iter);
        indient += 4;

        // Insert mma sync to compute Matrix Mul.
        // genMMASync();

        indient -= 4;
        function += insertIndient(indient);
        function += "}\n";

        indient -= 4;
        function += insertIndient(indient);
        function += "}\n";

        indient -= 4;
        function += insertIndient(indient);
        function += "}\n";

        // TODO: Store accum into smem_c;
        function += insertSyncnorize(indient, MemoryType::Shared);

        // TODO: Store smem_c into C;

        // Function End;
        indient -= 4;
        function += insertIndient(indient);
        function += "}\n";

        return function;
    }
}  // namespace tilegraph::kernel::cuda