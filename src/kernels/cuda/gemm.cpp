#include "kernels/cuda/gemm.hpp"
#include "kernels/cuda/sync.hpp"
#include "kernels/cuda/tensor_core.hpp"
#include "kernels/kernel_unit.hpp"
#include <memory>

namespace tilegraph::kernel::cuda {
    CudaGEMMKernel::CudaGEMMKernel(uint32_t M, uint32_t N, uint32_t K,
                                   uint32_t ShardedM, uint32_t ShardedN,
                                   uint32_t ShardedK, uint32_t WarpM,
                                   uint32_t WarpN, uint32_t WarpK,
                                   uint32_t WmmaM, uint32_t WmmaN,
                                   uint32_t WmmaK, bool transpose_a,
                                   bool transpose_b)
        : M(M),
          N(N),
          K(K),
          ShardedM(ShardedM),
          ShardedN(ShardedN),
          ShardedK(ShardedK),
          WarpM(WarpM),
          WarpN(WarpN),
          WarpK(WarpK),
          WmmaM(WmmaM),
          WmmaN(WmmaN),
          WmmaK(WmmaK),
          transpose_a(transpose_a),
          transpose_b(transpose_b) {}

    std::string CudaGEMMKernel::genTCGEMM(std::string name) {
        std::string function;
        // Define Functions;
        auto load_smem_a = std::make_shared<CudaFunction>(
            "loadSmemA", FuncType::Device, DataType::Void);
        auto load_smem_b = std::make_shared<CudaFunction>(
            "loadSmemB", FuncType::Device, DataType::Void);
        auto load_smem_c = std::make_shared<CudaFunction>(
            "loadSmemC", FuncType::Device, DataType::Void);
        auto store_smem_c = std::make_shared<CudaFunction>(
            "StoreSmemC", FuncType::Device, DataType::Void);
        auto load_frag_a = std::make_shared<CudaFunction>(
            "LoadFragA", FuncType::Device, DataType::Void);
        auto load_frag_b = std::make_shared<CudaFunction>(
            "LoadFragB", FuncType::Device, DataType::Void);
        auto store_accum = std::make_shared<CudaFunction>(
            "StoreAccum", FuncType::Device, DataType::Void);

        functions.insert(load_smem_a);
        functions.insert(load_smem_b);
        functions.insert(load_smem_c);
        functions.insert(store_smem_c);
        functions.insert(load_frag_a);
        functions.insert(load_frag_b);
        functions.insert(store_accum);

        for (auto func : functions) {
            func->declareFunction();
        }

        uint16_t indient = 0;
        std::vector<std::pair<std::string, std::string>> arguments;
        arguments.push_back(std::make_pair("half*", "A"));
        function += function_unit->declareGlobal(name, arguments);
        function += "{\n";
        indient += 4;

        // Generate tensor core gemm cuda implementation.

        // Decalre Sharded Memory.
        auto smem_a = std::make_shared<CudaVar>(
            MemoryType::Shared, DataType::Half, ShardedM * ShardedK, "SA");
        auto smem_b = std::make_shared<CudaVar>(
            MemoryType::Shared, DataType::Half, ShardedK * ShardedN, "SB");
        auto smem_c = std::make_shared<CudaVar>(
            MemoryType::Shared, DataType::Half, ShardedM * ShardedN, "SC");

        // Declare Warp variable.
        auto frag_a = std::make_shared<CudaVar>(
            MemoryType::Warp, DataType::Half, WarpM * WarpK, "FA");
        auto frag_b = std::make_shared<CudaVar>(
            MemoryType::Warp, DataType::Half, WarpK * WarpN, "FB");
        auto accum = std::make_shared<CudaVar>(MemoryType::Warp, DataType::Half,
                                               WarpM * WarpN, "AC");

        vars.insert(smem_a);
        vars.insert(smem_b);
        vars.insert(smem_c);
        vars.insert(frag_a);
        vars.insert(frag_b);
        vars.insert(accum);

        for (auto var : vars) {
            function += var->declareVar(indient);
        }

        // accum.initVar(indient);

        function += insertIndient(indient);
        function +=
            fmt::format("for (int ko = 0; ko < K / {}; ko += 1){{\n", ShardedK);
        indient += 4;
        // TODO: load sharded memory
        function += insertSyncnorize(indient, MemoryType::Shared);

        auto iter_m = std::make_unique<Iteration>(
            std::make_unique<CudaVar>(MemoryType::Shared, DataType::Int32, 0,
                                      "mii"),
            std::variant<int, std::shared_ptr<CudaVar>>(1),
            std::variant<int, std::shared_ptr<CudaVar>>(0),
            std::variant<int, std::shared_ptr<CudaVar>>((int)(WarpM / WmmaM)));

        auto iter_n = std::make_unique<Iteration>(
            std::make_unique<CudaVar>(MemoryType::Shared, DataType::Int32, 0,
                                      "nii"),
            std::variant<int, std::shared_ptr<CudaVar>>(1),
            std::variant<int, std::shared_ptr<CudaVar>>(0),
            std::variant<int, std::shared_ptr<CudaVar>>((int)(WarpN / WmmaN)));

        function += iter_m->genIter(indient);
        indient += 4;
        function += iter_n->genIter(indient);
        indient += 4;

        // Insert mma sync to compute Matrix Mul.
        auto warp_a = frag_a->getVarIndexByVar(iter_m->getIterVar());
        auto warp_b = frag_b->getVarIndexByVar(iter_n->getIterVar());
        auto warp_c = accum->getVarIndexByVar(
            fmt::format("mii * {} + nii", WarpN / WmmaN));
        function += genWmmaSync(indient, warp_a, warp_b, warp_c, warp_c);

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