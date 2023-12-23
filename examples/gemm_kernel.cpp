#include "kernels/cuda/gemm.hpp"
#include "core/type.hpp"
#include <fmt/core.h>

using namespace tilegraph;
using namespace tilegraph::kernel::cuda;

int main() {
    auto gemm_kernel =
        std::make_shared<CudaGEMMKernel>(128, 128, 32, 64, 64, 16, false, true);
    auto kernel = gemm_kernel->genTCGEMM("matmul");
    fmt::println("GEMM Kernel: \n {}", kernel);
}