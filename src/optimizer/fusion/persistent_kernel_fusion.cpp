#include "optimizer/fusion/persistent_kernel_fusion.hpp"

namespace tilegraph::fusion {
    bool PersistentKernelFusion::fusion(Graph::Pointer graph) {
        // GEMM, GEMM + RELU, GEMM + ADD + RELU
    }
}  // namespace tilegraph::fusion