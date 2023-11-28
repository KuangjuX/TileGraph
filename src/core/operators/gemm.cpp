#include "core/operators/gemm.hpp"
#include "common/common.hpp"

namespace tilegraph::operators {
    GEMM::GEMM(float alpha, float beta, bool transA, bool transB)
        : alpha(alpha), beta(beta), transA(transA), transB(transB) {}

    std::vector<Tensor::Pointer> GEMM::inferShape(
        std::vector<Tensor::Pointer> inputs) {
        // default Row Major Matrix
        ASSERT(inputs.size() == 2, "GEMM should have 2 inputs");
        ASSERT(inputs[0]->tensor_dimension.size() == 2,
               "GEMM input 0 should be 2D");
        ASSERT(inputs[1]->tensor_dimension.size() == 2,
               "GEMM input 1 should be 2D");
        ASSERT(inputs[0]->tensor_dimension[1] == inputs[1]->tensor_dimension[0],
               "GEMM input 0 and input 1 "
               "should have same shape "
               "except last dimension");
        auto output = std::make_shared<Tensor>(std::vector<int64_t>{
            inputs[0]->tensor_dimension[0], inputs[1]->tensor_dimension[1]});
        return {output};
    }

}  // namespace tilegraph::operators