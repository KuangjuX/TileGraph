#include "core/operators/gemm.hpp"
#include "common/common.h"

using namespace tilegraph::graph;

namespace tilegraph::operators {
    // GEMM::GEMM(){};  // default Row Major Matrix
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

    std::vector<GEdge::Pointer> GEMM::inferShape(
        std::vector<GEdge::Pointer> inputs) {
        // default Row Major Matrix
        ASSERT(inputs.size() == 2, "GEMM should have 2 inputs");
        ASSERT(inputs[0]->getTensor().get()->tensor_dimension.size() == 2,
               "GEMM input 0 should be 2D");
        ASSERT(inputs[1]->getTensor().get()->tensor_dimension.size() == 2,
               "GEMM input 1 should be 2D");
        ASSERT(inputs[0]->getTensor().get()->tensor_dimension[1] ==
                   inputs[1]->getTensor().get()->tensor_dimension[0],
               "GEMM input 0 and input 1 "
               "should have same shape "
               "except last dimension");
        auto output_tensors =
            this->inferShape({inputs[0]->getTensor(), inputs[1]->getTensor()});
        ASSERT(output_tensors.size() == 1, "GEMM should have 1 output");
        auto output = std::make_shared<GEdge>(output_tensors[0]);
        return {output};
    }

}  // namespace tilegraph::operators