#pragma once
#include "core/graph/graph.hpp"
#include "core/operators/operator.hpp"

namespace tilegraph::operators {

    class GEMM : public Operator {
       public:
        GEMM() {}
        ~GEMM() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;
        virtual std::vector<graph::GEdge::Pointer> inferShape(
            std::vector<graph::GEdge::Pointer> inputs) override;
    };

}  // namespace tilegraph::operators