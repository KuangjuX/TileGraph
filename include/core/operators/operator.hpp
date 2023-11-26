#pragma once
#include "core/graph/gedge.hpp"
#include "core/tensor.hpp"
#include "core/type.hpp"

namespace tilegraph::operators {
    class Operator {
       public:
        Operator(OperatorType op_type_value) : op_type(op_type_value) {}
        ~Operator() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) = 0;
        virtual std::vector<graph::GEdge::Pointer> inferShape(
            std::vector<graph::GEdge::Pointer> inputs) = 0;

        OperatorType op_type;
    };
}  // namespace tilegraph::operators