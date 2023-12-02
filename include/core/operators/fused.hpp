#pragma once
#include "core/operators/operator.hpp"

namespace tilegraph::operators {
    class FusedOp : public Operator {
       public:
        FusedOp(std::vector<Operator::OpBox> ops = {});
        ~FusedOp() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        std::vector<Operator::OpBox> ops;
    };
}  // namespace tilegraph::operators