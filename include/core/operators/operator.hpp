#pragma once
#include "core/tensor.hpp"
#include "core/type.hpp"

namespace tilegraph::operators {
    class Operator {
       public:
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs);

        OperatorType op_type;

        using OpBox = std::shared_ptr<Operator>;
    };
}  // namespace tilegraph::operators