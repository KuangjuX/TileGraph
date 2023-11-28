#include "core/operators/unary.hpp"
#include "common/common.hpp"

namespace tilegraph::operators {

    Unary::Unary(UnaryType type) : type(type) {}

    std::vector<Tensor::Pointer> Unary::inferShape(
        std::vector<Tensor::Pointer> inputs) {
        ASSERT(inputs.size() == 1, "Unary operator should have 1 input");
        return inputs;
    }

}  // namespace tilegraph::operators