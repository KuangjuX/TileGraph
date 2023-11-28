#include "core/operators/binary.hpp"
#include "common/common.hpp"

namespace tilegraph::operators {

    Binary::Binary(BinaryType type) : binary_type(type) {}

    std::vector<Tensor::Pointer> Binary::inferShape(
        std::vector<Tensor::Pointer> inputs) {
        ASSERT(inputs.size() == 2, "Binary operator should have 2 inputs");
        return inputs;
    }

}  // namespace tilegraph::operators