#include "core/operators/operator.hpp"

namespace tilegraph::operators {
    std::vector<Tensor::Pointer> Operator::inferShape(
        std::vector<Tensor::Pointer> inputs) {
        // Empty Implementation.
        return {};
    }
}  // namespace tilegraph::operators