#include "core/operators/operator.hpp"
#include "core/type.hpp"

namespace tilegraph::operators {
    class Unary : public Operator {
       public:
        Unary(OperatorType type);
        ~Unary() = default;
        std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        OperatorType type;
    };
}  // namespace tilegraph::operators