#include "core/operators/operator.hpp"
#include "core/type.hpp"

namespace tilegraph::operators {

    class Binary : public Operator {
       public:
        Binary(OperatorType type);
        ~Binary() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        OperatorType binary_type;
    };
}  // namespace tilegraph::operators