#include "core/operators/operator.hpp"

namespace tilegraph::operators {
    enum class BinaryType {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        And,
        Or,
        Xor,
    };
    class Binary : public Operator {
       public:
        Binary(BinaryType type);
        ~Binary() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        BinaryType binary_type;
    };
}  // namespace tilegraph::operators