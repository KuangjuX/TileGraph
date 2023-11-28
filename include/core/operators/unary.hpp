#include "core/operators/operator.hpp"

namespace tilegraph::operators {
    class Unary : public Operator {
        enum class UnaryType {
            Abs,
            Acos,
            Acosh,
            Asin,
            Asinh,
            Atan,
            Atanh,
            Cos,
            Cosh,
            Sin,
            Sinh,
            Tan,
            Tanh,
            Relu,
            Sqrt,
            Sigmoid,
            Erf,
            Log,
            Not,
            Neg,
            Identity,
        };

       public:
        Unary(UnaryType type);
        ~Unary() = default;
        std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        UnaryType type;
    };
}  // namespace tilegraph::operators