#pragma once
#include "core/operators/operator.hpp"

namespace tilegraph::operators {

    class GEMM : public Operator {
       public:
        GEMM(float alpha = 1.0f, float beta = 1.0f, bool transA = false,
             bool transB = false);
        ~GEMM() = default;
        virtual std::vector<Tensor::Pointer> inferShape(
            std::vector<Tensor::Pointer> inputs) override;

        float alpha, beta;
        bool transA, transB;
    };

}  // namespace tilegraph::operators