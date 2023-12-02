#include "core/operators/fused.hpp"
#include "common/common.hpp"

#include <fmtlog.h>

namespace tilegraph::operators {
    FusedOp::FusedOp(std::vector<Operator::OpBox> ops) : ops(ops) {}

    std::vector<Tensor::Pointer> FusedOp::inferShape(
        std::vector<Tensor::Pointer> inputs) {
        if (ops.empty()) {
            loge("[FusedOperator::inferShape] No operators in fused operator.");
            return {};
        } else {
            loge(
                "[FusedOperator::inferShape] Fused operator is not "
                "implemented.");
            UNREACHABLE();
            return {};
        }
    }

}  // namespace tilegraph::operators