#include "core/operators/gemm.hpp"

#include <gtest/gtest.h>

using namespace tilegraph;
using namespace tilegraph::operators;

TEST(OPERATORS, gemm) {
    auto gemm = std::make_shared<GEMM>();
    auto input0 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3});
    auto input1 = std::make_shared<Tensor>(std::vector<int64_t>{3, 4});
    auto output = gemm->inferShape({input0, input1});

    ASSERT_EQ(output.size(), 1);
    ASSERT_EQ(output[0]->tensor_dimension.size(), 2);
    ASSERT_EQ(output[0]->tensor_dimension[0], 2);
    ASSERT_EQ(output[0]->tensor_dimension[1], 4);
}
