#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"
#include "core/operators/gemm.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "optimizer/fusion/graph_gemm_fusion.hpp"

#include <gtest/gtest.h>

using namespace tilegraph;
using namespace tilegraph::operators;
using namespace tilegraph::fusion;

TEST(Fusion, gemm_relu) {
    Data* a = new Data({5120, 5120});
    Data* b = new Data({5120, 5120});
    Node* gemm = new Gemm({a, b});
    Data* gemm_out = gemm->getOutput(0);
    Node* relu = new RELU({gemm_out});
    Data* relu_out = relu->getOutput(0);

    Graph* graph = new Graph({gemm, relu}, {a, b}, {relu_out});
    std::shared_ptr<Graph> graph_ptr(graph);

    auto gemm_fusion = std::make_shared<GemmFusion>();
    gemm_fusion->fusion(graph_ptr);

    EXPECT_EQ(graph_ptr->operators.size(), 1);
    EXPECT_EQ(graph_ptr->operators[0]->getOperatorType(),
              OperatorType::SUBGRAPH);

    auto ordered_ops = graph_ptr->topoSort();
    EXPECT_EQ(ordered_ops.size(), 1);
    EXPECT_EQ(ordered_ops[0]->getOperatorType(), OperatorType::SUBGRAPH);
}
