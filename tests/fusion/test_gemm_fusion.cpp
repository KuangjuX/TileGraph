#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/graph/graph_base.hpp"
#include "core/graph/gnode.hpp"
#include "core/graph/gedge.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "optimizer/fusion/graph_gemm_fusion.hpp"

#include <gtest/gtest.h>

using namespace tilegraph;
using namespace tilegraph::fusion;
using namespace tilegraph::graph;

TEST(Fusion, gemm_relu) {
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_gemm = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto gemm = std::make_shared<GNode>(
        GNode({edge_a, edge_b}, {edge_out_gemm}, {OperatorType::GEMM}));

    auto edge_out_relu = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto relu = std::make_shared<GNode>(
        GNode({edge_out_gemm}, {edge_out_relu}, {OperatorType::RELU}));

    auto graph = std::make_shared<Graph>(
        Graph({gemm, relu}, {edge_a, edge_b}, {edge_out_relu}));
    graph->connect();

    auto gemm_fusion = std::make_shared<GemmFusion>();
    gemm_fusion->fusion(graph);

    auto ordered_ops = graph->topoSort();
    EXPECT_EQ(ordered_ops.size(), 1);
    EXPECT_EQ(ordered_ops[0]->getOperatorType(), OperatorType::GEMM_RELU);
}

TEST(Fusion, gemm_relu_softmax) {
    // GEMM -> Relu -> Softmax
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_gemm = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto gemm = std::make_shared<GNode>(
        GNode({edge_a, edge_b}, {edge_out_gemm}, {OperatorType::GEMM}));

    auto edge_out_relu = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto relu = std::make_shared<GNode>(
        GNode({edge_out_gemm}, {edge_out_relu}, {OperatorType::RELU}));

    auto edge_out_softmax = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto softmax = std::make_shared<GNode>(
        GNode({edge_out_relu}, {edge_out_softmax}, {OperatorType::SOFTMAX}));

    auto graph = std::make_shared<Graph>(
        Graph({gemm, relu, softmax}, {edge_a, edge_b}, {edge_out_softmax}));
    graph->connect();

    auto gemm_fusion = std::make_shared<GemmFusion>();
    gemm_fusion->fusion(graph);

    auto ordered_ops = graph->topoSort();
    EXPECT_EQ(ordered_ops.size(), 2);
    EXPECT_EQ(ordered_ops[0]->getOperatorType(), OperatorType::GEMM_RELU);
    EXPECT_EQ(ordered_ops[1]->getOperatorType(), OperatorType::SOFTMAX);
}

TEST(Fusion, relu_gemm_relu_softmax) {
    // Relu -> GEMM -> Relu -> Softmax
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_relu1 = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto relu1 = std::make_shared<GNode>(
        GNode({edge_a}, {edge_out_relu1}, {OperatorType::RELU}));

    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_gemm = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto gemm = std::make_shared<GNode>(
        GNode({edge_out_relu1, edge_b}, {edge_out_gemm}, {OperatorType::GEMM}));

    auto edge_out_relu2 = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto relu2 = std::make_shared<GNode>(
        GNode({edge_out_gemm}, {edge_out_relu2}, {OperatorType::RELU}));

    auto edge_out_softmax = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto softmax = std::make_shared<GNode>(
        GNode({edge_out_relu2}, {edge_out_softmax}, {OperatorType::SOFTMAX}));

    auto graph = std::make_shared<Graph>(Graph(
        {relu1, gemm, relu2, softmax}, {edge_a, edge_b}, {edge_out_softmax}));
    graph->connect();

    auto gemm_fusion = std::make_shared<GemmFusion>();
    gemm_fusion->fusion(graph);

    auto ordered_ops = graph->topoSort();
    ASSERT_EQ(ordered_ops.size(), 3);
    ASSERT_EQ(ordered_ops[0]->getOperatorType(), OperatorType::RELU);
    ASSERT_EQ(ordered_ops[1]->getOperatorType(), OperatorType::GEMM_RELU);
    EXPECT_EQ(ordered_ops[2]->getOperatorType(), OperatorType::SOFTMAX);
}
