#include "core/graph/graph.hpp"
#include "core/graph/graph_base.hpp"
#include "core/graph/gnode.hpp"
#include "core/graph/gedge.hpp"
#include "core/tensor.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace tilegraph;
using namespace tilegraph::graph;

TEST(Graph, graph_base_toposort) {
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_add = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto node_a = std::make_shared<GNode>(
        GNode({edge_a, edge_b}, {edge_out_add}, OperatorType::ADD));

    auto edge_out_relu = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto node_b = std::make_shared<GNode>(
        GNode({edge_out_add}, {edge_out_relu}, OperatorType::RELU));

    auto edge_out_softmax = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto node_c = std::make_shared<GNode>(
        GNode({edge_out_relu}, {edge_out_softmax}, OperatorType::SOFTMAX));

    auto graph = std::make_shared<GraphBase>(GraphBase(
        {node_a, node_b, node_c}, {edge_a, edge_b}, {edge_out_softmax}));

    graph->connect();
    auto sorted = graph->topoSort();
    EXPECT_EQ(sorted[0].get()->getOperatorType(), OperatorType::ADD);
    EXPECT_EQ(sorted[1].get()->getOperatorType(), OperatorType::RELU);
    EXPECT_EQ(sorted[2].get()->getOperatorType(), OperatorType::SOFTMAX);
}

TEST(Graph, graph_base_toposort_2) {
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto node_a =
        std::make_shared<GNode>(GNode({edge_a, edge_b}, {}, OperatorType::ADD));
    ASSERT_EQ(node_a->getOutputs().size(), 1);

    auto node_b = std::make_shared<GNode>(
        GNode(node_a->getOutputs(), {}, OperatorType::RELU));
    ASSERT_EQ(node_b->getOutputs().size(), 1);

    auto node_c = std::make_shared<GNode>(
        GNode(node_b->getOutputs(), {}, OperatorType::SOFTMAX));
    ASSERT_EQ(node_c->getOutputs().size(), 1);

    auto graph = std::make_shared<GraphBase>(GraphBase(
        {node_a, node_b, node_c}, {edge_a, edge_b}, node_c->getOutputs()));

    graph->connect();
    auto sorted = graph->topoSort();
    EXPECT_EQ(sorted[0].get()->getOperatorType(), OperatorType::ADD);
    EXPECT_EQ(sorted[1].get()->getOperatorType(), OperatorType::RELU);
    EXPECT_EQ(sorted[2].get()->getOperatorType(), OperatorType::SOFTMAX);
}
