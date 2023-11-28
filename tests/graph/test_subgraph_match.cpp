#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/graph/graph_base.hpp"
#include "core/graph/gnode.hpp"
#include "core/graph/gedge.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "optimizer/fusion/graph_gemm_fusion.hpp"
#include "optimizer/fusion/subgraph_fusion/gemm_relu_fusion.hpp"

#include <gtest/gtest.h>
#include <fmtlog.h>

using namespace tilegraph;
using namespace tilegraph::fusion;
using namespace tilegraph::graph;
using namespace tilegraph::fusion::subgraph;

TEST(SubGraphFuse, gemm_relu) {
    // Relu -> GEMM -> Relu -> Softmax
    fmtlog::setLogLevel(fmtlog::LogLevel::INF);
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

    auto gemm_relu_fusion = std::make_shared<GemmReluFusion>(graph);

    gemm_relu_fusion->create_subgraphs();
    gemm_relu_fusion->match_and_fuse_subgraph();
}