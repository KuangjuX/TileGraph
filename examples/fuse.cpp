#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"
#include "core/operators/gemm.hpp"
#include "optimizer/fusion/subgraph_fusion/gemm_relu_fusion.hpp"
#include "common/common.hpp"
#include <fmtlog.h>
#include <fmt/core.h>

using namespace tilegraph;
using namespace tilegraph::graph;
using namespace tilegraph::fusion;
using namespace tilegraph::fusion::subgraph;

int main() {
    fmtlog::setLogLevel(fmtlog::LogLevel::INF);
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

    auto gemm_relu_fusion = std::make_shared<GemmReluFusion>(graph);

    gemm_relu_fusion->create_subgraphs();
    gemm_relu_fusion->match_and_fuse_subgraph();

    auto ordered_ops = graph->topoSort();

    ASSERT(ordered_ops.size() == 3, "Graph node size is not 3");
    ASSERT(ordered_ops[0]->getOperatorType() == OperatorType::RELU,
           "Graph node type is not RELU");
    ASSERT(ordered_ops[1]->getOperatorType() == OperatorType::GEMM_RELU,
           "Graph node type is not GEMM_RELU");
    ASSERT(ordered_ops[2]->getOperatorType() == OperatorType::SOFTMAX,
           "Graph node type is not SOFTMAX");

    fmt::println("Fuse test passed!");
}