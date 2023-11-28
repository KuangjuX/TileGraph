#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"
#include "core/operators/gemm.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "optimizer/fusion/graph_gemm_fusion.hpp"
#include <fmtlog.h>

using namespace tilegraph::fusion;
using namespace tilegraph;

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

    auto gemm_fusion = std::make_shared<GemmFusion>();
    gemm_fusion->fusion(graph);

    auto ordered_ops = graph->topoSort();
}