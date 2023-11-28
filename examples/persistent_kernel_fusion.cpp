#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/graph/graph_base.hpp"
#include "core/graph/gnode.hpp"
#include "core/graph/gedge.hpp"
#include "optimizer/fusion/subgraph_fusion/gemm_relu_fusion.hpp"
#include "optimizer/fusion/persistent_kernel_fusion.hpp"
#include "common/common.hpp"

#include <fmtlog.h>

using namespace tilegraph;
using namespace tilegraph::fusion;
using namespace tilegraph::graph;
using namespace tilegraph::fusion::subgraph;

int main() {
    // GEMM -> GEMM -> SOFTMAX -> GEMM
    auto edge_a = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_b = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_c = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_d = std::make_shared<GEdge>(GEdge({5120, 5120}));

    auto edge_out_gemm1 = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_gemm2 = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_softmax = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto edge_out_gemm3 = std::make_shared<GEdge>(GEdge({5120, 5120}));
    auto gemm1 = std::make_shared<GNode>(
        GNode({edge_a, edge_b}, {edge_out_gemm1}, {OperatorType::GEMM}));
    auto gemm2 = std::make_shared<GNode>(GNode(
        {edge_out_gemm1, edge_c}, {edge_out_gemm2}, {OperatorType::GEMM}));
    auto softmax = std::make_shared<GNode>(
        GNode({edge_out_gemm2}, {edge_out_softmax}, {OperatorType::SOFTMAX}));
    auto gemm3 = std::make_shared<GNode>(GNode(
        {edge_out_softmax, edge_d}, {edge_out_gemm3}, {OperatorType::GEMM}));

    auto graph = std::make_shared<Graph>(Graph({gemm1, gemm2, softmax, gemm3},
                                               {edge_a, edge_b, edge_c, edge_d},
                                               {edge_out_gemm3}));

    graph->connect();

    auto persistent_kernel_fusion = std::make_shared<PersistentKernelFusion>();
    persistent_kernel_fusion->fusion(graph);
}