#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "operators/elementwise.hpp"
#include "operators/gemm.hpp"
#include "engine/fusion/graph_fusion_base.hpp"
#include "engine/fusion/graph_gemm_fusion.hpp"

#include <gtest/gtest.h>

using namespace tilegraph;
using namespace tilegraph::fusion;

TEST(Fusion, gemm) {
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
}
