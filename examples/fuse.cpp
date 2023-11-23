#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"
#include "core/operators/gemm.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "optimizer/fusion/graph_gemm_fusion.hpp"

int main() {
    using namespace tilegraph::fusion;
    using namespace tilegraph::operators;
    using namespace tilegraph;

    // Relu -> GEMM -> Relu
    Data* a = new Data({5120, 5120});
    Node* relu1 = new RELU({a});
    Data* b = new Data({5120, 5120});
    Node* gemm = new Gemm({relu1->getOutput(0), b});
    Node* relu2 = new RELU({gemm->getOutput(0)});

    Graph* graph =
        new Graph({relu1, gemm, relu2}, {a, b}, {relu2->getOutput(0)});
    std::shared_ptr<Graph> graph_ptr(graph);

    auto subgraph = std::make_shared<SubGraph>(SubGraph({gemm, relu2}));
    // create fused node
    Node* fused_node = new Node(gemm->inputs, relu2->inputs, subgraph);

    graph->fuseNode({gemm, relu2}, fused_node);

    return 0;
}