#include <memory>
#include <fmt/core.h>

#include "core/graph/graph_base.hpp"
#include "core/graph/gedge.hpp"
#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"
#include "common/common.hpp"

using namespace tilegraph;
using namespace tilegraph::graph;

int main() {
    auto tensor_a = std::make_shared<Tensor>(Tensor({5120, 5120}));
    auto tensor_b = std::make_shared<Tensor>(Tensor({5120, 5120}));
    auto edge_a = std::make_shared<GEdge>(GEdge(tensor_a));
    auto edge_b = std::make_shared<GEdge>(GEdge(tensor_b));
    auto tensor_out_add = std::make_shared<Tensor>(Tensor({5120, 5120}));
    auto edge_out_add = std::make_shared<GEdge>(GEdge(tensor_out_add));
    auto node_a = std::make_shared<GNode>(
        GNode({edge_a, edge_b}, {edge_out_add}, OperatorType::ADD));

    auto tensor_out_relu = std::make_shared<Tensor>(Tensor({5120, 5120}));
    auto edge_out_relu = std::make_shared<GEdge>(GEdge(tensor_out_relu));
    auto node_b = std::make_shared<GNode>(
        GNode({edge_out_add}, {edge_out_relu}, OperatorType::RELU));

    auto graph = std::make_shared<GraphBase>(
        GraphBase({node_a, node_b}, {edge_a, edge_b}, {edge_out_relu}));

    graph->connect();
    auto sorted = graph->topoSort();

    ASSERT(sorted.size() == 2, "Graph node size is not 2");
    ASSERT(sorted[0]->getOperatorType() == OperatorType::ADD,
           "Graph node type is not ADD");
    ASSERT(sorted[1]->getOperatorType() == OperatorType::RELU,
           "Graph node type is not RELU");
    fmt::println("Topo sort test passed!");
    return 0;
}