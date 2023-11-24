#include <memory>
#include <iostream>

#include "core/graph/graph_base.hpp"
#include "core/graph/gedge.hpp"
#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"

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

    // std::cout << "sorted[0]->getOperatorType() = "
    //           << sorted[0]->getOperatorType() << std::endl;
    // std::cout << "sorted[1]->getOperatorType() = "
    //           << sorted[1]->getOperatorType() << std::endl;
    std::cout << "sorted.size() = " << sorted.size() << std::endl;
    // print name
    for (auto node : sorted) {
        std::cout << node.get()->name << std::endl;
    }

    return 0;
}