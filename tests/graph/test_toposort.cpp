#include "core/graph/graph.hpp"
#include "core/type.hpp"
#include "core/operators/elementwise.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace tilegraph;
using namespace tilegraph::graph;
using namespace tilegraph::operators;

TEST(Graph, toposort) {
    Data* a = new Data({100});
    Data* b = new Data({100});
    Node* op1 = new ADD({a, b});
    Data* out1 = op1->getOutput(0);
    Node* op2 = new RELU({out1});
    Data* out2 = op2->getOutput(0);

    Graph* graph = new Graph({op1, op2}, {a, b}, {out2});

    std::vector<Node*> sorted = graph->topoSort();

    EXPECT_EQ(sorted[0]->getOperatorType(), OperatorType::ADD);
    EXPECT_EQ(sorted[1]->getOperatorType(), OperatorType::RELU);
}
