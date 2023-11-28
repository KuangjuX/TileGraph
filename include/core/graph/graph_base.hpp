#pragma once
#include <vector>
#include <string>
#include <algorithm>

#include "core/graph/gedge.hpp"
#include "core/graph/gnode.hpp"

namespace tilegraph::graph {
    class GraphBase {
        static int64_t graph_count;

       public:
        std::string name;
        const int64_t index;
        std::vector<std::shared_ptr<GNode>> operators;
        std::vector<std::shared_ptr<GEdge>> inputs;
        std::vector<std::shared_ptr<GEdge>> outputs;
        std::vector<std::shared_ptr<GEdge>> inter_edges;

        GraphBase(std::vector<std::shared_ptr<GNode>> operators_list = {},
                  std::vector<std::shared_ptr<GEdge>> inputs_list = {},
                  std::vector<std::shared_ptr<GEdge>> outputs_list = {},
                  std::string name_value = "");
        ~GraphBase() = default;
        // Connect the graph based nodes and edges.
        void connect();
        // Topological sort the graph.
        std::vector<std::shared_ptr<GNode>> topoSort();
        // Earse the node from the graph.
        bool earseNode(GNode::Pointer node);
        // Add the node to the graph.
        bool addNode(GNode::Pointer node);
        // Fuse the subgraph into a single node.
        bool fuseNode(std::vector<std::shared_ptr<GNode>> old_nodes,
                      std::shared_ptr<GNode> subgraph_node);

        using Pointer = std::shared_ptr<GraphBase>;

       private:
        // Disconnect the node from the graph.
        bool disconect(GNode::Pointer node);
    };
}  // namespace tilegraph::graph