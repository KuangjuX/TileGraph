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
        void connect();
        std::vector<std::shared_ptr<GNode>> topoSort();
        bool fuseNode(std::vector<std::shared_ptr<GNode>> old_nodes,
                      std::shared_ptr<GNode> subgraph_node);
    };
}  // namespace tilegraph::graph