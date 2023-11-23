#include "optimizer/fusion/graph_gemm_fusion.hpp"

#include <fmtlog.h>

using namespace tilegraph::graph;

namespace tilegraph {
    namespace fusion {

        struct GemmGroup {
            int group_id;
            bool can_merge;
            std::vector<std::shared_ptr<Node>> nodes;
            std::vector<std::shared_ptr<Graph>> sub_groups;
        };

        bool GemmFusion::fusion(std::shared_ptr<Graph> graph) {
            auto nodes = graph->topoSort();

            // Step 1: find all the GEMM operators and group them using their
            // hash value.
            std::unordered_map<size_t, int> hash_to_gid;
            std::vector<std::shared_ptr<GemmGroup>> merge_groups;

            hash_to_gid.clear();
            merge_groups.clear();

            for (auto node : nodes) {
                if (node->getOperatorType() == OperatorType::GEMM) {
                    auto successor_nodes = node->successors;
                    if (successor_nodes.size() == 1) {
                        auto successor_node = successor_nodes[0];
                        if (successor_node->getOperatorType() ==
                            OperatorType::RELU) {
                            // create subgraph
                            auto subgraph = std::make_shared<SubGraph>(
                                SubGraph({node, successor_node}));
                            // crate fused node
                            Node* fused_node =
                                new Node(node->inputs, successor_node->outputs,
                                         subgraph);
                            // remove original nodes
                            if (graph->removeNode(node->index) &&
                                graph->removeNode(successor_node->index)) {
                                // add fused node
                                graph->addNode(fused_node);
                            } else {
                                FMTLOG(fmtlog::ERR, "Failed to remove nodes");
                            }
                        }
                    }
                }
            }

            return true;
        }

    }  // namespace fusion
}  // namespace tilegraph