#include "optimizer/fusion/graph_gemm_fusion.hpp"

#include <fmtlog.h>

using namespace tilegraph::graph;

namespace tilegraph {
    namespace fusion {

        struct GemmGroup {
            int group_id;
            bool can_merge;
            std::vector<std::shared_ptr<GNode>> nodes;
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
                            // create fused node
                            std::shared_ptr<GNode> fused_node =
                                std::make_shared<GNode>(
                                    GNode(node->inputs, successor_node->outputs,
                                          OperatorType::GEMM_RELU));
                            // remove original nodes
                            if (graph->fuseNode({node, successor_node},
                                                fused_node)) {
                                // add fused node
                                FMTLOG(fmtlog::INF, "Fused GEMM and RELU");
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