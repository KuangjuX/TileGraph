#include "engine/fusion/graph_gemm_fusion.hpp"

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
                                }
            }

            return true;
        }
    }  // namespace fusion
}  // namespace tilegraph