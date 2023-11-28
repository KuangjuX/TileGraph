#include "optimizer/fusion/persistent_kernel_fusion.hpp"
#include "core/type.hpp"

#include <fmtlog.h>

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    bool PersistentKernelFusion::fusion(Graph::Pointer graph) {
        // GEMM, GEMM + RELU, GEMM + ADD + RELU
        // <node_id, node>
        std::unordered_map<std::size_t, GNode::Pointer> nodes;
        // <node, node_id>
        std::unordered_map<GNode::Pointer, std::size_t> node_map;
        // <node_id, group>
        std::unordered_map<std::size_t, std::size_t> node_to_group;
        // std::vector<GNode::Pointer> gemm_nodes;

        // Step 1. Fuse GEMM_ADD, GEMM_ADD_RELU, GEMM_ADD_GELU and so on.
        // TODO:

        // Step 2. Find all GEMM and fused operators.
        auto check_node = [](std::shared_ptr<GNode> gnode) -> bool {
            auto op_type = gnode->getOperatorType();
            if (op_type == OperatorType::GEMM ||
                op_type == OperatorType::GEMM_RELU) {
                return true;
            }
            return false;
        };

        size_t counter = 0;
        auto ordered_ops = graph->topoSort();
        for (auto op : ordered_ops) {
            if (check_node(op)) {
                nodes[counter] = op;
                node_map[op] = counter;
                node_to_group[counter] = counter;
                counter++;
            }
        }

        // Step 3. Union GEMM and group them.
        for (auto op : nodes) {
            auto successors = op.second->successors;
            for (auto successor : successors) {
                if (check_node(successor)) {
                    node_to_group[node_map[successor]] = op.first;
                }
            }
        }

        // Step 4. Find all groups.
        auto groups = find_groups(node_to_group, nodes, node_map);

        // Step 5. Fuse all groups.
        for (auto group : groups) {
            auto group_idx = group.first;
            auto group_nodes = group.second;
            for (auto node : group_nodes) {
                logi("group index {}, node: {} {}", group_idx, node->name,
                     toString(node->getOperatorType()));
            }
        }
    }

    std::size_t PersistentKernelFusion::find_root(
        std::unordered_map<std::size_t, std::size_t> node_to_group,
        std::size_t node_idx) {
        while (node_to_group[node_idx] != node_idx) {
            node_idx = node_to_group[node_idx];
        }
        return node_idx;
    }

    std::unordered_map<std::size_t, std::vector<GNode::Pointer>>
    PersistentKernelFusion::find_groups(
        std::unordered_map<std::size_t, std::size_t> node_to_group,
        std::unordered_map<std::size_t, GNode::Pointer> nodes,
        std::unordered_map<GNode::Pointer, std::size_t> node_map) {
        std::unordered_map<std::size_t, std::vector<GNode::Pointer>> groups;
        for (auto node : nodes) {
            auto root = find_root(node_to_group, node.first);
            groups[root].push_back(node.second);
        }
        return groups;
    }
}  // namespace tilegraph::fusion