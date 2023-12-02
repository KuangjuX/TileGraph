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
        auto groups = findGroups(node_to_group, nodes, node_map);

        // Step 5. Fuse all groups.
        for (auto group : groups) {
            auto group_idx = group.first;
            auto group_nodes = group.second;
            for (auto node : group_nodes) {
                logi(
                    "[PersistentKernelFusion::fusion] group idx {}, node: {} "
                    "{}",
                    group_idx, node->name, toString(node->getOperatorType()));
            }

            if (group_nodes.size() > 1) {
                auto in_out_edges = searchInOut(group_nodes);
                auto in_edges = in_out_edges.first;
                auto out_edges = in_out_edges.second;

                // Create fused nodes
                auto fused_node = std::make_shared<GNode>(
                    GNode(in_edges, out_edges, OperatorType::FUSED));

                graph->fuseNode(group_nodes, fused_node);
            }
        }

        return true;
    }

    std::size_t PersistentKernelFusion::findRoot(
        std::unordered_map<std::size_t, std::size_t> node_to_group,
        std::size_t node_idx) {
        while (node_to_group[node_idx] != node_idx) {
            node_idx = node_to_group[node_idx];
        }
        return node_idx;
    }

    std::unordered_map<std::size_t, std::vector<GNode::Pointer>>
    PersistentKernelFusion::findGroups(
        std::unordered_map<std::size_t, std::size_t> node_to_group,
        std::unordered_map<std::size_t, GNode::Pointer> nodes,
        std::unordered_map<GNode::Pointer, std::size_t> node_map) {
        std::unordered_map<std::size_t, std::vector<GNode::Pointer>> groups;
        for (auto node : nodes) {
            auto root = findRoot(node_to_group, node.first);
            groups[root].push_back(node.second);
        }
        return groups;
    }

    std::pair<std::vector<GEdge::Pointer>, std::vector<GEdge::Pointer>>
    PersistentKernelFusion::searchInOut(std::vector<GNode::Pointer> group) {
        std::vector<GEdge::Pointer> in_edges;
        std::vector<GEdge::Pointer> out_edges;
        for (auto node : group) {
            // Search in edges by producer.
            auto inputs = node->getInputs();
            for (auto input : inputs) {
                // Check producer if in this group.
                auto producer = input->getProducer();
                if (std::find(group.begin(), group.end(), producer) ==
                    group.end()) {
                    in_edges.push_back(input);
                }
            }
            // Search out edges by consumers.
            auto outputs = node->getOutputs();
            for (auto output : outputs) {
                // Check consumers if in this group.
                auto consumers = output->getConsumers();
                for (auto consumer : consumers) {
                    if (std::find(group.begin(), group.end(), consumer) ==
                        group.end()) {
                        out_edges.push_back(output);
                    }
                }
            }
        }
        return std::make_pair(in_edges, out_edges);
    }
}  // namespace tilegraph::fusion