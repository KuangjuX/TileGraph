#include "core/graph/graph_base.hpp"

#include <fmt/core.h>
#include <fmtlog.h>

namespace tilegraph::graph {
    int64_t GraphBase::graph_count = 0;

    GraphBase::GraphBase(std::vector<std::shared_ptr<GNode>> operators_list,
                         std::vector<std::shared_ptr<GEdge>> inputs_list,
                         std::vector<std::shared_ptr<GEdge>> outputs_list,
                         std::string name_value)
        : name(name_value),
          index(graph_count++),
          nodes(operators_list),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Graph_" + std::to_string(index) : name);
    }

    void GraphBase::connect() {
        for (auto node : nodes) {
            auto outputs = node->getOutputs();
            if (outputs.empty()) {
                if (node->inferShape().isErr()) {
                    loge("[GraphBase::connect] Failed to infer node shape.");
                } else {
                    node->outputs = node->inferShape().unwrap();
                }
            }
            for (auto edge : node->inputs) {
                auto it = edge.get();
                it->addConsumer(node);
                if (it->producer != NULL) {
                    node.get()->predecessors.push_back(it->producer);
                    it->producer.get()->successors.push_back(node);
                }
            }
            for (auto it : node.get()->outputs) {
                it->setProducer(node);
            }
            for (auto it : node.get()->inputs) {
                node->in_degree += it->producer == NULL ? 0 : 1;
            }
        }
    }

    bool GraphBase::earseNode(GNode::Pointer node) {
        if (!disconect(node)) {
            loge("Failed to disconect node.");
            return false;
        }
        // Remove node form operators.
        auto operators_iter = std::find(nodes.begin(), nodes.end(), node);
        if (operators_iter != nodes.end()) {
            nodes.erase(operators_iter);
            return true;
        }

        loge("Failed to remove node from operators.");
        return false;
    }

    bool GraphBase::addNode(GNode::Pointer node) {
        auto input_edges = node->getInputs();
        auto output_edges = node->getOutputs();
        for (auto input : input_edges) {
            // Add node to input edges' consumer list.
            input->addConsumer(node);
            if (input->producer != NULL) {
                node->predecessors.push_back(input->producer);
                input->producer->successors.push_back(node);
            }
        }

        for (auto output : output_edges) {
            output->setProducer(node);
            if (output->consumers.size() > 0) {
                for (auto consumer : output->consumers) {
                    node->successors.push_back(consumer);
                    consumer->predecessors.push_back(node);
                }
            }
        }

        for (auto input : input_edges) {
            node->in_degree += input->producer == NULL ? 0 : 1;
        }

        nodes.push_back(node);
        return true;
    }

    bool GraphBase::disconect(GNode::Pointer node) {
        auto input_edges = node->getInputs();
        auto output_edges = node->getOutputs();

        for (auto input : input_edges) {
            // Remove node from consumer list.
            if (!input->earseConsumer(node)) {
                loge("Failed to remove node from consumer list.");
                return false;
            }
            // Remove node from input edges' producer list.
            if (input->producer != NULL) {
                if (!input->producer->earseSuccessor(node)) {
                    loge(
                        "Failed to remove node from inputs' producer node's "
                        "successor list.");
                    return false;
                }
            }
        }

        for (auto output : output_edges) {
            // Remove node from producer list.
            output->producer = NULL;
            if (output->consumers.size() != 0) {
                for (auto consumer : output->consumers) {
                    if (!consumer->earsePredecessor(node)) {
                        loge(
                            "Failed to remove node from outputs' consumer "
                            "node's predecessor list.");
                        return false;
                    }
                }
            }
        }
        return true;
    }

    std::vector<std::shared_ptr<GNode>> GraphBase::topoSort() {
        std::unordered_map<std::shared_ptr<GNode>, int64_t> operators_indegree;
        for (auto op : nodes) {
            operators_indegree[op] = op->in_degree;
        }
        std::vector<std::shared_ptr<GNode>> result;
        while (!operators_indegree.empty()) {
            for (auto op = operators_indegree.begin();
                 op != operators_indegree.end(); ++op) {
                if (op->second == 0) {
                    result.push_back(op->first);
                    for (auto successor : (op->first)->successors) {
                        --operators_indegree[successor];
                    }
                    operators_indegree.erase(op->first);
                    break;
                }
            }
        }
        return result;
    }

    bool GraphBase::fuseNode(std::vector<std::shared_ptr<GNode>> old_nodes,
                             std::shared_ptr<GNode> subgraph_node) {
        subgraph_node->in_degree = 0;
        subgraph_node->predecessors.clear();
        subgraph_node->successors.clear();

        for (auto old_node : old_nodes) {
            earseNode(old_node);
        }
        addNode(subgraph_node);
        return true;
    }

}  // namespace tilegraph::graph