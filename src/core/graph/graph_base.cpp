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
          operators(operators_list),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Graph_" + std::to_string(index) : name);
    }

    void GraphBase::connect() {
        for (auto node : operators) {
            auto outputs = node.get()->getOutputs();
            auto outputs_num = node.get()->outputs_num;
            if (outputs.empty()) {
                std::shared_ptr<Tensor> inter_edge;
                for (auto i = 0; i < outputs_num; ++i) {
                    inter_edge = std::make_shared<Tensor>(
                        inputs[0].get()->getTensor().get()->tensor_dimension,
                        inputs[0].get()->getTensor().get()->name,
                        inputs[0].get()->getTensor().get()->tensor_datatype,
                        inputs[0].get()->getTensor().get()->tensor_type);
                    outputs.push_back(std::make_shared<GEdge>(inter_edge));
                }
            }
            for (auto edge : node.get()->inputs) {
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
                node.get()->indegree += it->producer == NULL ? 0 : 1;
            }

            // print node info
            // fmt::print(
            //     "node->name: {}, node->indegree: {}, node->outputs_num:
            //     {}\n", node.get()->name, node.get()->indegree,
            //     node.get()->outputs_num);
            // logi("connect: node->name: {}, node->indegree: {},
            // node->outputs_num: {}",
            //      node.get()->name, node.get()->indegree,
            //      node.get()->outputs_num
        }
    }

    bool GraphBase::earseNode(GNode::Pointer node) {
        if (!disconect(node)) {
            loge("Failed to disconect node.");
            return false;
        }
        // Remove node form operators.
        auto operators_iter =
            std::find(operators.begin(), operators.end(), node);
        if (operators_iter != operators.end()) {
            operators.erase(operators_iter);
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
            node->indegree += input->producer == NULL ? 0 : 1;
        }

        operators.push_back(node);
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
        for (auto op : operators) {
            operators_indegree[op] = op->indegree;
            // logi("op->indegree: {}, name: {}", op->indegree, op->name);
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
        subgraph_node->indegree = 0;
        subgraph_node->predecessors.clear();
        subgraph_node->successors.clear();

        for (auto old_node : old_nodes) {
            earseNode(old_node);
        }
        addNode(subgraph_node);
        return true;
    }

}  // namespace tilegraph::graph