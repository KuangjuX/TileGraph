#include "core/graph/graph_base.hpp"

#include <fmt/core.h>

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
            fmt::print(
                "node->name: {}, node->indegree: {}, node->outputs_num: {}\n",
                node.get()->name, node.get()->indegree,
                node.get()->outputs_num);
        }
    }

    std::vector<std::shared_ptr<GNode>> GraphBase::topoSort() {
        std::unordered_map<std::shared_ptr<GNode>, int64_t> operators_indegree;
        for (auto op : operators) {
            operators_indegree[op] = op->indegree;
            fmt::println("op->indegree: {}, name: {}", op->indegree, op->name);
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
        // Replace some nodes with subgraph_node
        auto subgraph_input_tensors = subgraph_node->inputs;
        auto subgraph_output_tensors = subgraph_node->outputs;

        fmt::println("subgraph_input_tensors.size(): {}",
                     subgraph_input_tensors.size());

        // Clear subgraph node indgree, predecessors and successors
        subgraph_node->indegree = 0;
        subgraph_node->predecessors.clear();
        subgraph_node->successors.clear();

        // Update input and output tensors.
        for (auto tensor : subgraph_input_tensors) {
            // Remove old nodes from consumers
            auto consumers = tensor->consumers;
            auto consumers_iter =
                std::find(consumers.begin(), consumers.end(), old_nodes[0]);
            if (consumers_iter != consumers.end()) {
                consumers.erase(consumers_iter);
            }
            // Add subgraph_node to consumers
            tensor->consumers.push_back(subgraph_node);
            if (tensor->producer != NULL) {
                subgraph_node->predecessors.push_back(tensor->producer);
                tensor->producer->successors.push_back(subgraph_node);
                // Remove old nodes from predecessors.
                for (auto old_node : old_nodes) {
                    auto predecessors_iter =
                        std::find(tensor->producer->successors.begin(),
                                  tensor->producer->successors.end(), old_node);
                    if (predecessors_iter !=
                        tensor->producer->successors.end()) {
                        tensor->producer->successors.erase(predecessors_iter);
                    }
                }
            }
        }

        for (auto tensor : subgraph_output_tensors) {
            if (tensor->consumers.size() > 0) {
                for (auto consumer : tensor->consumers) {
                    // Add subgraph node successors
                    subgraph_node->successors.push_back(consumer);
                    consumer->predecessors.push_back(subgraph_node);

                    // Remove old nodes from consumers
                    for (auto old_node : old_nodes) {
                        auto consumers_iter =
                            std::find(consumer->predecessors.begin(),
                                      consumer->predecessors.end(), old_node);
                        if (consumers_iter != consumer->predecessors.end()) {
                            consumer->predecessors.erase(consumers_iter);
                        }
                    }
                }
            }
            tensor->setProducer(subgraph_node);
        }

        for (auto tensor : subgraph_input_tensors) {
            subgraph_node->indegree += tensor->producer == NULL ? 0 : 1;
        }

        // Add subgraph_node to operators
        operators.push_back(subgraph_node);

        // Remove old nodes from operators
        for (auto old_node : old_nodes) {
            auto operators_iter =
                std::find(operators.begin(), operators.end(), old_node);
            if (operators_iter != operators.end()) {
                operators.erase(operators_iter);
            }
        }
    }

}  // namespace tilegraph::graph