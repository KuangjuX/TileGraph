#include <algorithm>
#include <iostream>

#include <fmt/core.h>

#include "core/graph/graph.hpp"
#include "core/graph/subgraph.hpp"

namespace tilegraph::graph {

    int64_t Node::count = 0;
    int64_t Data::count = 0;
    int64_t Graph::count = 0;

    // Operator implementation
    Node::Node(std::vector<Data*> inputs_list, std::vector<Data*> outputs_list,
               std::string name_value, int64_t outputs_num_value)
        : name(name_value),
          index(count++),
          indegree(0),
          outputs_num(outputs_num_value),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Operator_" + std::to_string(index) : name);
        if (outputs.empty()) {
            Data* temp;
            for (auto i = 0; i < outputs_num; ++i) {
                temp = new Data(inputs[0]->tensor_dimension, inputs[0]->name,
                                inputs[0]->tensor_datatype,
                                inputs[0]->tensor_type);
                outputs.push_back(temp);
            }
        }
        for (auto it : inputs) {
            it->addConsumer(this);
            it->remaining += 1;
            if (it->producer != NULL) {
                predecessors.push_back(it->producer);
                it->producer->successors.push_back(this);
            }
        }
        for (auto it : outputs) {
            it->setProducer(this);
        }
        for (auto it : inputs) {
            indegree += it->producer == NULL ? 0 : 1;
        }
    }

    Node::Node(std::vector<Data*> inputs_list, std::vector<Data*> outputs_list,
               std::shared_ptr<SubGraph> subgraph, std::string name_value,
               int64_t outputs_num_value)
        : name(name_value),
          index(count++),
          indegree(0),
          outputs_num(outputs_num_value),
          inputs(inputs_list),
          outputs(outputs_list),
          subgraph(subgraph) {
        name =
            (name == "" ? "SubGraph_Operator_" + std::to_string(index) : name);
        this->operator_type = OperatorType::SUBGRAPH;
    }

    Data* Node::getOutput(int64_t index) { return outputs[index]; }

    std::vector<Data*> Node::getOutputs() { return outputs; }

    OperatorType Node::getOperatorType() { return operator_type; }

    Data::Data(const std::vector<int64_t>& dimension, std::string name_value,
               TensorDatatype dtype, TensorType type)
        : tensor_dimension(dimension),
          tensor_datatype(dtype),
          tensor_type(type),
          name(name_value),
          index(count++),
          producer(NULL),
          remaining(0) {
        name = (name == "" ? "Data_" + std::to_string(index) : name);
    }

    void Data::setProducer(Node* producer_value) { producer = producer_value; }

    void Data::addConsumer(Node* consumer_value) {
        consumers.push_back(consumer_value);
    }

    // Graph implementation
    Graph::Graph(std::vector<Node*> operators_list,
                 std::vector<Data*> inputs_list,
                 std::vector<Data*> outputs_list, std::string name_value)
        : name(name_value),
          index(count++),
          operators(operators_list),
          inputs(inputs_list),
          outputs(outputs_list),
          platform(Platform::CUDA) {
        name = (name == "" ? "Graph_" + std::to_string(index) : name);
    }

    std::vector<Node*> Graph::topoSort() {
        std::unordered_map<Node*, int64_t> operators_indegree;
        for (auto op : operators) {
            operators_indegree[op] = op->indegree;
            fmt::println("op->name: {}, op->indegree: {}", op->name,
                         op->indegree);
        }
        std::vector<Node*> result;
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

    void Graph::printGraph() {
        //   std::string info_string = "";
        //   info_string += "Graph ";
        //   info_string += "Name: [";
        //   info_string += name;
        //   info_string += "] ";
        //   LOG(INFO) << info_string;
        //   LOG(INFO) << "==== Operators ====";
        //   for (auto it : operators) {
        //     it->printLink();
        //   }
        //   LOG(INFO) << "==== Inputs ====";
        //   for (auto it : inputs) {
        //     it->printLink();
        //   }
        //   LOG(INFO) << "==== Temps ====";
        //   for (auto it : temps) {
        //     it->printLink();
        //   }
        //   LOG(INFO) << "==== Outputs ====";
        //   for (auto it : outputs) {
        //     it->printLink();
        //   }
        fmt::print("Graph Name: {}\n", name);
    }

    bool Graph::fuseNode(std::vector<Node*> old_nodes, Node* subgraph_node) {
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
                tensor->remaining -= 1;
            }
            // Add subgraph_node to consumers
            tensor->consumers.push_back(subgraph_node);
            tensor->remaining += 1;
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