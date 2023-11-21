#include <algorithm>
#include <fmt/core.h>

#include "core/graph/graph.hpp"
#include "core/graph/subgraph.hpp"

namespace tilegraph {

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
        for (auto op : operators) {
            for (auto data : op->inputs) {
                remaining_data.insert(data);
            }
            for (auto data : op->outputs) {
                auto outputs_iter =
                    std::find(outputs.begin(), outputs.end(), data);
                if (outputs_iter == outputs.end()) {
                    temps.push_back(data);
                }
            }
        }
    }

    std::vector<Node*> Graph::topoSort() {
        std::unordered_map<Node*, int64_t> operators_temp;
        for (auto op : operators) {
            operators_temp[op] = op->indegree;
        }
        std::vector<Node*> result;
        while (!operators_temp.empty()) {
            for (auto op = operators_temp.begin(); op != operators_temp.end();
                 ++op) {
                if (op->second == 0) {
                    result.push_back(op->first);
                    for (auto successor : (op->first)->successors) {
                        --operators_temp[successor];
                    }
                    operators_temp.erase(op->first);
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

    bool Graph::removeNode(int64_t node_index) {
        for (auto it = operators.begin(); it != operators.end(); ++it) {
            if ((*it)->index == node_index) {
                operators.erase(it);
                return true;
            }
        }
        return false;
    }

    void Graph::addNode(Node* node) { this->operators.push_back(node); }

}  // namespace tilegraph