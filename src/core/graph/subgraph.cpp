#include "core/graph/subgraph.hpp"

#include <algorithm>

namespace tilegraph::graph {
    int64_t SubGraph::count = 0;
    // Graph implementation
    SubGraph::SubGraph(std::vector<Node*> operators_list,
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

    std::vector<Node*> SubGraph::topoSort() {
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

    void SubGraph::printGraph() {
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
        // fmt::print("Graph Name: {}\n", name);
    }

    bool SubGraph::removeNode(int64_t node_index) {
        for (auto it = operators.begin(); it != operators.end(); ++it) {
            if ((*it)->index == node_index) {
                operators.erase(it);
                return true;
            }
        }
        return false;
    }

    void SubGraph::addNode(Node* node) { this->operators.push_back(node); }
}  // namespace tilegraph::graph
