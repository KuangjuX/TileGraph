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

}  // namespace tilegraph::graph