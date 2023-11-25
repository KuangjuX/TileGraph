#pragma once
#include <vector>
#include <string>
#include <memory>

#include "core/type.hpp"
#include "core/graph/subgraph.hpp"
// #include "core/graph/gedge.hpp"

namespace tilegraph::graph {
    class GEdge;
    // class SubGraph;
    class GNode {
       private:
        static int64_t node_count;

       public:
        std::string name;
        const int64_t index;
        int64_t indegree;
        int64_t outputs_num;
        std::vector<std::shared_ptr<GEdge>> inputs;
        std::vector<std::shared_ptr<GEdge>> outputs;
        std::vector<std::shared_ptr<GNode>> predecessors;
        std::vector<std::shared_ptr<GNode>> successors;
        OperatorType operator_type;
        // std::shared_ptr<SubGraph> subgraph;

       public:
        GNode(std::vector<std::shared_ptr<GEdge>> inputs_list = {},
              std::vector<std::shared_ptr<GEdge>> outputs_list = {},
              OperatorType op_type = OperatorType::ADD,
              std::string name_value = "", int64_t outputs_num_value = 1);
        // GNode(std::vector<std::shared_ptr<GEdge>> inputs_list = {},
        //       std::vector<std::shared_ptr<GEdge>> outputs_list = {},
        //       std::shared_ptr<SubGraph> = nullptr, std::string name_value =
        //       "", int64_t outputs_num_value = 1);
        ~GNode() = default;
        // std::shared_ptr<Edge> getOutput(int64_t index);
        // std::vector<std::shared_ptr<Edge>> getOutputs();
        int64_t getIndex();
        std::shared_ptr<GEdge> getOutput(int64_t index);
        std::vector<std::shared_ptr<GEdge>> getInputs();
        std::vector<std::shared_ptr<GEdge>> getOutputs();
        OperatorType getOperatorType();
    };
}  // namespace tilegraph::graph