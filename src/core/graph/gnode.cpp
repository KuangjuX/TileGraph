#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"

#include <algorithm>

namespace tilegraph::graph {

    int64_t GNode::node_count = 0;

    GNode::GNode(std::vector<std::shared_ptr<GEdge>> inputs_list,
                 std::vector<std::shared_ptr<GEdge>> outputs_list,
                 OperatorType op_type, std::shared_ptr<Operator::OpBox> op,
                 std::string name_value, int64_t outputs_num_value)
        : name(name_value),
          index(node_count++),
          in_degree(0),
          op_type(op_type),
          op(op),
          outputs_num(outputs_num_value),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Operator_" + std::to_string(index) : name);
    }

    int64_t GNode::getIndex() { return index; }

    std::shared_ptr<GEdge> GNode::getOutput(int64_t index) {
        return outputs[index];
    }

    std::vector<std::shared_ptr<GEdge>> GNode::getInputs() { return inputs; }

    std::vector<std::shared_ptr<GEdge>> GNode::getOutputs() { return outputs; }

    OperatorType GNode::getOperatorType() { return op_type; }

    bool GNode::earseSuccessor(Pointer node) {
        auto it = std::find(successors.begin(), successors.end(), node);
        if (it != successors.end()) {
            successors.erase(it);
            return true;
        }
        return false;
    }

    bool GNode::earsePredecessor(Pointer node) {
        auto it = std::find(predecessors.begin(), predecessors.end(), node);
        if (it != predecessors.end()) {
            predecessors.erase(it);
            return true;
        }
        return false;
    }

}  // namespace tilegraph::graph