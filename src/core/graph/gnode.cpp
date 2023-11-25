#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"

namespace tilegraph::graph {

    int64_t GNode::node_count = 0;

    GNode::GNode(std::vector<std::shared_ptr<GEdge>> inputs_list,
                 std::vector<std::shared_ptr<GEdge>> outputs_list,
                 OperatorType op_type, std::string name_value,
                 int64_t outputs_num_value)
        : name(name_value),
          index(node_count++),
          indegree(0),
          operator_type(op_type),
          outputs_num(outputs_num_value),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Operator_" + std::to_string(index) : name);
        // if (outputs.empty()) {
        //     std::shared_ptr<Tensor> inter_edge;
        //     for (auto i = 0; i < outputs_num; ++i) {
        //         inter_edge = std::make_shared<Tensor>(
        //             inputs[0].get()->getTensor().get()->tensor_dimension,
        //             inputs[0].get()->getTensor().get()->name,
        //             inputs[0].get()->getTensor().get()->tensor_datatype,
        //             inputs[0].get()->getTensor().get()->tensor_type);
        //         outputs.push_back(std::make_shared<Edge>(inter_edge));
        //     }
        // }
        // for (auto edge : inputs) {
        //     auto it = edge.get();
        //     it->addConsumer(std::make_shared<Node>(this));
        //     if (it->producer != NULL) {
        //         predecessors.push_back(it->producer);
        //         it->producer.get()->successors.push_back(
        //             std::make_shared<Node>(this));
        //     }
        // }
        // for (auto it : outputs) {
        //     // it->setProducer(this);
        //     it->setProducer(std::make_shared<Node>(this));
        // }
        // for (auto it : inputs) {
        //     indegree += it->producer == NULL ? 0 : 1;
        // }
    }

    // GNode::GNode(std::vector<std::shared_ptr<GEdge>> inputs_list,
    //              std::vector<std::shared_ptr<GEdge>> outputs_list,
    //              std::shared_ptr<SubGraph>, std::string name_value,
    //              int64_t outputs_num_value)
    //     : name(name_value),
    //       index(node_count++),
    //       indegree(0),
    //       outputs_num(outputs_num_value),
    //       inputs(inputs_list),
    //       outputs(outputs_list),
    //       subgraph(subgraph) {
    //     name =
    //         (name == "" ? "SubGraph_Operator_" + std::to_string(index) :
    //         name);
    //     this->operator_type = OperatorType::SUBGRAPH;
    // }

    int64_t GNode::getIndex() { return index; }

    std::shared_ptr<GEdge> GNode::getOutput(int64_t index) {
        return outputs[index];
    }

    std::vector<std::shared_ptr<GEdge>> GNode::getInputs() { return inputs; }

    std::vector<std::shared_ptr<GEdge>> GNode::getOutputs() { return outputs; }

    OperatorType GNode::getOperatorType() { return operator_type; }

}  // namespace tilegraph::graph