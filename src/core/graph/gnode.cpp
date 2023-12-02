#include "core/graph/gnode.hpp"
#include "core/graph/gedge.hpp"
#include "core/tensor.hpp"
#include "core/operators/gemm.hpp"
#include "core/operators/binary.hpp"
#include "core/operators/unary.hpp"

#include <algorithm>
#include <fmtlog.h>

namespace tilegraph::graph {
    using namespace tilegraph::operators;

    int64_t GNode::node_count = 0;

    GNode::GNode(std::vector<std::shared_ptr<GEdge>> inputs_list,
                 std::vector<std::shared_ptr<GEdge>> outputs_list,
                 OperatorType op_type, Operator::OpBox op,
                 std::string name_value)
        : name(name_value),
          index(node_count++),
          in_degree(0),
          op_type(op_type),
          op(op),
          inputs(inputs_list),
          outputs(outputs_list) {
        name = (name == "" ? "Operator_" + std::to_string(index) : name);
        if (op == nullptr) {
            switch (op_type) {
                // Default GEMM operator.
                case OperatorType::GEMM:
                    this->op = std::make_shared<GEMM>();
                    break;
                // Unary operators.
                case OperatorType::SIN:
                case OperatorType::COS:
                case OperatorType::SQRT:
                case OperatorType::RELU:
                case OperatorType::SOFTMAX:
                case OperatorType::SIGMOID:
                case OperatorType::TANH:
                    this->op = std::make_shared<Unary>(op_type);
                    break;
                // Binary operators.
                case OperatorType::ADD:
                case OperatorType::SUB:
                case OperatorType::MUL:
                case OperatorType::DIV:
                    this->op = std::make_shared<Binary>(op_type);
                default:
                    loge("[GNode::GNode] Operator type is not supported.");
                    break;
            }
        }
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

    Result<std::vector<std::shared_ptr<GEdge>>, GNode::GNodeError>
    GNode::inferShape() {
        if (this->op != nullptr) {
            // Get tensors from inputs.
            std::vector<std::shared_ptr<Tensor>> input_tensors;
            for (auto &input : this->inputs) {
                input_tensors.push_back(input->getTensor());
            }
            auto output_tensors = this->op->inferShape(input_tensors);

            std::vector<std::shared_ptr<GEdge>> outputs;
            for (auto &output_tensor : output_tensors) {
                auto output = std::make_shared<GEdge>(output_tensor);
                outputs.push_back(output);
            }
            // this->outputs = outputs;
            return Ok(outputs);
        } else {
            loge("[GNode::inferShape] Operator is nullptr.");
            return Err(GNode::GNodeError{GNode::GNodeError::Kind::InferError});
        }
    }

}  // namespace tilegraph::graph