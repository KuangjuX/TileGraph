#pragma once
#include <vector>
#include <string>
#include <memory>
#include <result.h>

#include "core/type.hpp"
#include "core/operators/operator.hpp"
#include "core/graph/subgraph.hpp"

namespace tilegraph::graph {
    using namespace tilegraph::operators;
    class GEdge;
    class GNode {
       public:
        struct GNodeError {
            enum class Kind { InferError };

            Kind kind;
        };
        using Pointer = std::shared_ptr<GNode>;

        std::string name;
        const int64_t index;
        int64_t in_degree;
        OperatorType op_type;
        // virtual class.
        Operator::OpBox op;
        std::vector<std::shared_ptr<GEdge>> inputs;
        std::vector<std::shared_ptr<GEdge>> outputs;
        std::vector<std::shared_ptr<GNode>> predecessors;
        std::vector<std::shared_ptr<GNode>> successors;

       public:
        GNode(std::vector<std::shared_ptr<GEdge>> inputs_list = {},
              std::vector<std::shared_ptr<GEdge>> outputs_list = {},
              OperatorType op_type = OperatorType::ADD,
              Operator::OpBox op = nullptr, std::string name_value = "");
        ~GNode() = default;
        int64_t getIndex();
        std::shared_ptr<GEdge> getOutput(int64_t index);
        std::vector<std::shared_ptr<GEdge>> getInputs();
        std::vector<std::shared_ptr<GEdge>> getOutputs();
        OperatorType getOperatorType();

        bool earseSuccessor(Pointer node);
        bool earsePredecessor(Pointer node);

        Result<std::vector<std::shared_ptr<GEdge>>, GNodeError> inferShape();

       private:
        static int64_t node_count;
    };
}  // namespace tilegraph::graph