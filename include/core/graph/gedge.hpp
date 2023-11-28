#pragma once
#include <vector>
#include <string>
#include <memory>

#include "core/graph/gnode.hpp"
#include "core/tensor.hpp"

namespace tilegraph::graph {
    class GNode;
    class GEdge {
       private:
        static int64_t edge_count;

       public:
        std::string name;
        const int64_t index;
        std::shared_ptr<GNode> producer;
        std::vector<std::shared_ptr<GNode>> consumers;
        std::shared_ptr<Tensor> tensor;

       public:
        GEdge() = delete;
        GEdge(std::shared_ptr<Tensor> tensor_value,
              std::string name_value = "");
        GEdge(const std::vector<int64_t> &dimension,
              std::string name_value = "", std::string tensor_name_value = "",
              TensorDatatype dtype = TensorDatatype::FLOAT,
              TensorType type = TensorType::VARIABLE);

        ~GEdge() = default;
        void setProducer(std::shared_ptr<GNode> node);
        void addConsumer(std::shared_ptr<GNode> node);
        bool earseConsumer(GNode::Pointer node);
        std::shared_ptr<GNode> getProducer();
        std::vector<std::shared_ptr<GNode>> getConsumers();
        std::shared_ptr<Tensor> getTensor();

        using Pointer = std::shared_ptr<GEdge>;
    };
}  // namespace tilegraph::graph
