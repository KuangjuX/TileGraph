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

        ~GEdge() = default;
        void setProducer(std::shared_ptr<GNode> producer_value);
        void addConsumer(std::shared_ptr<GNode> consumer_value);
        std::shared_ptr<Tensor> getTensor();
    };
}  // namespace tilegraph::graph
