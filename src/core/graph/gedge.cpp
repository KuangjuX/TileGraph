#include "core/graph/gedge.hpp"

#include <algorithm>

namespace tilegraph::graph {

    int64_t GEdge::edge_count = 0;

    GEdge::GEdge(std::shared_ptr<Tensor> tensor_value, std::string name_value)
        : index(edge_count++), name(name_value), tensor(tensor_value) {}

    GEdge::GEdge(const std::vector<int64_t> &dimension, std::string name_value,
                 std::string tensor_name_value, TensorDatatype dtype,
                 TensorType type)
        : index(edge_count++),
          name(name_value),
          tensor(std::make_shared<Tensor>(dimension, tensor_name_value, dtype,
                                          type)) {}

    void GEdge::addConsumer(std::shared_ptr<GNode> node) {
        consumers.push_back(node);
    }

    bool GEdge::earseConsumer(GNode::Pointer node) {
        auto it = std::find(consumers.begin(), consumers.end(), node);
        if (it != consumers.end()) {
            consumers.erase(it);
            return true;
        }
        return false;
    }

    void GEdge::setProducer(std::shared_ptr<GNode> node) { producer = node; }

    std::shared_ptr<GNode> GEdge::getProducer() { return producer; }

    std::vector<std::shared_ptr<GNode>> GEdge::getConsumers() {
        return consumers;
    }

    std::shared_ptr<Tensor> GEdge::getTensor() { return tensor; }

}  // namespace tilegraph::graph