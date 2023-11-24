#include "core/graph/gedge.hpp"

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

    void GEdge::addConsumer(std::shared_ptr<GNode> consumer_value) {
        consumers.push_back(consumer_value);
    }

    void GEdge::setProducer(std::shared_ptr<GNode> producer_value) {
        producer = producer_value;
    }

    std::shared_ptr<Tensor> GEdge::getTensor() { return tensor; }

}  // namespace tilegraph::graph