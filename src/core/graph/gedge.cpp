#include "core/graph/gedge.hpp"

namespace tilegraph::graph {

    int64_t GEdge::edge_count = 0;

    GEdge::GEdge(std::shared_ptr<Tensor> tensor_value, std::string name_value)
        : index(edge_count++), name(name_value), tensor(tensor_value) {}

    void GEdge::addConsumer(std::shared_ptr<GNode> consumer_value) {
        consumers.push_back(consumer_value);
    }

    void GEdge::setProducer(std::shared_ptr<GNode> producer_value) {
        producer = producer_value;
    }

    std::shared_ptr<Tensor> GEdge::getTensor() { return tensor; }

}  // namespace tilegraph::graph