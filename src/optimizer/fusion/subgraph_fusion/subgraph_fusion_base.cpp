#include "optimizer/fusion/subgraph_fusion/subgraph_fusion_base.hpp"

#include <fmt/core.h>

namespace tilegraph::fusion::subgraph {
    SubgraphFusionBase::SubgraphFusionBase(graph::Graph::Pointer graph)
        : graph(graph) {
        this->subgraph_match = std::make_shared<graph::SubGraphMatch>(graph);
    }

    Result<void, SubgraphFusionBase::FusionError>
    SubgraphFusionBase::match_and_fuse_subgraph() {
        for (auto subgraph : this->subgraphs) {
            if (this->subgraph_match->Match(subgraph)) {
                auto records = this->subgraph_match->get_matched_subgraph();
                logi("Matched record size: {}", records.size());
                for (auto record : records) {
                    this->fuse_subgraph(record);
                }
            } else {
                return Err(SubgraphFusionBase::FusionError{
                    SubgraphFusionBase::FusionError::Kind::UnmatchedSubgraph});
            }
        }
        return Ok();
    }
}  // namespace tilegraph::fusion::subgraph