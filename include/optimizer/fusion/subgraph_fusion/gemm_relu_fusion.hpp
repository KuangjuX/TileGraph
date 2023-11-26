#pragma once
#include "core/graph/graph.hpp"
#include "core/graph/subgraph_match.hpp"
#include "optimizer/fusion/subgraph_fusion/subgraph_fusion_base.hpp"

namespace tilegraph::fusion::subgraph {
    // class GemmReluFusion {
    //    public:
    //     GemmReluFusion(std::shared_ptr<graph::Graph> graph);
    //     virtual ~GemmReluFusion() = default;

    //     bool create_subgraphs();
    //     bool match();
    //     bool fuse_subgraph(graph::SubGraphRecord::Pointer subgraph_record);

    //     std::vector<graph::SubGraph::Pointer> subgraphs;
    //     std::shared_ptr<graph::SubGraphMatch> subgraph_match;
    //     std::shared_ptr<graph::Graph> graph;
    // };

    class GemmReluFusion : public SubgraphFusionBase {
       public:
        GemmReluFusion(std::shared_ptr<graph::Graph> graph);
        virtual ~GemmReluFusion() = default;

        virtual void create_subgraphs() override;
        virtual Result<void, FusionError> fuse_subgraph(
            graph::SubGraphRecord::Pointer subgraph_record) override;
    };
}  // namespace tilegraph::fusion::subgraph