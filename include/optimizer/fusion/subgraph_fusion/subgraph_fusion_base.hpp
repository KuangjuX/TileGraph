#pragma once
#include "core/graph/graph.hpp"
#include "core/graph/subgraph_match.hpp"

#include <result.h>

namespace tilegraph::fusion::subgraph {
    class SubgraphFusionBase {
       public:
        struct FusionError {
            enum class Kind {
                UnmatchedSubgraph,
            };

            Kind kind;
        };
        SubgraphFusionBase(graph::Graph::Pointer graph);
        virtual ~SubgraphFusionBase() = default;

        virtual void create_subgraphs() = 0;
        virtual Result<void, FusionError> fuse_subgraph(
            graph::SubGraphRecord::Pointer subgraph_record) = 0;
        Result<void, FusionError> match_and_fuse_subgraph();

        std::vector<graph::SubGraph::Pointer> subgraphs;
        std::shared_ptr<graph::SubGraphMatch> subgraph_match;
        std::shared_ptr<graph::Graph> graph;
    };
}  // namespace tilegraph::fusion::subgraph