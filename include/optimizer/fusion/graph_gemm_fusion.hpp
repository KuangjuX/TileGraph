#pragma once
#include "optimizer/fusion/graph_fusion_base.hpp"
#include "core/graph/subgraph.hpp"

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    class GemmFusion : public GraphFusionBase {
       public:
        bool fusion(std::shared_ptr<Graph> graph) override;
    };
}  // namespace tilegraph::fusion