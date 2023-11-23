#pragma once
#include "core/graph/graph.hpp"

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    class GraphFusionBase {
       public:
        virtual bool fusion(std::shared_ptr<Graph> graph) { return true; }
    };
}  // namespace tilegraph::fusion