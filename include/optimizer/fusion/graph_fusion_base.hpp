#pragma once
#include "core/graph/graph.hpp"

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    class GraphFusionBase {
       public:
        virtual bool fusion(Graph::Pointer graph) { return true; }
    };
}  // namespace tilegraph::fusion