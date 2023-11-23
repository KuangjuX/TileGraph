#pragma once
#include "core/graph/graph.hpp"

namespace tilegraph {
    namespace fusion {
        class GraphFusionBase {
           public:
            virtual bool fusion(std::shared_ptr<Graph> graph) { return true; }
        };
    }  // namespace fusion
}  // namespace tilegraph