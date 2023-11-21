#pragma once
#include "core/graph/graph.hpp"

namespace tilegraph {
    namespace engine {
        namespace fusion {
            class GraphGusionBase {
               public:
                virtual bool fusion(std::shared_ptr<Graph> graph) {
                    return true;
                }
            };
        }  // namespace fusion
    }      // namespace engine
}  // namespace tilegraph