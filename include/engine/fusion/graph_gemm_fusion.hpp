#pragma once
#include "engine/fusion/graph_fusion_base.hpp"

namespace tilegraph {
    namespace engine {
        namespace fusion {
            class GemmFusion : public GraphFusionBase {
               public:
                GemmFusion() = default;

                ~GemmFusion() = default;
            };
        }  // namespace fusion
    }      // namespace engine
}  // namespace tilegraph