#include "core/graph/graph.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    class PersistentKernelFusion : public GraphFusionBase {
       public:
        PersistentKernelFusion() = default;
        PersistentKernelFusion(Graph::Pointer graph);
        bool fusion(Graph::Pointer graph) override;
    };
}  // namespace tilegraph::fusion