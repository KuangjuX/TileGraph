#include "core/graph/graph.hpp"
#include "optimizer/fusion/graph_fusion_base.hpp"

#include <map>

using namespace tilegraph::graph;

namespace tilegraph::fusion {
    class PersistentKernelFusion : public GraphFusionBase {
       public:
        PersistentKernelFusion() = default;
        PersistentKernelFusion(Graph::Pointer graph);
        bool fusion(Graph::Pointer graph) override;

       private:
        std::size_t find_root(
            std::unordered_map<std::size_t, std::size_t> node_to_group,
            std::size_t node_idx);
        std::unordered_map<std::size_t, std::vector<GNode::Pointer>>
        find_groups(std::unordered_map<std::size_t, std::size_t> node_to_group,
                    std::unordered_map<std::size_t, GNode::Pointer> nodes,
                    std::unordered_map<GNode::Pointer, std::size_t> node_map);
    };
}  // namespace tilegraph::fusion