#include "optimizer/fusion/subgraph_fusion/gemm_relu_fusion.hpp"
#include "core/graph/gnode.hpp"
#include "core/type.hpp"

#include <fmt/core.h>

namespace tilegraph::fusion::subgraph {

    GemmReluFusion::GemmReluFusion(std::shared_ptr<graph::Graph> graph)
        : graph(graph) {
        subgraph_match = std::make_shared<graph::SubGraphMatch>(graph);
    }

    bool GemmReluFusion::create_subgraphs() {
        using namespace graph;
        auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
            if (gnode->getOperatorType() != OperatorType::GEMM) {
                return false;
            }
            return true;
        };

        SubGraph::Pointer s_gemm_relu = std::make_shared<SubGraph>();
        s_gemm_relu->name = "GEMM_RELU";
        s_gemm_relu->check_starting_node = check_root;

        {
            Pattern::Pointer p_gemm_relu = std::make_shared<Pattern>();
            std::vector<OperatorType> ops{OperatorType::GEMM,
                                          OperatorType::RELU};
            p_gemm_relu->descriptions.push_back(std::make_pair(ops, 1));
            p_gemm_relu->reverse_order = false;
            auto check_gemm_relu = [](const PatternRecord& pr) -> bool {
                return true;
            };
            p_gemm_relu->check.push_back(check_gemm_relu);

            s_gemm_relu->patterns.push_back(p_gemm_relu);
        }

        subgraphs.push_back(s_gemm_relu);

        return true;
    }

    bool GemmReluFusion::match() {
        for (auto subgraph : subgraphs) {
            if (subgraph_match->Match(subgraph)) {
                auto records = subgraph_match->get_matched_subgraph();
                fmt::println("Matched subgraph: {}", subgraph->name);
            }
        }
        return true;
    }

    bool GemmReluFusion::fuse_subgraph(
        graph::SubGraphRecord::Pointer subgraph_record) {}
}  // namespace tilegraph::fusion::subgraph