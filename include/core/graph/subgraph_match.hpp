#pragma once
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include <fmtlog.h>

#include "core/graph/gnode.hpp"
#include "core/graph/graph.hpp"
#include "common/common.hpp"

namespace tilegraph::graph {

    struct PatternRecord;
    struct Pattern;
    struct SubGraph;
    struct SubGraphRecord;
    class SubGraphMatch;

    struct Pattern {
        // pair<op types, starting node index for next pattern>
        // eg. { {{"Add", "Mul"}, 1},  {{"Add", "Convert","Div"}, 2}}
        std::vector<std::pair<std::vector<OperatorType>, size_t>> descriptions;
        // pattern search order
        bool reverse_order = false;

        /*
        cutomized check for the pattern
        for example, if we want to ensure that the second node of the pattern
        has 2 outputs, since such information is not included in description,
        we could add check function to check.
        */
        std::vector<std::function<bool(const PatternRecord&)>> check;
        // PatternRecord::Pointer pattern_record;
        using Pointer = std::shared_ptr<Pattern>;
    };

    struct PatternRecord {
       public:
        PatternRecord(Pattern::Pointer p) : pattern(p) {}

        std::shared_ptr<GNode> get_next_start_node() {
            ASSERT(pattern_description_idx < pattern->descriptions.size(),
                   "Pattern description index out of range.");
            size_t idx = pattern->descriptions[pattern_description_idx].second;
            ASSERT(idx < nodes.size(), "Index out of range.");
            return nodes[idx];
        }

        bool is_valid() {
            if (pattern == nullptr || nodes.empty()) return false;

            for (auto func : pattern->check) {
                if (!func(*this)) return false;
            }

            return true;
        }

        std::string get_symbol() {
            std::string identity;
            for (auto node : nodes) {
                auto id = node->getIndex();
                identity += std::to_string(id) + "_";
            }
            return identity;
        }

        /*
        pair< pattern nodes, pattern index>
        Pattern.descriptions may contain several descriptions with equal logic
        meaning, PatternRecord.nodes contains all nodes of matched pattern. For
        each pattern descrpition in Pattern.descriptions, there may exist
        several pattern in the graph, pair.second is used to indentify which
        description pair.first matches.
        */
        std::vector<std::shared_ptr<GNode>> nodes;
        void set_pattern_description_idx(size_t idx) {
            pattern_description_idx = idx;
        }
        size_t get_pattern_description_idx() const {
            return pattern_description_idx;
        }
        Pattern::Pointer pattern;
        using Pointer = std::shared_ptr<PatternRecord>;

       private:
        size_t pattern_description_idx;
    };

    /*
    SubGraph consists of many Pattern.
    for a subgraph like
            A
        /       \
        B          C
        |           |
        D           F
        |           /
        E          /
        \        /
            G

    , we can description it as pattern(A->B->D->E->G) with starting node A
    in non-reverse order followed by pattern(G->F->C) in reverse order.
    */
    struct SubGraph {
        std::string name;
        std::function<bool(std::shared_ptr<GNode>)> check_starting_node;
        std::vector<Pattern::Pointer> patterns;
        std::vector<std::function<bool(const SubGraphRecord&)>> check;
        using Pointer = std::shared_ptr<SubGraph>;
    };

    struct SubGraphRecord {
       public:
        SubGraphRecord(std::shared_ptr<GNode> sn, SubGraph::Pointer sg)
            : starting_node(sn), subgraph(sg) {}

        bool is_valid() {
            if (subgraph == nullptr || pattern_records.empty()) return false;

            if (!subgraph->check_starting_node(starting_node)) return false;

            for (auto pr : pattern_records) {
                if (!pr->is_valid()) return false;
            }

            for (auto func : subgraph->check) {
                if (!func(*this)) return false;
            }

            return true;
        }

        void set_starting_node(std::shared_ptr<GNode> node) {
            starting_node = node;
        }
        const std::shared_ptr<GNode>& get_starting_node() const {
            return starting_node;
        }
        std::vector<PatternRecord::Pointer> pattern_records;
        SubGraph::Pointer subgraph;
        using Pointer = std::shared_ptr<SubGraphRecord>;

       private:
        std::shared_ptr<GNode> starting_node;
    };

    class SubGraphMatch {
       public:
        SubGraphMatch(std::shared_ptr<Graph> g) : m_graph(g) {}

        bool Match(SubGraph::Pointer subgraph);
        bool FindSubGraph(SubGraphRecord::Pointer subgraph_record,
                          SubGraph::Pointer subgraph,
                          std::shared_ptr<GNode> start);
        bool SearchSubGraph(SubGraphRecord::Pointer subgraph_record,
                            SubGraph::Pointer subgraph,
                            PatternRecord::Pointer cur_pr, size_t idx);
        bool FindPattern(Pattern::Pointer pattern,
                         std::vector<PatternRecord::Pointer>& pattern_records,
                         std::shared_ptr<GNode> start);
        void SearchPattern(std::shared_ptr<GNode> cur_node,
                           size_t description_idx, size_t idx,
                           std::vector<PatternRecord::Pointer>& pattern_records,
                           std::vector<std::shared_ptr<GNode>>& pattern_nodes,
                           Pattern::Pointer pattern);
        const std::vector<SubGraphRecord::Pointer>& get_matched_subgraph()
            const {
            return m_matched_records;
        }
        void clear_matched_records() { m_matched_records.clear(); };

       private:
        std::shared_ptr<Graph> m_graph;
        std::vector<SubGraphRecord::Pointer> m_matched_records;
        std::unordered_set<std::shared_ptr<GNode>> m_starting_nodes;
    };
}  // namespace tilegraph::graph