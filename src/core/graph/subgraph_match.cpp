#include <stack>

#include "core/graph/subgraph_match.hpp"

using namespace tilegraph;
using namespace tilegraph::graph;

bool SubGraphMatch::Match(SubGraph::Pointer subgraph) {
    // check starting node
    for (auto node : m_graph->topoSort()) {
        if (subgraph->check_starting_node(node) &&
            m_starting_nodes.find(node) == m_starting_nodes.end()) {
            SubGraphRecord::Pointer subgraph_record =
                std::make_shared<SubGraphRecord>(node, subgraph);
            if (FindSubGraph(subgraph_record, subgraph, node)) {
                m_matched_records.push_back(subgraph_record);
                m_starting_nodes.insert(node);
            }
        }
    }

    return !m_matched_records.empty();
}

bool SubGraphMatch::FindSubGraph(SubGraphRecord::Pointer subgraph_record,
                                 SubGraph::Pointer subgraph,
                                 std::shared_ptr<GNode> start) {
    ASSERT(subgraph->patterns.size() > 0, "Subgraph patterns is empty!");
    auto init_pattern = subgraph->patterns[0];
    std::vector<PatternRecord::Pointer> init_pattern_records;
    if (FindPattern(init_pattern, init_pattern_records, start)) {
        for (auto pr : init_pattern_records) {
            if (SearchSubGraph(subgraph_record, subgraph, pr, 1) &&
                subgraph_record->is_valid()) {
                return true;  // return true when we find the first subgraph
            } else if (!subgraph_record->is_valid()) {
                loge("Subgraph Invalid!");
                return false;
            }
        }
    }

    return false;
}

bool SubGraphMatch::SearchSubGraph(SubGraphRecord::Pointer subgraph_record,
                                   SubGraph::Pointer subgraph,
                                   PatternRecord::Pointer cur_pr, size_t idx) {
    std::stack<PatternRecord::Pointer> s;
    std::vector<PatternRecord::Pointer> matched_pattern_records;
    std::unordered_set<std::string> pr_symbols;
    s.push(cur_pr);
    while (!s.empty()) {
        cur_pr = s.top();
        s.pop();
        subgraph_record->pattern_records.push_back(cur_pr);
        pr_symbols.insert(cur_pr->get_symbol());

        if (idx == subgraph->patterns.size() &&
            subgraph_record->pattern_records.size() ==
                subgraph->patterns.size()) {
            return true;
        } else {
            auto start = cur_pr->get_next_start_node();
            auto next_pattern = subgraph->patterns[idx];
            // to ensure that subgraoh_record->pattern_records does not contain
            // the same record.
            bool is_valid = false;
            if (FindPattern(next_pattern, matched_pattern_records, start)) {
                for (auto pr : matched_pattern_records) {
                    auto symbol = pr->get_symbol();
                    if (pr_symbols.find(symbol) == pr_symbols.end()) {
                        s.push(pr);
                        is_valid = true;
                    }
                }

                idx++;
            }

            if (!is_valid) {
                auto back_pr = subgraph_record->pattern_records.back();
                subgraph_record->pattern_records.pop_back();
                pr_symbols.erase(back_pr->get_symbol());
            }
        }
    }

    return false;
}

bool SubGraphMatch::FindPattern(
    Pattern::Pointer pattern,
    std::vector<PatternRecord::Pointer>& pattern_records,
    std::shared_ptr<GNode> start) {
    pattern_records.clear();
    std::vector<std::shared_ptr<GNode>> pattern_nodes;
    pattern_nodes.push_back(start);
    for (size_t i = 0; i < pattern->descriptions.size(); i++) {
        SearchPattern(start, i, 1, pattern_records, pattern_nodes, pattern);
    }

    if (pattern_records.empty()) {
        return false;
    }
    return true;
}

void SubGraphMatch::SearchPattern(
    std::shared_ptr<GNode> cur_node, size_t description_idx, size_t idx,
    std::vector<PatternRecord::Pointer>& pattern_records,
    std::vector<std::shared_ptr<GNode>>& pattern_nodes,
    Pattern::Pointer pattern) {
    auto description_ops = pattern->descriptions[description_idx].first;
    if (idx == description_ops.size() &&
        pattern_nodes.size() == description_ops.size()) {
        PatternRecord::Pointer pr = std::make_shared<PatternRecord>(pattern);
        pr->nodes = pattern_nodes;
        pr->set_pattern_description_idx(description_idx);
        if (pr->is_valid())
            pattern_records.push_back(pr);
        else {
            loge("PatternRecord Invalid!");
        }
    } else {
        std::vector<std::shared_ptr<GEdge>> edges;
        if (pattern->reverse_order) {
            edges = cur_node->getInputs();
        } else {
            edges = cur_node->getOutputs();
        }

        for (auto edge : edges) {
            std::vector<std::shared_ptr<GNode>> sub_nodes = {};
            if (pattern->reverse_order) {
                sub_nodes.push_back(edge->getProducer());

            } else {
                sub_nodes = edge->getConsumers();
            }

            for (auto sub_node : sub_nodes) {
                if (sub_node->getOperatorType() == description_ops[idx]) {
                    logi("Find Pattern: {}",
                         toString(sub_node->getOperatorType()));
                    pattern_nodes.push_back(sub_node);
                    SearchPattern(sub_node, description_idx, idx + 1,
                                  pattern_records, pattern_nodes, pattern);
                    pattern_nodes.pop_back();
                } else {
                    logi("Not Find Pattern: {}",
                         toString(sub_node->getOperatorType()));
                }
            }
        }
    }
}