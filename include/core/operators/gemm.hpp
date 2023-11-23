#pragma once
#include "core/graph/graph.hpp"

namespace tilegraph::operators {
    using namespace tilegraph::graph;

    class Gemm : public Node {
       public:
        // Constructor
        Gemm(std::vector<Data *> inputs_list = {},
             std::vector<Data *> outputs_list = {}, std::string name_value = "",
             int64_t outputs_num_value = 1);
        // Destructor
        ~Gemm() = default;
    };

}  // namespace tilegraph::operators