#pragma once
#include <variant>
#include <string>
#include <vector>

namespace tilegraph {

using Attribute =
    std::variant<int32_t, int64_t, bool, float, double, void *, std::string,
                 std::vector<int32_t>, std::vector<int64_t>, std::vector<bool>,
                 std::vector<float>, std::vector<double>, std::vector<void *>,
                 std::vector<std::string>>;

}