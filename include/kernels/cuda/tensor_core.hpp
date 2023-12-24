#pragma once
#include <fmt/core.h>
#include <string>

namespace tilegraph::kernel::cuda {
    std::string genWmmaSync(int indient, std::string a, std::string b,
                            std::string c, std::string d);
}  // namespace tilegraph::kernel::cuda