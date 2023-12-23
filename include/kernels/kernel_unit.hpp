#pragma once
#include <string>
namespace tilegraph::kernel {
    // class KernelUnit {};

    std::string insertIndient(int indient) {
        std::string res;
        for (int i = 0; i < indient; i++) {
            res += " ";
        }
        return res;
    }
}  // namespace tilegraph::kernel