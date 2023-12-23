#pragma once
#include <fmt/core.h>
#include <string>

namespace tilegraph::kernel::cuda {
    std::string genMMASync(int indient, std::string a, std::string b,
                           std::string c, std::string d) {
        std::string mma_sync;
        for (int i = 0; i < indient; i++) {
            mma_sync += " ";
        }

        mma_sync +=
            fmt::format("nvcuda::wmma::mma_sync({}, {}, {}, {})", d, a, b, c);

        return mma_sync;
    }
}  // namespace tilegraph::kernel::cuda