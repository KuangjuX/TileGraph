#include "kernels/cuda/tensor_core.hpp"

namespace tilegraph::kernel::cuda {
    std::string genWmmaSync(int indient, std::string a, std::string b,
                            std::string c, std::string d) {
        std::string mma_sync;
        for (int i = 0; i < indient; i++) {
            mma_sync += " ";
        }

        // D = A * B + C / C = A * B + C
        mma_sync += fmt::format("nvcuda::wmma::mma_sync({}, {}, {}, {});\n;", d,
                                a, b, c);

        return mma_sync;
    }
}  // namespace tilegraph::kernel::cuda