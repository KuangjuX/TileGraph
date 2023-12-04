#include "kernels/kernel_unit.hpp"
#include "kernels/cuda/header.hpp"
#include <set>

namespace tilegraph::kernel::cuda {
    class CudaKernelUnit : public KernelUnit {
       public:
        CudaKernelUnit() = default;
        ~CudaKernelUnit() = default;

        std::set<CudaHeader> headers;

        void addHeader(CudaHeader header) { headers.insert(header); }
    };
}  // namespace tilegraph::kernel