#pragma once

#include "kernels/gemm.hpp"

namespace tilegraph::kernel::cuda::cute {
    class CuteGEMMKernel : public GEMMKernel {};
}  // namespace tilegraph::kernel::cuda::cute