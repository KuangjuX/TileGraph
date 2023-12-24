#pragma once
#include <string>

namespace tilegraph::kernel::cuda {
    enum class CudaHeader {
        cuda,
        cuda_runtime,
        cublas,
        cudnn,
        cutlass,
    };

    std::string generateHeader(CudaHeader header) {
        switch (header) {
            case CudaHeader::cuda:
                return "#include <cuda.h>\n";
            case CudaHeader::cuda_runtime:
                return "#include <cuda_runtime.h>\n";
            case CudaHeader::cublas:
                return "#include <cublas_v2.h>\n";
            case CudaHeader::cudnn:
                return "#include <cudnn.h>\n";
            case CudaHeader::cutlass:
                return "#include <cutlass/cutlass.h>\n";
            default:
                return "";
        }
    }

}  // namespace tilegraph::kernel::cuda