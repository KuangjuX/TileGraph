#include <dlfcn.h>

#include "codegen/cuda_compiler.hpp"
#include "common/common.hpp"

namespace tilegraph::codegen {

    auto CudaCompiler::hardware() const noexcept -> std::string_view {
        return "CUDA";
    }
    auto CudaCompiler::extension() const noexcept -> std::string_view {
        return "cu";
    }
    void *CudaCompiler::_compile(std::filesystem::path const &src,
                                 const char *symbol) {
        auto out = src, so = src;
        out.replace_extension("o");
        so.replace_filename("libkernel.so");
        {
            std::string command;
            command = fmt::format("nvcc -Xcompiler \"-fPIC\" {} -c -o {}",
                                  src.c_str(), out.c_str());
            std::system(command.c_str());
            command =
                fmt::format("nvcc -shared {} -o {}", out.c_str(), so.c_str());
            std::system(command.c_str());
        }

        auto handle = dlopen(so.c_str(), RTLD_LAZY);
        ASSERT(handle, "Failed to load kernel library: {}", dlerror());
        auto function = dlsym(handle, symbol);
        ASSERT(function, "Failed to load kernel function: {}", dlerror());
        return function;
    }

}  // namespace tilegraph::codegen