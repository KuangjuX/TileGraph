#pragma once
#include "codegen/compiler.hpp"

namespace tilegraph::codegen {

    class CudaCompiler final : public Compiler {
       protected:
        std::string_view hardware() const noexcept final;
        std::string_view extension() const noexcept final;
        void *_compile(std::filesystem::path const &src,
                       const char *symbol) final;
    };

}  // namespace tilegraph::codegen