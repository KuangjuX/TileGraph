#pragma once
#include <string>
#include <memory>
namespace tilegraph::codegen {
    class Generator {
       public:
        std::string code;
        virtual ~Generator() = default;
        virtual void generate_head();
        virtual void generate_kernel();
        virtual void generate_host();
    };
}  // namespace tilegraph::codegen