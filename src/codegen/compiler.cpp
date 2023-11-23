#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unistd.h>

#include "codegen/compiler.hpp"
#include "common/common.h"

namespace tilegraph::codegen {
    namespace fs = std::filesystem;

    auto Compiler::repo_path() -> fs::path const & {
        static std::once_flag pathFlag;
        static fs::path path;
        std::call_once(pathFlag, [] {
            auto codegenDir = getenv("CODEGEN_DIR");
            path = fs::path(codegenDir ? codegenDir : "build") / "code_repo" /
                   std::to_string(getpid());
            ASSERT(fs::create_directories(path),
                   "Failed to create directory \"{}\" for code generation",
                   path.c_str());
        });
        return path;
    }

    void *Compiler::compile(const char *dir_, const char *code,
                            const char *symbol) {
        auto [it, ok] = _dirs.try_emplace(dir_, nullptr);
        if (!ok) {
            return it->second;
        }
        auto dir = repo_path() / hardware() / dir_;
        auto src = dir / fmt::format("lib.{}", extension());
        fs::create_directories(dir);
        std::ofstream(src) << code;
        return it->second = _compile(src, symbol);
    }
    void *Compiler::fetch(const char *dir_) { return _dirs.at(dir_); }

}  // namespace tilegraph::codegen