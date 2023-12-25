#pragma once
#include "kernels/var.hpp"
#include <variant>
#include <memory>

namespace tilegraph::kernel {
    template <class VarType>
    class Iteration {
       public:
        std::unique_ptr<VarType> iter_var;
        // The step of the iteration.
        std::variant<int, std::shared_ptr<VarType>> step;
        // The start and end of the iteration.
        std::variant<int, std::shared_ptr<VarType>> start;
        std::variant<int, std::shared_ptr<VarType>> end;

       public:
        Iteration(std::unique_ptr<VarType> iter_var,
                  std::variant<int, std::shared_ptr<VarType>> step,
                  std::variant<int, std::shared_ptr<VarType>> start,
                  std::variant<int, std::shared_ptr<VarType>> end);

        virtual std::string genIter(int indient);
        virtual std::string getIterVar();
    };
}  // namespace tilegraph::kernel