#pragma once
#include "core/type.hpp"

namespace tilegraph::kernel {
    class Var {
       public:
        MemoryType memory_level;
        DataType data_type;
        uint32_t len;
        std::string name;

        Var(MemoryType memory_level, DataType data_type, uint32_t len,
            std::string name);
        ~Var() = default;

        virtual std::string declareVar(int indient) = 0;
        virtual std::string initVar(int indient) = 0;
        virtual std::string getVarIndex(uint32_t index) = 0;
        virtual std::string getVarIndexByVar(std::string index) = 0;
    };
}  // namespace tilegraph::kernel