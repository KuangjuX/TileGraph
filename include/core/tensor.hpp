#pragma once
#include <vector>
#include <string>

#include "core/type.hpp"

namespace tilegraph {
    class Tensor {
       private:
        static int64_t tensor_count;

       public:
        std::string name;
        const int64_t index;
        TensorDatatype tensor_datatype;
        TensorType tensor_type;
        std::vector<int64_t> tensor_dimension;

       public:
        Tensor() = delete;
        Tensor(const std::vector<int64_t> &dimension,
               std::string name_value = "",
               TensorDatatype dtype = TensorDatatype::FLOAT,
               TensorType type = TensorType::VARIABLE);

        ~Tensor() = default;
    };
}  // namespace tilegraph