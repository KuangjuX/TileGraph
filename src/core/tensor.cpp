#include "core/tensor.hpp"

namespace tilegraph {

    int64_t Tensor::tensor_count = 0;

    Tensor::Tensor(const std::vector<int64_t> &dimension,
                   std::string name_value, TensorDatatype dtype,
                   TensorType type)
        : name(name_value),
          index(tensor_count++),
          tensor_datatype(dtype),
          tensor_type(type),
          tensor_dimension(dimension) {}

}  // namespace tilegraph