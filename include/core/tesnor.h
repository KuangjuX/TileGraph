#pragma once
#include "core/type.h"
#include "core/split.h"
#include "core/tile.h"
#include <vector>
#include <map>

namespace tilegraph {

class Tensor {
 public:
  // Self information
  TensorDatatype tensor_datatype;         // element type
  TensorType tensor_type;                 // const or variable
  TensorLayout tensor_layout;             // array or bchw or bhwc
  std::vector<int64_t> tensor_dimension;  // tensor dim
  std::vector<int64_t> tensor_stride;     // tensor stride
  std::string tensor_name;
  int64_t data_offset;
  bool is_contiguous;

 public:
  // Constructor
  Tensor() = delete;
  Tensor(const std::vector<int64_t>& dimension, TensorDatatype dtype,
         TensorType type, TensorLayout layout, std::string name,
         int64_t offset = 0);
  Tensor(const std::vector<int64_t>& dimension,
         const std::vector<int64_t>& stride, TensorDatatype dtype,
         TensorType type, TensorLayout layout, std::string name,
         int64_t offset = 0);
  // Destructor
  ~Tensor() = default;
  // Tiling with split
  TileTensor tiling(const Split& split);
  //   // Tiling with tile size
  //   TileTensor tiling(const Tile& Tile);
  // Information
  void printInformation();
  void printSummary();
  // Get Function;
  bool isContiguous();
  // Easy Funciton;
  void flatten(int64_t start = 0, int64_t end = -1);
};

}  // namespace tilegraph