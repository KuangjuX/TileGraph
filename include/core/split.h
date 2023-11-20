#pragma once
#include "core/type.h"
#include <vector>

namespace tilegraph {

class Split {
 public:
  // Self information
  std::vector<int64_t> split_dimension;

 public:
  // Constructor
  Split() = default;
  Split(const std::vector<int64_t>& dimension);
  // Destructor
  ~Split() = default;
  // Information
  void printInformation();
  void printSummary();
};

}  // namespace tilegraph