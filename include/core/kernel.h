#pragma once
#include "core/type.h"
#include <vector>

namespace tilegraph {

class Kernel {
 public:
  // Self information
  KernelType kernel_type;

 public:
  // Constructor
  Kernel() = delete;
  Kernel(KernelType type);
  // Destructor
  ~Kernel() = default;
  // Generator
  virtual std::string generatorCodeOnCUDA(std::vector<std::string> args) = 0;
  virtual std::string generatorCodeOnBANG(std::vector<std::string> args) = 0;
  // Information
  void printInformation();
  void printSummary();
};

}  // namespace tilegraph