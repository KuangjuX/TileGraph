#pragma once
#include "core/graph/graph.h"

namespace tilegraph {

class Binary : public Node {
 public:
  // Constructor
  Binary(OperatorType type, std::vector<Data *> inputs_list = {},
         std::vector<Data *> outputs_list = {}, std::string name_value = "",
         int64_t outputs_num_value = 1);
  // Destructor
  ~Binary() = default;
};

class Unary : public Node {
 public:
  // Constructor
  Unary(OperatorType type, std::vector<Data *> inputs_list = {},
        std::vector<Data *> outputs_list = {}, std::string name_value = "",
        int64_t outputs_num_value = 1);
  // Destructor
  ~Unary() = default;
};

#define DEFINE_BINARY(OP_NAME)                                                 \
  class OP_NAME : public Binary {                                              \
   public:                                                                     \
    OP_NAME(std::vector<Data *> inputs_list = {},                              \
            std::vector<Data *> outputs_list = {},                             \
            std::string name_value = "", int64_t outputs_num_value = 1)        \
        : Binary(OperatorType::OP_NAME, inputs_list, outputs_list, name_value, \
                 outputs_num_value) {}                                         \
  };

#define DEFINE_UNARY(OP_NAME)                                                 \
  class OP_NAME : public Unary {                                              \
   public:                                                                    \
    OP_NAME(std::vector<Data *> inputs_list = {},                             \
            std::vector<Data *> outputs_list = {},                            \
            std::string name_value = "", int64_t outputs_num_value = 1)       \
        : Unary(OperatorType::OP_NAME, inputs_list, outputs_list, name_value, \
                outputs_num_value) {}                                         \
  };

// Binary OPs
DEFINE_BINARY(ADD)
DEFINE_BINARY(SUB)
DEFINE_BINARY(MUL)
DEFINE_BINARY(DIV)
DEFINE_BINARY(EQ)
DEFINE_BINARY(GE)
DEFINE_BINARY(GT)
DEFINE_BINARY(LE)
DEFINE_BINARY(LT)
DEFINE_BINARY(NE)
DEFINE_BINARY(AND)
DEFINE_BINARY(OR)
DEFINE_BINARY(XOR)
DEFINE_BINARY(FLOORMOD)
DEFINE_BINARY(FLOORDIV)
#undef DEFINE_BINARY

// Unary OPs
DEFINE_UNARY(SQRT)
DEFINE_UNARY(RSQRT)
DEFINE_UNARY(RELU)
DEFINE_UNARY(RECIP)
DEFINE_UNARY(SIGMOID)
DEFINE_UNARY(SIN)
DEFINE_UNARY(COS)
DEFINE_UNARY(TANH)
#undef DEFINE_UNARY

}  // namespace tilegraph