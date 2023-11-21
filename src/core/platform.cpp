#include "core/platform.h"

namespace tilegraph {

#define CASE(TYPE, STR) \
  case Platform::TYPE:  \
    return STR

const char *Platform::toString() const {
  switch (type) {
    CASE(CUDA, "CUDA");
    CASE(BANG, "BANG");
    default:
      return "Unknown";
  }
}

bool Platform::isCUDA() const { return type == Platform::CUDA; }

bool Platform::isBANG() const { return type == Platform::BANG; }

}  // namespace tilegraph