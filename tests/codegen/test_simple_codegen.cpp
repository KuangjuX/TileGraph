#include <gtest/gtest.h>

#include "codegen/cuda_compiler.hpp"

using namespace ::tilegraph::codegen;

constexpr static const char *code = R"~(
#include <cstdio>
__global__ void kernel() {
    printf("Hello World from GPU!\n");
}
extern "C" {
void launchKernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
}
)~";

TEST(generator, CudaCodeRepo) {
    CudaCompiler nvcc;
    auto function = nvcc.compile("helloWorld", code, "launchKernel");
    reinterpret_cast<void (*)()>(function)();
}