#include <gtest/gtest.h>

#include "codegen/cuda_compiler.hpp"

using namespace ::tilegraph::codegen;

constexpr static const char *code = R"~(
#include <cstdio>
__global__ void kernel(float* a) {
    a[threadIdx.x] += 1.0;
}
extern "C" {
void launchKernel(float* a) {
    float* dev_a;
    cudaMalloc(&dev_a, 100 * sizeof(float));
    cudaMemcpy(dev_a, a, 100 * sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<1, 100>>>(dev_a);
    cudaDeviceSynchronize();
    cudaMemcpy(a, dev_a, 100 * sizeof(float), cudaMemcpyDeviceToHost);
}
}
)~";

TEST(Codegen, simple_cuda) {
    CudaCompiler nvcc;
    auto function = nvcc.compile("add", code, "launchKernel");

    float *a = new float[100];
    memset(a, 0, 100 * sizeof(float));
    reinterpret_cast<void (*)(float *)>(function)(a);
    EXPECT_EQ(a[0], 1.0);
}