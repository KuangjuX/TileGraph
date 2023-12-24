#pragma once
#include <tuple>
#include <string>

namespace tilegraph {
    // Cacheline
    using Cacheline = std::tuple<std::string, std::string, int>;
    // MemoryDispatch
    enum class MemoryDispatch { RANDOM, FIFO, LRU, LFU };
    enum class TensorDatatype { HALF, FLOAT, DOUBLE, INT32 };
    enum class TensorLayout { NCHW, NHWC, ARRAY };
    enum class TensorType { CONST, VARIABLE };
    // OperatorType
    enum class OperatorType {
        // Binary
        ADD,
        SUB,
        MUL,
        DIV,
        EQ,
        GE,
        GT,
        LE,
        LT,
        NE,
        AND,
        OR,
        XOR,
        FLOORMOD,
        FLOORDIV,

        // Unary
        SIGMOID,
        RELU,
        SQRT,
        RSQRT,
        RECIP,
        SIN,
        COS,
        TANH,
        GELU,
        // Memory
        LOAD,
        ALLOCATE,
        STORE,
        FREE,
        // Sync
        SYNC,
        // GEMM
        GEMM,
        SOFTMAX,

        GEMM_RELU,

        FUSED
    };

    // enum class MemoryType { Register, Warp, Shared, Global };
    enum class MemoryType { Register, Warp, Shared, Global };

    enum class FuncType { Device, Global, Host };

    enum class DataType { Void, Half, Float, Int32 };

    // KernelType
    enum class KernelType {
        BINARY,
        UNARY,
        REDUCE,
        BROADCAST,
        MEMORY,
        FMA,
        SYNC
    };

    // CacheType
    enum class CacheType { CACHE, LDRAM };

    // CacheHitLocation
    enum class CacheHitLocation { CACHE, LDRAM, NOT_FOUND, ERROR };

    std::string toString(OperatorType op_type);

}  // namespace tilegraph