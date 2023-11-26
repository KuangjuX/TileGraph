# Design

## 设计思想

TileGraph 是一个实验性的 DNN 静态代码生成框架，主要关注于算子之间的融合与高效率的代码生成。对于计算密集型算子主要依靠子图匹配技术对于一些特定模式的算子进行融合，例如 Bolt 中提到的 Persistent Kernel Fusion 以及 Attention Fusion 等。随后融合后变为优化过的图，这里融合后的节点对于一些模式固定的优化可以使用新的算子类型代替子图。

进行一次算子融合 Pass 后变为优化过的子图，随后对于图信息进行下降，下降后主要关注关注算子间的内存分配信息与内存层级。首先进行 tensor 的 tiling，随后对于整个图进行一次 Welder Pass 做启发式的内存算子融合，对于不同层级的内存分别用不同层级的子图结构进行标识。随后使用 Perf profiler 对于一些参数进行选择，最终按照 Kernel Graph 进行代码生成，对于每个子图都要进行递归地代码生成，退出子图后需要插入同步原语，例如 `__syncthreads()`。最终生成 CUDA 代码。