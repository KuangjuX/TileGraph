set(SRC
    elementwise.cpp
    gemm.cpp
)


add_library(tilegraph_operatorz STATIC ${SRC})
target_include_directories(tilegraph_operators SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(tilegraph_operators PRIVATE nnfusion_serialize)