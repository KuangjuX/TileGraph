set(SRC
    platform.cpp 
    graph/graph.cpp
    graph/subgraph.cpp
)


add_library(tilegraph_core STATIC ${SRC})
target_include_directories(tilegraph_core SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(tilegraph_core PRIVATE nnfusion_serialize)