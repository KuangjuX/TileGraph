cmake_minimum_required(VERSION 3.17)
project(InfiniTensor C CXX)

option(USE_CUDA "Support CUDA GPU" OFF)
option(USE_BANG "Support BANG MLU" OFF)

option(BUILD_TEST "Build test code" ON)
option(BUILD_ASAN "Build code whith ASAN" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

################################################################################
# ASAN Check
################################################################################
if(BUILD_ASAN)
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
endif()

include_directories(include)
include_directories(3rd-party/fmt/include)
include_directories(3rd-party/result)
include_directories(3rd-party/fmtlog)
add_definitions(-D FMTLOG_HEADER_ONLY)
add_definitions(-D FMT_HEADER_ONLY)

if(BUILD_TEST)
    include_directories(3rd-party/googletest/googletest/include)
    add_subdirectory(3rd-party/googletest)
endif()

add_compile_options(-Wall)

enable_testing()


file(GLOB_RECURSE SOURCE 
  src/core/*.cpp 
  src/core/graph/*.cpp 
  src/core/operators/*.cpp 
  src/optimizer/fusion/*.cpp
  srs/optimizer/fusion/subgraph_fusion/*.cpp
  src/optimizer/tilling/*.cpp
  src/codegen/*.cpp
)

add_library(tilegraph SHARED ${SOURCE})

add_compile_options(-Wall)
# add_compile_options(-Werror)

target_link_directories(tilegraph PRIVATE 3rd-party/fmt/build)
target_link_libraries(tilegraph PRIVATE fmt)

function(build_test files)
  # Non-recursive glob for skip failed tests
  file(GLOB TEST_SOURCES ${files})
  foreach(testsourcefile ${TEST_SOURCES})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} tilegraph GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach(testsourcefile ${TEST_SOURCES})
endfunction()

if(BUILD_TEST)
    build_test(tests/operators/*.cpp)
    build_test(tests/graph/*.cpp)
    build_test(tests/fusion/*.cpp)
    build_test(tests/codegen/*.cpp)
endif()
