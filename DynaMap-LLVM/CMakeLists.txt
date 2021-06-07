cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_library(MemAccInstPass MODULE
    # List source files
    mem_acc_inst.cpp
)

add_library(GvnOptPass MODULE
    # List source files
    gvn_opt.cpp
)

set_target_properties( MemAccInstPass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)

set_target_properties( GvnOptPass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)
