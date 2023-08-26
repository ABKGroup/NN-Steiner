cmake_minimum_required(VERSION 3.0.0)
project(rmst VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(../../../extern/pybind11 build/)
set (RMST_SRC
    include/dist.c
    include/err.c
    include/heap.c
    include/mst2.c
    include/neighbors.c
    include/sd2.c
    include/unixtimer.c
    rmst.cpp
)
pybind11_add_module(rmst pybind11.cpp ${RMST_SRC})
add_executable(dbg main.cpp ${RMST_SRC})

target_include_directories( dbg
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}
)

target_include_directories( rmst
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}
)
