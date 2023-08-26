cmake_minimum_required(VERSION 3.0.0)
project(geosteiner VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(../../../extern/pybind11 build/)
pybind11_add_module(geosteiner geosteiner.cpp)


find_library(LIBGEOSTEINER libgeosteiner.a ./geosteiner-5.3/)
find_library(LIBLPS libLPS.a ./geosteiner-5.3/lp_solve_2.3/)

message("${LIBGEOSTEINER}")
message("${LIBLPS}")

target_include_directories( ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries( ${PROJECT_NAME}
    PUBLIC
    ${LIBGEOSTEINER}
    ${LIBLPS}
    gmp
    )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)