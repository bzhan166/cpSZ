cmake_minimum_required (VERSION 3.10)
project(sz)
set (sz_VERSION_MAJOR 1)
set (sz_VERSION_MINOR 0)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-g -O2")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/build/bin)

find_package(zstd REQUIRED)
find_package(FTK REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ftk/install/lib/cmake/FTK")

add_subdirectory (src)
add_subdirectory (test)

