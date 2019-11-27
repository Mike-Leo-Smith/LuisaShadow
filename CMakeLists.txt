cmake_minimum_required(VERSION 3.15)
project(LuisaShadow LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA 9.0 REQUIRED)
link_libraries(${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake)
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})
link_libraries(optix_prime)

file(GLOB SOURCE_FILES *.cpp *.h *.cu)
add_executable(LuisaShadow ${SOURCE_FILES})
