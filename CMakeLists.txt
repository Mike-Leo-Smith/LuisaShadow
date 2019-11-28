cmake_minimum_required(VERSION 3.15)
project(LuisaShadow LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS OFF CACHE BOOL "" FORCE)

find_package(CUDA 9.0 REQUIRED)
link_libraries(${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake)
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})
link_libraries(optix_prime)

include_directories(thirdparty)
include_directories(thirdparty/glfw/include)
include_directories(thirdparty/glm)
include_directories(thirdparty/stb/repo)
include_directories(thirdparty/tinyexr/repo)
include_directories(thirdparty/tinyobjloader/repo)
include_directories(thirdparty/glsl-optimizer/src)

set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_CXX_17 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_FAST_MATH ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_LANG_EXTENSIONS ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_SSE2 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_SSE3 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_SSSE3 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_SSE4_1 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_SSE4_2 ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_AVX ON CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE_SIMD_AVX2 ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glm)
link_libraries(glm)

set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glfw)
link_libraries(glfw)

add_subdirectory(thirdparty/tinyexr)
link_libraries(tinyexr)

add_subdirectory(thirdparty/tinyobjloader)
link_libraries(tinyobjloader)

add_subdirectory(thirdparty/stb)
link_libraries(stb)

add_subdirectory(thirdparty/glad)
link_libraries(glad)

set(DEBUG OFF CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 11)
add_subdirectory(thirdparty/glsl-optimizer)
link_libraries(glsl_optimizer)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

file(GLOB SOURCE_FILES src/*.cpp src/*.h src/*.cu)
add_executable(LuisaShadow ${SOURCE_FILES})
