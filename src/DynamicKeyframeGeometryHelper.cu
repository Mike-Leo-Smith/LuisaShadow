//
// Created by Mike on 11/28/2019.
//

#include <cuda_runtime_api.h>
#include <optix_math.h>
#include <cstdio>
#include "DynamicKeyframeGeometryHelper.h"

__device__ float3 normal_slerp(float3 n1, float3 n2, float t) {
    if (dot(n1, n2) > 0.995f) { return normalize(lerp(n1, n2, t)); }
    auto theta = acosf(dot(n1, n2));
    auto sin_t_theta = sinf(t * theta);
    return normalize(lerp(n1, n2, sin_t_theta / (sin_t_theta + sinf((1 - t) * theta))));
}

__global__ void do_update_positions(optix::float4 *buffer, size_t size, const optix::float4 *prev, const optix::float4 *next, float t) {
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        auto p0 = prev[index];
        auto p1 = next[index];
        buffer[index] = optix::make_float4(optix::lerp(optix::make_float3(p0), optix::make_float3(p1), t), 1.0f);
    }
}

__global__ void do_update_normals(optix::float4 *buffer, size_t size, const optix::float4 *prev, const optix::float4 *next, float t) {
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        auto n0 = prev[index];
        auto n1 = next[index];
        buffer[index] = optix::make_float4(normal_slerp(optix::make_float3(n0), optix::make_float3(n1), t), 1.0f);
    }
}

void dynamic_keyframe_geometry_update_positions(
        optix::float4 *position_vbo, size_t size,
        optix::float4 *prev_positions, optix::float4 *next_positions, float t) {
    
    constexpr auto block_width = 128ul;
    auto block_count = (size + block_width - 1) / block_width * block_width;
    
    do_update_positions<<<block_count, block_width>>>(position_vbo, size, prev_positions, next_positions, t);
    
    
}

void dynamic_keyframe_geometry_update_normals(
        optix::float4 *normal_vbo, size_t size, optix::float4 *prev_normals,
        optix::float4 *next_normals, float t) {
    
    constexpr auto block_width = 128ul;
    auto block_count = (size + block_width - 1) / block_width * block_width;
    
    do_update_normals<<<block_count, block_width>>>(normal_vbo, size, prev_normals, next_normals, t);
    
}
