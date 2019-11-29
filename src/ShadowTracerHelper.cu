//
// Created by Mike on 11/29/2019.
//

#include <cstdio>
#include "ShadowTracerHelper.h"

__global__ void do_generate_rays(
        ShadowRay *ray_buffer, const optix::float4 *position_buffer, size_t count,
        optix::float3 light_position) {
    
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < count) {
        auto P = optix::make_float3(position_buffer[index]);
        auto L = light_position - P;
        ray_buffer[index] = { P, 1e-4f, optix::normalize(L), optix::length(L)};
    }

}

void shadow_tracer_generate_rays(
        ShadowRay *ray_buffer, const optix::float4 *position_buffer, size_t count,
        optix::float3 light_position) {
    
    constexpr auto block_width = 128ul;
    auto block_count = (count + block_width - 1) / block_width * block_width;
    
    do_generate_rays<<<block_count, block_width>>>(ray_buffer, position_buffer, count, light_position);

}