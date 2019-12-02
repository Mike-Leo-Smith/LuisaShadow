//
// Created by Mike on 11/29/2019.
//

#include <cstdio>
#include <cstdint>
#include "ShadowTracerHelper.h"

constexpr auto block_width = 128ul;

template<uint32_t base>
__device__ constexpr float halton(uint32_t offset) {
    auto f = 1.0f;
    auto inv_b = 1.0f / base;
    auto r = 0.0f;
    for (auto i = offset; i != 0u; i /= base) {
        f = f * inv_b;
        r = r + f * static_cast<float>(i % base);
    }
    return optix::clamp(r, 0.0f, 1.0f);
}

__device__ inline optix::float2 uniform_sample_disk(optix::float2 u) {
    auto r = sqrtf(u.x);
    auto theta = M_PIf * 2.0f * u.y;
    return {r * cosf(theta), r * sinf(theta)};
}

__global__ void do_generate_rays(
        ShadowRay *ray_buffer, const optix::float4 *position_buffer, uint32_t *seed_buffer, size_t count,
        optix::float3 light_position) {
    
    constexpr auto light_radius = 1.5f;
    
    auto index = block_width * blockIdx.x + threadIdx.x;
    if (index < count) {
        auto P = optix::make_float3(position_buffer[index]);
        auto seed = seed_buffer[index];
        auto offset = uniform_sample_disk(optix::make_float2(halton<2>(seed), halton<3>(seed))) * light_radius;
        seed_buffer[index] = seed + 1;
        auto L = light_position + optix::make_float3(offset.x, 0.0f, offset.y) - P;
        ray_buffer[index] = { P, 1e-4f, optix::normalize(L), optix::length(L)};
    }
}

__global__ void do_accumulate_shadow(float *hit_accum, const float *hit, int n, size_t count) {
    auto index = block_width * blockIdx.x + threadIdx.x;
    if (index < count) {
        auto shadow = static_cast<float>(hit[index] > 0.0f);
        hit_accum[index] = optix::lerp(hit_accum[index], shadow, 1.0f / n);
    }
}

void shadow_tracer_generate_rays(
        ShadowRay *ray_buffer, const optix::float4 *position_buffer, uint32_t *seeds, size_t count,
        optix::float3 light_position) {
    
    auto block_count = (count + block_width - 1) / block_width;
    
    do_generate_rays<<<block_count, block_width>>>(ray_buffer, position_buffer, seeds, count, light_position);
}

void shadow_tracer_accumulate_shadow(float *hit_accum, const float *hit, int n, size_t count) {
    
    auto block_count = (count + block_width - 1) / block_width;
    
    do_accumulate_shadow<<<block_count, block_width>>>(hit_accum, hit, n, count);
}
