//
// Created by Mike on 11/29/2019.
//

#pragma once

#include "ShadowRay.h"

void shadow_tracer_generate_rays(
        ShadowRay *ray_buffer, const optix::float4 *position_buffer, uint32_t *seeds,
        size_t count,
        optix::float3 light_position);

void shadow_tracer_accumulate_shadow(float *hit_accum, const float *hit, int n, size_t count);
