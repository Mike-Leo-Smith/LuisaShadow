//
// Created by Mike on 11/29/2019.
//

#pragma once

#include "ShadowRay.h"

void shadow_tracer_generate_rays(ShadowRay *ray_buffer, const optix::float4 *position_buffer, size_t count,
                                 optix::float3 light_position);
