//
// Created by Mike on 11/28/2019.
//

#pragma once

#include <optix_math.h>

void dynamic_keyframe_geometry_update_positions(
        optix::float4 *position_vbo, size_t size,
        optix::float4 *prev_positions, optix::float4 *next_positions, float t);

void dynamic_keyframe_geometry_update_normals(
        optix::float4 *normal_vbo, size_t size,
        optix::float4 *prev_normals, optix::float4 *next_normals, float t);