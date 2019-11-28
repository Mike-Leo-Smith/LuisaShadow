//
// Created by Mike on 11/28/2019.
//

#pragma once

#include <optix_math.h>

struct ShadowRay {
    optix::float3 origin;
    float t_min;
    optix::float3 direction;
    float t_max;
};
