//
// Created by Mike on 11/29/2019.
//

#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <optix_prime/optix_primepp.h>
#include <optix_math.h>
#include <cuda_gl_interop.h>

#include "check_cuda.h"
#include "ShadowRay.h"

class ShadowTracer {

private:
    size_t _width;
    size_t _height;
    optix::prime::Context _context;
    optix::prime::Model _model;
    optix::prime::Query _query;
    float *_hit_buffer{nullptr};
    ShadowRay *_ray_buffer{nullptr};
    cudaGraphicsResource_t _shadow_resource{nullptr};
    uint32_t _shadow_texture{0u};
    optix::prime::BufferDesc _geometry_buffer_desc;
    
public:
    ShadowTracer(size_t width, size_t height);
    void bind_geometry(optix::float4 *position_buffer, size_t vertex_count);
    void update();
    void execute(const optix::float4 *position_buffer, glm::vec3 light_position);
    [[nodiscard]] uint32_t shadow_texture() const noexcept { return _shadow_texture; }
};
