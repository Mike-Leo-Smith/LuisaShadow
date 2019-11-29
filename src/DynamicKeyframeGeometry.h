//
// Created by Mike on 11/28/2019.
//

#pragma once

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <optix_math.h>
#include <tiny_obj_loader.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "DynamicKeyframeGeometryHelper.h"
#include "check_cuda.h"

class DynamicKeyframeGeometry {

private:
    size_t _vertex_count{0ul};
    std::vector<optix::float4 *> _position_buffers;
    std::vector<optix::float4 *> _normal_buffers;
    cudaGraphicsResource_t _position_resource{nullptr};
    cudaGraphicsResource_t _normal_resource{nullptr};
    std::vector<glm::vec4> _colors;
    size_t _resource_offset{0ul};
    float _animation_interval{0.1f};

public:
    [[nodiscard]] static DynamicKeyframeGeometry load(const std::vector<std::string> &paths, glm::mat4 transform, glm::vec3 color);
    [[nodiscard]] size_t vertex_count() const noexcept { return _vertex_count; }
    void bind_vbo(uint32_t position_vbo, uint32_t normal_vbo, uint32_t color_vbo, size_t offset);
    
    void update(float time);
    
    template<typename F>
    void with_position_pointer(F &&func) {
        auto buffer_size = _vertex_count * sizeof(optix::float4);
        optix::float4 *position_ptr = nullptr;
        CHECK_CUDA(cudaGraphicsMapResources(1, &_position_resource));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&position_ptr), &buffer_size, _position_resource));
        func(position_ptr + _resource_offset);
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &_position_resource));
    }
    
    template<typename F>
    void with_normal_pointer(F &&func) {
        auto buffer_size = _vertex_count * sizeof(optix::float4);
        optix::float4 *normal_ptr = nullptr;
        CHECK_CUDA(cudaGraphicsMapResources(1, &_normal_resource));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&normal_ptr), &buffer_size, _normal_resource));
        func(normal_ptr + _resource_offset);
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &_normal_resource));
    }
};
