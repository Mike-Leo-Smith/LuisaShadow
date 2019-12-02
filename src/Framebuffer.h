//
// Created by Mike on 11/29/2019.
//

#pragma once

#include <cstdint>
#include <vector>
#include <thread>
#include <iostream>
#include <glad/glad.h>
#include <optix_math.h>
#include <cuda_gl_interop.h>

class Framebuffer {

private:
    uint32_t _width{0};
    uint32_t _height{0};
    uint32_t _fbo{0};
    uint32_t _beauty_texture{0};
    uint32_t _albedo_texture{0};
    uint32_t _position_texture{0};
    uint32_t _normal_texture{0};
    cudaGraphicsResource_t _position_resource{nullptr};
    optix::float4 *_position_buffer{nullptr};

public:
    Framebuffer(uint32_t width, uint32_t height);
    
    template<typename F>
    void with(F &&render) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
        render();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    template<typename F>
    void with_position_pointer(F &&func) {
        cudaArray_t array;
        CHECK_CUDA(cudaGraphicsMapResources(1, &_position_resource));
        CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&array, _position_resource, 0, 0));
        CHECK_CUDA(cudaMemcpyFromArrayAsync(_position_buffer, array, 0, 0, _width * _height * sizeof(optix::float4), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &_position_resource));
        func(const_cast<optix::float4 *>(_position_buffer));
    }
    
    [[nodiscard]] uint32_t beauty_texture() const noexcept { return _beauty_texture; }
    [[nodiscard]] uint32_t position_texture() const noexcept { return _position_texture; }
    
};
