//
// Created by Mike on 11/29/2019.
//

#include "check_cuda.h"
#include "Framebuffer.h"

Framebuffer::Framebuffer(uint32_t width, uint32_t height)
        : _width{width}, _height{height} {
    
    auto fbo = 0u;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    
    uint32_t textures[2];
    glGenTextures(2, textures);
    for (auto texture : textures) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textures[1], 0);
    
    GLenum targets[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, targets);
    
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _width, _height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "G-Buffer Framebuffer not complete!" << std::endl;
    }
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    _fbo = fbo;
    _beauty_texture = textures[0];
    _position_texture = textures[1];
    
    CHECK_CUDA(cudaGraphicsGLRegisterImage(&_position_resource, _position_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&_position_buffer), _width * _height * sizeof(float4)));
}
