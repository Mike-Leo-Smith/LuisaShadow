//
// Created by Mike on 11/29/2019.
//

#include <random>

#include "ShadowTracer.h"
#include "ShadowTracerHelper.h"

ShadowTracer::ShadowTracer(size_t width, size_t height)
        : _width{width}, _height{height} {
    
    _context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    _model = _context->createModel();
    
    size_t size = width * height;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&_seed_buffer), size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&_hit_buffer), size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&_hit_accum_buffer), size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&_ray_buffer), size * sizeof(ShadowRay)));
    
    std::vector<uint32_t> random_seeds(size);
    std::default_random_engine random_engine{std::random_device{}()};
    for (auto &&seed : random_seeds) {
        seed = random_engine() << 8u;
    }
    CHECK_CUDA(cudaMemcpy(_seed_buffer, random_seeds.data(), size, cudaMemcpyHostToDevice));
    
    _query = _model->createQuery(RTP_QUERY_TYPE_ANY);
    _query->setRays(size, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR,
                    _ray_buffer);
    _query->setHits(size, RTP_BUFFER_FORMAT_HIT_T, RTP_BUFFER_TYPE_CUDA_LINEAR, _hit_buffer);
    
    glGenTextures(1, &_shadow_texture);
    glBindTexture(GL_TEXTURE_2D, _shadow_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, _width, _height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    CHECK_CUDA(cudaGraphicsGLRegisterImage(&_shadow_resource, _shadow_texture, GL_TEXTURE_2D,
                                           cudaGraphicsRegisterFlagsNone));
}

void ShadowTracer::execute(const optix::float4 *position_buffer, glm::vec3 light_position) {
    
    size_t size = _width * _height;
    for (auto i = 0; i < 128; i++) {  // TODO: support sampling area lights
        shadow_tracer_generate_rays(_ray_buffer, position_buffer, _seed_buffer, size,
                                    optix::make_float3(light_position.x, light_position.y, light_position.z));
        _query->execute(RTP_QUERY_HINT_ASYNC);
        shadow_tracer_accumulate_shadow(_hit_accum_buffer, _hit_buffer, i + 1, size);
    }
    
    cudaArray_t shadow_array;
    CHECK_CUDA(cudaGraphicsMapResources(1, &_shadow_resource));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&shadow_array, _shadow_resource, 0, 0));
    CHECK_CUDA(cudaMemcpyToArrayAsync(shadow_array, 0, 0, _hit_accum_buffer, size * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &_shadow_resource));
}

void ShadowTracer::update() {
    _model->setTriangles(_geometry_buffer_desc);
    _model->update(RTP_MODEL_HINT_ASYNC);
}

void ShadowTracer::bind_geometry(optix::float4 *position_buffer, size_t vertex_count) {
    _geometry_buffer_desc = _context->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT4, RTP_BUFFER_TYPE_CUDA_LINEAR, position_buffer);
    _geometry_buffer_desc->setRange(0, vertex_count);
}
