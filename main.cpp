#include <iostream>

#include <optix_prime/optix_primepp.h>
#include <cuda_runtime_api.h>

#include "test.h"

int main() {
    
    float vertices[] {
        0.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    };
    
    float *vertex_buffer = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&vertex_buffer), sizeof(vertices));
    cudaMemcpy(vertex_buffer, vertices, sizeof(vertices), cudaMemcpyHostToDevice);
    
    auto context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    auto model = context->createModel();
    model->setTriangles(1, RTP_BUFFER_TYPE_CUDA_LINEAR, vertex_buffer, sizeof(float4));
    model->update(RTP_MODEL_HINT_NONE);
    
    float rays[] {
        0.0f, 0.0f, -1.0f,  // origin
        1e-3f,              // tmin
        0.0f, 0.0f, 1.0f,   // direction
        1e3f,               // tmax
    };
    
    float *ray_buffer = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&ray_buffer), sizeof(rays));
    cudaMemcpy(ray_buffer, rays, sizeof(rays), cudaMemcpyHostToDevice);
    
    uint8_t *hit_buffer = nullptr;
    cudaMallocManaged(reinterpret_cast<void **>(&hit_buffer), sizeof(uint8_t));
    cudaMemset(hit_buffer, 0, sizeof(uint8_t));
    
    auto query = model->createQuery(RTP_QUERY_TYPE_ANY);
    query->setRays(1, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, ray_buffer);
    query->setHits(1, RTP_BUFFER_FORMAT_HIT_BITMASK, RTP_BUFFER_TYPE_CUDA_LINEAR, hit_buffer);
    
    query->execute(RTP_QUERY_HINT_NONE);
    
    std::cout << static_cast<uint32_t>(*hit_buffer) << std::endl;
    
    test();
    return 0;
}
