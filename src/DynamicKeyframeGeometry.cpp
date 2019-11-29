//
// Created by Mike on 11/28/2019.
//

#include <iostream>

#include "DynamicKeyframeGeometry.h"
#include "DynamicKeyframeGeometryHelper.h"

DynamicKeyframeGeometry DynamicKeyframeGeometry::load(const std::vector<std::string> &paths, glm::mat4 transform, glm::vec3 color) {
    
    auto normal_matrix = glm::transpose(glm::inverse(glm::mat3{transform}));
    
    auto dynamic_vertex_count = 0ul;
    std::vector<optix::float4 *> position_buffers;
    std::vector<optix::float4 *> normal_buffers;
    for (auto &&path : paths) {
        
        std::cout << "Loading " << path << std::endl;
        
        tinyobj::ObjReader reader;
        reader.ParseFromFile(path);
        
        std::vector<glm::vec4> flattened_positions;
        std::vector<glm::vec4> flattened_normals;
        
        auto positions = reinterpret_cast<const glm::vec3 *>(reader.GetAttrib().vertices.data());
        auto normals = reinterpret_cast<const glm::vec3 *>(reader.GetAttrib().normals.data());
        for (auto &&shape : reader.GetShapes()) {
            for (auto index : shape.mesh.indices) {
                auto p = transform * glm::vec4{positions[index.vertex_index], 1.0f};
                auto n = normal_matrix * normals[index.normal_index];
                flattened_positions.emplace_back(glm::vec3{p} / p.w, 1.0f);
                flattened_normals.emplace_back(n, 1.0f);
            }
        }
        
        dynamic_vertex_count = flattened_positions.size();
        std::cout << "vertex_count = " << dynamic_vertex_count << std::endl;
        
        optix::float4 *position_buffer = nullptr;
        optix::float4 *normal_buffer = nullptr;
        auto position_buffer_size = sizeof(glm::vec4) * dynamic_vertex_count;
        auto normal_buffer_size = sizeof(glm::vec4) * dynamic_vertex_count;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&position_buffer), position_buffer_size));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&normal_buffer), normal_buffer_size));
        CHECK_CUDA(cudaMemcpy(position_buffer, flattened_positions.data(), position_buffer_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(normal_buffer, flattened_normals.data(), normal_buffer_size, cudaMemcpyHostToDevice));
        
        position_buffers.emplace_back(position_buffer);
        normal_buffers.emplace_back(normal_buffer);
    }
    
    DynamicKeyframeGeometry geometry;
    geometry._vertex_count = dynamic_vertex_count;
    geometry._position_buffers = std::move(position_buffers);
    geometry._normal_buffers = std::move(normal_buffers);
    geometry._colors.resize(dynamic_vertex_count, glm::vec4{color, 1.0f});
    
    return geometry;
}

void DynamicKeyframeGeometry::bind_vbo(uint32_t position_vbo, uint32_t normal_vbo, uint32_t color_vbo, size_t offset) {
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&_position_resource, position_vbo, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&_normal_resource, normal_vbo, cudaGraphicsRegisterFlagsNone));
    _resource_offset = offset;
    glNamedBufferSubData(color_vbo, offset * sizeof(glm::vec4), _vertex_count * sizeof(glm::vec4), _colors.data());
}

void DynamicKeyframeGeometry::update(float time) {
    
    auto progress = time / _animation_interval;
    auto index = std::floor(progress);
    auto prev_index = static_cast<size_t>(index) % _position_buffers.size();
    auto next_index = (prev_index + 1ul) % _position_buffers.size();
    auto t = progress - index;
    
    with_position_pointer([&](optix::float4 *p) {
        dynamic_keyframe_geometry_update_positions(p, _vertex_count, _position_buffers[prev_index], _position_buffers[next_index], t);
    });
    
    with_normal_pointer([&](optix::float4 *p) {
        dynamic_keyframe_geometry_update_normals(p, _vertex_count, _normal_buffers[prev_index], _normal_buffers[next_index], t);
    });
}
