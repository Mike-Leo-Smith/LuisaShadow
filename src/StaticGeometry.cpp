//
// Created by Mike on 11/28/2019.
//

#include <glad/glad.h>
#include <tiny_obj_loader.h>
#include "StaticGeometry.h"

StaticGeometry StaticGeometry::load(
        const std::vector<std::string> &paths,
        const std::vector<glm::mat4> &transforms,
        const std::vector<glm::vec3> &colors) {
    
    std::vector<glm::vec4> loaded_positions;
    std::vector<glm::vec4> loaded_normals;
    std::vector<glm::vec4> loaded_colors;
    
    for (auto i = 0ull; i < paths.size(); i++) {
        auto &&path = paths[i];
        auto transform = transforms[i];
        auto normal_matrix = glm::transpose(glm::inverse(glm::mat3{transform}));
        tinyobj::ObjReader reader;
        reader.ParseFromFile(path);
        auto positions = reinterpret_cast<const glm::vec3 *>(reader.GetAttrib().vertices.data());
        auto normals = reinterpret_cast<const glm::vec3 *>(reader.GetAttrib().normals.data());
        for (auto &&shape : reader.GetShapes()) {
            for (auto index : shape.mesh.indices) {
                auto p = transform * glm::vec4{positions[index.vertex_index], 1.0f};
                auto n = normal_matrix * normals[index.normal_index];
                loaded_positions.emplace_back(glm::vec3{p} / p.w, 1.0f);
                loaded_normals.emplace_back(n, 1.0f);
            }
        }
        loaded_colors.resize(loaded_positions.size(), glm::vec4{colors[i], 1.0f});
    }
    
    StaticGeometry geometry;
    geometry._positions = std::move(loaded_positions);
    geometry._normals = std::move(loaded_normals);
    geometry._colors = std::move(loaded_colors);
    
    return geometry;
}

void StaticGeometry::bind_vbo(uint32_t position_vbo, uint32_t normal_vbo, uint32_t color_vbo, size_t offset) {
    auto offset_bytes = offset * sizeof(glm::vec4);
    auto buffer_size = _positions.size() * sizeof(glm::vec4);
    glNamedBufferSubData(position_vbo, offset_bytes, buffer_size, _positions.data());
    glNamedBufferSubData(normal_vbo, offset_bytes, buffer_size, _normals.data());
    glNamedBufferSubData(color_vbo, offset_bytes, buffer_size, _colors.data());
}
