//
// Created by Mike on 11/28/2019.
//

#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

class StaticGeometry {

private:
    std::vector<glm::vec4> _positions;
    std::vector<glm::vec4> _normals;
    std::vector<glm::vec4> _colors;

public:
    [[nodiscard]] static StaticGeometry load(const std::vector<std::string> &paths, const std::vector<glm::mat4> &transforms, const std::vector<glm::vec3> &colors);
    void bind_vbo(uint32_t position_vbo, uint32_t normal_vbo, uint32_t color_vbo, size_t offset);
    [[nodiscard]] size_t vertex_count() const noexcept { return _positions.size(); }
};
