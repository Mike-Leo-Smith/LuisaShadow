#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <tiny_obj_loader.h>

#include <optix_prime/optix_primepp.h>

#include "DynamicKeyframeGeometry.h"
#include "StaticGeometry.h"
#include "Shader.h"
#include "ShadowRay.h"

int main() {
    
    // initialize GLFW
    if (glfwInit() == 0) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        exit(-1);
    }
    
    constexpr auto WINDOW_WIDTH = 800;
    constexpr auto WINDOW_HEIGHT = 600;
    
    // create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, 0);
    glfwWindowHint(GLFW_SRGB_CAPABLE, 1);
    auto window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "LuisaShadow", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    
    // load OpenGL
    if (gladLoadGL() == 0) {
        std::cerr << "Failed to initialize OpenGL." << std::endl;
        exit(-1);
    }
    
    // load shaders
    Shader shader{"resources/shaders/diffuse.vert", "resources/shaders/diffuse.frag"};
    
    // create geometry
    std::vector<std::string> elephant_file_names;
    for (auto i = 1; i <= 48; i++) {
        elephant_file_names.emplace_back(serialize("resources/elephant/elephant-gallop-", (i < 10 ? "0" : ""), i, ".obj"));
    }
    
    auto elephant_translation = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -9.9f, 0.0f});
    auto elephant_rotation = glm::rotate(glm::mat4{1.0f}, glm::radians(60.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    auto elephant_scaling = glm::scale(glm::mat4{1.0f}, glm::vec3{10.0f, 10.0f, 10.0f});
    auto elephant_transform = elephant_translation * elephant_rotation * elephant_scaling;
    
    glm::vec3 elephant_color{1.0f};
    auto elephant_geometry = DynamicKeyframeGeometry::load(elephant_file_names, elephant_transform, elephant_color);
    
    std::vector<std::string> wall_paths{
        "resources/cube/cube.obj",
        "resources/cube/cube.obj",
        "resources/cube/cube.obj",
        "resources/cube/cube.obj",
        "resources/cube/cube.obj",
    };
    
    std::vector<glm::mat4> wall_transforms{
        glm::translate(glm::mat4{1.0f}, glm::vec3{-20.0f, 0.0f, 0.0f}) * glm::scale(glm::mat4{1.0f}, glm::vec3{20.1f}),  // left
        glm::translate(glm::mat4{1.0f}, glm::vec3{20.0f, 0.0f, 0.0f}) * glm::scale(glm::mat4{1.0f}, glm::vec3{20.1f}),   // right
        glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 20.0f, 0.0f}) * glm::scale(glm::mat4{1.0f}, glm::vec3{20.1f}),   // top
        glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -20.0f, 0.0f}) * glm::scale(glm::mat4{1.0f}, glm::vec3{20.1f}),  // bottom
        glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, -20.0f}) * glm::scale(glm::mat4{1.0f}, glm::vec3{20.1f}),  // back
    };
    
    std::vector<glm::vec3> wall_colors{
        glm::vec3{1.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 1.0f, 0.0f},
        glm::vec3{1.0f, 1.0f, 1.0f},
        glm::vec3{1.0f, 1.0f, 1.0f},
        glm::vec3{1.0f, 1.0f, 1.0f},
    };
    
    auto wall_geometry = StaticGeometry::load(wall_paths, wall_transforms, wall_colors);
    
    auto total_vertex_count = elephant_geometry.vertex_count() + wall_geometry.vertex_count();
    
    auto vao = 0u;
    auto position_vbo = 0u;
    auto normal_vbo = 0u;
    auto color_vbo = 0u;
    
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferData(GL_ARRAY_BUFFER, total_vertex_count * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    
    glGenBuffers(1, &normal_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo);
    glBufferData(GL_ARRAY_BUFFER, total_vertex_count * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    
    glGenBuffers(1, &color_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
    glBufferData(GL_ARRAY_BUFFER, total_vertex_count * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), nullptr);
    
    elephant_geometry.bind_vbo(position_vbo, normal_vbo, color_vbo, 0ul);
    wall_geometry.bind_vbo(position_vbo, normal_vbo, color_vbo, elephant_geometry.vertex_count());
    
    glm::vec3 light_position{0.0f, 7.0f, 0.0f};
    glm::vec3 light_emission{300.0f};
    
    glm::vec3 eye{0.0f, 0.0f, 28.0f};
    auto view_matrix = glm::lookAt(eye, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    auto projection_matrix = glm::perspective(glm::radians(45.0f), static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT), 0.1f, 100.0f);
    
    // initialize ray tracer
    auto context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    auto model = context->createModel();
    elephant_geometry.with_position_pointer([&](optix::float4 *p) {
        model->setTriangles(1, RTP_BUFFER_TYPE_CUDA_LINEAR, p, sizeof(float4));
    });
    
    float *hit_buffer = nullptr;
    ShadowRay *ray_buffer = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&hit_buffer), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&ray_buffer), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(ShadowRay));
    
    auto query = model->createQuery(RTP_QUERY_TYPE_ANY);
    query->setRays(1, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, ray_buffer);
    query->setHits(1, RTP_BUFFER_FORMAT_HIT_T, RTP_BUFFER_TYPE_CUDA_LINEAR, hit_buffer);
    
    auto initial_time = glfwGetTime();
    
    // main loop
    while (!glfwWindowShouldClose(window)) {
        
        glfwPollEvents();
        
        auto time = static_cast<float>(glfwGetTime() - initial_time);
        elephant_geometry.update(time);
        model->update(RTP_MODEL_HINT_ASYNC);
        
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
        
        glClearColor(1.0f, 0.5f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        shader.use();
        shader.setMat4("view", view_matrix);
        shader.setMat4("projection", projection_matrix);
        shader.setVec3("light_position", light_position);
        shader.setVec3("light_emission", light_emission);
        
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, total_vertex_count);
        
        query->execute(RTP_QUERY_HINT_ASYNC);
        
        glfwSwapBuffers(window);
    }
    
    return 0;
}
