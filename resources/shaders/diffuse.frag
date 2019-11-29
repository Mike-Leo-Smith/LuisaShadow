#version 410 core

in vec3 Position;
in vec3 Normal;
in vec3 Albedo;

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 FragPosition;

uniform vec3 light_position;
uniform vec3 light_emission;

#define M_PIf   3.14159265358979323846f
#define M_1_PIf 0.318309886183790671538f

void main() {

    vec3 N = normalize(Normal);
    vec3 L = light_position - Position;
    float attenuation = 1.0f / max(dot(L, L), 1e-3f);
    L = normalize(L);

    vec3 Lo = vec3(0.0f);
    float cos_theta = dot(L, N);
    if (cos_theta > 1e-3f) {
        Lo = Albedo * light_emission * attenuation * M_1_PIf * cos_theta;
    }

    FragColor = vec4(Lo, 1.0f);
    FragPosition = vec4(Position, 1.0f);
}
