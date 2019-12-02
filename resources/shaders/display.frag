#version 410 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screen;
uniform sampler2D shadow;

void main() {
    float Shadow = texture(shadow, TexCoords).r;
    FragColor = vec4((1.0f - Shadow) * texture(screen, TexCoords).rgb, 1.0f);
}
