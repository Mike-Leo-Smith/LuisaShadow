#version 410 core

layout (location = 0) in vec2 aPosition;

out vec2 TexCoords;

void main() {
    TexCoords = aPosition * 0.5f + 0.5f;
    gl_Position = vec4(aPosition, 0.0f, 1.0f);
}
