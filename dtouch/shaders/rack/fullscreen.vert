#version 330 core
// One clipped triangle covering the viewport; every rack pass uses it.
in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
