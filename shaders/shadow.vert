#version 450

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform PushConsts {
    mat4 model;
    mat4 light_view_proj;
} push;

void main() {
    gl_Position = push.light_view_proj * push.model * vec4(in_position, 1.0);
}