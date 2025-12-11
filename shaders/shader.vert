#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv; // Pass UV coordinates

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix; // For transforming normals correctly
    // Material is included here in the struct
};

void main() {
    vec4 world_position = model * vec4(v_position, 1.0);
    // Transform normal using normal matrix (transpose of inverse of model's 3x3)
    vec4 world_normal = normal_matrix * vec4(v_normal, 0.0);

    gl_Position = view_projection * world_position;

    f_position = world_position.xyz;
    f_normal = world_normal.xyz; // Pass world-space normal
    f_uv = v_uv; // Pass UV coordinates
}