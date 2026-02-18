#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_light_space_pos; // Позиция фрагмента глазами света

struct LightColors {
    vec3 ambient; float _pad0;
    vec3 diffuse; float _pad1;
    vec3 specular; float _pad2;
};

struct DirectionalLight {
    vec3 direction;
    float intensity;
    LightColors colors;
};

struct AmbientLight {
    vec3 color;
    float intensity;
    vec3 specular_color;
    float shininess;
};

layout(std140, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection; // Матрица вида-проекции света
    vec3 camera_position;
    uint num_point_lights;
    uint num_spot_lights;
    uint _pad_align0;
    uint _pad_align1;
    uint _pad_align2;
    DirectionalLight directional_light;
    AmbientLight ambient_light;
} scene;

layout(std140, binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad0;
} material;

void main() {
    vec4 world_pos = material.model * vec4(in_position, 1.0);
    f_position = world_pos.xyz;

    // Корректный расчет нормали с учетом масштабирования
    mat3 normal_matrix = transpose(inverse(mat3(material.model)));
    f_normal = normalize(normal_matrix * in_normal);

    f_uv = in_uv;

    // Перевод координаты фрагмента в пространство света
    f_light_space_pos = scene.light_view_projection * world_pos;

    gl_Position = scene.view_projection * world_pos;
}