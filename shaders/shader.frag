#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;
layout(location = 3) in vec4 f_light_space_pos;

layout(location = 0) out vec4 out_color;

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
    mat4 light_view_projection;
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

struct PointLight {
    vec3 position; float _pad0;
    LightColors colors;
    float linear;
    float quadratic;
    float _pad1;
    float _pad2;
};

layout(std430, binding = 2) readonly buffer PointLightsBuffer {
    PointLight lights[];
} point_lights;

struct SpotLight {
    PointLight point_light;
    vec3 direction;
    float cut_off;
    float outer_cut_off;
    float _pad1;
    float _pad2;
};

layout(std430, binding = 3) readonly buffer SpotLightsBuffer {
    SpotLight lights[];
} spot_lights;

layout (binding = 4) uniform sampler2D albedo_texture;
layout (binding = 5) uniform sampler2DShadow shadow_map;

float calculate_shadow(vec4 light_pos, vec3 normal, vec3 light_dir) {
    vec3 proj_coords = light_pos.xyz / light_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if(proj_coords.z > 1.0 || proj_coords.x > 1.0 || proj_coords.x < 0.0 || proj_coords.y > 1.0 || proj_coords.y < 0.0) {
        return 1.0;
    }

    float bias = max(0.005 * (1.0 - dot(normal, light_dir)), 0.0005);

    float shadow = texture(shadow_map, vec3(proj_coords.xy, proj_coords.z - bias));
    return shadow;
}

vec3 calc_dir_light(vec3 N, vec3 V, vec3 albedo, vec3 specular_color, float shininess, float shadow_factor) {
    vec3 L = normalize(-scene.directional_light.direction);
    float NdotL = max(dot(N, L), 0.0);

    vec3 diffuse = scene.directional_light.colors.diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float spec_intensity = (NdotL > 0.0) ? pow(NdotH, shininess) : 0.0;
    vec3 specular = scene.directional_light.colors.specular * spec_intensity;

    vec3 light_res = (diffuse + specular) * shadow_factor;

    vec3 ambient = scene.directional_light.colors.ambient;

    return (ambient + light_res) * scene.directional_light.intensity * albedo;
}

vec3 calc_point_light(PointLight light, vec3 N, vec3 V, vec3 P, vec3 albedo, vec3 specular_col, float shininess) {
    vec3 L = light.position - P;
    float dist = length(L);
    L = normalize(L);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light.colors.diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = light.colors.specular * pow(NdotH, shininess);

    float attenuation = 1.0 / (1.0 + light.linear * dist + light.quadratic * dist * dist);
    vec3 ambient = light.colors.ambient;

    return (ambient + diffuse + specular) * albedo * attenuation;
}

vec3 calc_spot_light(SpotLight light, vec3 N, vec3 V, vec3 P, vec3 albedo, vec3 specular_col, float shininess) {
    vec3 L = light.point_light.position - P;
    float dist = length(L);
    L = normalize(L);

    vec3 spotDir = normalize(-light.direction);
    float theta = dot(L, spotDir);
    float epsilon = light.cut_off - light.outer_cut_off;
    float intensity = clamp((theta - light.outer_cut_off) / max(epsilon, 1e-6), 0.0, 1.0);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light.point_light.colors.diffuse * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = light.point_light.colors.specular * pow(NdotH, shininess);

    float attenuation = 1.0 / (1.0 + light.point_light.linear * dist + light.point_light.quadratic * dist * dist);
    vec3 ambient = light.point_light.colors.ambient;

    return (ambient + diffuse + specular) * albedo * attenuation * intensity;
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 P = f_position;
    vec3 V = normalize(scene.camera_position - P);

    vec4 texel = texture(albedo_texture, f_uv);
    vec3 albedo = material.albedo_color * texel.rgb;
    vec3 specular_col = material.specular_color;
    float shininess = material.shininess;

    vec3 color = vec3(0.0);

    color += albedo * scene.ambient_light.color * scene.ambient_light.intensity;

    vec3 L_dir = normalize(-scene.directional_light.direction);
    float shadow = calculate_shadow(f_light_space_pos, N, L_dir);
    color += calc_dir_light(N, V, albedo, specular_col, shininess, shadow);

    for (uint i = 0u; i < scene.num_point_lights; ++i) {
        color += calc_point_light(point_lights.lights[i], N, V, P, albedo, specular_col, shininess);
    }

    for (uint i = 0u; i < scene.num_spot_lights; ++i) {
        color += calc_spot_light(spot_lights.lights[i], N, V, P, albedo, specular_col, shininess);
    }

    out_color = vec4(color, 1.0);
}