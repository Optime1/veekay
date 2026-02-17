#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv; // Receive UV coordinates

layout (location = 0) out vec4 final_color;

// Define max number of lights (must match C++)
#define MAX_POINT_LIGHTS 10

struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct PointLight {
    vec3 position;
    float intensity; // Intensity for attenuation calculation
    vec3 color;
};

struct Material {
    vec3 albedo_factor; // Factor to multiply texture color (or base color if no texture)
    float _pad0; // Pad to align to 16 bytes
    vec3 specular_color;
    float shininess;
};

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    Material material; // Include material properties
};

layout (binding = 2, std140) uniform LightingUniforms {
    DirectionalLight directional_light;
    PointLight point_lights[MAX_POINT_LIGHTS];
    uint num_point_lights;
};

// Declare the texture sampler
layout (binding = 3) uniform sampler2D texSampler; // Binding 3 for texture + sampler

// Blinn-Phong Lighting Functions
vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, Material mat) {
    // Normalize light direction
    vec3 lightDir = normalize(-light.direction); // Direction *from* light *to* surface

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff; // Don't multiply by albedo here yet

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), mat.shininess);
    vec3 specular = light.color * spec * mat.specular_color;

    return diffuse + specular;
}

vec3 calculatePointLight(PointLight light, vec3 fragPos, vec3 normal, vec3 viewDir, Material mat) {
    vec3 lightDir = normalize(light.position - fragPos);

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff; // Don't multiply by albedo here yet

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), mat.shininess);
    vec3 specular = light.color * spec * mat.specular_color;

    // Attenuation (Inverse Square Law)
    float distance = length(light.position - fragPos);
    // Ensure distance is not zero to avoid division by zero
    float attenuation = light.intensity / max(distance * distance, 0.001); // Use intensity directly

    diffuse *= attenuation;
    specular *= attenuation;

    return diffuse + specular;
}

void main() {
    vec3 norm = normalize(f_normal);
    if (length(norm) < 0.99) {
        norm = vec3(0.0, 0.0, 1.0); // Fallback normal if input is invalid
    }
    vec3 viewDir = normalize(camera_position - f_position);

    vec3 result = vec3(0.0);

    result += calculateDirectionalLight(directional_light, norm, viewDir, material);

    for(uint i = 0u; i < num_point_lights; i++) {
        result += calculatePointLight(point_lights[i], f_position, norm, viewDir, material);
    }

    vec4 tex_color = texture(texSampler, f_uv);
    vec3 lit_color = result * tex_color.rgb * material.albedo_factor;
    float alpha = tex_color.a;

    final_color = vec4(lit_color, alpha);
}