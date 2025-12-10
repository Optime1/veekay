#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// Define max number of lights (must match C++)
#define MAX_POINT_LIGHTS 10

struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct PointLight {
    vec3 position;
    float intensity; // Intensity for attenuation
    vec3 color;
};

struct Material {
    vec3 albedo_color;
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
    Material material;
};

layout (binding = 2, std140) uniform LightingUniforms {
    DirectionalLight directional_light;
    PointLight point_lights[MAX_POINT_LIGHTS];
    uint num_point_lights;
};

// Blinn-Phong Lighting Functions
vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, Material mat) {
    vec3 lightDir = normalize(-light.direction); // Direction *from* light *to* surface

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = light.color * diff * mat.albedo_color;

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
    vec3 diffuse = light.color * diff * mat.albedo_color;

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), mat.shininess);
    vec3 specular = light.color * spec * mat.specular_color;

    // Attenuation (Inverse Square Law)
    float distance = length(light.position - fragPos);
    // Using intensity for attenuation: I / (constant + linear * d + quadratic * d^2)
    // Simplified: intensity / (distance^2) -> I = intensity, const=0, linear=0, quad=1
    // Or: intensity / (1 + 0.1*d + 0.01*d^2) etc. Adjust coefficients as needed.
    float attenuation = light.intensity / (distance * distance);
    // Optional: Clamping to prevent extreme intensity at very close distances
    // attenuation = min(attenuation, 100.0); // Example clamp

    diffuse *= attenuation;
    specular *= attenuation;

    return diffuse + specular;
}

void main() {
    vec3 norm = normalize(f_normal);
    vec3 viewDir = normalize(camera_position - f_position);

    vec3 result = vec3(0.0); // Ambient term can be added here if desired

    // Calculate Directional Light
    result += calculateDirectionalLight(directional_light, norm, viewDir, material);

    // Calculate Point Lights
    for(uint i = 0u; i < num_point_lights; i++) {
        result += calculatePointLight(point_lights[i], f_position, norm, viewDir, material);
    }

    final_color = vec4(result, 1.0);
}