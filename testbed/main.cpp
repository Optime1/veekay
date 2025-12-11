#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm> // For std::clamp
#include <string> // For std::string
#include <filesystem> // For path manipulation

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h> // Include lodepng header

// --- Define TINYOBJLOADER_IMPLEMENTATION before including ---
#define TINYOBJLOADER_IMPLEMENTATION
// --- End of definition ---

#include "tiny_obj_loader.h" // Include tinyobjloader header

namespace {

// --- NEW STRUCTS FOR LIGHTING AND MATERIALS ---

struct DirectionalLight {
    veekay::vec3 direction; float _pad0; // Pad to align to 16 bytes
    veekay::vec3 color;     float _pad1; // Pad to align to 16 bytes
};

struct PointLight {
    veekay::vec3 position;  float intensity; // Intensity for attenuation calculation
    veekay::vec3 color;     float _pad0;     // Pad to align to 16 bytes
};

// Define max number of lights
constexpr uint32_t max_directional_lights = 1;
constexpr uint32_t max_point_lights = 10; // Adjust as needed

struct LightingUniforms {
    DirectionalLight directional_light;
    PointLight point_lights[max_point_lights];
    uint32_t num_point_lights;
    float _pad0[3]; // Pad to align total size to 16-byte boundary if necessary
};

struct Material {
    veekay::vec3 albedo_factor; // Factor to multiply texture color (or base color if no texture)
    float _pad0; // Pad to align to 16 bytes
    veekay::vec3 specular_color;
    float shininess;
};

// --- ORIGINAL STRUCTS (Updated) ---

constexpr uint32_t max_models = 1024;

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
    // NOTE: You can add more attributes
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 camera_position; // Needed for specular calculation
    float _pad0; // Pad to align to 16 bytes
};

struct ModelUniforms {
    veekay::mat4 model;
    veekay::mat4 normal_matrix; // For transforming normals correctly
    Material material; // Include material properties
};

struct Mesh {
    veekay::graphics::Buffer* vertex_buffer;
    veekay::graphics::Buffer* index_buffer;
    uint32_t indices;
};

struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};

    // NOTE: Model matrix (translation, rotation and scaling)
    veekay::mat4 matrix() const;
};

struct Model {
    Mesh mesh;
    Transform transform;
    Material material; // Use Material struct
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;
    constexpr static float mouse_sensitivity = 0.1f; // Adjust sensitivity
    constexpr static float movement_speed = 2.0f;    // Movement speed (units per second)

    veekay::vec3 position = {};
    veekay::vec3 rotation = {}; // Pitch (x), Yaw (y), Roll (z) in radians

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    // NOTE: View matrix of camera (inverse of a transform)
    veekay::mat4 view() const;

    // NOTE: View and projection composition
    veekay::mat4 view_projection(float aspect_ratio) const;
};

// --- Scene Objects ---
inline namespace {
    Camera camera{
        .position = {0.0f, -0.5f, -3.0f}
    };

    std::vector<Model> models;

    // Lighting data
    LightingUniforms lighting_uniforms = {};
    veekay::graphics::Buffer* lighting_uniforms_buffer = nullptr;

    // --- Animation Variables ---
    bool animate_light = false;
    float animation_time = 0.0f;
    float animation_amplitude = 3.0f; // How far the light moves
    float animation_speed = 2.0f;    // How fast the light moves

    // --- Model Placement Variables ---
    // Default values for initial placement
    veekay::vec3 default_model_position = {30.0f, 0.0f, -30.0f}; // Changed from {0.0f, 0.0f, 0.0f}
    float default_model_scale = 0.5f; // Changed from 0.1f
    // Runtime adjustable values
    veekay::vec3 model_position = default_model_position; // Start with default
    float model_scale = default_model_scale; // Start with default
}

// --- Vulkan Objects ---
inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout; // Updated layout
    VkDescriptorSet descriptor_set;              // Updated set

    VkPipelineLayout pipeline_layout; // Updated layout
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;

    // --- Texturing Objects ---
    veekay::graphics::Texture* loaded_texture = nullptr; // Loaded texture
    VkSampler loaded_texture_sampler = VK_NULL_HANDLE;   // Loaded sampler

    veekay::graphics::Texture* missing_texture;
    VkSampler missing_texture_sampler;
}

float toRadians(float degrees) {
    return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
    // Create rotation matrix (Euler angles ZYX order)
    auto rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    auto rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    auto rot_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
    auto rotation_matrix = rot_z * rot_y * rot_x; // Order matters

    // Create scaling matrix
    auto scaling_matrix = veekay::mat4::scaling(scale);

    // Create translation matrix
    auto translation_matrix = veekay::mat4::translation(position);

    // Combine: T * R * S
    return translation_matrix * rotation_matrix * scaling_matrix;
}

veekay::mat4 Camera::view() const {
    // Calculate camera orientation from rotation angles (Euler angles ZYX)
    // Yaw (Y-axis) -> Yaw
    // Pitch (X-axis) -> Pitch
    // Roll (Z-axis) -> Roll (often 0 for FPS cameras)

    float cos_yaw = std::cos(rotation.y);
    float sin_yaw = std::sin(rotation.y);
    float cos_pitch = std::cos(rotation.x);
    float sin_pitch = std::sin(rotation.x);

    // Calculate camera axes
    veekay::vec3 front;
    front.x = sin_yaw * cos_pitch;
    front.y = -sin_pitch; // Y is down in Vulkan view space convention, so negative pitch gives forward
    front.z = cos_yaw * cos_pitch;
    front = veekay::vec3::normalized(front);

    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, {0.0f, -1.0f, 0.0f})); // Use world down
    veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

    // Build Look-At matrix: Inverse of the camera's world transform
    veekay::mat4 view_matrix = veekay::mat4::identity();

    view_matrix[0][0] = right.x;
    view_matrix[1][0] = right.y;
    view_matrix[2][0] = right.z;

    view_matrix[0][1] = up.x;
    view_matrix[1][1] = up.y;
    view_matrix[2][1] = up.z;

    view_matrix[0][2] = -front.x; // Negative because we look along the -Z axis in view space
    view_matrix[1][2] = -front.y;
    view_matrix[2][2] = -front.z;

    // Apply translation (inverse of camera position)
    view_matrix[3][0] = -veekay::vec3::dot(right, position);
    view_matrix[3][1] = -veekay::vec3::dot(up, position);
    view_matrix[3][2] = -veekay::vec3::dot(-front, position); // Note the negative front here

    return view_matrix;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
    auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    return view() * projection;
}

// --- Load OBJ Model (Updated) ---
std::vector<Vertex> loadObjModel(const std::string& path, std::vector<uint32_t>& out_indices, std::string& out_texture_path, std::vector<tinyobj::material_t>& out_materials) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    // std::vector<tinyobj::material_t> materials; // Pass materials vector from outside
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &out_materials, &warn, &err, path.c_str())) {
        std::cerr << "tinyobjloader error: " << err << std::endl;
        return {}; // Return empty vector on error
    }

    // --- Extract texture path from materials (Updated) ---
    out_texture_path = ""; // Initialize to empty
    for (const auto& mat : out_materials) {
        if (!mat.diffuse_texname.empty()) {
            out_texture_path = mat.diffuse_texname; // Use the first diffuse texture found
            std::cout << "Found diffuse texture in MTL: " << out_texture_path << std::endl;
            // Optional: resolve relative path based on the .obj file path
            // std::filesystem::path obj_path(path);
            // std::filesystem::path tex_path(mat.diffuse_texname);
            // out_texture_path = (obj_path.parent_path() / tex_path).string();
            break; // Stop at first texture
        }
    }
    if (out_texture_path.empty()) {
        std::cout << "No diffuse texture found in MTL file." << std::endl;
    }
    // --- End of texture path extraction ---

    std::vector<Vertex> vertices;
    out_indices.clear(); // Clear the output vector

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex = {};

            // Position
            vertex.position.x = attrib.vertices[3 * index.vertex_index + 0];
            vertex.position.y = attrib.vertices[3 * index.vertex_index + 1];
            vertex.position.z = attrib.vertices[3 * index.vertex_index + 2];

            // Normal (if available)
            if (index.normal_index >= 0) {
                vertex.normal.x = attrib.normals[3 * index.normal_index + 0];
                vertex.normal.y = attrib.normals[3 * index.normal_index + 1];
                vertex.normal.z = attrib.normals[3 * index.normal_index + 2];
            } else {
                // Set zero normal or calculate later if needed
                vertex.normal = {0.0f, 0.0f, 0.0f};
            }

            // UV-coordinates (if available)
            if (index.texcoord_index >= 0) {
                vertex.uv.x = attrib.texcoords[2 * index.texcoord_index + 0];
                vertex.uv.y = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]; // Flip Y
            } else {
                // Set (0,0) or default value
                vertex.uv = {0.0f, 0.0f};
            }

            vertices.push_back(vertex);
            out_indices.push_back(static_cast<uint32_t>(vertices.size() - 1));
        }
    }

    return vertices;
}

// --- Load Texture from PNG (Updated) ---
veekay::graphics::Texture* loadTextureFromPNG(const std::string& filename, VkCommandBuffer cmd) {
    if (filename.empty()) {
        std::cerr << "Texture filename is empty, cannot load." << std::endl;
        return nullptr;
    }
    std::vector<unsigned char> image;
    unsigned width, height;

    unsigned error = lodepng::decode(image, width, height, filename);
    if (error) {
        std::cerr << "LodePNG error loading '" << filename << "': " << lodepng_error_text(error) << std::endl;
        return nullptr;
    }

    // LodePNG loads to RGBA by default
    auto* texture = new veekay::graphics::Texture(
        cmd,
        width,
        height,
        VK_FORMAT_R8G8B8A8_UNORM, // Or VK_FORMAT_B8G8R8A8_UNORM depending on byte order
        image.data()
    );

    return texture;
}

// --- Load Shader Module ---
VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << path << "\n";
        return nullptr;
    }
    size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module from: " << path << "\n";
        return nullptr;
    }

    return result;
}

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;
    VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

    { // --- Build Graphics Pipeline ---
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (!vertex_shader_module) {
            std::cerr << "Failed to load Vulkan vertex shader from file\n";
            veekay::app.running = false;
            return;
        }

        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (!fragment_shader_module) {
            std::cerr << "Failed to load Vulkan fragment shader from file\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stage_infos[2];

        stage_infos[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };

        stage_infos[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        VkVertexInputAttributeDescription attributes[] = {
            { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
            { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
            { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
            .pVertexAttributeDescriptions = attributes,
        };

        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = false,
            .minSampleShading = 1.0f,
        };

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(veekay::app.window_width),
            .height = static_cast<float>(veekay::app.window_height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = {veekay::app.window_width, veekay::app.window_height},
        };

        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        VkPipelineColorBlendAttachmentState attachment_info{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachment_info
        };

        { // --- Descriptor Pool ---
            VkDescriptorPoolSize pools[] = {
                { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8 },
                { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 8 },
                { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 8 }, // Add pool for sampler
            };

            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2, // Need 2 sets now (Scene+Model+Lighting and Texture+Sampler)
                .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                .pPoolSizes = pools,
            };

            if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor pool\n";
                veekay::app.running = false;
                return;
            }
        }

        { // --- Descriptor Set Layout (Updated) ---
            VkDescriptorSetLayoutBinding bindings[] = {
                { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT }, // Lighting
                { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT }, // Texture + Sampler
            };

            VkDescriptorSetLayoutCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                .pBindings = bindings,
            };

            if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor set layout\n";
                veekay::app.running = false;
                return;
            }
        }

        { // --- Allocate Descriptor Sets ---
            VkDescriptorSetAllocateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate Vulkan descriptor set\n";
                veekay::app.running = false;
                return;
            }
        }

        { // --- Pipeline Layout (Updated) ---
            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };

            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }
        }

        VkGraphicsPipelineCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };

        if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline\n";
            veekay::app.running = false;
            return;
        }
    }

    // --- Create Buffers ---
    scene_uniforms_buffer = new veekay::graphics::Buffer(
        sizeof(SceneUniforms),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    model_uniforms_buffer = new veekay::graphics::Buffer(
        max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    lighting_uniforms_buffer = new veekay::graphics::Buffer(
        sizeof(LightingUniforms),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // --- Load OBJ Model (Updated) ---
    std::vector<uint32_t> obj_indices;
    std::vector<tinyobj::material_t> loaded_materials; // Vector to store materials
    std::string texture_path_from_mtl; // Variable to store texture path
    std::vector<Vertex> obj_vertices = loadObjModel("model.obj", obj_indices, texture_path_from_mtl, loaded_materials); // Pass the path variable and materials vector

    if (!obj_vertices.empty() && !obj_indices.empty()) {
        Mesh obj_mesh;
        obj_mesh.vertex_buffer = new veekay::graphics::Buffer(
            obj_vertices.size() * sizeof(Vertex), obj_vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        obj_mesh.index_buffer = new veekay::graphics::Buffer(
            obj_indices.size() * sizeof(uint32_t), obj_indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        obj_mesh.indices = uint32_t(obj_indices.size());

        // --- Determine Material Properties from MTL (Updated) ---
        Material model_material = Material{
            .albedo_factor = {1.0f, 1.0f, 1.0f}, // Default factor
            .specular_color = {0.5f, 0.5f, 0.5f},
            .shininess = 32.0f
        };

        // If tinyobjloader loaded materials, you can access them here
        if (!loaded_materials.empty()) {
            const auto& mat = loaded_materials[0]; // Use the first material
            if (!texture_path_from_mtl.empty()) {
                 // If texture is present, set albedo factor to white (1,1,1)
                 // The texture will provide the color
                 model_material.albedo_factor = {1.0f, 1.0f, 1.0f};
                 std::cout << "Using texture '" << texture_path_from_mtl << "', setting albedo factor to white." << std::endl;
            } else {
                 // If no texture, use the diffuse color from the material as the albedo factor
                 model_material.albedo_factor = {mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]};
                 std::cout << "No texture, using material diffuse color as albedo factor: (" << mat.diffuse[0] << ", " << mat.diffuse[1] << ", " << mat.diffuse[2] << ")" << std::endl;
            }
            // Use the specular color and shininess from the material if available
            model_material.specular_color = {mat.specular[0], mat.specular[1], mat.specular[2]};
            model_material.shininess = mat.shininess > 0.0f ? mat.shininess : 32.0f; // Default if shininess is 0
        } else {
             std::cout << "No materials found in MTL, using default material properties." << std::endl;
        }
        // --- End of Material Determination ---

        // --- Add Model with Default Transform ---
        models.emplace_back(Model{
            .mesh = obj_mesh,
            .transform = Transform{
                .position = default_model_position, // Set initial position
                .scale = {default_model_scale, default_model_scale, default_model_scale} // Set initial scale
            },
            .material = model_material
        });
        std::cout << "Loaded OBJ model with " << obj_vertices.size() << " vertices and " << obj_indices.size() << " indices.\n";
    } else {
        std::cerr << "Failed to load OBJ model or model is empty. Exiting.\n";
        veekay::app.running = false;
        return;
    }


    // --- Load Texture from MTL Path (Updated) ---
    // Use the path found by tinyobjloader, or fallback to a default name if empty
    std::string texture_to_load = texture_path_from_mtl.empty() ? "texture.png" : texture_path_from_mtl;
    loaded_texture = loadTextureFromPNG(texture_to_load, cmd); // Load the texture from the MTL path or fallback
    if (!loaded_texture) {
         std::cerr << "Failed to load texture from '" << texture_to_load << "'. Exiting.\n";
         // If texture loading fails, you might want to continue with just material colors
         // For now, let's treat it as an error
         veekay::app.running = false;
         return;
    }

    // --- Create Sampler (Updated - Correct Field Order) ---
    VkSamplerCreateInfo sampler_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr, // Явно указать, если не используется
        .flags = 0,       // Явно указать, если не используются флаги
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, // <-- Перемещено сюда
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0f, // Явно указать, если не используется смещение LOD
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16.0f,
        .compareEnable = VK_FALSE, // <-- Перемещено сюда
        .compareOp = VK_COMPARE_OP_ALWAYS, // <-- Перемещено сюда
        .minLod = 0.0f, // Явно указать, если не используется
        .maxLod = VK_LOD_CLAMP_NONE, // Или конкретное значение, если используется
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    if (vkCreateSampler(device, &sampler_info, nullptr, &loaded_texture_sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler\n";
        veekay::app.running = false;
        return;
    }

    // --- Initialize Lighting Data ---
    // Directional Light
    lighting_uniforms.directional_light.direction = veekay::vec3::normalized({0.0f, 1.0f, 0.0f}); // Changed to (0, 1, 0) - light from below
    lighting_uniforms.directional_light.color = {1.0f, 1.0f, 1.0f}; // Changed to (255, 255, 255) in float

    // Point Lights
    lighting_uniforms.point_lights[0] = PointLight{
        .position = {0.0f, 0.0f, 0.0f}, // Initial position will be overridden by animation if active
        .intensity = 10.0f, // Higher intensity for noticeable attenuation
        .color = {1.0f, 0.0f, 0.0f} // Red
    };
    lighting_uniforms.point_lights[1] = PointLight{
        .position = {2.0f, 1.0f, -2.0f},
        .intensity = 10.0f,
        .color = {0.0f, 1.0f, 0.0f} // Green
    };
    lighting_uniforms.point_lights[2] = PointLight{
        .position = {-2.0f, 1.0f, 2.0f},
        .intensity = 10.0f,
        .color = {0.0f, 0.0f, 1.0f} // Blue
    };
    lighting_uniforms.num_point_lights = 3; // Set initial number of active lights

    // --- Initialize Missing Texture and Sampler (fallback) (Updated) ---
    {
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan texture sampler\n";
            veekay::app.running = false;
            return;
        }

        uint32_t pixels[] = {
            0xff000000, 0xffff00ff,
            0xffff00ff, 0xff000000,
        };

        missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
    }

    // --- Update Descriptor Sets (Updated) ---
    {
        VkDescriptorBufferInfo buffer_infos[] = {
            { .buffer = scene_uniforms_buffer->buffer, .offset = 0, .range = sizeof(SceneUniforms) },
            { .buffer = model_uniforms_buffer->buffer, .offset = 0, .range = sizeof(ModelUniforms) }, // Update range
            { .buffer = lighting_uniforms_buffer->buffer, .offset = 0, .range = sizeof(LightingUniforms) },
        };

        VkDescriptorImageInfo image_info = { // Updated
            .sampler = loaded_texture_sampler,
            .imageView = loaded_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_infos[] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_infos[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffer_infos[1] }, // Update binding
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_infos[2] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_info }, // Updated
        };

        vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]), write_infos, 0, nullptr);
    }
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    // Delete sampler
    vkDestroySampler(device, loaded_texture_sampler, nullptr);

    // Delete loaded texture
    delete loaded_texture;

    // Delete missing texture and sampler
    vkDestroySampler(device, missing_texture_sampler, nullptr);
    delete missing_texture;

    // Delete buffers
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;
    delete lighting_uniforms_buffer;

    // Delete mesh buffers (if any were created)
    for (auto& model : models) {
        delete model.mesh.vertex_buffer;
        delete model.mesh.index_buffer;
    }

    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    // --- Update Animation Time ---
    animation_time = static_cast<float>(time); // Use actual application time

    // --- ImGUI Controls ---
    ImGui::Begin("Controls:");
    ImGui::SliderFloat("Camera Pitch", &camera.rotation.x, static_cast<float>(-M_PI) / 2.0f + 0.1f, static_cast<float>(M_PI) / 2.0f - 0.1f);
    // Sliders for runtime adjustment of model position and scale
    ImGui::SliderFloat3("Model Position", model_position.elements, -20.0f, 20.0f); // Adjust range as needed
    ImGui::SliderFloat("Model Scale", &model_scale, 0.01f, 10.0f); // Keep the scale slider
    ImGui::End();

    ImGui::Begin("Lighting Controls");
    ImGui::Text("Directional Light");
    // Color and Direction sliders for directional light
    ImGui::ColorEdit3("Dir Color", lighting_uniforms.directional_light.color.elements);
    ImGui::SliderFloat3("Dir Direction", lighting_uniforms.directional_light.direction.elements, -1.0f, 1.0f);
    // Normalize the direction vector after editing
    if (ImGui::IsItemDeactivated()) { // Update only after the item is finished being edited
         lighting_uniforms.directional_light.direction = veekay::vec3::normalized(lighting_uniforms.directional_light.direction);
    }

    ImGui::Text("Point Lights");
    ImGui::SliderInt("Num Point Lights", (int*)&lighting_uniforms.num_point_lights, 0, max_point_lights);
    ImGui::Checkbox("Animate Light 0", &animate_light); // Add checkbox for animation
    if (animate_light) {
        // Optionally allow adjusting animation parameters
        ImGui::SliderFloat("Anim Amplitude", &animation_amplitude, 0.1f, 10.0f);
        ImGui::SliderFloat("Anim Speed", &animation_speed, 0.1f, 10.0f);
    }

    for (uint32_t i = 0; i < std::min(lighting_uniforms.num_point_lights, max_point_lights); ++i) {
        ImGui::PushID(i);
        ImGui::Text("Light %d", i);
        ImGui::ColorEdit3("Color", lighting_uniforms.point_lights[i].color.elements);
        // Disable slider for X position of light 0 if animation is on
        if (i == 0 && animate_light) {
             ImGui::Text("X Pos (Animated): %.2f", lighting_uniforms.point_lights[0].position.x);
             // Update animated position here, before UI might try to change it
             lighting_uniforms.point_lights[0].position.x = sinf(animation_time * animation_speed) * animation_amplitude;
             // Allow Y and Z to be modified even when animating X
             ImGui::SliderFloat("Y", &lighting_uniforms.point_lights[0].position.y, -10.0f, 10.0f);
             ImGui::SliderFloat("Z", &lighting_uniforms.point_lights[0].position.z, -10.0f, 10.0f);
        } else {
            ImGui::SliderFloat3("Position", lighting_uniforms.point_lights[i].position.elements, -10.0f, 10.0f);
        }
        ImGui::SliderFloat("Intensity", &lighting_uniforms.point_lights[i].intensity, 0.0f, 50.0f);
        ImGui::PopID();
    }
    ImGui::End();

    // --- Camera Controls ---
    if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        using namespace veekay::input;

        if (mouse::isButtonDown(mouse::Button::left)) {
            auto move_delta = mouse::cursorDelta();

            // Update camera rotation based on mouse delta
            camera.rotation.y -= move_delta.x * Camera::mouse_sensitivity * 0.01f; // Yaw
            camera.rotation.x -= move_delta.y * Camera::mouse_sensitivity * 0.01f; // Pitch
            // Clamp pitch to avoid flipping, explicitly cast M_PI to float
            camera.rotation.x = std::clamp(camera.rotation.x, static_cast<float>(-M_PI) / 2.0f + 0.1f, static_cast<float>(M_PI) / 2.0f - 0.1f);

            // Calculate camera axes from view matrix
            auto view = camera.view();
            veekay::vec3 right = {view[0][0], view[1][0], view[2][0]};
            veekay::vec3 up = {view[0][1], view[1][1], view[2][1]};
            veekay::vec3 front = {-view[0][2], -view[1][2], -view[2][2]}; // Negative Z from view matrix

            // Movement speed based on time delta for consistent speed
            // float delta_time = static_cast<float>(glfwGetTime() - last_frame_time); // Use actual delta time if available
            float delta_time = 1.0f / 60.0f; // Approximate, use actual delta time if available
            float speed = Camera::movement_speed * delta_time;

            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position += front * speed;
            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position -= front * speed;
            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right * speed;
            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right * speed;
            if (keyboard::isKeyDown(keyboard::Key::q)) // Up
                camera.position += up * speed;
            if (keyboard::isKeyDown(keyboard::Key::z)) // Down
                camera.position -= up * speed;
        }
    }

    // --- Update Uniform Buffers ---
    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .camera_position = camera.position,
    };

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        Model& model = models[i]; // Use reference to allow modification
        ModelUniforms& uniforms = model_uniforms[i];

        // Apply position and scale from UI to the model's transform before calculating the matrix
        model.transform.position = model_position;
        model.transform.scale = {model_scale, model_scale, model_scale};

        uniforms.model = model.transform.matrix();
        // Calculate normal matrix (transpose of inverse of upper 3x3 of model matrix)
        auto model_mat4 = uniforms.model;
        auto normal_mat3x3 = veekay::mat4::identity();
        for (int j = 0; j < 3; ++j) {
             for (int k = 0; k < 3; ++k) {
                 normal_mat3x3[j][k] = model_mat4[j][k];
             }
        }
        uniforms.normal_matrix = veekay::mat4::transpose(normal_mat3x3);
        uniforms.normal_matrix[3][0] = 0.0f; uniforms.normal_matrix[3][1] = 0.0f; uniforms.normal_matrix[3][2] = 0.0f; uniforms.normal_matrix[3][3] = 1.0f;

        uniforms.material = model.material; // Assign material properties
    }

    // Update buffers
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
    *(LightingUniforms*)lighting_uniforms_buffer->mapped_region = lighting_uniforms;

    const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        const ModelUniforms& uniforms = model_uniforms[i];
        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &info);

    VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
    VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
    VkClearValue clear_values[] = {clear_color, clear_depth};

    VkRenderPassBeginInfo rp_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = veekay::app.vk_render_pass,
        .framebuffer = framebuffer,
        .renderArea = {
            .extent = {veekay::app.window_width, veekay::app.window_height}
        },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize zero_offset = 0;

    VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
    VkBuffer current_index_buffer = VK_NULL_HANDLE;

    const size_t model_uniforms_alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        const Mesh& mesh = model.mesh;

        if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
            current_vertex_buffer = mesh.vertex_buffer->buffer;
            vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
        }

        if (current_index_buffer != mesh.index_buffer->buffer) {
            current_index_buffer = mesh.index_buffer->buffer;
            vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
        }

        uint32_t offset = i * model_uniforms_alignment;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);

        vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}