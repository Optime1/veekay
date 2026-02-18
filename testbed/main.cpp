#include "veekay/input.hpp"
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>
#include <veekay/veekay.hpp>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>
#include <ranges>
#ifdef _WIN32
#include <windows.h>
#endif
constexpr uint32_t SHADOW_MAP_SIZE = 2048;
size_t aligned_sizeof;
enum class SamplerMode {
Repeat,
MirroredRepeat,
ClampToEdge,
ClampToBorder
};
namespace {
constexpr uint32_t MAX_MODELS = 1024;
constexpr uint32_t MAX_LIGHTS = 4;
struct Vertex {
veekay::vec3 position;
veekay::vec3 normal;
veekay::vec2 uv;
};
struct LightColors {
veekay::vec3 ambient;
float _pad0;
veekay::vec3 diffuse;
float _pad1;
veekay::vec3 specular;
float _pad2;
};
struct DirectionalLight {
veekay::vec3 direction;
float intensity;
LightColors colors;
};
struct PointLight {
veekay::vec3 position = {};
float _pad0;
LightColors colors = {
.ambient = {0.0f, 0.0f, 0.0f},
.diffuse = {1.0f, 1.0f, 1.0f},
.specular = {1.0f, 1.0f, 1.0f}
};
float linear = 0.09f;
float quadratic = 0.032f;
float _pad1;
float _pad2;
};
struct AmbientLight {
veekay::vec3 color;
float intensity;
veekay::vec3 specular_color;
float shininess;
};
struct alignas(16) SceneUniforms {
veekay::mat4 view_projection{};
veekay::mat4 light_view_projection{};
veekay::vec3 camera_position{};
uint32_t num_point_lights = 0;
uint32_t num_spot_lights = 0;
uint32_t _pad_align[3]{};
DirectionalLight directional_light{};
AmbientLight ambient_light{};
};
struct ShadowPushConstants {
veekay::mat4 model;
veekay::mat4 light_view_proj;
};
struct ModelUniforms {
veekay::mat4 model{};
veekay::vec3 albedo_color{};
float shininess = 10.0f;
veekay::vec3 specular_color{};
float _pad0{};
};
struct Mesh {
veekay::graphics::Buffer *vertex_buffer = nullptr;
veekay::graphics::Buffer *index_buffer = nullptr;
uint32_t indices = 0;
};
class Transform {
public:
veekay::vec3 position = {};
veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
veekay::vec3 rotation = {};
[[nodiscard]] veekay::mat4 matrix() const;
};
struct TextureAsset {
veekay::graphics::Texture *texture = nullptr;
std::string path;
};
struct Model {
Mesh mesh;
Transform transform;
veekay::vec3 albedo_color = {1.0f, 1.0f, 1.0f};
veekay::vec3 specular_color = {0.5f, 0.5f, 0.5f};
float shininess = 10.0f;
veekay::graphics::Texture *texture_ref = nullptr;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
SamplerMode sampler_mode = SamplerMode::Repeat;
};
class Camera {
public:
constexpr static float DEFAULT_FOV = 60.0f;
constexpr static float DEFAULT_NEAR_PLANE = 0.01f;
constexpr static float DEFAULT_FAR_PLANE = 100.0f;
veekay::vec3 position = {};
veekay::vec3 rotation = {};
float fov = DEFAULT_FOV;
float near_plane = DEFAULT_NEAR_PLANE;
float far_plane = DEFAULT_FAR_PLANE;
[[nodiscard]] veekay::mat4 view() const;
[[nodiscard]] veekay::mat4 view_projection(float aspect_ratio) const;
static float toRadians(float degrees) {
return degrees * static_cast<float>(M_PI) / 180.0f;
}
[[nodiscard]] veekay::vec3 getFront() const {
const float pitch = rotation.x;
const float yaw = rotation.y;
const veekay::vec3 front = {
std::cos(pitch) * std::sin(yaw),
std::sin(pitch),
-std::cos(pitch) * std::cos(yaw)
};
return veekay::vec3::normalized(front);
}
};
struct SpotLight {
PointLight point_light;
veekay::vec3 direction;
float cut_off = std::cos(Camera::toRadians(12.5f));
float outer_cut_off = std::cos(Camera::toRadians(15.0f));
float _pad1;
float _pad2;
};
class Renderer {
struct MeshPart {
Mesh mesh;
std::string material_name;
};
std::map<SamplerMode, VkSampler> samplers;
VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkShaderModule shadow_vertex_shader_module = VK_NULL_HANDLE;
VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
VkPipeline shadow_pipeline = VK_NULL_HANDLE;
VkRenderPass shadow_render_pass = VK_NULL_HANDLE;
VkFramebuffer shadow_framebuffer = VK_NULL_HANDLE;
VkImage shadow_image = VK_NULL_HANDLE;
VkDeviceMemory shadow_image_memory = VK_NULL_HANDLE;
VkImageView shadow_image_view = VK_NULL_HANDLE;
VkSampler shadow_sampler = VK_NULL_HANDLE;
veekay::graphics::Buffer *scene_uniforms_buffer = nullptr;
veekay::graphics::Buffer *model_uniforms_buffer = nullptr;
veekay::graphics::Buffer *point_lights_buffer = nullptr;
veekay::graphics::Buffer *spot_lights_buffer = nullptr;
Mesh plane_mesh;
Mesh cube_mesh;
Mesh cone_mesh;
veekay::graphics::Texture *missing_texture = nullptr;
VkSampler texture_sampler = VK_NULL_HANDLE;
std::vector<TextureAsset> loaded_textures;
veekay::graphics::Texture *default_texture = nullptr;
veekay::graphics::Texture *loadTexture(VkCommandBuffer cmd, const char *path);
static VkShaderModule loadShaderModule(const char *path);
void createMeshes(VkCommandBuffer cmd);
void createUniformsAndDescriptors(VkCommandBuffer cmd);
void allocateDescriptorsForModels(VkCommandBuffer cmd);
static VkSampler createSampler(VkSamplerAddressMode addressMode);
void createShadowResources();
void createShadowRenderPass();
void createShadowFramebuffer();
void createShadowPipeline();
static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
public:
Renderer() = default;
Renderer(const Renderer &) = delete;
Renderer &operator=(const Renderer &) = delete;
void initialize(VkCommandBuffer cmd);
void shutdown() const;
void update_uniforms(const Camera &camera, const std::vector<Model> &models, float aspect_ratio,
const std::vector<PointLight> &point_lights,
const std::vector<SpotLight> &spot_lights,
const DirectionalLight &dir_light,
const AmbientLight &ambient_light) const;
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer, const std::vector<Model> &models) const;
[[nodiscard]] const Mesh &getPlaneMesh() const { return plane_mesh; }
[[nodiscard]] const Mesh &getCubeMesh() const { return cube_mesh; }
[[nodiscard]] const Mesh &getConeMesh() const { return cone_mesh; }
void createDefaultTexture(VkCommandBuffer cmd);
};
namespace Scene {
Camera camera{
.position = {0.0f, 2.0f, -5.0f},
.rotation = {0.0f, 0.0f, 0.0f}
};
std::vector<Model> models;
Renderer renderer;
DirectionalLight dir_light{
.direction = {-0.5f, -1.0f, -0.1f},
.intensity = 1.0f,
.colors = {
.ambient = {0.2f, 0.2f, 0.2f},
.diffuse = {1.0f, 1.0f, 1.0f},
.specular = {1.0f, 1.0f, 1.0f}
}
};
AmbientLight ambient_light{
.color = {0.1f, 0.1f, 0.1f},
.intensity = 1.0f,
.specular_color = {0.0f, 0.0f, 0.0f},
.shininess = 8.0f,
};
std::vector<PointLight> point_lights = {};
std::vector<SpotLight> spot_lights = {};
}
veekay::mat4 Transform::matrix() const {
const auto s = veekay::mat4::scaling(scale);
const auto rot_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
const auto rot_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
const auto rot_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
const auto r = rot_z * rot_y * rot_x;
const auto t = veekay::mat4::translation(position);
return t * r * s;
}
veekay::mat4 Camera::view() const {
const veekay::vec3 front = getFront();
const veekay::vec3 target = position + front;
veekay::vec3 up = {0.0f, -1.0f, 0.0f}; // Изменено для переворота сцены
return veekay::mat4::lookAt(position, target, up);
}
veekay::mat4 Camera::view_projection(float aspect_ratio) const {
auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
return view() * projection;
}
VkShaderModule Renderer::loadShaderModule(const char *path) {
std::ifstream file(path, std::ios::binary | std::ios::ate);
if (!file.is_open()) {
return VK_NULL_HANDLE;
}
const size_t size = file.tellg();
std::vector<uint32_t> buffer(size / sizeof(uint32_t));
file.seekg(0);
file.read(reinterpret_cast<char *>(buffer.data()), size);
file.close();
VkShaderModuleCreateInfo info{
.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
.codeSize = size,
.pCode = buffer.data(),
};
VkShaderModule result;
if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
return VK_NULL_HANDLE;
}
return result;
}
void Renderer::createMeshes(VkCommandBuffer cmd) {
{
std::vector<Vertex> vertices = {
{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
{{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
{{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}},
{{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}},
};
std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
vertices.size() * sizeof(Vertex), vertices.data(),
VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
plane_mesh.index_buffer = new veekay::graphics::Buffer(
indices.size() * sizeof(uint32_t), indices.data(),
VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
plane_mesh.indices = static_cast<uint32_t>(indices.size());
}
{
std::vector<Vertex> vertices = {
{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
};
std::vector<uint32_t> indices = {
0, 1, 2, 2, 3, 0,
4, 5, 6, 6, 7, 4,
8, 9, 10, 10, 11, 8,
12, 13, 14, 14, 15, 12,
16, 17, 18, 18, 19, 16,
20, 21, 22, 22, 23, 20,
};
cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
vertices.size() * sizeof(Vertex), vertices.data(),
VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
cube_mesh.index_buffer = new veekay::graphics::Buffer(
indices.size() * sizeof(uint32_t), indices.data(),
VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
cube_mesh.indices = static_cast<uint32_t>(indices.size());
}
{
std::vector<Vertex> vertices;
std::vector<uint32_t> indices;
const float radius = 0.5f;
const float height = 1.0f;
const int segments = 32;
vertices.push_back({{0.0f, height, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.0f}});
for (int i = 0; i <= segments; ++i) {
float theta = 2.0f * 3.14159265359f * float(i) / float(segments);
float x = radius * std::cos(theta);
float z = radius * std::sin(theta);
veekay::vec3 side_normal = veekay::vec3::normalized({x, height / radius, z});
vertices.push_back({{x, 0.0f, z}, side_normal, {float(i) / segments, 1.0f}});
}
int base_center_index = vertices.size();
vertices.push_back({{0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});
for (int i = 0; i <= segments; ++i) {
float theta = 2.0f * 3.14159265359f * float(i) / float(segments);
float x = radius * std::cos(theta);
float z = radius * std::sin(theta);
vertices.push_back({{x, 0.0f, z}, {0.0f, -1.0f, 0.0f}, {float(i) / segments, 1.0f}});
}
for (int i = 1; i <= segments; ++i) {
indices.push_back(0);
indices.push_back(i);
indices.push_back(i + 1);
}
for (int i = 0; i < segments; ++i) {
indices.push_back(base_center_index);
indices.push_back(base_center_index + 1 + i + 1);
indices.push_back(base_center_index + 1 + i);
}
cone_mesh.vertex_buffer = new veekay::graphics::Buffer(
vertices.size() * sizeof(Vertex), vertices.data(),
VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
cone_mesh.index_buffer = new veekay::graphics::Buffer(
indices.size() * sizeof(uint32_t), indices.data(),
VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
cone_mesh.indices = static_cast<uint32_t>(indices.size());
}
}
uint32_t Renderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
VkPhysicalDeviceMemoryProperties memProperties;
vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);
for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
return i;
}
}
throw std::runtime_error("failed to find suitable memory type!");
}
void Renderer::createShadowResources() {
VkDevice device = veekay::app.vk_device;
VkImageCreateInfo imageInfo{
.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
.imageType = VK_IMAGE_TYPE_2D,
.format = VK_FORMAT_D32_SFLOAT,
.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
.mipLevels = 1,
.arrayLayers = 1,
.samples = VK_SAMPLE_COUNT_1_BIT,
.tiling = VK_IMAGE_TILING_OPTIMAL,
.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
};
if (vkCreateImage(device, &imageInfo, nullptr, &shadow_image) != VK_SUCCESS)
throw std::runtime_error("Failed to create shadow image");
VkMemoryRequirements memReqs;
vkGetImageMemoryRequirements(device, shadow_image, &memReqs);
VkMemoryAllocateInfo allocInfo{
.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
.allocationSize = memReqs.size,
.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
};
if (vkAllocateMemory(device, &allocInfo, nullptr, &shadow_image_memory) != VK_SUCCESS)
throw std::runtime_error("Failed to allocate shadow image memory");
vkBindImageMemory(device, shadow_image, shadow_image_memory, 0);
VkImageViewCreateInfo viewInfo{
.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
.image = shadow_image,
.viewType = VK_IMAGE_VIEW_TYPE_2D,
.format = VK_FORMAT_D32_SFLOAT,
.subresourceRange = {
.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
.baseMipLevel = 0,
.levelCount = 1,
.baseArrayLayer = 0,
.layerCount = 1
}
};
if (vkCreateImageView(device, &viewInfo, nullptr, &shadow_image_view) != VK_SUCCESS)
throw std::runtime_error("Failed to create shadow image view");
constexpr VkSamplerCreateInfo samplerInfo{
.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
.magFilter = VK_FILTER_LINEAR,
.minFilter = VK_FILTER_LINEAR,
.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
.mipLodBias = 0.0f,
.anisotropyEnable = VK_FALSE,
.compareEnable = VK_TRUE,
.compareOp = VK_COMPARE_OP_LESS,
.minLod = 0.0f,
.maxLod = 1.0f,
.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
};
if (vkCreateSampler(device, &samplerInfo, nullptr, &shadow_sampler) != VK_SUCCESS)
throw std::runtime_error("Failed to create shadow sampler");
}
void Renderer::createShadowRenderPass() {
VkDevice device = veekay::app.vk_device;
VkAttachmentDescription depthAttachment{};
depthAttachment.format = VK_FORMAT_D32_SFLOAT;
depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
VkAttachmentReference depthRef{};
depthRef.attachment = 0;
depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
VkSubpassDescription subpass{};
subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
subpass.pDepthStencilAttachment = &depthRef;
VkRenderPassCreateInfo renderPassInfo{};
renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
renderPassInfo.attachmentCount = 1;
renderPassInfo.pAttachments = &depthAttachment;
renderPassInfo.subpassCount = 1;
renderPassInfo.pSubpasses = &subpass;
if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadow_render_pass) != VK_SUCCESS)
throw std::runtime_error("Failed to create shadow render pass!");
}
void Renderer::createShadowFramebuffer() {
VkDevice device = veekay::app.vk_device;
VkFramebufferCreateInfo framebufferInfo{};
framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
framebufferInfo.renderPass = shadow_render_pass;
framebufferInfo.attachmentCount = 1;
framebufferInfo.pAttachments = &shadow_image_view;
framebufferInfo.width = SHADOW_MAP_SIZE;
framebufferInfo.height = SHADOW_MAP_SIZE;
framebufferInfo.layers = 1;
if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadow_framebuffer) != VK_SUCCESS)
throw std::runtime_error("Failed to create shadow framebuffer!");
}
void Renderer::createUniformsAndDescriptors(VkCommandBuffer cmd) {
VkDevice &device = veekay::app.vk_device;
VkPhysicalDeviceProperties props;
vkGetPhysicalDeviceProperties(veekay::app.vk_physical_device, &props);
uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;
scene_uniforms_buffer = new veekay::graphics::Buffer(
sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
model_uniforms_buffer = new veekay::graphics::Buffer(
MAX_MODELS * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
point_lights_buffer = new veekay::graphics::Buffer(
MAX_LIGHTS * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
spot_lights_buffer = new veekay::graphics::Buffer(
MAX_LIGHTS * sizeof(SpotLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
{
samplers[SamplerMode::Repeat] = createSampler(VK_SAMPLER_ADDRESS_MODE_REPEAT);
samplers[SamplerMode::MirroredRepeat] = createSampler(VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
samplers[SamplerMode::ClampToEdge] = createSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
samplers[SamplerMode::ClampToBorder] = createSampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
}
{
VkDescriptorPoolSize pools[] = {
{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_MODELS},
{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = MAX_MODELS},
{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = MAX_MODELS * 2},
{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_MODELS * 2}
};
VkDescriptorPoolCreateInfo info{
.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
.maxSets = MAX_MODELS, .poolSizeCount = std::size(pools), .pPoolSizes = pools,
};
if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS)
throw std::runtime_error("Failed to create Vulkan descriptor pool.");
}
{
VkDescriptorSetLayoutBinding bindings[] = {
{
.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
},
{
.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
},
{
.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
},
{
.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
},
{
.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
},
{
.binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
}
};
VkDescriptorSetLayoutCreateInfo info{
.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
.bindingCount = std::size(bindings), .pBindings = bindings,
};
if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
throw std::runtime_error("Failed to create Vulkan descriptor set layout.");
}
}
void Renderer::allocateDescriptorsForModels(VkCommandBuffer cmd) {
VkDevice &device = veekay::app.vk_device;
for (auto &model: Scene::models) {
VkDescriptorSetAllocateInfo alloc_info{
.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
.descriptorPool = descriptor_pool,
.descriptorSetCount = 1,
.pSetLayouts = &descriptor_set_layout,
};
if (vkAllocateDescriptorSets(device, &alloc_info, &model.descriptor_set) != VK_SUCCESS) {
throw std::runtime_error("Failed to allocate descriptor set!");
}
veekay::graphics::Texture *tex = model.texture_ref ? model.texture_ref : default_texture;
VkSampler selected_sampler = samplers[model.sampler_mode];
VkDescriptorBufferInfo buffer_infos[] = {
{.buffer = scene_uniforms_buffer->buffer, .offset = 0, .range = sizeof(SceneUniforms)},
{.buffer = model_uniforms_buffer->buffer, .offset = 0, .range = sizeof(ModelUniforms)},
{.buffer = point_lights_buffer->buffer, .offset = 0, .range = MAX_LIGHTS * sizeof(PointLight)},
{.buffer = spot_lights_buffer->buffer, .offset = 0, .range = MAX_LIGHTS * sizeof(SpotLight)},
};
VkDescriptorImageInfo image_info{
.sampler = selected_sampler,
.imageView = tex->view,
.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
};
VkDescriptorImageInfo shadow_info{
.sampler = shadow_sampler,
.imageView = shadow_image_view,
.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
};
VkWriteDescriptorSet writes[] = {
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 0,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
.pBufferInfo = &buffer_infos[0]
},
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 1,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
.pBufferInfo = &buffer_infos[1]
},
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 2,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
.pBufferInfo = &buffer_infos[2]
},
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 3,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
.pBufferInfo = &buffer_infos[3]
},
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 4,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
.pImageInfo = &image_info
},
{
.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = model.descriptor_set, .dstBinding = 5,
.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
.pImageInfo = &shadow_info
}
};
vkUpdateDescriptorSets(device, std::size(writes), writes, 0, nullptr);
}
}
void Renderer::createShadowPipeline() {
VkDevice device = veekay::app.vk_device;
shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
if (!shadow_vertex_shader_module) throw std::runtime_error("Failed to load shadow vertex shader");
VkPipelineShaderStageCreateInfo vertStage{
.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
.stage = VK_SHADER_STAGE_VERTEX_BIT,
.module = shadow_vertex_shader_module,
.pName = "main"
};
VkPushConstantRange pushConstant{
.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
.offset = 0,
.size = sizeof(ShadowPushConstants)
};
VkPipelineLayoutCreateInfo pipelineLayoutInfo{
.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
.setLayoutCount = 0,
.pushConstantRangeCount = 1,
.pPushConstantRanges = &pushConstant
};
vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &shadow_pipeline_layout);
VkVertexInputBindingDescription bindingDescription{
.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
};
VkVertexInputAttributeDescription attributeDescriptions[] = {
{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)}
};
VkPipelineVertexInputStateCreateInfo vertexInputInfo{
.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
.vertexBindingDescriptionCount = 1,
.pVertexBindingDescriptions = &bindingDescription,
.vertexAttributeDescriptionCount = 1,
.pVertexAttributeDescriptions = attributeDescriptions
};
VkPipelineInputAssemblyStateCreateInfo inputAssembly{
.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
};
VkPipelineViewportStateCreateInfo viewportState{
.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
.viewportCount = 1, .scissorCount = 1
};
VkPipelineRasterizationStateCreateInfo rasterizer{
.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
.polygonMode = VK_POLYGON_MODE_FILL,
.cullMode = VK_CULL_MODE_BACK_BIT,
.frontFace = VK_FRONT_FACE_CLOCKWISE,
.depthBiasEnable = VK_TRUE,
.depthBiasConstantFactor = 1.25f,
.depthBiasSlopeFactor = 1.75f,
.lineWidth = 1.0f
};
VkPipelineMultisampleStateCreateInfo multisampling{
.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
};
VkPipelineDepthStencilStateCreateInfo depthStencil{
.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
.depthTestEnable = VK_TRUE,
.depthWriteEnable = VK_TRUE,
.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
};
VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
VkPipelineDynamicStateCreateInfo dynamicState{
.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
.dynamicStateCount = 2,
.pDynamicStates = dynamicStates
};
VkGraphicsPipelineCreateInfo pipelineInfo{
.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
.pNext = nullptr,
.stageCount = 1,
.pStages = &vertStage,
.pVertexInputState = &vertexInputInfo,
.pInputAssemblyState = &inputAssembly,
.pViewportState = &viewportState,
.pRasterizationState = &rasterizer,
.pMultisampleState = &multisampling,
.pDepthStencilState = &depthStencil,
.pColorBlendState = nullptr,
.pDynamicState = &dynamicState,
.layout = shadow_pipeline_layout,
.renderPass = shadow_render_pass
};
if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadow_pipeline) != VK_SUCCESS)
throw std::runtime_error("failed to create shadow pipeline!");
}
void Renderer::initialize(VkCommandBuffer cmd) {
VkDevice &device = veekay::app.vk_device;
vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
if (!vertex_shader_module) throw std::runtime_error("Failed to load Vulkan vertex shader from file.");
fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
if (!fragment_shader_module) throw std::runtime_error("Failed to load Vulkan fragment shader from file.");
createShadowResources();
createShadowRenderPass();
createShadowFramebuffer();
createUniformsAndDescriptors(cmd);
createDefaultTexture(cmd);
createMeshes(cmd);
createShadowPipeline();
VkPipelineShaderStageCreateInfo stage_infos[2];
stage_infos[0] = {
.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
.module = vertex_shader_module, .pName = "main"
};
stage_infos[1] = {
.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
.module = fragment_shader_module, .pName = "main"
};
VkVertexInputBindingDescription buffer_binding{
.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
};
VkVertexInputAttributeDescription attributes[] = {
{
.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
.offset = offsetof(Vertex, position)
},
{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv)},
};
VkPipelineVertexInputStateCreateInfo input_state_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
.pVertexBindingDescriptions = &buffer_binding, .vertexAttributeDescriptionCount = std::size(attributes),
.pVertexAttributeDescriptions = attributes
};
VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
};
VkPipelineRasterizationStateCreateInfo raster_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
.polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
.frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f
};
VkPipelineMultisampleStateCreateInfo sample_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, .sampleShadingEnable = false, .minSampleShading = 1.0f
};
VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
VkPipelineDynamicStateCreateInfo dynamic_state_info = {
.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
.dynamicStateCount = std::size(dynamic_states),
.pDynamicStates = dynamic_states
};
VkPipelineViewportStateCreateInfo viewport_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1
};
VkPipelineDepthStencilStateCreateInfo depth_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = true,
.depthWriteEnable = true, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
};
VkPipelineColorBlendAttachmentState attachment_info{
.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
VK_COLOR_COMPONENT_A_BIT
};
VkPipelineColorBlendStateCreateInfo blend_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .logicOpEnable = false,
.logicOp = VK_LOGIC_OP_COPY, .attachmentCount = 1, .pAttachments = &attachment_info
};
VkPipelineLayoutCreateInfo layout_info{
.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
.pSetLayouts = &descriptor_set_layout
};
if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
throw std::runtime_error("Failed to create Vulkan pipeline layout.");
VkGraphicsPipelineCreateInfo info{
.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stage_infos,
.pVertexInputState = &input_state_info, .pInputAssemblyState = &assembly_state_info,
.pViewportState = &viewport_info,
.pRasterizationState = &raster_info, .pMultisampleState = &sample_info,
.pDepthStencilState = &depth_info,
.pColorBlendState = &blend_info, .pDynamicState = &dynamic_state_info,
.layout = pipeline_layout, .renderPass = veekay::app.vk_render_pass,
};
if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
throw std::runtime_error("Failed to create Vulkan pipeline.");
Scene::models.push_back(Model{
.mesh = getCubeMesh(),
.transform = Transform{.position = {-1.0f, 0.5f, 0.0f}, .scale = {1.0f, 1.0f, 1.0f}},
.albedo_color = veekay::vec3{1.0f, 0.2f, 0.2f},
.specular_color = veekay::vec3{0.5f, 0.5f, 0.5f},
.shininess = 32.0f,
.texture_ref = nullptr,
.sampler_mode = SamplerMode::Repeat,
});
Scene::models.push_back(Model{
.mesh = getConeMesh(),
.transform = Transform{.position = {1.0f, 0.5f, 0.0f}, .scale = {1.0f, 1.0f, 1.0f}},
.albedo_color = veekay::vec3{0.2f, 0.2f, 1.0f},
.specular_color = veekay::vec3{0.5f, 0.5f, 0.5f},
.shininess = 32.0f,
.texture_ref = nullptr,
.sampler_mode = SamplerMode::Repeat,
});
Scene::models.emplace_back(Model{
.mesh = getPlaneMesh(),
.transform = Transform{},
.albedo_color = veekay::vec3{0.3f, 0.3f, 0.3f},
.specular_color = veekay::vec3{0.05f, 0.05f, 0.05f},
.shininess = 32.0f,
.texture_ref = nullptr,
.sampler_mode = SamplerMode::Repeat,
});
allocateDescriptorsForModels(cmd);
}
void Renderer::shutdown() const {
VkDevice &device = veekay::app.vk_device;
vkDestroyFramebuffer(device, shadow_framebuffer, nullptr);
vkDestroyRenderPass(device, shadow_render_pass, nullptr);
vkDestroyImageView(device, shadow_image_view, nullptr);
vkFreeMemory(device, shadow_image_memory, nullptr);
vkDestroyImage(device, shadow_image, nullptr);
vkDestroySampler(device, shadow_sampler, nullptr);
vkDestroyPipeline(device, shadow_pipeline, nullptr);
vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
vkDestroySampler(device, texture_sampler, nullptr);
delete default_texture;
for (auto &asset: loaded_textures) {
delete asset.texture;
}
for (const auto &sampler: samplers | std::views::values) {
vkDestroySampler(device, sampler, nullptr);
}
delete missing_texture;
for (const auto &model: Scene::models) {
bool is_plane = (model.mesh.vertex_buffer == plane_mesh.vertex_buffer);
bool is_cube = (model.mesh.vertex_buffer == cube_mesh.vertex_buffer);
bool is_cone = (model.mesh.vertex_buffer == cone_mesh.vertex_buffer);
if (!is_plane && !is_cube && !is_cone) {
delete model.mesh.index_buffer;
delete model.mesh.vertex_buffer;
}
}
delete cone_mesh.index_buffer;
delete cone_mesh.vertex_buffer;
delete cube_mesh.index_buffer;
delete cube_mesh.vertex_buffer;
delete plane_mesh.index_buffer;
delete plane_mesh.vertex_buffer;
delete spot_lights_buffer;
delete point_lights_buffer;
delete model_uniforms_buffer;
delete scene_uniforms_buffer;
vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
vkDestroyPipeline(device, pipeline, nullptr);
vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
vkDestroyShaderModule(device, fragment_shader_module, nullptr);
vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}
veekay::graphics::Texture *Renderer::loadTexture(VkCommandBuffer cmd, const char *path) {
for (auto &asset: loaded_textures) {
if (asset.path == path) return asset.texture;
}
std::vector<unsigned char> pixels;
unsigned int width, height;
if (unsigned error = lodepng::decode(pixels, width, height, path)) {
return default_texture;
}
auto *new_tex = new veekay::graphics::Texture(
cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data()
);
loaded_textures.push_back({new_tex, path});
return new_tex;
}
void Renderer::createDefaultTexture(VkCommandBuffer cmd) {
uint32_t pixels[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
default_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
}
VkSampler Renderer::createSampler(VkSamplerAddressMode addressMode) {
VkSamplerCreateInfo info{
.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
.magFilter = VK_FILTER_LINEAR,
.minFilter = VK_FILTER_LINEAR,
.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
.addressModeU = addressMode,
.addressModeV = addressMode,
.addressModeW = addressMode,
.anisotropyEnable = VK_TRUE,
.maxAnisotropy = 16.0f,
.minLod = 0.0f,
.maxLod = VK_LOD_CLAMP_NONE,
};
VkSampler sampler;
if (vkCreateSampler(veekay::app.vk_device, &info, nullptr, &sampler) != VK_SUCCESS) {
throw std::runtime_error("Failed to create sampler");
}
return sampler;
}
void Renderer::update_uniforms(const Camera &camera, const std::vector<Model> &models, float aspect_ratio,
const std::vector<PointLight> &point_lights,
const std::vector<SpotLight> &spot_lights,
const DirectionalLight &dir_light,
const AmbientLight &ambient_light) const {
veekay::vec3 lightDir = veekay::vec3::normalized(dir_light.direction);
veekay::vec3 lightPos = -lightDir * 20.0f;
veekay::mat4 lightView = veekay::mat4::lookAt(lightPos, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
float orthoSize = 10.0f;
veekay::mat4 lightProj = veekay::mat4::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, 50.0f);
veekay::mat4 lightVP = lightView * lightProj;
SceneUniforms scene_uniforms{
.view_projection = camera.view_projection(aspect_ratio),
.light_view_projection = lightVP,
.camera_position = camera.position,
.num_point_lights = static_cast<uint32_t>(point_lights.size()),
.num_spot_lights = static_cast<uint32_t>(spot_lights.size()),
._pad_align = {0, 0, 0},
.directional_light = dir_light,
.ambient_light = ambient_light
};
std::vector<ModelUniforms> model_uniforms(models.size());
for (size_t i = 0; i < models.size(); ++i) {
const Model &model = models[i];
ModelUniforms &uniforms = model_uniforms[i];
uniforms.model = model.transform.matrix();
uniforms.albedo_color = model.albedo_color;
uniforms.specular_color = model.specular_color;
uniforms.shininess = model.shininess;
}
if (scene_uniforms_buffer->mapped_region)
*static_cast<SceneUniforms *>(scene_uniforms_buffer->mapped_region) = scene_uniforms;
if (model_uniforms_buffer->mapped_region) {
auto* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
for (size_t i = 0; i < models.size(); ++i) {
memcpy(base + i * aligned_sizeof, &model_uniforms[i], sizeof(ModelUniforms));
}
}
if (point_lights_buffer->mapped_region)
std::ranges::copy(point_lights,
static_cast<PointLight *>(point_lights_buffer->mapped_region));
if (spot_lights_buffer->mapped_region)
std::ranges::copy(spot_lights,
static_cast<SpotLight *>(spot_lights_buffer->mapped_region));
}
void Renderer::render(VkCommandBuffer cmd, VkFramebuffer framebuffer, const std::vector<Model> &models) const {
vkResetCommandBuffer(cmd, 0);
VkCommandBufferBeginInfo info{
.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
};
vkBeginCommandBuffer(cmd, &info);
auto* scene_data = static_cast<SceneUniforms*>(scene_uniforms_buffer->mapped_region);
veekay::mat4 lightVP = scene_data->light_view_projection;
{
VkClearValue clearDepth{};
clearDepth.depthStencil = {1.0f, 0};
VkRenderPassBeginInfo renderPassInfo{};
renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
renderPassInfo.renderPass = shadow_render_pass;
renderPassInfo.framebuffer = shadow_framebuffer;
renderPassInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
renderPassInfo.clearValueCount = 1;
renderPassInfo.pClearValues = &clearDepth;
vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
VkViewport viewport{0, 0, static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE), 0.0f, 1.0f};
VkRect2D scissor{{0,0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
vkCmdSetViewport(cmd, 0, 1, &viewport);
vkCmdSetScissor(cmd, 0, 1, &scissor);
VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
VkBuffer current_index_buffer = VK_NULL_HANDLE;
VkDeviceSize zero_offset = 0;
for (const auto& model : models) {
if (current_vertex_buffer != model.mesh.vertex_buffer->buffer) {
current_vertex_buffer = model.mesh.vertex_buffer->buffer;
vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
}
if (current_index_buffer != model.mesh.index_buffer->buffer) {
current_index_buffer = model.mesh.index_buffer->buffer;
vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
}
ShadowPushConstants push{
.model = model.transform.matrix(),
.light_view_proj = lightVP
};
vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstants), &push);
vkCmdDrawIndexed(cmd, model.mesh.indices, 1, 0, 0, 0);
}
vkCmdEndRenderPass(cmd);
}
{
VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
VkClearValue clear_values[] = {clear_color, clear_depth};
VkRenderPassBeginInfo begin_info{
.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
.renderPass = veekay::app.vk_render_pass,
.framebuffer = framebuffer,
.renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
.clearValueCount = std::size(clear_values),
.pClearValues = clear_values,
};
vkCmdBeginRenderPass(cmd, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
}
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
VkViewport viewport{
.x = 0.0f, .y = 0.0f, .width = static_cast<float>(veekay::app.window_width),
.height = static_cast<float>(veekay::app.window_height),
.minDepth = 0.0f, .maxDepth = 1.0f,
};
VkRect2D scissor{
.offset = {0, 0}, .extent = {veekay::app.window_width, veekay::app.window_height},
};
vkCmdSetViewport(cmd, 0, 1, &viewport);
vkCmdSetScissor(cmd, 0, 1, &scissor);
VkDeviceSize zero_offset = 0;
VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
VkBuffer current_index_buffer = VK_NULL_HANDLE;
for (size_t i = 0, n = models.size(); i < n; ++i) {
const Model &model = models[i];
const Mesh &mesh = model.mesh;
if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
current_vertex_buffer = mesh.vertex_buffer->buffer;
vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
}
if (current_index_buffer != mesh.index_buffer->buffer) {
current_index_buffer = mesh.index_buffer->buffer;
vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
}
auto offset = static_cast<uint32_t>(i * aligned_sizeof);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
0, 1, &model.descriptor_set, 1, &offset);
vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
}
vkCmdEndRenderPass(cmd);
vkEndCommandBuffer(cmd);
}
void initialize(VkCommandBuffer cmd) {
try {
Scene::renderer.initialize(cmd);
} catch (const std::exception& e) {
veekay::app.running = false;
}
}
void shutdown() {
Scene::renderer.shutdown();
}
void update(double time) {
ImGui::Begin("Controls:");
ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", Scene::camera.position.x, Scene::camera.position.y,
Scene::camera.position.z);
ImGui::Separator();
if (ImGui::CollapsingHeader("Directional Light")) {
ImGui::SliderFloat3("Dir Direction", &Scene::dir_light.direction.x, -1.0f, 1.0f);
ImGui::SliderFloat("Intensity", &Scene::dir_light.intensity, 0.0f, 10.0f);
ImGui::ColorEdit3("Dir Ambient", &Scene::dir_light.colors.ambient.x);
ImGui::ColorEdit3("Dir Diffuse", &Scene::dir_light.colors.diffuse.x);
ImGui::ColorEdit3("Dir Specular", &Scene::dir_light.colors.specular.x);
}
if (ImGui::CollapsingHeader("Ambient Light")) {
ImGui::ColorEdit3("Ambient Color", &Scene::ambient_light.color.x);
ImGui::SliderFloat("Ambient Intensity", &Scene::ambient_light.intensity, 0.0f, 5.0f);
ImGui::ColorEdit3("Ambient Specular Color", &Scene::ambient_light.specular_color.x);
ImGui::SliderFloat("Ambient Shininess", &Scene::ambient_light.shininess, 1.0f, 256.0f);
}
ImGui::End();
if (!ImGui::IsWindowHovered()) {
using namespace veekay::input;
Camera &camera = Scene::camera;
constexpr float move_speed = 0.1f;
if (mouse::isButtonDown(mouse::Button::left)) {
constexpr float sensitivity = 0.003f;
auto move_delta = mouse::cursorDelta();
camera.rotation.y += move_delta[0] * sensitivity;
camera.rotation.x += move_delta[1] * sensitivity;
}
const veekay::vec3 front = camera.getFront();
veekay::vec3 up = {0.0f, 1.0f, 0.0f};
const veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, up));
up = veekay::vec3::normalized(veekay::vec3::cross(right, front));
if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * move_speed;
if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * move_speed;
if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * move_speed;
if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * move_speed;
if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * move_speed;
if (keyboard::isKeyDown(keyboard::Key::z)) camera.position += up * move_speed;
}
float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
Scene::renderer.update_uniforms(Scene::camera, Scene::models, aspect_ratio,
Scene::point_lights, Scene::spot_lights,
Scene::dir_light, Scene::ambient_light);
}
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
Scene::renderer.render(cmd, framebuffer, Scene::models);
}
}
int main() {
return veekay::run({
.init = initialize,
.shutdown = shutdown,
.update = update,
.render = render,
});
}