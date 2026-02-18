cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(veekay LANGUAGES C CXX)

add_library(${PROJECT_NAME} source/veekay.cpp source/input.cpp source/graphics.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
	$<BUILD_INTERFACE:${veekay_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

target_compile_options(${PROJECT_NAME} PRIVATE PRIVATE /W4)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG 3.4
)

FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
	GIT_TAG v1.4.321
)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui
	GIT_TAG v1.92.3
)

FetchContent_Declare(
		tinyobjloader
		GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
		GIT_TAG v2.0.0rc13
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(glfw vk-bootstrap imgui tinyobjloader)

add_subdirectory(testbed)

target_link_libraries(${PROJECT_NAME} PRIVATE
	glfw
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
)

target_include_directories(${PROJECT_NAME} PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_sources(${PROJECT_NAME} PRIVATE
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${tinyobjloader_SOURCE_DIR})

target_sources(${PROJECT_NAME} PRIVATE ${tinyobjloader_SOURCE_DIR}/tiny_obj_loader.cc)