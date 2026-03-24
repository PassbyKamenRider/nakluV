#include "S72Loader.hpp"

#include "../VK.hpp"

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb_image.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>

void S72Loader::traverse_scene(S72::Node *node, glm::mat4 const &parent_from_world)
{
	glm::mat4 world_matrix = parent_from_world * (glm::translate(glm::mat4(1.0f), node->translation) *
												  glm::mat4_cast(node->rotation) *
												  glm::scale(glm::mat4(1.0f), node->scale));

	if (node->mesh)
	{
		auto it = mesh_data.find(node->mesh->name);
		if (it != mesh_data.end())
		{
			ObjectInstance inst;
			inst.vertices = it->second;
			inst.transform.WORLD_FROM_LOCAL = world_matrix;
			inst.transform.WORLD_FROM_LOCAL_NORMAL = glm::transpose(glm::inverse(world_matrix));

			auto mat_it = material_data.find(node->mesh->material->name);
			if (mat_it != material_data.end()) {
				inst.material = &mat_it->second;

				auto env_it = environment_data.end();
				
				if (node->environment) {
					env_it = environment_data.find(node->environment->name);
				} else if (!environment_data.empty()) {
					env_it = environment_data.begin();
				}

				if (env_it != environment_data.end()) {
					inst.environment_cube_set = env_it->second.env_descriptors; 
				}
			}

			inst.transform.MATERIAL_TYPE.materialType = inst.material->type;

			if (rtg.configuration.culling)
			{
				glm::mat4 view_from_world = glm::inverse(active_camera->world_from_local);
				glm::mat4 vs_transform = view_from_world * world_matrix;

				if (SAT_visibility_test(active_frustum, vs_transform, it->second.aabb))
				{
					object_instances.emplace_back(inst);
				}
			}
			else
			{
				object_instances.emplace_back(inst);
			}

			if (isDebugMode)
			{
				auto aabb = it->second.aabb;
				glm::vec3 local_corners[8] = {
					{aabb.min.x, aabb.min.y, aabb.min.z},
					{aabb.max.x, aabb.min.y, aabb.min.z},
					{aabb.min.x, aabb.max.y, aabb.min.z},
					{aabb.max.x, aabb.max.y, aabb.min.z},
					{aabb.min.x, aabb.min.y, aabb.max.z},
					{aabb.max.x, aabb.min.y, aabb.max.z},
					{aabb.min.x, aabb.max.y, aabb.max.z},
					{aabb.max.x, aabb.max.y, aabb.max.z}};

				glm::vec3 world_corners[8];
				for (int i = 0; i < 8; ++i)
				{
					world_corners[i] = glm::vec3(world_matrix * glm::vec4(local_corners[i], 1.0f));
				}

				static const uint32_t indices[] = {
					0, 1, 1, 3, 3, 2, 2, 0,
					4, 5, 5, 7, 7, 6, 6, 4,
					0, 4, 1, 5, 2, 6, 3, 7};

				for (uint32_t idx : indices)
				{
					PosColVertex v;
					v.Position = {world_corners[idx].x, world_corners[idx].y, world_corners[idx].z};
					v.Color = {0, 255, 0, 255};
					lines_vertices.push_back(v);
				}
			}
		}
	}

	if (node->camera)
	{
		auto cam_it = cameras.find(node->camera->name);
		if (cam_it != cameras.end())
		{
			cam_it->second.world_from_local = world_matrix;
		}

		if (isDebugMode)
		{
			auto const &p_data = std::get<S72::Camera::Perspective>(node->camera->projection);

			glm::mat4 P = glm::perspective(p_data.vfov, p_data.aspect, p_data.near, p_data.far);

			glm::mat4 WORLD_FROM_CLIP = world_matrix * glm::inverse(P);

			std::array<glm::vec4, 8> ndc_corners = {
				glm::vec4{-1, -1, 0, 1}, glm::vec4{1, -1, 0, 1}, glm::vec4{-1, 1, 0, 1}, glm::vec4{1, 1, 0, 1},
				glm::vec4{-1, -1, 1, 1}, glm::vec4{1, -1, 1, 1}, glm::vec4{-1, 1, 1, 1}, glm::vec4{1, 1, 1, 1}};

			std::array<glm::vec3, 8> world_corners;
			for (int i = 0; i < 8; ++i)
			{
				glm::vec4 p = WORLD_FROM_CLIP * ndc_corners[i];
				world_corners[i] = glm::vec3(p) / p.w;
			}

			static const uint32_t indices[] = {
				0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7};

			for (uint32_t idx : indices)
			{
				PosColVertex v;
				v.Position = {world_corners[idx].x, world_corners[idx].y, world_corners[idx].z};
				v.Color = {255, 255, 0, 255};
				lines_vertices.push_back(v);
			}
		}
	}

	if (node->light) {
		auto const &light = *(node->light);

		if (std::holds_alternative<S72::Light::Sun>(light.source)) {
			glm::vec3 direction = glm::normalize(glm::vec3(world_matrix[2]));
			auto const &sun = std::get<S72::Light::Sun>(light.source);
			glm::vec3 energy = glm::vec3(light.tint.r, light.tint.g, light.tint.b) * sun.strength;

			if (sun.angle > 3.0f) {
				world.SKY_DIRECTION.x = direction.x;
				world.SKY_DIRECTION.y = direction.y;
				world.SKY_DIRECTION.z = direction.z;
				world.SKY_ENERGY.r = energy.r;
				world.SKY_ENERGY.g = energy.g;
				world.SKY_ENERGY.b = energy.b;
			} else {
				world.SUN_DIRECTION.x = direction.x;
				world.SUN_DIRECTION.y = direction.y;
				world.SUN_DIRECTION.z = direction.z;
				world.SUN_ENERGY.r = energy.r;
				world.SUN_ENERGY.g = energy.g;
				world.SUN_ENERGY.b = energy.b;
			}
		} else {
			glm::vec3 tint = glm::vec3(light.tint.r, light.tint.g, light.tint.b);

			if (auto const* sphere = std::get_if<S72::Light::Sphere>(&light.source)) {
				LightInstance inst = {};
				inst.world_from_local = world_matrix;
				inst.tint_shadow = glm::vec4(tint, (float)light.shadow);
				inst.type_power = glm::vec4(1.0f, sphere->power, 0.0f, 0.0f);
				inst.params = glm::vec4(sphere->radius, sphere->limit, 0.0f, 0.0f);
				light_data.push_back(inst);

			} else if (auto const* spot = std::get_if<S72::Light::Spot>(&light.source)) {
				float half_fov = spot->fov * 0.5f;
				float cosOuter = std::cos(half_fov);
				float cosInner = std::cos(half_fov * (1.0f - spot->blend));

				if (light.shadow > 0) {
					ShadowLightInstance inst = {};
					inst.world_from_local = world_matrix;
					inst.tint_shadow = glm::vec4(tint, (float)light.shadow);
					inst.type_power = glm::vec4(2.0f, spot->power, 0.0f, 0.0f);
					inst.params = glm::vec4(spot->radius, spot->limit, cosInner, cosOuter);

					float far_dist = std::isfinite(spot->limit) ? spot->limit : 100.0f;
					glm::vec3 light_pos = glm::vec3(world_matrix[3]);
					glm::vec3 light_dir = -glm::normalize(glm::vec3(world_matrix[2]));
					glm::vec3 up = (std::abs(glm::dot(light_dir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f)
						? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
					glm::mat4 view = glm::lookAt(light_pos, light_pos + light_dir, up);
					glm::mat4 proj = glm::perspective(spot->fov, 1.0f, 0.5f, far_dist);
					inst.light_VP = proj * view;

					shadow_light_data.push_back(inst);
				} else {
					LightInstance inst = {};
					inst.world_from_local = world_matrix;
					inst.tint_shadow = glm::vec4(tint, 0.0f);
					inst.type_power = glm::vec4(2.0f, spot->power, 0.0f, 0.0f);
					inst.params = glm::vec4(spot->radius, spot->limit, cosInner, cosOuter);
					light_data.push_back(inst);
				}
			}
		}
	}

	for (auto *child : node->children)
	{
		traverse_scene(child, world_matrix);
	}
}

uint32_t S72Loader::create_1x1_texture(float r, float g, float b, float a, bool is_srgb)
{
    auto toUint8 = [](float val) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(val, 0.0f, 1.0f) * 255.0f + 0.5f);
    };

    uint8_t pixel[4] = { toUint8(r), toUint8(g), toUint8(b), toUint8(a) };

    VkFormat vk_format = VK_FORMAT_R8G8B8A8_UNORM;

    textures.emplace_back(rtg.helpers.create_image(
        {1, 1}, 
        vk_format, 
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        Helpers::Unmapped
    ));
    
    rtg.helpers.transfer_to_image(pixel, 4, textures.back());
    
    return static_cast<uint32_t>(textures.size() - 1);
}

uint32_t S72Loader::load_texture(std::string path, S72::Texture::Format format)
{
    int width, height, channels;
    stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    
    if (!pixels) {
        throw std::runtime_error("Failed to load texture file: " + path);
    }

    uint32_t img_w = uint32_t(width);
    uint32_t img_h = uint32_t(height);

    VkFormat vk_format = (format == S72::Texture::Format::srgb) 
                         ? VK_FORMAT_R8G8B8A8_SRGB 
                         : VK_FORMAT_R8G8B8A8_UNORM;

    textures.emplace_back(
        rtg.helpers.create_image(
            VkExtent2D{.width = img_w, .height = img_h},
            vk_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped));

    rtg.helpers.transfer_to_image(pixels, img_w * img_h * 4, textures.back());

    stbi_image_free(pixels);

    VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = textures.back().handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = vk_format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    return (uint32_t)(textures.size() - 1);
}

inline glm::vec3 rgbe_to_float(glm::u8vec4 col) {
    if (col == glm::u8vec4(0,0,0,0)) return glm::vec3(0.0f);

    int exp = int(col.a) - 128;
    return glm::vec3(
        std::ldexp((col.r + 0.5f) / 256.0f, exp),
        std::ldexp((col.g + 0.5f) / 256.0f, exp),
        std::ldexp((col.b + 0.5f) / 256.0f, exp)
    );
}

uint32_t S72Loader::load_cubemap(std::string path, uint32_t max_lod) {
    int width, height, channels;
    stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Failed to load cubemap at " << path << std::endl;
        return 0;
    }

    uint32_t mipLevels = max_lod + 1;
    uint32_t img_w = uint32_t(width);
    uint32_t img_h = uint32_t(height / 6);

    textures_cube.emplace_back(
        rtg.helpers.create_image(
            VkExtent2D{.width = img_w, .height = img_h},
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped,
            6,
            VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
            mipLevels
        )
    );

    size_t dot = path.find_last_of('.');
    std::string prefix = path.substr(0, dot);
    std::string ext = path.substr(dot);

    for (uint32_t level = 0; level < mipLevels; ++level) {
        stbi_uc* current_pixels = nullptr;
        int cur_w, cur_h, cur_c;

        if (level == 0) {
            current_pixels = pixels;
            cur_w = width;
            cur_h = height;
        } else {
            std::string level_path = prefix + "." + std::to_string(level) + ext;
            current_pixels = stbi_load(level_path.c_str(), &cur_w, &cur_h, &cur_c, STBI_rgb_alpha);
            if (!current_pixels) break; 
        }

        std::vector<glm::vec4> float_pixels;
        float_pixels.reserve(cur_w * cur_h);

        for (int i = 0; i < cur_w * cur_h; ++i) {
            glm::u8vec4 rgbe(
                current_pixels[i * 4 + 0],
                current_pixels[i * 4 + 1],
                current_pixels[i * 4 + 2],
                current_pixels[i * 4 + 3]
            );

            glm::vec3 color = rgbe_to_float(rgbe);
            
            float_pixels.emplace_back(color.r, color.g, color.b, 1.0f);
        }

        rtg.helpers.transfer_to_image(
            float_pixels.data(), 
            float_pixels.size() * sizeof(glm::vec4), 
            textures_cube.back(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            level 
        );

        if (level > 0) stbi_image_free(current_pixels);
    }
    stbi_image_free(pixels);

    return (uint32_t)(textures_cube.size() - 1);
}

void S72Loader::update_animations(float current_time)
{
	for (auto &driver : scene.drivers)
	{
		uint32_t left_idx = 0;
		while (left_idx + 1 < driver.times.size() && driver.times[left_idx + 1] <= current_time)
		{
			left_idx++;
		}
		uint32_t right_idx = (left_idx + 1 < driver.times.size()) ? left_idx + 1 : left_idx;

		float alpha = 0.0f;
		if (left_idx != right_idx)
		{
			alpha = (current_time - driver.times[left_idx]) / (driver.times[right_idx] - driver.times[left_idx]);
		}

		if (driver.interpolation == S72::Driver::Interpolation::STEP)
			alpha = 0.0f;

		if (driver.channel == S72::Driver::Channel::translation)
		{
			glm::vec3 v1 = glm::make_vec3(driver.values.data() + (3 * left_idx));
			glm::vec3 v2 = glm::make_vec3(driver.values.data() + (3 * right_idx));
			driver.node.translation = glm::mix(v1, v2, alpha);
		}
		else if (driver.channel == S72::Driver::Channel::scale)
		{
			glm::vec3 v1 = glm::make_vec3(driver.values.data() + (3 * left_idx));
			glm::vec3 v2 = glm::make_vec3(driver.values.data() + (3 * right_idx));
			driver.node.scale = glm::mix(v1, v2, alpha);
		}
		else if (driver.channel == S72::Driver::Channel::rotation)
		{
			glm::quat q1 = glm::make_quat(driver.values.data() + (4 * left_idx));
			glm::quat q2 = glm::make_quat(driver.values.data() + (4 * right_idx));

			if (driver.interpolation == S72::Driver::Interpolation::SLERP)
			{
				driver.node.rotation = glm::slerp(q1, q2, alpha);
			}
			else
			{
				driver.node.rotation = glm::normalize(glm::lerp(q1, q2, alpha));
			}
		}
	}
}

S72Loader::S72Loader(RTG &rtg_) : rtg(rtg_)
{
	try
	{
		scene = S72::load(rtg_.configuration.scene_path);
	}
	catch (std::exception &e)
	{
		std::cerr << "Scene loading failed:\n"
				  << e.what() << std::endl;
	}

	{ // make some textures
		textures.reserve(1);

		{ // texture 0 will be the default albedo map
			uint32_t size = 1;
			uint8_t pixel_data[] = { 127, 127, 216, 255 };

			// make a place for the texture to live on the GPU:
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{.width = size, .height = size},
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped));

			// transfer data:
			rtg.helpers.transfer_to_image(pixel_data, sizeof(pixel_data), textures.back());
		}

		{ // texture 1 will be the default normal map
			uint8_t normal_pixel[] = { 128, 128, 255, 255 };

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{1, 1},
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped));

			rtg.helpers.transfer_to_image(normal_pixel, sizeof(normal_pixel), textures.back());
		}

		{ // texture 2 will be the default roughness map
			uint8_t normal_pixel[] = { 255, 255, 255, 255 };

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{1, 1},
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped));

			rtg.helpers.transfer_to_image(normal_pixel, sizeof(normal_pixel), textures.back());
		}

		{ // texture 3 will be the default metalness map
			uint8_t normal_pixel[] = { 0, 0, 0, 255 };

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{1, 1},
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped));

			rtg.helpers.transfer_to_image(normal_pixel, sizeof(normal_pixel), textures.back());
		}
	}

	// load data from S72
	std::unordered_map<std::string, std::vector<char>> loaded_data;
	for (auto const &[src, datafile] : scene.data_files)
	{
		std::ifstream file(datafile.path, std::ios::binary | std::ios::ate);
		if (!file)
			throw std::runtime_error("Could not open data file: " + datafile.path);
		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);
		std::vector<char> data(size);
		file.read(data.data(), size);
		loaded_data[src] = std::move(data);
	}

	for (auto &[name, cam] : scene.cameras)
	{
		CameraInstance inst;
		inst.props = std::get<S72::Camera::Perspective>(cam.projection);
		cameras[name] = inst;
		camera_list.push_back(&cameras[name]);

		if (name == rtg.configuration.camera_name)
		{
			active_camera = &cameras[name];
		}
	}

	if (active_camera == nullptr)
	{
		camera_mode = CameraMode::Free;
		active_camera = camera_list.empty() ? nullptr : camera_list[0];
	}

	std::vector<PosNorTanTexVertex> vertices;
	for (auto const &[name, mesh] : scene.meshes)
	{
		ObjectVertices info;
		info.first = (uint32_t)(vertices.size());
		info.count = mesh.count;
		info.aabb.min = glm::vec3(std::numeric_limits<float>::infinity());
		info.aabb.max = glm::vec3(-std::numeric_limits<float>::infinity());

		vertices.resize(info.first + info.count);

		if (auto it = mesh.attributes.find("POSITION"); it != mesh.attributes.end())
		{
			auto const &attr = it->second;
			const char *src_ptr = loaded_data.at(attr.src.src).data() + attr.offset;

			for (uint32_t i = 0; i < mesh.count; ++i)
			{
				glm::vec3 pos;
				std::memcpy(&pos, src_ptr + i * attr.stride, sizeof(glm::vec3));

				vertices[info.first + i].Position = {pos.x, pos.y, pos.z};

				info.aabb.min = glm::min(info.aabb.min, pos);
				info.aabb.max = glm::max(info.aabb.max, pos);
			}
		}

		if (auto it = mesh.attributes.find("NORMAL"); it != mesh.attributes.end())
		{
			auto const &attr = it->second;
			const char *src_ptr = loaded_data.at(attr.src.src).data() + attr.offset;
			for (uint32_t i = 0; i < mesh.count; ++i)
			{
				std::memcpy(&vertices[info.first + i].Normal, src_ptr + i * attr.stride, sizeof(glm::vec3));
			}
		}

		if (auto it = mesh.attributes.find("TANGENT"); it != mesh.attributes.end())
        {
            auto const &attr = it->second;
            const char *src_ptr = loaded_data.at(attr.src.src).data() + attr.offset;
            for (uint32_t i = 0; i < mesh.count; ++i)
            {
                std::memcpy(&vertices[info.first + i].Tangent, src_ptr + i * attr.stride, sizeof(float) * 4);
            }
        }

		if (auto it = mesh.attributes.find("TEXCOORD"); it != mesh.attributes.end())
		{
			auto const &attr = it->second;
			const char *src_ptr = loaded_data.at(attr.src.src).data() + attr.offset;
			for (uint32_t i = 0; i < mesh.count; ++i)
			{
				std::memcpy(&vertices[info.first + i].TexCoord, src_ptr + i * attr.stride, sizeof(glm::vec2));
				vertices[info.first + i].TexCoord.t = 1.0f - vertices[info.first + i].TexCoord.t;
			}
		}
		mesh_data[name] = info;
	}

	for (auto const& [name, env] : scene.environments) {
		EnvironmentInstance inst;

		std::string base_path = env.radiance->path;

		inst.radiance_idx = load_cubemap(base_path, 5);

		size_t dot = base_path.find_last_of('.');
		std::string prefix = base_path.substr(0, dot);
		std::string ext = base_path.substr(dot);
		std::string lambertian_path = prefix + ".lambertian" + ext;

		inst.lambertian_idx = load_cubemap(lambertian_path);

		environment_data[name] = inst;
	}

	std::unordered_map<std::string, uint32_t> path_to_texture_index;

	for (auto const &[name, mat] : scene.materials)
	{
		MaterialInstance inst;

		if (mat.normal_map) {
			const S72::Texture *n_tex = mat.normal_map;
			if (path_to_texture_index.find(n_tex->src) == path_to_texture_index.end()) {
				path_to_texture_index[n_tex->src] = this->load_texture(n_tex->path, n_tex->format);
			}
			inst.normal_index = (int)path_to_texture_index[n_tex->src];
		}

		if (mat.displacement_map) {
			const S72::Texture *n_tex = mat.displacement_map;
			if (path_to_texture_index.find(n_tex->src) == path_to_texture_index.end()) {
				path_to_texture_index[n_tex->src] = this->load_texture(n_tex->path, n_tex->format);
			}
			inst.displacement_idx = (int)path_to_texture_index[n_tex->src];
		}

		if (auto const *pbr = std::get_if<S72::Material::PBR>(&mat.brdf)) 
		{
			inst.type = 0;

			if (auto const *const *tex_ptr = std::get_if<S72::Texture *>(&pbr->albedo)) {
				const S72::Texture *tex = *tex_ptr;
				if (path_to_texture_index.find(tex->src) == path_to_texture_index.end()) {
					path_to_texture_index[tex->src] = this->load_texture(tex->path, tex->format);
				}
				inst.albedo_index = path_to_texture_index[tex->src];
			} else if (auto const *col_ptr = std::get_if<S72::color>(&pbr->albedo)) {
				inst.albedo_index = create_1x1_texture(col_ptr->r, col_ptr->g, col_ptr->b, 1.0f, true);
			}

			if (auto const *const *tex_ptr = std::get_if<S72::Texture *>(&pbr->roughness)) {
				const S72::Texture *tex = *tex_ptr;
				if (path_to_texture_index.find(tex->src) == path_to_texture_index.end()) {
					path_to_texture_index[tex->src] = this->load_texture(tex->path, tex->format);
				}
				inst.roughness_index = path_to_texture_index[tex->src];
			} else if (auto const *val_ptr = std::get_if<float>(&pbr->roughness)) {
				float v = *val_ptr;
    			inst.roughness_index = create_1x1_texture(v, v, v, 1.0f, false);;
			}

			if (auto const *const *tex_ptr = std::get_if<S72::Texture *>(&pbr->metalness)) {
				const S72::Texture *tex = *tex_ptr;
				if (path_to_texture_index.find(tex->src) == path_to_texture_index.end()) {
					path_to_texture_index[tex->src] = this->load_texture(tex->path, tex->format);
				}
				inst.metalness_index = path_to_texture_index[tex->src];
			} else if (auto const *val_ptr = std::get_if<float>(&pbr->metalness)) {
				float v = *val_ptr;
    			inst.metalness_index = create_1x1_texture(v, v, v, 1.0f, false);
			}
		}
		else if (auto const *lambert = std::get_if<S72::Material::Lambertian>(&mat.brdf))
		{
			inst.type = 1;
			if (auto const *const *tex_ptr = std::get_if<S72::Texture *>(&lambert->albedo)) {
				const S72::Texture *tex = *tex_ptr;
				if (path_to_texture_index.find(tex->src) == path_to_texture_index.end()) {
					path_to_texture_index[tex->src] = this->load_texture(tex->path, tex->format);
				}
				inst.albedo_index = path_to_texture_index[tex->src];
			} else {
				const auto &col = std::get<S72::color>(lambert->albedo);
    			inst.albedo_index = create_1x1_texture(col.r, col.g, col.b, 1.0f, true);
			}
		}
		else if (std::holds_alternative<S72::Material::Mirror>(mat.brdf)) {
			inst.type = 2;
		}
		else {
			inst.type = 3;
		}

		material_data[name] = inst;
	}

	object_vertices = rtg.helpers.create_buffer(
		vertices.size() * sizeof(PosNorTanTexVertex),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped);
	rtg.helpers.transfer_to_buffer(vertices.data(), vertices.size() * sizeof(PosNorTanTexVertex), object_vertices);

	// select a depth format
	// at least one of these two must be supported, according to the spec; but neither are required
	depth_format = rtg.helpers.find_image_format(
		{VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32},
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

	{ // create render pass
		std::array<VkAttachmentDescription, 2> attachments{
			VkAttachmentDescription{
				// color attachment:
				.format = rtg.surface_format.format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = rtg.present_layout,
			},
			VkAttachmentDescription{
				// depth attachment:
				.format = depth_format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
		};

		VkAttachmentReference color_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_attachment_ref{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};

		std::array<VkSubpassDependency, 2> dependencies{
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			},
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			}};

		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};

		VK(vkCreateRenderPass(rtg.device, &create_info, nullptr, &render_pass));
	}

	{ // create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK(vkCreateCommandPool(rtg.device, &create_info, nullptr, &command_pool));
	}

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass, 0);
	shadow_pipeline.create(rtg, depth_format, objects_pipeline.set1_Transforms);

	{
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());

		std::array<VkDescriptorPoolSize, 3> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 3 * per_workspace,
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 3 * per_workspace,
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = ShadowPipeline::MAX_SHADOW_MAPS * per_workspace,
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 16 * per_workspace,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool));
	}

	workspaces.resize(rtg.workspaces.size());
	for (Workspace &workspace : workspaces)
	{
		{ // allocate command buffer
			VkCommandBufferAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &workspace.command_buffer));
		}

		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped);
		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped);

		{ // allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lines_pipeline.set0_Camera,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Camera_descriptors));
		}

		workspace.World_src = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped);

		{ // allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors));
		}

		size_t lights_buffer_size = std::max(light_data.size(), size_t(1)) * sizeof(LightInstance);
		workspace.Lights_src = rtg.helpers.create_buffer(
			lights_buffer_size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped);

		workspace.Lights = rtg.helpers.create_buffer(
			lights_buffer_size, 
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped);

		{ // allocate descriptor set for Lights descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set4_Lights,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Lights_descriptors));
		}

		{ // allocate descriptor set for shadow maps
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set5_Shadows,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Shadow_descriptors));
		}

		{ // allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors));
		}

		{
			VkDescriptorBufferInfo Camera_info{
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};

			VkDescriptorBufferInfo Light_info{
				.buffer = workspace.Lights.handle,
				.offset = 0,
				.range = workspace.Lights.size,
			};

			std::array<VkWriteDescriptorSet, 4> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Camera_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &Camera_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.World_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &World_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Lights_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Light_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Lights_descriptors,
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Light_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()),
				writes.data(),
				0,
				nullptr);

			std::array<VkDescriptorImageInfo, ShadowPipeline::MAX_SHADOW_MAPS> dummy_image_infos;
			for (uint32_t i = 0; i < ShadowPipeline::MAX_SHADOW_MAPS; ++i) {
				dummy_image_infos[i] = VkDescriptorImageInfo{
					.sampler = shadow_pipeline.sampler,
					.imageView = shadow_pipeline.dummy_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				};
			}

			VkWriteDescriptorSet shadow_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = workspace.Shadow_descriptors,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = ShadowPipeline::MAX_SHADOW_MAPS,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = dummy_image_infos.data(),
			};

			vkUpdateDescriptorSets(rtg.device, 1, &shadow_write, 0, nullptr);
		}
	}

	{ // make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures)
		{
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK(vkCreateImageView(rtg.device, &create_info, nullptr, &image_view));

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ // make image views for the cubemaps
		texture_views_cube.clear();
		texture_views_cube.reserve(textures_cube.size());
		for (Helpers::AllocatedImage const &image : textures_cube) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
				.format = image.format,
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = image.mip_levels,
					.baseArrayLayer = 0,
					.layerCount = 6,
				},
			};
			VkImageView image_view;
			VK(vkCreateImageView(rtg.device, &create_info, nullptr, &image_view));
			texture_views_cube.emplace_back(image_view);
		}
	}

	{ // make a sampler for the 2D textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f,
			.compareEnable = VK_FALSE,
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK(vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler));
	}

	{ // make a sampler for the cubemap textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f,
			.compareEnable = VK_FALSE,
			.minLod = 0.0f,
			.maxLod = 5.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK(vkCreateSampler(rtg.device, &create_info, nullptr, &cube_sampler));
	}

	{ // create the texture descriptor pool
        uint32_t material_count = uint32_t(material_data.size());
        uint32_t env_count = uint32_t(scene.environments.size());
        
        uint32_t total_sampler_descriptors = material_count * 5 + env_count * 3;
        uint32_t total_uniform_descriptors = material_count;

        std::array<VkDescriptorPoolSize, 2> pool_sizes{
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = total_sampler_descriptors, 
            },
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = total_uniform_descriptors, 
            },
        };

        VkDescriptorPoolCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = material_count + env_count,
            .poolSizeCount = uint32_t(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        };

        VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool));
    }

	{ // allocate and write the texture descriptor sets
		for (auto &[name, mat] : material_data)
		{
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set2_TEXTURE,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &mat.descriptor_set));

			VkDescriptorImageInfo albedo_info{
				.sampler = texture_sampler,
				.imageView = texture_views.at(mat.albedo_index),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkDescriptorImageInfo normal_info{
				.sampler = texture_sampler,
				.imageView = texture_views.at(mat.normal_index),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkDescriptorImageInfo roughness_info{
				.sampler = texture_sampler,
				.imageView = texture_views.at(mat.roughness_index),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkDescriptorImageInfo metalness_info{
				.sampler = texture_sampler,
				.imageView = texture_views.at(mat.metalness_index),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			VkDescriptorImageInfo displacement_info{
				.sampler = texture_sampler,
				.imageView = texture_views.at(mat.displacement_idx),
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			std::array<VkWriteDescriptorSet, 5> mat_writes{};

			mat_writes[0] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = mat.descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &albedo_info,
			};

			mat_writes[1] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = mat.descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &normal_info,
			};
			
			mat_writes[2] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = mat.descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &roughness_info,
			};

			mat_writes[3] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = mat.descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &metalness_info,
			};

			mat_writes[4] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = mat.descriptor_set,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &displacement_info,
			};

			vkUpdateDescriptorSets(rtg.device, uint32_t(mat_writes.size()), mat_writes.data(), 0, nullptr);
		}
	}

	VkImageView brdf_lut_view = VK_NULL_HANDLE;
	{ //read brdf lookup table
		std::string lut_path = "brdf_lut.bin";
		
		std::ifstream file(lut_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open BRDF LUT");
		}

		size_t fileSize = (size_t)file.tellg();
		file.seekg(0);
		std::vector<char> data(fileSize);
		if (!file.read(data.data(), fileSize)) {
			throw std::runtime_error("Failed to read BRDF LUT.");
		}
		file.close();

		Helpers::AllocatedImage brdf_lut_image = rtg.helpers.create_image(
			{512, 512},
			VK_FORMAT_R32G32B32A32_SFLOAT,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		rtg.helpers.transfer_to_image(data.data(), fileSize, brdf_lut_image);

		textures.emplace_back(std::move(brdf_lut_image));

		VkImageViewCreateInfo view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = textures.back().handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VK(vkCreateImageView(rtg.device, &view_info, nullptr, &brdf_lut_view));
		
		texture_views.emplace_back(brdf_lut_view); 
	}

	{ // allocate and write the environment descriptor sets
		for (auto& [name, inst] : environment_data) {
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set3_Environment, 
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &inst.env_descriptors));

			std::array<VkDescriptorImageInfo, 3> env_infos{
				VkDescriptorImageInfo{
					.sampler = cube_sampler,
					.imageView = texture_views_cube.at(inst.radiance_idx),
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				VkDescriptorImageInfo{
					.sampler = cube_sampler,
					.imageView = texture_views_cube.at(inst.lambertian_idx),
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				VkDescriptorImageInfo{
					.sampler = cube_sampler,
					.imageView = brdf_lut_view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				}
			};

			std::array<VkWriteDescriptorSet, 3> env_writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = inst.env_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &env_infos[0],
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = inst.env_descriptors,
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &env_infos[1],
				},
					VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = inst.env_descriptors,
					.dstBinding = 2,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &env_infos[2],
				}
			};

			vkUpdateDescriptorSets(rtg.device, uint32_t(env_writes.size()), env_writes.data(), 0, nullptr);
		}
	}
}

S72Loader::~S72Loader()
{
	// just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS)
	{
		std::cerr << "Failed to vkDeviceWaitIdle in S72Loader::~S72Loader [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	if (texture_descriptor_pool)
	{
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = nullptr;

		texture_descriptors.clear();
	}

	if (texture_sampler)
	{
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	if (cube_sampler)
	{
		vkDestroySampler(rtg.device, cube_sampler, nullptr);
		cube_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView &view : texture_views)
	{
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (VkImageView &view : texture_views_cube)
	{
		if (view != VK_NULL_HANDLE) {
			vkDestroyImageView(rtg.device, view, nullptr);
			view = VK_NULL_HANDLE;
		}
	}
	texture_views_cube.clear();

	for (auto &texture : textures)
	{
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	rtg.helpers.destroy_buffer(std::move(object_vertices));

	for (auto &sm : shadow_maps) {
		if (sm.framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(rtg.device, sm.framebuffer, nullptr);
		if (sm.depth_view != VK_NULL_HANDLE) vkDestroyImageView(rtg.device, sm.depth_view, nullptr);
		if (sm.depth_image.handle != VK_NULL_HANDLE) rtg.helpers.destroy_image(std::move(sm.depth_image));
	}
	shadow_maps.clear();

	shadow_pipeline.destroy(rtg);

	if (swapchain_depth_image.handle != VK_NULL_HANDLE)
	{
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces)
	{
		if (workspace.command_buffer != VK_NULL_HANDLE)
		{
			vkFreeCommandBuffers(rtg.device, command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}

		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}
		if (workspace.Camera_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}
		if (workspace.World_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		if (workspace.Transforms_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		if (workspace.ShadowLights_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.ShadowLights_src));
		}
		if (workspace.ShadowLights.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.ShadowLights));
		}
	}
	workspaces.clear();

	if (descriptor_pool)
	{
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
	}

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);

	if (command_pool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}

void S72Loader::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain)
{
	// clean up existing framebuffers
	if (swapchain_depth_image.handle != VK_NULL_HANDLE)
	{
		destroy_framebuffers();
	}

	// allocate depth image for framebuffers to share
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent,
		depth_format,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped);

	{ // create depth image view:
		VkImageViewCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_depth_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1},
		};

		VK(vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view));
	}

	// create framebuffers pointing to each swapchain image view and the shared depth image view
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i)
	{
		std::array<VkImageView, 2> attachments{
			swapchain.image_views[i],
			swapchain_depth_image_view,
		};
		VkFramebufferCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = render_pass,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.width = swapchain.extent.width,
			.height = swapchain.extent.height,
			.layers = 1,
		};

		VK(vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]));
	}
}

void S72Loader::destroy_framebuffers()
{
	for (VkFramebuffer &framebuffer : swapchain_framebuffers)
	{
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}

void S72Loader::render(RTG &rtg_, RTG::RenderParams const &render_params)
{
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	Workspace &workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	VK(vkResetCommandBuffer(workspace.command_buffer, 0));
	{ // begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK(vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}

	if (!lines_vertices.empty())
	{ // upload lines vertices:
		//[re-]allocate lines buffers if needed:
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE || workspace.lines_vertices_src.size < needed_bytes)
		{
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.lines_vertices_src.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}

			workspace.lines_vertices_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			std::cout << "Re-allocated lines buffers to " << new_bytes << " bytes." << std::endl;
		}

		assert(workspace.lines_vertices_src.size == workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);

		// host-side copy into lines_vertices_src:
		assert(workspace.lines_vertices_src.allocation.mapped);
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		// device-side copy from lines_vertices_src -> lines_vertices:
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.handle, 1, &copy_region);
	}

	if (!object_instances.empty())
	{
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size < needed_bytes)
		{
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;

			if (workspace.Transforms_src.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}

			workspace.Transforms_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped);

			workspace.Transforms = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped);

			// update descriptor set
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};

			std::array<VkWriteDescriptorSet, 1> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Transforms_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Transforms_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()), writes.data(),
				0, nullptr);

			std::cout << "Re-allocated object transforms buffers to " << new_bytes << " bytes." << std::endl;
		}

		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);

		{ // copy transforms into transforms_src
			assert(workspace.Transforms_src.allocation.mapped);
			ObjectsPipeline::Transform *out = reinterpret_cast<ObjectsPipeline::Transform *>(workspace.Transforms_src.allocation.data());

			for (ObjectInstance const &inst : object_instances)
			{
				*out = inst.transform;
				++out;
			}
		}

		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle, workspace.Transforms.handle, 1, &copy_region);
	}

	{
		if (!light_data.empty()) 
		{
			size_t needed_bytes = light_data.size() * sizeof(LightInstance);
			
			if (workspace.Lights_src.handle == VK_NULL_HANDLE || workspace.Lights_src.size < needed_bytes) 
			{
				size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
				
				if (workspace.Lights_src.handle) rtg.helpers.destroy_buffer(std::move(workspace.Lights_src));
				if (workspace.Lights.handle)     rtg.helpers.destroy_buffer(std::move(workspace.Lights));

				workspace.Lights_src = rtg.helpers.create_buffer(
					new_bytes,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					Helpers::Mapped
				);
				workspace.Lights = rtg.helpers.create_buffer(
					new_bytes,
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					Helpers::Unmapped
				);

				VkDescriptorBufferInfo lights_info{
					.buffer = workspace.Lights.handle,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				};

				VkWriteDescriptorSet write{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Lights_descriptors,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &lights_info,
				};

				vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
				std::cout << "Re-allocated lights buffers to " << new_bytes << " bytes." << std::endl;
			}

			assert(workspace.Lights_src.allocation.mapped);
			std::memcpy(workspace.Lights_src.allocation.data(), light_data.data(), needed_bytes);

			VkBufferCopy copy_region{
				.srcOffset = 0,
				.dstOffset = 0,
				.size = needed_bytes,
			};
			vkCmdCopyBuffer(workspace.command_buffer, workspace.Lights_src.handle, workspace.Lights.handle, 1, &copy_region);
		}
	}

	{
		LinesPipeline::Camera camera{
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD};
		assert(workspace.Camera_src.size == sizeof(camera));

		memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

		assert(workspace.Camera_src.size == workspace.Camera.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Camera_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle, workspace.Camera.handle, 1, &copy_region);
	}

	{ // upload world info:
		assert(workspace.World_src.size == sizeof(world));

		memcpy(workspace.World_src.allocation.data(), &world, sizeof(world));

		assert(workspace.World_src.size == workspace.World.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.World_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.World_src.handle, workspace.World.handle, 1, &copy_region);
	}

	{ // memory barrier
		VkMemoryBarrier memory_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		};

		vkCmdPipelineBarrier(workspace.command_buffer,
							 VK_PIPELINE_STAGE_TRANSFER_BIT,
							 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							 0,
							 1, &memory_barrier,
							 0, nullptr,
							 0, nullptr);
	}

	{
		size_t needed_bytes = shadow_light_data.empty()
			? sizeof(ShadowLightInstance)
			: shadow_light_data.size() * sizeof(ShadowLightInstance);

		if (workspace.ShadowLights_src.handle == VK_NULL_HANDLE || workspace.ShadowLights_src.size < needed_bytes)
		{
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;

			if (workspace.ShadowLights_src.handle) rtg.helpers.destroy_buffer(std::move(workspace.ShadowLights_src));
			if (workspace.ShadowLights.handle)     rtg.helpers.destroy_buffer(std::move(workspace.ShadowLights));

			workspace.ShadowLights_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped
			);
			workspace.ShadowLights = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			VkDescriptorBufferInfo shadow_lights_info{
				.buffer = workspace.ShadowLights.handle,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			};

			VkWriteDescriptorSet write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = workspace.Lights_descriptors,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &shadow_lights_info,
			};
			vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
		}

		if (!shadow_light_data.empty()) {
			size_t copy_bytes = shadow_light_data.size() * sizeof(ShadowLightInstance);
			assert(workspace.ShadowLights_src.allocation.mapped);
			std::memcpy(workspace.ShadowLights_src.allocation.data(), shadow_light_data.data(), copy_bytes);

			VkBufferCopy copy_region{
				.srcOffset = 0,
				.dstOffset = 0,
				.size = copy_bytes,
			};
			vkCmdCopyBuffer(workspace.command_buffer, workspace.ShadowLights_src.handle, workspace.ShadowLights.handle, 1, &copy_region);
		}
	}

	{
		active_shadow_count = uint32_t(shadow_light_data.size());

		while (shadow_maps.size() < active_shadow_count) {
			shadow_maps.push_back(ShadowMap{});
		}

		for (uint32_t si = 0; si < active_shadow_count; ++si) {
			uint32_t map_size = uint32_t(shadow_light_data[si].tint_shadow.w);

			ShadowMap &sm = shadow_maps[si];

			if (sm.size != map_size) {
				if (sm.framebuffer != VK_NULL_HANDLE) {
					vkDestroyFramebuffer(rtg.device, sm.framebuffer, nullptr);
					sm.framebuffer = VK_NULL_HANDLE;
				}
				if (sm.depth_view != VK_NULL_HANDLE) {
					vkDestroyImageView(rtg.device, sm.depth_view, nullptr);
					sm.depth_view = VK_NULL_HANDLE;
				}
				if (sm.depth_image.handle != VK_NULL_HANDLE) {
					rtg.helpers.destroy_image(std::move(sm.depth_image));
				}

				sm.size = map_size;
				sm.depth_image = rtg.helpers.create_image(
					{map_size, map_size},
					depth_format,
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
				);

				VkImageViewCreateInfo view_info{
					.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					.image = sm.depth_image.handle,
					.viewType = VK_IMAGE_VIEW_TYPE_2D,
					.format = depth_format,
					.subresourceRange{
						.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
						.baseMipLevel = 0, .levelCount = 1,
						.baseArrayLayer = 0, .layerCount = 1,
					},
				};
				VK(vkCreateImageView(rtg.device, &view_info, nullptr, &sm.depth_view));

				VkFramebufferCreateInfo fb_info{
					.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
					.renderPass = shadow_pipeline.render_pass,
					.attachmentCount = 1,
					.pAttachments = &sm.depth_view,
					.width = map_size,
					.height = map_size,
					.layers = 1,
				};
				VK(vkCreateFramebuffer(rtg.device, &fb_info, nullptr, &sm.framebuffer));
			}

			VkClearValue depth_clear{.depthStencil{.depth = 1.0f, .stencil = 0}};
			VkRenderPassBeginInfo shadow_begin{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = shadow_pipeline.render_pass,
				.framebuffer = sm.framebuffer,
				.renderArea{.offset = {0, 0}, .extent = {map_size, map_size}},
				.clearValueCount = 1,
				.pClearValues = &depth_clear,
			};

			vkCmdBeginRenderPass(workspace.command_buffer, &shadow_begin, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport shadow_viewport{
				.x = 0.0f, .y = 0.0f,
				.width = float(map_size), .height = float(map_size),
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};
			VkRect2D shadow_scissor{.offset = {0, 0}, .extent = {map_size, map_size}};
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &shadow_viewport);
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &shadow_scissor);

			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline.handle);

			vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				shadow_pipeline.layout, 0, 1, &workspace.Transforms_descriptors, 0, nullptr);

			glm::mat4 light_vp = shadow_light_data[si].light_VP;
			vkCmdPushConstants(workspace.command_buffer, shadow_pipeline.layout,
				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &light_vp);

			VkBuffer vertex_buffers[]{object_vertices.handle};
			VkDeviceSize offsets[]{0};
			vkCmdBindVertexBuffers(workspace.command_buffer, 0, 1, vertex_buffers, offsets);

			for (ObjectInstance const &inst : object_instances) {
				uint32_t index = uint32_t(&inst - &object_instances[0]);
				vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
			}

			vkCmdEndRenderPass(workspace.command_buffer);
		}

		{
			std::array<VkDescriptorImageInfo, ShadowPipeline::MAX_SHADOW_MAPS> image_infos;
			for (uint32_t i = 0; i < ShadowPipeline::MAX_SHADOW_MAPS; ++i) {
				if (i < active_shadow_count) {
					image_infos[i] = VkDescriptorImageInfo{
						.sampler = shadow_pipeline.sampler,
						.imageView = shadow_maps[i].depth_view,
						.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					};
				} else {
					image_infos[i] = VkDescriptorImageInfo{
						.sampler = shadow_pipeline.sampler,
						.imageView = shadow_pipeline.dummy_view,
						.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					};
				}
			}

			VkWriteDescriptorSet write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = workspace.Shadow_descriptors,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = ShadowPipeline::MAX_SHADOW_MAPS,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = image_infos.data(),
			};
			vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
		}
	}

	{ // render pass
		std::array<VkClearValue, 2> clear_values{
			VkClearValue{.color{.float32{0.0f, 0.0f, 0.0f, 1.0f}}},
			VkClearValue{.depthStencil{.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			},
			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};

		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		{
			VkRect2D scissor{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			};
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &scissor);
		}

		{
			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = float(rtg.swapchain_extent.width),
				.height = float(rtg.swapchain_extent.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &viewport);
		}

		{ // draw with background pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background_pipeline.handle);

			{
				BackgroundPipeline::Push push{
					.time = time,
				};

				vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
			}

			vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		}

		{ // draw with lines pipeline
			if (!lines_vertices.empty())
			{
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);

				{
					std::array<VkBuffer, 1> vertex_buffers{workspace.lines_vertices.handle};
					std::array<VkDeviceSize, 1> offsets{0};
					vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
				}

				{
					std::array<VkDescriptorSet, 1> descriptor_sets{
						workspace.Camera_descriptors,
					};
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						lines_pipeline.layout,
						0,
						uint32_t(descriptor_sets.size()), descriptor_sets.data(),
						0, nullptr);
				}

				vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
			}
		}

		{
			if (!object_instances.empty())
			{ // draw with the objects pipeline:
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.handle);

				{ // use object_vertices (offset 0) as vertex buffer binding 0:
					std::array<VkBuffer, 1> vertex_buffers{object_vertices.handle};
					std::array<VkDeviceSize, 1> offsets{0};
					vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
				}

				{ // bind World and Transforms descriptor sets:
					std::array<VkDescriptorSet, 2> descriptor_sets{
						workspace.World_descriptors,
						workspace.Transforms_descriptors,
					};
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						0,
						uint32_t(descriptor_sets.size()), descriptor_sets.data(),
						0, nullptr);
					
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						4,
						1, &workspace.Lights_descriptors,
						0, nullptr);

					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						5,
						1, &workspace.Shadow_descriptors,
						0, nullptr);
				}

				// draw all instances:
                for (ObjectInstance const &inst : object_instances)
				{
					uint32_t index = uint32_t(&inst - &object_instances[0]);

					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						2,
						1, &inst.material->descriptor_set,
						0, nullptr);

					if (inst.environment_cube_set != VK_NULL_HANDLE) {
						vkCmdBindDescriptorSets(
							workspace.command_buffer,
							VK_PIPELINE_BIND_POINT_GRAPHICS,
							objects_pipeline.layout,
							3,
							1, &inst.environment_cube_set,
							0, nullptr);
					}

					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
				}
			}
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}

	// end recording
	VK(vkEndCommandBuffer(workspace.command_buffer));

	{ // submit `workspace.command buffer` for the GPU to run:
		std::array<VkSemaphore, 1> wait_semaphores{
			render_params.image_available};
		std::array<VkPipelineStageFlags, 1> wait_stages{
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		static_assert(wait_semaphores.size() == wait_stages.size(), "every semaphore needs a stage");

		std::array<VkSemaphore, 1> signal_semaphores{
			render_params.image_done};
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = uint32_t(wait_semaphores.size()),
			.pWaitSemaphores = wait_semaphores.data(),
			.pWaitDstStageMask = wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &workspace.command_buffer,
			.signalSemaphoreCount = uint32_t(signal_semaphores.size()),
			.pSignalSemaphores = signal_semaphores.data(),
		};

		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, render_params.workspace_available));
	}
}

// TODO:move this helper somewhere else
inline glm::mat4 orbit(
	float target_x, float target_y, float target_z,
	float azimuth, float elevation, float radius)
{
	float ca = std::cos(azimuth);
	float sa = std::sin(azimuth);
	float ce = std::cos(elevation);
	float se = std::sin(elevation);

	glm::vec3 right = glm::vec3(-sa, ca, 0.0f);
	glm::vec3 up = glm::vec3(-se * ca, -se * sa, ce);
	glm::vec3 out = glm::vec3(ce * ca, ce * sa, se);

	glm::vec3 target = glm::vec3(target_x, target_y, target_z);
	glm::vec3 eye = target + radius * out;

	return glm::mat4(
		right.x, up.x, out.x, 0.0f,
		right.y, up.y, out.y, 0.0f,
		right.z, up.z, out.z, 0.0f,
		-glm::dot(right, eye), -glm::dot(up, eye), -glm::dot(out, eye), 1.0f);
}

bool S72Loader::SAT_visibility_test(const CullingFrustum &frustum, const glm::mat4 &world_to_view, const AABB &aabb)
{
	float z_near = frustum.near_plane;
	float z_far = frustum.far_plane;
	float x_near = frustum.near_right;
	float y_near = frustum.near_top;

	glm::vec3 corners[] = {
		{aabb.min.x, aabb.min.y, aabb.min.z},
		{aabb.max.x, aabb.min.y, aabb.min.z},
		{aabb.min.x, aabb.max.y, aabb.min.z},
		{aabb.min.x, aabb.min.y, aabb.max.z},
	};

	for (int i = 0; i < 4; i++)
	{
		corners[i] = glm::vec3(world_to_view * glm::vec4(corners[i], 1.0f));
	}

	OBB obb;
	obb.axes[0] = corners[1] - corners[0];
	obb.axes[1] = corners[2] - corners[0];
	obb.axes[2] = corners[3] - corners[0];

	obb.center = corners[0] + 0.5f * (obb.axes[0] + obb.axes[1] + obb.axes[2]);

	obb.extents = glm::vec3{glm::length(obb.axes[0]), glm::length(obb.axes[1]), glm::length(obb.axes[2])};
	obb.axes[0] /= obb.extents.x;
	obb.axes[1] /= obb.extents.y;
	obb.axes[2] /= obb.extents.z;
	obb.extents *= 0.5f;

	{
		float MoC = obb.center.z;
		float radius = 0.0f;
		for (int i = 0; i < 3; i++)
			radius += fabsf(obb.axes[i].z) * obb.extents[i];

		float obb_min = MoC - radius;
		float obb_max = MoC + radius;

		if (obb_min > z_near || obb_max < z_far)
			return false;
	}

	{
		const glm::vec3 M[] = {
			{0.0f, -z_near, y_near},
			{0.0f, z_near, y_near},
			{-z_near, 0.0f, x_near},
			{z_near, 0.0f, x_near},
		};

		for (int m = 0; m < 4; m++)
		{
			float MoX = fabsf(M[m].x);
			float MoY = fabsf(M[m].y);
			float MoZ = M[m].z;
			float MoC = glm::dot(M[m], obb.center);

			float obb_radius = 0.0f;
			for (int i = 0; i < 3; i++)
			{
				obb_radius += fabsf(glm::dot(M[m], obb.axes[i])) * obb.extents[i];
			}
			float obb_min = MoC - obb_radius;
			float obb_max = MoC + obb_radius;

			float p = x_near * MoX + y_near * MoY;
			float tau_0 = z_near * MoZ - p;
			float tau_1 = z_near * MoZ + p;

			if (tau_0 < 0.0f)
				tau_0 *= z_far / z_near;
			if (tau_1 > 0.0f)
				tau_1 *= z_far / z_near;

			if (obb_min > tau_1 || obb_max < tau_0)
				return false;
		}
	}

	return true;
}

void S72Loader::update(float dt)
{
	time = std::fmod(time + dt, 60.0f);

	update_animations(playback_time);

	playback_time += dt;

	{ // sun and sky defaults:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.0f;
		world.SKY_ENERGY.g = 0.0f;
		world.SKY_ENERGY.b = 0.0f;

		world.SUN_DIRECTION.x = 0.0f;
		world.SUN_DIRECTION.y = 0.0f;
		world.SUN_DIRECTION.z = 1.0f;

		world.SUN_ENERGY.r = 0.0f;
		world.SUN_ENERGY.g = 0.0f;
		world.SUN_ENERGY.b = 0.0f;
	}

	{
		object_instances.clear();
		lines_vertices.clear();
		light_data.clear();
		shadow_light_data.clear();

		glm::mat4 identity = glm::mat4(1.0f);

		for (auto *root_node : scene.scene.roots)
		{
			traverse_scene(root_node, identity);
		}
	}

	{
		if (camera_mode == CameraMode::Scene && active_camera != nullptr)
		{
			auto const &props = active_camera->props;
			glm::mat4 const &m = active_camera->world_from_local;

			glm::vec3 eye = glm::vec3(m[3]);
			glm::vec3 forward = -glm::vec3(m[2]);
			glm::vec3 up = glm::vec3(m[1]);

			glm::mat4 view = glm::lookAt(eye, eye + forward, up);

			float aspect = rtg.swapchain_extent.width / (float)rtg.swapchain_extent.height;
			glm::mat4 projection = glm::perspective(props.vfov, aspect, props.near, props.far);

			projection[1][1] *= -1.0f;

			CLIP_FROM_WORLD = projection * view;

			// frustum
			active_frustum.near_plane = -props.near;
			active_frustum.far_plane = -props.far;

			active_frustum.near_top = props.near * tanf(props.vfov * 0.5f);
			active_frustum.near_right = active_frustum.near_top * props.aspect;
		}
		else if (camera_mode == CameraMode::Free)
		{
			glm::mat4 projection = glm::perspective(
				free_camera.fov,
				rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
				free_camera.near,
				free_camera.far);

			projection[1][1] *= -1.0f;

			glm::mat4 view = orbit(
				free_camera.target_x, free_camera.target_y, free_camera.target_z,
				free_camera.azimuth, free_camera.elevation, free_camera.radius);

			CLIP_FROM_WORLD = projection * view;
		}
	}

	{
		if (camera_mode == CameraMode::Free) {
			glm::mat4 world_from_camera = glm::inverse(orbit(
				free_camera.target_x, free_camera.target_y, free_camera.target_z,
				free_camera.azimuth, free_camera.elevation, free_camera.radius
			));

			world.EYE.x = world_from_camera[3].x;
			world.EYE.y = world_from_camera[3].y;
			world.EYE.z = world_from_camera[3].z;
		} else {
			world.EYE.x = active_camera->world_from_local[3].x;
			world.EYE.y = active_camera->world_from_local[3].y;
			world.EYE.z = active_camera->world_from_local[3].z;
		}

		world.EXPOSURE.exposure = float(std::pow(2.0f, rtg.configuration.exposure));
		world.TONEMAPPING.tone_mapping_mode = rtg.configuration.tone_mapping_mode;
	}

	{
		for (auto &inst : object_instances)
		{
			inst.transform.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * inst.transform.WORLD_FROM_LOCAL;
		}
	}
}

void S72Loader::on_input(InputEvent const &evt)
{
	if (action)
	{
		action(evt);
		return;
	}

	// general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB)
	{
		if (!camera_list.empty())
		{
			camera_mode = CameraMode::Scene;
			current_camera_index = (current_camera_index + 1) % camera_list.size();
			active_camera = camera_list[current_camera_index];
		}
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_F)
	{
		camera_mode = CameraMode::Free;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_SLASH)
	{
		isDebugMode = !isDebugMode;
	}

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_SPACE)
	{
		playback_time = 0.0f;
	}

	// free camera controls:
	if (camera_mode == CameraMode::Free)
	{

		if (evt.type == InputEvent::MouseWheel)
		{
			free_camera.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			free_camera.radius = std::max(free_camera.radius, 0.5f * free_camera.near);
			free_camera.radius = std::min(free_camera.radius, 2.0f * free_camera.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT))
		{
			// start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// image height at plane of target point:
					float height = 2.0f * std::tan(free_camera.fov * 0.5f) * free_camera.radius;

					// motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height * height;

					// compute camera transform to extract right (first row) and up (second row):
					glm::mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius);

					// move the desired distance:
					free_camera.target_x = init_camera.target_x - dx * camera_from_world[0][0] - dy * camera_from_world[1][0];
					free_camera.target_y = init_camera.target_y - dx * camera_from_world[0][1] - dy * camera_from_world[1][1];
					free_camera.target_z = init_camera.target_z - dx * camera_from_world[0][2] - dy * camera_from_world[1][2];
					return;
				}
			};

			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
		{
			// start tumbling

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// motion, normalized so 1.0 is window height:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height;

					// rotate camera based on motion:
					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					// reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					free_camera.azimuth -= std::round(free_camera.azimuth / twopi) * twopi;
					free_camera.elevation -= std::round(free_camera.elevation / twopi) * twopi;
					return;
				}
			};

			return;
		}
	}
}