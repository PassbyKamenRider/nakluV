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
			inst.texture = (mat_it != material_data.end()) ? mat_it->second.texture_index : 0;

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

	if (node->light)
	{
		auto const &light = *(node->light);
		if (std::holds_alternative<S72::Light::Sun>(light.source))
		{
			glm::vec3 direction = glm::normalize(glm::vec3(world_matrix[2]));

			auto const &sun = std::get<S72::Light::Sun>(light.source);
			glm::vec3 energy = glm::vec3(light.tint.r, light.tint.g, light.tint.b) * sun.strength;

			if (sun.angle > 3.0f)
			{
				world.SKY_DIRECTION = {direction.x, direction.y, direction.z, 0.0f};
				world.SKY_ENERGY = {energy.r, energy.g, energy.b, 0.0f};
			}
			else
			{
				world.SUN_DIRECTION = {direction.x, direction.y, direction.z, 0.0f};
				world.SUN_ENERGY = {energy.r, energy.g, energy.b, 0.0f};
			}
		}
	}

	for (auto *child : node->children)
	{
		traverse_scene(child, world_matrix);
	}
}

uint32_t S72Loader::load_texture(std::string path, S72::Texture::Format format)
{
	int width, height, channels;
	stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

	uint32_t img_w = uint32_t(width);
	uint32_t img_h = uint32_t(height);

	VkFormat vk_format;
	if (format == S72::Texture::Format::srgb)
	{
		vk_format = VK_FORMAT_R8G8B8A8_SRGB;
	}
	else
	{
		vk_format = VK_FORMAT_R8G8B8A8_UNORM;
	}

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

	return (uint32_t)(textures.size() - 1);
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

	std::vector<PosNorTexVertex> vertices;
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

	std::unordered_map<std::string, uint32_t> path_to_texture_index;
	for (auto const &[name, mat] : scene.materials)
	{
		uint32_t tex_idx = 0;
		auto handle_albedo = [&](auto const &albedo)
		{
			if (auto const *const *tex_ptr = std::get_if<S72::Texture *>(&albedo))
			{
				const S72::Texture *tex = *tex_ptr;
				if (tex)
				{
					if (path_to_texture_index.find(tex->src) == path_to_texture_index.end())
					{
						path_to_texture_index[tex->src] = this->load_texture(tex->path, tex->format);
					}
					return path_to_texture_index[tex->src];
				}
			}
			else
			{
				const auto &col = std::get<S72::color>(albedo);
				uint8_t pixel[4] = {(uint8_t)(col.r * 255.0f), (uint8_t)(col.g * 255.0f), (uint8_t)(col.b * 255.0f), 255};
				textures.emplace_back(rtg.helpers.create_image({1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
															   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped));
				rtg.helpers.transfer_to_image(pixel, 4, textures.back());
				return (uint32_t)(textures.size() - 1);
			}
			return 0u;
		};

		if (auto const *pbr = std::get_if<S72::Material::PBR>(&mat.brdf))
		{
			tex_idx = handle_albedo(pbr->albedo);
		}
		else if (auto const *lambert = std::get_if<S72::Material::Lambertian>(&mat.brdf))
		{
			tex_idx = handle_albedo(lambert->albedo);
		}
		material_data[name] = MaterialInstance{.texture_index = tex_idx};
	}

	object_vertices = rtg.helpers.create_buffer(
		vertices.size() * sizeof(PosNorTexVertex),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped);
	rtg.helpers.transfer_to_buffer(vertices.data(), vertices.size() * sizeof(PosNorTexVertex), object_vertices);

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

	{
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());

		std::array<VkDescriptorPoolSize, 2> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2 * per_workspace,
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1 * per_workspace,
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 3 * per_workspace,
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
			// NOTE: will actually fill in this descriptor set just a bit lower
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

			std::array<VkWriteDescriptorSet, 2> writes{
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
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()),
				writes.data(),
				0,
				nullptr);
		}
	}

	{ // make some textures
		textures.reserve(1);

		{ // texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
			// make the texture:
			uint32_t size = 128;
			std::vector<uint32_t> data;
			data.reserve(size * size);
			for (uint32_t y = 0; y < size; ++y)
			{
				float fy = (y + 0.5f) / float(size);
				for (uint32_t x = 0; x < size; ++x)
				{
					float fx = (x + 0.5f) / float(size);
					if (fx < 0.05f && fy < 0.05f)
						data.emplace_back(0xff0000ff);
					else if ((fx < 0.5f) == (fy < 0.5f))
						data.emplace_back(0xff444444);
					else
						data.emplace_back(0xffbbbbbb);
				}
			}
			assert(data.size() == size * size);

			// make a place for the texture to live on the GPU:
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{.width = size, .height = size},
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped));

			// transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
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

	{ // make a sampler for the textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK(vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler));
	}

	{ // create the texture descriptor pool
		uint32_t per_texture = uint32_t(textures.size());

		std::array<VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1 * 1 * per_texture,
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 1 * per_texture,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool));
	}

	{ // allocate and write the texture descriptor sets
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_TEXTURE,
		};
		texture_descriptors.assign(textures.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet &descriptor_set : texture_descriptors)
		{
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set));
		}

		// write descriptors for textures
		std::vector<VkDescriptorImageInfo> infos(textures.size());
		std::vector<VkWriteDescriptorSet> writes(textures.size());

		for (Helpers::AllocatedImage const &image : textures)
		{
			size_t i = &image - &textures[0];

			infos[i] = VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[i],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			writes[i] = VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture_descriptors[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &infos[i],
			};
		}

		vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
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

	for (VkImageView &view : texture_views)
	{
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto &texture : textures)
	{
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	rtg.helpers.destroy_buffer(std::move(object_vertices));

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
	// assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	// get more convenient names for the current workspace and target framebuffer:
	Workspace &workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	// record (into `workspace.command_buffer`) commands that run a `render_pass` that just clears `framebuffer`:
	// refsol::Tutorial_render_record_blank_frame(rtg, render_pass, framebuffer, &workspace.command_buffer);
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
			// round to next multiple of 4k to avoid re-allocating continuously if vertex count grows slowly:
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
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,											// going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // host-visible memory, coherent (no special sync needed)
				Helpers::Mapped																// get a pointer to the memory
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // going to use as vertex buffer, also going to have GPU into this memory
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,								  // GPU-local memory
				Helpers::Unmapped													  // don't get a pointer to the memory
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
		assert(workspace.Camera_src.size == sizeof(world));

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
							 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
							 0,
							 1, &memory_barrier,
							 0, nullptr,
							 0, nullptr);
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
				}

				// Camera descriptor set is still bound, but unused

				// draw all instances:
				for (ObjectInstance const &inst : object_instances)
				{
					uint32_t index = uint32_t(&inst - &object_instances[0]);

					// bind texture descriptor set
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						2,
						1, &texture_descriptors[inst.texture],
						0, nullptr);

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

	{ // sun and sky:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.1f;
		world.SKY_ENERGY.g = 0.1f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = 6.0f / 23.0f;
		world.SUN_DIRECTION.y = 13.0f / 23.0f;
		world.SUN_DIRECTION.z = 18.0f / 23.0f;

		world.SUN_ENERGY.r = 1.0f;
		world.SUN_ENERGY.g = 1.0f;
		world.SUN_ENERGY.b = 0.9f;
	}

	{ // objects
		object_instances.clear();
		lines_vertices.clear();

		glm::mat4 identity = glm::mat4(1.0f);

		for (auto *root_node : scene.scene.roots)
		{
			traverse_scene(root_node, identity);
		}

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