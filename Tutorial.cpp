#include "Tutorial.hpp"

#include "VK.hpp"
#include "refsol.hpp"

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_) {
	refsol::Tutorial_constructor(rtg, &depth_format, &render_pass, &command_pool);

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass, 0);

	{
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());
		
		std::array<VkDescriptorPoolSize, 2> pool_sizes {
			VkDescriptorPoolSize {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2 * per_workspace,
			},
			VkDescriptorPoolSize {
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1 * per_workspace,
			},
		};

		VkDescriptorPoolCreateInfo create_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 3 * per_workspace,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool));
	}

	workspaces.resize(rtg.workspaces.size());
	for (Workspace &workspace : workspaces) {
		refsol::Tutorial_constructor_workspace(rtg, command_pool, &workspace.command_buffer);

		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info {
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
			Helpers::Mapped
		);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};

			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors) );
			//NOTE: will actually fill in this descriptor set just a bit lower
		}

		{ //allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors));
		}

		{
			VkDescriptorBufferInfo Camera_info {
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};

			std::array<VkWriteDescriptorSet, 2> writes {
				VkWriteDescriptorSet {
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
				nullptr
			);
		}
	}

	
	{ //create object vertices
		std::vector<PosNorTexVertex> vertices;

		{ //A [-1,1]x[-1,1]x{0} quadrilateral:
			plane_vertices.first = uint32_t(vertices.size());
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 0.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
				.TexCoord{ .s = 1.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 0.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});

			plane_vertices.count = uint32_t(vertices.size()) - plane_vertices.first;
		}

		{ //A torus:
			torus_vertices.first = uint32_t(vertices.size());

			//will parameterize with (u,v) where:
			// - u is angle around main axis (+z)
			// - v is angle around the tube

			constexpr float R1 = 0.75f; //main radius
			constexpr float R2 = 0.15f; //tube radius

			constexpr uint32_t U_STEPS = 20;
			constexpr uint32_t V_STEPS = 16;

			//texture repeats around the torus:
			constexpr float V_REPEATS = 2.0f;
			constexpr float U_REPEATS = int(V_REPEATS / R2 * R1 + 0.999f); //approximately square, rounded up

			auto emplace_vertex = [&](uint32_t ui, uint32_t vi) {
				//convert steps to angles:
				// (doing the mod since trig on 2 M_PI may not exactly match 0)
				float ua = (ui % U_STEPS) / float(U_STEPS) * 2.0f * float(M_PI);
				float va = (vi % V_STEPS) / float(V_STEPS) * 2.0f * float(M_PI);

				vertices.emplace_back( PosNorTexVertex{
					.Position{
						.x = (R1 + R2 * std::cos(va)) * std::cos(ua),
						.y = (R1 + R2 * std::cos(va)) * std::sin(ua),
						.z = R2 * std::sin(va),
					},
					.Normal{
						.x = std::cos(va) * std::cos(ua),
						.y = std::cos(va) * std::sin(ua),
						.z = std::sin(va),
					},
					.TexCoord{
						.s = ui / float(U_STEPS) * U_REPEATS,
						.t = vi / float(V_STEPS) * V_REPEATS,
					},
				});
			};

			for (uint32_t ui = 0; ui < U_STEPS; ++ui) {
				for (uint32_t vi = 0; vi < V_STEPS; ++vi) {
					emplace_vertex(ui, vi);
					emplace_vertex(ui+1, vi);
					emplace_vertex(ui, vi+1);

					emplace_vertex(ui, vi+1);
					emplace_vertex(ui+1, vi);
					emplace_vertex(ui+1, vi+1);
				}
			}

			torus_vertices.count = uint32_t(vertices.size()) - torus_vertices.first;
		}

		{ // A pyramid:
			pyramid_vertices.first = uint32_t(vertices.size());
			
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 0.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f },
				.TexCoord{ .s = 1.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 0.0f, .t = 1.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.0f, .z = 1.0f},
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});

			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = -0.894f, .y = 0.0f, .z = 0.447f},
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = -0.894f, .y = 0.0f, .z = 0.447f},
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 0.0f, .y = 0.0f, .z = 2.0f },
				.Normal{ .x = -0.894f, .y = 0.0f, .z = 0.447f},
				.TexCoord{ .s = 0.5f, .t = 1.0f },
			});

			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = -0.894f, .z = 0.447f },
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = -0.894f, .z = 0.447f },
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 0.0f, .y = 0.0f, .z = 2.0f },
				.Normal{ .x = 0.0f, .y = -0.894f, .z = 0.447f },
				.TexCoord{ .s = 0.5f, .t = 1.0f },
			});

			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = -1.0f, .z = 0.0f },
				.Normal{ .x = 0.894f, .y = 0.0f, .z = 0.447f },
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.894f, .y = 0.0f, .z = 0.447f },
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 0.0f, .y = 0.0f, .z = 2.0f },
				.Normal{ .x = 0.894f, .y = 0.0f, .z = 0.447f },
				.TexCoord{ .s = 0.5f, .t = 1.0f },
			});

			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.894f, .z = 0.447f },
				.TexCoord{ .s = 0.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = -1.0f, .y = 1.0f, .z = 0.0f },
				.Normal{ .x = 0.0f, .y = 0.894f, .z = 0.447f },
				.TexCoord{ .s = 1.0f, .t = 0.0f },
			});
			vertices.emplace_back(PosNorTexVertex{
				.Position{ .x = 0.0f, .y = 0.0f, .z = 2.0f },
				.Normal{ .x = 0.0f, .y = 0.894f, .z = 0.447f },
				.TexCoord{ .s = 0.5f, .t = 1.0f },
			});

			pyramid_vertices.count = uint32_t(vertices.size()) - pyramid_vertices.first;
		}

		size_t bytes = vertices.size() * sizeof(vertices[0]);

		object_vertices = rtg.helpers.create_buffer(
			bytes,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}

	{ //make some textures
		textures.reserve(2);

		{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
			//make the texture:
			uint32_t size = 128;
			std::vector< uint32_t > data;
			data.reserve(size * size);
			for (uint32_t y = 0; y < size; ++y) {
				float fy = (y + 0.5f) / float(size);
				for (uint32_t x = 0; x < size; ++x) {
					float fx = (x + 0.5f) / float(size);
					if      (fx < 0.05f && fy < 0.05f) data.emplace_back(0xff0000ff);
					else if ( (fx < 0.5f) == (fy < 0.5f)) data.emplace_back(0xff444444);
					else data.emplace_back(0xffbbbbbb);
				}
			}
			assert(data.size() == size*size);

			//make a place for the texture to live on the GPU:
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = size , .height = size },
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		}

		{ //texture 1 will be a classic 'xor' texture:
			//actually make the texture:
			uint32_t size = 256;
			std::vector< uint32_t > data;
			data.reserve(size * size);
			for (uint32_t y = 0; y < size; ++y) {
				for (uint32_t x = 0; x < size; ++x) {
					uint8_t r = uint8_t(x) ^ uint8_t(y);
					uint8_t g = uint8_t(x + 128) ^ uint8_t(y);
					uint8_t b = uint8_t(x) ^ uint8_t(y + 27);
					uint8_t a = 0xff;
					data.emplace_back( uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24) );
				}
			}
			assert(data.size() == size*size);

			//make a place for the texture to live on the GPU:
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = size , .height = size },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		}

		{ //pyramid texture
			int width, height, channels;
			stbi_uc* pixels = stbi_load("pyramid_texture.png", &width, &height, &channels, STBI_rgb_alpha);

			uint32_t img_w = uint32_t(width);
			uint32_t img_h = uint32_t(height);

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = img_w, .height = img_h },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(pixels, img_w * img_h * 4, textures.back());

			stbi_image_free(pixels);
		}
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures) {
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
			VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view) );

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ //make a sampler for the textures
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
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler) );
	}
		
	{ //create the texture descriptor pool
		uint32_t per_texture = uint32_t(textures.size());

		std::array< VkDescriptorPoolSize, 1> pool_sizes{
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

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool) );
	}

	{ //allocate and write the texture descriptor sets
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_TEXTURE,
		};
		texture_descriptors.assign(textures.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet &descriptor_set : texture_descriptors) {
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set));
		}

		//write descriptors for textures
		std::vector<VkDescriptorImageInfo> infos(textures.size());
		std::vector<VkWriteDescriptorSet> writes(textures.size());

		for(Helpers::AllocatedImage const &image : textures) {
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

Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	if (texture_descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = nullptr;

		texture_descriptors.clear();
	}

	if (texture_sampler) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView &view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto &texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces) {
		refsol::Tutorial_destructor_workspace(rtg, command_pool, &workspace.command_buffer);

		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}
		if (workspace.Camera_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}
		if (workspace.World_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		if (workspace.Transforms_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
	}
	workspaces.clear();

	if (descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
	}

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);

	refsol::Tutorial_destructor(rtg, &render_pass, &command_pool);
}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain) {
	//[re]create framebuffers:
	refsol::Tutorial_on_swapchain(rtg, swapchain, depth_format, render_pass, &swapchain_depth_image, &swapchain_depth_image_view, &swapchain_framebuffers);
}

void Tutorial::destroy_framebuffers() {
	refsol::Tutorial_destroy_framebuffers(rtg, &swapchain_depth_image, &swapchain_depth_image_view, &swapchain_framebuffers);
}


void Tutorial::render(RTG &rtg_, RTG::RenderParams const &render_params) {
	//assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	//get more convenient names for the current workspace and target framebuffer:
	Workspace &workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	//record (into `workspace.command_buffer`) commands that run a `render_pass` that just clears `framebuffer`:
	//refsol::Tutorial_render_record_blank_frame(rtg, render_pass, framebuffer, &workspace.command_buffer);
	VK( vkResetCommandBuffer(workspace.command_buffer, 0) );
	{ //begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK( vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}

	if (!object_instances.empty()) {
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size < needed_bytes) {
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;

			if (workspace.Transforms_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}

			workspace.Transforms_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped
			);

			workspace.Transforms = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			//update descriptor set
			VkDescriptorBufferInfo Transforms_info {
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
				0, nullptr
			);

			std::cout << "Re-allocated object transforms buffers to " << new_bytes << " bytes." << std::endl;
		}

		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);

		{ //copy transforms into transforms_src
			assert(workspace.Transforms_src.allocation.mapped);
			ObjectsPipeline::Transform *out = reinterpret_cast<ObjectsPipeline::Transform*>(workspace.Transforms_src.allocation.data());

			for (ObjectInstance const &inst : object_instances) {
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
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD
		};
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

	{ //upload world info:
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

	{ //memory barrier
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
			0, nullptr
		);
	}

	{ //render pass
		std::array< VkClearValue, 2 > clear_values{
			VkClearValue{ .color{ .float32{0.4f, 0.8f, 1.0f, 1.0f} } },
			VkClearValue{ .depthStencil{ .depth = 1.0f, .stencil = 0 } },
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
				.offset = { .x = 0, .y = 0 },
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

		{ //draw with background pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background_pipeline.handle);

			{
				BackgroundPipeline::Push push {
					.time = time,
				};

				vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
			}

			vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		}

		{ //draw with lines pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);

			{
				std::array<VkBuffer, 1> vertex_buffers{ workspace.lines_vertices.handle };
				std::array<VkDeviceSize, 1> offsets { 0 };
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
					0, nullptr
				);
			}

			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
		}

		{
			if (!object_instances.empty()) { //draw with the objects pipeline:
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.handle);

				{ //use object_vertices (offset 0) as vertex buffer binding 0:
					std::array<VkBuffer, 1> vertex_buffers{ object_vertices.handle };
					std::array<VkDeviceSize, 1> offsets{ 0 };
					vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
				}

				{ //bind World and Transforms descriptor sets:
					std::array< VkDescriptorSet, 2 > descriptor_sets{
						workspace.World_descriptors,
						workspace.Transforms_descriptors,
					};
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						0,
						uint32_t(descriptor_sets.size()), descriptor_sets.data(),
						0, nullptr
					);
				}

				//Camera descriptor set is still bound, but unused

				//draw all instances:
				for (ObjectInstance const &inst : object_instances) {
					uint32_t index = uint32_t(&inst - &object_instances[0]);

					//bind texture descriptor set
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						objects_pipeline.layout,
						2,
						1, &texture_descriptors[inst.texture],
						0, nullptr
					);

					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
				}
			}
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}

	//end recording
	VK ( vkEndCommandBuffer(workspace.command_buffer));

	//submit `workspace.command buffer` for the GPU to run:
	refsol::Tutorial_render_submit(rtg, render_params, workspace.command_buffer);
}

void Tutorial::update(float dt) {
	time = std::fmod(time + dt, 60.0f);

	{
		if (camera_mode == CameraMode::Scene) {
			//camera rotating around the origin:
			float ang = float(M_PI) * 2.0f * 10.0f * (time / 60.0f);
			CLIP_FROM_WORLD = perspective(
				60.0f * float(M_PI) / 180.0f,
				rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
				0.1f,
				1000.0f
			) * look_at(
				3.0f * std::cos(ang), 3.0f * std::sin(ang), 1.0f,
				0.0f, 0.0f, 0.5f,
				0.0f, 0.0f, 1.0f
			);
		} else if (camera_mode == CameraMode::Free) {
			CLIP_FROM_WORLD = perspective(
				free_camera.fov,
				rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
				free_camera.near,
				free_camera.far
			) * orbit(
				free_camera.target_x, free_camera.target_y, free_camera.target_z,
				free_camera.azimuth, free_camera.elevation, free_camera.radius
			);
		} else {
			assert(0 && "only two camera modes");
		}
	}

	{ //sun and sky:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.1f;
		world.SKY_ENERGY.g = 0.1f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = std::cos(time / 3.0f);
		world.SUN_DIRECTION.y = std::sin(time / 3.0f);
		world.SUN_DIRECTION.z = 0.0f;

		world.SUN_ENERGY.r = 0.5f + 0.5f * std::cos(time * 2.0f);
		world.SUN_ENERGY.g = 0.5f + 0.5f * std::cos(time * 2.0f + 2.0f);
		world.SUN_ENERGY.b = 0.5f + 0.5f * std::cos(time * 2.0f + 4.0f);
	}

	{ //lines
		lines_vertices.clear();
		constexpr size_t count = 2 * 30 * 30 + 2 * 30 * 30;
		lines_vertices.reserve(count);

		auto get_spherized_pos = [&](float x, float y) {
			float u = (x + 1.0f) * 0.5f;
			float v = (y + 1.0f) * 0.5f;

			float theta = float(M_PI) * u; 
			float phi = 2.0f * float(M_PI) * v;

			float nx = std::sin(theta) * std::cos(phi);
			float ny = std::sin(theta) * std::sin(phi);
			float nz = std::cos(theta);

			uint8_t r = (uint8_t) (cos(time * 0.3f) * 255);
			uint8_t g = (uint8_t) (cos(time * 0.4f) * 255);
			uint8_t b = (uint8_t) (cos(time * 0.5f) * 255);

			return PosColVertex{
				.Position{ .x = nx, .y = ny, .z = nz },
				.Color{ .r = r, .g = g, .b = b, .a = 0xff }
			};
		};

		for (uint32_t i = 0; i < 30; ++i) {
			float y = (i + 0.5f) / 30 * 2.0f - 1.0f;
			for (uint32_t j = 0; j < 30; ++j) {
				float x0 = j / 30.0f * 2.0f - 1.0f;
				float x1 = (j + 1) / 30.0f  * 2.0f - 1.0f;
				lines_vertices.emplace_back(get_spherized_pos(x0, y));
				lines_vertices.emplace_back(get_spherized_pos(x1, y));
			}
		}

		for (uint32_t i = 0; i < 30; ++i) {
			float x = (i + 0.5f) / 30 * 2.0f - 1.0f;
			for (uint32_t j = 0; j < 30; ++j) {
				float y0 = j / 30.0f  * 2.0f - 1.0f;
				float y1 = (j + 1) / 30.0f  * 2.0f - 1.0f;
				lines_vertices.emplace_back(get_spherized_pos(x, y0));
				lines_vertices.emplace_back(get_spherized_pos(x, y1));
			}
		}

		assert(lines_vertices.size() == count);
	}

	{ //objects
		object_instances.clear();

		// { //plane translated +x by one unit:
		// 	mat4 WORLD_FROM_LOCAL{
		// 		1.0f, 0.0f, 0.0f, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		0.0f, 0.0f, 1.0f, 0.0f,
		// 		1.0f, 0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = plane_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 	});
		// }

		// { //torus translated -x by one unit and rotated CCW around +y:
		// 	float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
		// 	float ca = std::cos(ang);
		// 	float sa = std::sin(ang);
		// 	mat4 WORLD_FROM_LOCAL{
		// 		  ca, 0.0f,  -sa, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		  sa, 0.0f,   ca, 0.0f,
		// 		-1.0f,0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = torus_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 	});
		// }

		{ //pyramid stacks
			uint32_t pyramid_size = 5;
			float spacing = 2.0f;

			for (uint32_t z = 0; z < 3; ++z) {
				uint32_t current_size = pyramid_size - 2 * z;
				float z_offset = z * spacing;

				for (uint32_t y = 0; y < current_size; ++y) {
					float y_offset = (y - (current_size - 1) * 0.5f) * spacing;
					
					for (uint32_t x = 0; x < current_size; ++x) {
						float x_offset = (x - (current_size - 1) * 0.5f) * spacing;

						mat4 WORLD_FROM_LOCAL{
							1.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 1.0f, 0.0f,
							x_offset, y_offset, z_offset, 1.0f,
						};

						object_instances.emplace_back(ObjectInstance{
							.vertices = pyramid_vertices,
							.transform{
								.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
								.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
								.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
							},
							.texture = 2,
						});
					}
				}
			}
		}
	}
}

void Tutorial::on_input(InputEvent const &evt) {
	if (action) {
		action(evt);
		return;
	}

	//general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB) {
		camera_mode = CameraMode((int(camera_mode) + 1) % 2);
		return;
	}

	//free camera controls:
	if (camera_mode == CameraMode::Free) {

		if (evt.type == InputEvent::MouseWheel) {
			free_camera.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			free_camera.radius = std::max(free_camera.radius, 0.5f * free_camera.near);
			free_camera.radius = std::min(free_camera.radius, 2.0f * free_camera.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT)) {
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//image height at plane of target point:
					float height = 2.0f * std::tan(free_camera.fov * 0.5f) * free_camera.radius;

					//motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height * height;

					//compute camera transform to extract right (first row) and up (second row):
					mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius
					);

					//move the desired distance:
					free_camera.target_x = init_camera.target_x - dx * camera_from_world[0] - dy * camera_from_world[1];
					free_camera.target_y = init_camera.target_y - dx * camera_from_world[4] - dy * camera_from_world[5];
					free_camera.target_z = init_camera.target_z - dx * camera_from_world[8] - dy * camera_from_world[9];
					return;
				}
			};

			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
			//start tumbling

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;
			
			action = [this,init_x,init_y,init_camera](InputEvent const &evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//motion, normalized so 1.0 is window height:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height;

					//rotate camera based on motion:
					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					//reduce azimuth and elevation to [-pi,pi] range:
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