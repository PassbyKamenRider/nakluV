#include "Tutorial.hpp"

#include "VK.hpp"
#include "refsol.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_) {
	refsol::Tutorial_constructor(rtg, &depth_format, &render_pass, &command_pool);

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);

	workspaces.resize(rtg.workspaces.size());
	for (Workspace &workspace : workspaces) {
		refsol::Tutorial_constructor_workspace(rtg, command_pool, &workspace.command_buffer);
	}
}

Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

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
	}
	workspaces.clear();

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);

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

	if (!lines_vertices.empty()) {
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE || workspace.lines_vertices_src.size < needed_bytes) {
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;

			if (workspace.lines_vertices_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle) {
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

		assert(workspace.lines_vertices_src.allocation.mapped);
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		VkBufferCopy copy_reign {
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.handle, 1, &copy_reign);
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

			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
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
	}

	{
		lines_vertices.clear();
		constexpr size_t count = 2 * 30 * 30 + 2 * 30 * 30;
		lines_vertices.reserve(count);

		auto get_spherized_pos = [&](float x, float y) {
			float u = (x + 1.0f) * 0.5f;
			float v = (y + 1.0f) * 0.5f;

			float theta = (float(M_PI) / 2.0f) * u; 
			float phi = 2.0f * float(M_PI) * v;

			float nx = std::sin(theta) * std::cos(phi);
			float ny = std::sin(theta) * std::sin(phi);
			float nz = std::cos(theta);

			uint8_t r = (uint8_t) (sin(time * 0.2f) * 255);
			uint8_t g = (uint8_t) (cos(time * 0.2f) * 255);
			uint8_t b = (uint8_t) (sin(time * 0.5f) * 255);

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

	for (PosColVertex &v : lines_vertices) {
		vec4 res = CLIP_FROM_WORLD * vec4{v.Position.x, v.Position.y, v.Position.z, 1.0f};
		v.Position.x = res[0] / res[3];
		v.Position.y = res[1] / res[3];
		v.Position.z = res[2] / res[3];
	}
}

void Tutorial::on_input(InputEvent const &) {
}