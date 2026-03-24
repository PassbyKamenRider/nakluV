#include "../a1/S72Loader.hpp"

#include "../Helpers.hpp"
#include "../VK.hpp"

static uint32_t shadow_vert_code[] =
#include "../spv/shadow.vert.inl"
;

void S72Loader::ShadowPipeline::create(RTG &rtg, VkFormat depth_format, VkDescriptorSetLayout transforms_layout) {

	{ // shadow render pass
		VkAttachmentDescription depth_attachment{
			.format = depth_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		VkAttachmentReference depth_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass_desc{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 0,
			.pDepthStencilAttachment = &depth_ref,
		};

		VkSubpassDependency dependency{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};

		VkRenderPassCreateInfo rp_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &depth_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass_desc,
			.dependencyCount = 1,
			.pDependencies = &dependency,
		};

		VK(vkCreateRenderPass(rtg.device, &rp_info, nullptr, &render_pass));
	}

	{ // shadow sampler
		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.compareEnable = VK_TRUE,
			.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};
		VK(vkCreateSampler(rtg.device, &sampler_info, nullptr, &sampler));
	}

	{ // dummy shadow image (used to occupy unused shadowmap slots)
		dummy_image = rtg.helpers.create_image(
			{1, 1},
			depth_format,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		);

		VK(vkResetCommandBuffer(rtg.helpers.transfer_command_buffer, 0));
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK(vkBeginCommandBuffer(rtg.helpers.transfer_command_buffer, &begin_info));

		VkImageMemoryBarrier barrier1{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = dummy_image.handle,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0, .levelCount = 1,
				.baseArrayLayer = 0, .layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(rtg.helpers.transfer_command_buffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier1);

		VkClearDepthStencilValue clear_val{.depth = 1.0f, .stencil = 0};
		VkImageSubresourceRange range{
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0, .levelCount = 1,
			.baseArrayLayer = 0, .layerCount = 1,
		};
		vkCmdClearDepthStencilImage(rtg.helpers.transfer_command_buffer,
			dummy_image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			&clear_val, 1, &range);

		VkImageMemoryBarrier barrier2{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = dummy_image.handle,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0, .levelCount = 1,
				.baseArrayLayer = 0, .layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(rtg.helpers.transfer_command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier2);

		VK(vkEndCommandBuffer(rtg.helpers.transfer_command_buffer));

		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &rtg.helpers.transfer_command_buffer,
		};
		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
		VK(vkQueueWaitIdle(rtg.graphics_queue));

		// Create image view
		VkImageViewCreateInfo view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = dummy_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0, .levelCount = 1,
				.baseArrayLayer = 0, .layerCount = 1,
			},
		};
		VK(vkCreateImageView(rtg.device, &view_info, nullptr, &dummy_view));
	}

	{ // set0_Lights descriptor set layout (for shadow pipeline)
		VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		};

		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1,
			.pBindings = &binding,
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_Lights));
	}

	{ // shadow pipeline layout
		std::array<VkDescriptorSetLayout, 2> layouts{
			set0_Lights,         // set0: shadow lights SSBO
			transforms_layout,   // set1: transforms SSBO
		};

		VkPushConstantRange push_constant{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(uint32_t),
		};

		VkPipelineLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(layouts.size()),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constant,
		};

		VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
	}

	{ // shadow pipeline
		VkShaderModule shadow_vert_module = rtg.helpers.create_shader_module(shadow_vert_code);

		VkPipelineShaderStageCreateInfo stage{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shadow_vert_module,
			.pName = "main",
		};

		std::array<VkDynamicState, 2> dynamic_states{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		};

		VkPipelineDynamicStateCreateInfo dynamic_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = uint32_t(dynamic_states.size()),
			.pDynamicStates = dynamic_states.data(),
		};

		VkPipelineInputAssemblyStateCreateInfo input_assembly{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		VkPipelineViewportStateCreateInfo viewport_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		VkPipelineRasterizationStateCreateInfo rasterization_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.depthBiasEnable = VK_TRUE,
			.depthBiasConstantFactor = 1.25f,
			.depthBiasSlopeFactor = 1.75f,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo multisample_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
		};

		VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

		VkPipelineColorBlendStateCreateInfo color_blend_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = 0,
			.pAttachments = nullptr,
		};

		VkGraphicsPipelineCreateInfo pipeline_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 1,
			.pStages = &stage,
			.pVertexInputState = &ObjectsPipeline::Vertex::array_input_state,
			.pInputAssemblyState = &input_assembly,
			.pViewportState = &viewport_state,
			.pRasterizationState = &rasterization_state,
			.pMultisampleState = &multisample_state,
			.pDepthStencilState = &depth_stencil_state,
			.pColorBlendState = &color_blend_state,
			.pDynamicState = &dynamic_state,
			.layout = layout,
			.renderPass = render_pass,
			.subpass = 0,
		};

		VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &handle));

		vkDestroyShaderModule(rtg.device, shadow_vert_module, nullptr);
	}
}

void S72Loader::ShadowPipeline::destroy(RTG &rtg) {
	if (dummy_view != VK_NULL_HANDLE) {
		vkDestroyImageView(rtg.device, dummy_view, nullptr);
		dummy_view = VK_NULL_HANDLE;
	}
	if (dummy_image.handle != VK_NULL_HANDLE) {
		rtg.helpers.destroy_image(std::move(dummy_image));
	}
	if (sampler != VK_NULL_HANDLE) {
		vkDestroySampler(rtg.device, sampler, nullptr);
		sampler = VK_NULL_HANDLE;
	}
	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}
	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}
	if (set0_Lights != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_Lights, nullptr);
		set0_Lights = VK_NULL_HANDLE;
	}
	if (render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}
