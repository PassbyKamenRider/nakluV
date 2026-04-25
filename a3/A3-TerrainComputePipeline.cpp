#include "A3-TerrainComputePipeline.hpp"

#include "../Helpers.hpp"
#include "../VK.hpp"

static uint32_t terrain_code[] =
#include "../spv/terrain.comp.inl"
;

void TerrainComputePipeline::create(RTG &rtg) {
	VkShaderModule module = rtg.helpers.create_shader_module(terrain_code);

	{ // set0: output vertex buffer
		std::array<VkDescriptorSetLayoutBinding, 1> bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			},
		};

		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_vertices));
	}

	{ // pipeline layout
		std::array<VkDescriptorSetLayout, 1> layouts{set0_vertices};

		VkPushConstantRange push_range{
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = sizeof(PushConstants),
		};

		VkPipelineLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(layouts.size()),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};

		VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
	}

	{ // compute pipeline
		VkComputePipelineCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = module,
				.pName = "main",
			},
			.layout = layout,
		};

		VK(vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
	}

	vkDestroyShaderModule(rtg.device, module, nullptr);
}

void TerrainComputePipeline::destroy(RTG &rtg) {
	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}
	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}
	if (set0_vertices != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_vertices, nullptr);
		set0_vertices = VK_NULL_HANDLE;
	}
}
