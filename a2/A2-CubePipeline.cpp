#include "A2-CubePipeline.hpp"

#include "../Helpers.hpp"
#include "../VK.hpp"

static uint32_t cube_code[] =
#include "../spv/cube.comp.lambertian.inl"
;

void CubePipeline::create(RTG &rtg) {
	VkShaderModule module = rtg.helpers.create_shader_module(cube_code);
	
	{ //the set0_in layout holds input face info:
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
			VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set01_face) );
	}

	{ //the set2 layout holds roughness info (and maybe, someday, more brdf params):
		std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set2_params) );
	}


	{ //create pipeline layout:
		std::array< VkDescriptorSetLayout, 3 > layouts{
			set01_face,
			set01_face,
			set2_params,
		};

		VkPipelineLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(layouts.size()),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr,
		};

		VK( vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout) );
	}

	{ //create pipelines:

		VkComputePipelineCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.flags = VK_PIPELINE_CREATE_DISPATCH_BASE,
			.stage = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = module,
				.pName = "main"
			},
			.layout = layout,
		};

		VK( vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle) );

	}

	//modules no longer needed now that pipeline is created:
	vkDestroyShaderModule(rtg.device, module, nullptr);
}

void CubePipeline::destroy(RTG &rtg) {
	if (handle != VK_NULL_HANDLE) {
		vkDestroyPipeline(rtg.device, handle, nullptr);
		handle = VK_NULL_HANDLE;
	}

	if (layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(rtg.device, layout, nullptr);
		layout = VK_NULL_HANDLE;
	}

	if (set2_params != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set2_params, nullptr);
		set2_params = VK_NULL_HANDLE;
	}

	if (set01_face != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set01_face, nullptr);
		set01_face = VK_NULL_HANDLE;
	}
}