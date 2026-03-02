#include "A2-GGXPipeline.hpp"
#include "../Helpers.hpp"
#include "../VK.hpp"

static uint32_t cube_code[] = 
#include "../spv/ggx.comp.inl" 
;

void GGXPipeline::create(RTG &rtg) {
    VkShaderModule module = rtg.helpers.create_shader_module(cube_code);
    
    {
        std::array<VkDescriptorSetLayoutBinding, 1> bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
        };
        
        VkDescriptorSetLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = uint32_t(bindings.size()),
            .pBindings = bindings.data(),
        };

        VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_env) );
    }

    {
        std::array<VkDescriptorSetLayoutBinding, 1> bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };
        
        VkDescriptorSetLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = uint32_t(bindings.size()),
            .pBindings = bindings.data(),
        };

        VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_out) );
    }

    VkPushConstantRange push_constant_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(float),
    };

    {
        std::array<VkDescriptorSetLayout, 2> layouts{
            set0_env,
            set1_out
        };

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = uint32_t(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
        };

        VK( vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout) );
    }

    {
        VkComputePipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
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

    vkDestroyShaderModule(rtg.device, module, nullptr);
}

void GGXPipeline::destroy(RTG &rtg) {
    if (handle != VK_NULL_HANDLE) vkDestroyPipeline(rtg.device, handle, nullptr);
    if (layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(rtg.device, layout, nullptr);
    if (set0_env != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(rtg.device, set0_env, nullptr);
    if (set1_out != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(rtg.device, set1_out, nullptr);
    
    handle = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    set0_env = VK_NULL_HANDLE;
    set1_out = VK_NULL_HANDLE;
}