#include "A2-LutPipeline.hpp"
#include "../Helpers.hpp"
#include "../VK.hpp"

static uint32_t lut_code[] =
#include "../spv/lut.comp.inl"
;

void LutPipeline::create(RTG &rtg) {
    VkShaderModule module = rtg.helpers.create_shader_module(lut_code);

    {
        std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
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

        VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_lut) );
    }

    {
        std::array< VkDescriptorSetLayout, 1 > layouts{
            set1_lut
        };

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = uint32_t(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 0,
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

void LutPipeline::destroy(RTG &rtg) {
    if (handle != VK_NULL_HANDLE) {
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if (set1_lut != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(rtg.device, set1_lut, nullptr);
        set1_lut = VK_NULL_HANDLE;
    }
}