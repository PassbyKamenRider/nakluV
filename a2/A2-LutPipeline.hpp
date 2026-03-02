#pragma once

#include "../RTG.hpp"

struct LutPipeline {
    void create(RTG &rtg);
    void destroy(RTG &rtg);

    VkDescriptorSetLayout set1_lut = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
};