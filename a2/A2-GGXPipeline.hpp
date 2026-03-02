#pragma once

#include "../RTG.hpp"
#include <glm/glm.hpp>

struct GGXPipeline {
    struct PushConstants {
        float roughness;
    };

    VkDescriptorSetLayout set0_env = VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_out = VK_NULL_HANDLE;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &rtg);
    void destroy(RTG &rtg);

    GGXPipeline() = default;
    GGXPipeline(GGXPipeline const &) = delete;
    GGXPipeline &operator=(GGXPipeline const &) = delete;
};