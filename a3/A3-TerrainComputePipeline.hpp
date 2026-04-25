#pragma once

#include "../RTG.hpp"

struct TerrainComputePipeline {
	VkDescriptorSetLayout set0_vertices = VK_NULL_HANDLE;

	struct PushConstants {
		float block_origin_x, block_origin_y, block_origin_z;
		float block_size;
		uint32_t resolution;
		float isovalue;
		uint32_t octaves;
		float persistence;
		float lacunarity;
		float frequency;
	};

	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkPipeline handle = VK_NULL_HANDLE;

	void create(RTG &);
	void destroy(RTG &);
};
