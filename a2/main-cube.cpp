#define _CRT_SECURE_NO_WARNINGS

#include "../RTG.hpp"
#include "A2-CubePipeline.hpp"
#include "A2-LutPipeline.hpp"
#include "A2-GGXPipeline.hpp"
#include "../VK.hpp"

#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb_image.h"

static glm::vec4 decode_rgbe(uint8_t r, uint8_t g, uint8_t b, uint8_t e) {
    float exponent = float(e);
    if (exponent == 0.0f) return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    float scale = std::pow(2.0f, exponent - 128.0f);
    return glm::vec4(float(r) / 255.0f * scale, float(g) / 255.0f * scale, float(b) / 255.0f * scale, 1.0f);
}

static void encode_rgbe(glm::vec3 rgb, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& e) {
    float maxComponent = std::max({rgb.r, rgb.g, rgb.b});
    if (maxComponent < 1e-9f) {
        r = g = b = e = 0;
        return;
    }
    float expVal = std::floor(std::log2(maxComponent)) + 1.0f;
    float scale = std::pow(2.0f, expVal);
    r = (uint8_t)std::clamp(std::round((rgb.r / scale) * 255.0f), 0.0f, 255.0f);
    g = (uint8_t)std::clamp(std::round((rgb.g / scale) * 255.0f), 0.0f, 255.0f);
    b = (uint8_t)std::clamp(std::round((rgb.b / scale) * 255.0f), 0.0f, 255.0f);
    e = (uint8_t)std::clamp(std::round(expVal + 128.0f), 0.0f, 255.0f);
}

struct GPUFace {
    Helpers::AllocatedImage image;
    Helpers::AllocatedBuffer buffer;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;

    void create(RTG &rtg, VkDescriptorPool descriptor_pool, CubePipeline const &pipeline, uint32_t const height, uint32_t const width, glm::vec4 * const data) {
        image = rtg.helpers.create_image(
            VkExtent2D{ .width = width, .height = height },
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped
        );

        if (data) {
            rtg.helpers.transfer_to_image(data, sizeof(glm::vec4) * height * width, image, VK_IMAGE_LAYOUT_GENERAL);
        }

        buffer = rtg.helpers.create_buffer(
            sizeof(CubePipeline::Face),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            Helpers::Unmapped
        );

        CubePipeline::Face face_info{};
        glm::vec3 s = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 t = glm::vec3(0.0f,-1.0f, -0.0f);
        glm::vec3 center = glm::vec3(1.0f, 0.0f, 0.0f);

        face_info.WORLD_FROM_PX.m0 = 2.0f * s.x / float(width);
        face_info.WORLD_FROM_PX.m1 = 2.0f * s.y / float(width);
        face_info.WORLD_FROM_PX.m2 = 2.0f * s.z / float(width);
        face_info.WORLD_FROM_PX.m3 = 2.0f * t.x / float(width);
        face_info.WORLD_FROM_PX.m4 = 2.0f * t.y / float(width);
        face_info.WORLD_FROM_PX.m5 = 2.0f * t.z / float(width);

        float corner = 1.0f - 2.0f / float(width) * 0.5f;
        face_info.WORLD_FROM_PX.m6 = center.x - corner * s.x - corner * t.x;
        face_info.WORLD_FROM_PX.m7 = center.y - corner * s.y - corner * t.y;
        face_info.WORLD_FROM_PX.m8 = center.z - corner * s.z - corner * t.z;

        rtg.helpers.transfer_to_buffer( &face_info, sizeof(face_info), buffer );

        VkImageViewCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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
        VK( vkCreateImageView(rtg.device, &create_info, nullptr, &view) );

        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &pipeline.set01_face,
        };
        VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptors) );

        VkDescriptorBufferInfo b_info{ .buffer = buffer.handle, .offset = 0, .range = buffer.size };
        VkDescriptorImageInfo i_info{ .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        std::array< VkWriteDescriptorSet, 2 > writes{
            VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptors, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &b_info },
            VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptors, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &i_info }
        };
        vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    }

    void destroy(RTG &rtg) {
        vkDestroyImageView(rtg.device, view, nullptr);
        rtg.helpers.destroy_buffer(std::move(buffer));
        rtg.helpers.destroy_image(std::move(image));
    }
};

void download_gpu_image(RTG &rtg, VkCommandBuffer cmd, VkQueue queue, Helpers::AllocatedImage &img, uint32_t w, uint32_t h, const char* filename) {
    Helpers::AllocatedBuffer staging = rtg.helpers.create_buffer(
        sizeof(glm::vec4) * w * h,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        Helpers::Mapped
    );

    VK( vkResetCommandBuffer(cmd, 0) );
    VkCommandBufferBeginInfo begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK( vkBeginCommandBuffer(cmd, &begin_info) );

    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = img.handle,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy copy_region{ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } };
    vkCmdCopyImageToBuffer(cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.handle, 1, &copy_region);
    VK( vkEndCommandBuffer(cmd) );

    VkSubmitInfo submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    VK( vkQueueSubmit(queue, 1, &submit, nullptr) );
    VK( vkDeviceWaitIdle(rtg.device) );

    glm::vec4* pixels = reinterpret_cast<glm::vec4*>(staging.allocation.mapped);
    std::vector<uint8_t> out8(w * h * 4);
    for (uint32_t i = 0; i < w * h; ++i) {
        encode_rgbe(glm::vec3(pixels[i]), out8[i*4+0], out8[i*4+1], out8[i*4+2], out8[i*4+3]);
    }
    stbi_write_png(filename, w, h, 4, out8.data(), w * 4);
    rtg.helpers.destroy_buffer(std::move(staging));
}

void download_and_save_lut(RTG &rtg, VkCommandBuffer cmd, VkQueue queue, Helpers::AllocatedImage &img, uint32_t w, uint32_t h, const std::string& base_path) {
    Helpers::AllocatedBuffer staging = rtg.helpers.create_buffer(
        sizeof(glm::vec4) * w * h,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        Helpers::Mapped
    );

    VK( vkResetCommandBuffer(cmd, 0) );
    VkCommandBufferBeginInfo begin_info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK( vkBeginCommandBuffer(cmd, &begin_info) );

    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = img.handle,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy copy_region{ .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { w, h, 1 } };
    vkCmdCopyImageToBuffer(cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.handle, 1, &copy_region);
    VK( vkEndCommandBuffer(cmd) );

    VkSubmitInfo submit{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    VK( vkQueueSubmit(queue, 1, &submit, nullptr) );
    VK( vkDeviceWaitIdle(rtg.device) );

    std::ofstream out(base_path + ".bin", std::ios::binary);
    if (out) {
        out.write(reinterpret_cast<const char*>(staging.allocation.mapped), staging.size);
        out.close();
    }

    glm::vec4* pixels = reinterpret_cast<glm::vec4*>(staging.allocation.mapped);
    std::vector<uint8_t> out8(w * h * 4);
    for (uint32_t i = 0; i < w * h; ++i) {
        out8[i*4+0] = (uint8_t)(glm::clamp(pixels[i].r, 0.0f, 1.0f) * 255.0f);
        out8[i*4+1] = (uint8_t)(glm::clamp(pixels[i].g, 0.0f, 1.0f) * 255.0f);
        out8[i*4+2] = (uint8_t)(glm::clamp(pixels[i].b, 0.0f, 1.0f) * 255.0f);
        out8[i*4+3] = (uint8_t)(glm::clamp(pixels[i].a, 0.0f, 1.0f) * 255.0f);
    }
    stbi_write_png((base_path + ".png").c_str(), w, h, 4, out8.data(), w * 4);
    rtg.helpers.destroy_buffer(std::move(staging));
}

int main(int argc, char **argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  cube <in.png> --lambertian <out.png>\n"
                  << "  cube --lut <out.png>\n"
                  << "  cube <in.png> --ggx <out.png>" << std::endl;
        return 1;
    }

    std::string input_path = "";
    std::string output_path = "";
    bool mode_lambertian = false;
    bool mode_lut = false;
    bool mode_ggx = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--lambertian") {
            mode_lambertian = true;
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--lut") {
            mode_lut = true;
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--ggx") {
            mode_ggx = true;
            if (i + 1 < argc) output_path = argv[++i];
        } else {
            input_path = arg;
        }
    }

    try {
        RTG::Configuration configuration;
        configuration.headless = true;
        RTG rtg(configuration);

        // --- Descriptor Pool ---
        std::array<VkDescriptorPoolSize, 2> pool_sizes{
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 20
            },
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 20
            }
        };

        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 40,
            .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data()
        };

        VkDescriptorPool descriptor_pool;
        VK( vkCreateDescriptorPool(
            rtg.device, 
            &pool_info, 
            nullptr, 
            &descriptor_pool) 
        );

        // --- Command Pool ---
        VkCommandPoolCreateInfo cp_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = rtg.graphics_queue_family.value()
        };

        VkCommandPool command_pool;
        VK( vkCreateCommandPool(
            rtg.device, 
            &cp_info, 
            nullptr, 
            &command_pool) 
        );
        
        // --- Command Buffer ---
        VkCommandBufferAllocateInfo cb_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

        VkCommandBuffer cmd;
        VK( vkAllocateCommandBuffers(
            rtg.device, 
            &cb_info, 
            &cmd) 
        );

        // --- Common Resource Loading ---
        if (mode_lambertian || mode_ggx) {
            int w, h, ch;
            uint8_t* raw_pixels = stbi_load(input_path.c_str(), &w, &h, &ch, 4);
            if (!raw_pixels) throw std::runtime_error("Failed to load input: " + input_path);
            
            std::vector<glm::vec4> float_data(w * h);
            for (int i = 0; i < w * h; ++i) {
                float_data[i] = decode_rgbe(
                    raw_pixels[i*4+0], 
                    raw_pixels[i*4+1], 
                    raw_pixels[i*4+2], 
                    raw_pixels[i*4+3]
                );
            }
            stbi_image_free(raw_pixels);

            if (mode_lambertian) {
                CubePipeline cube_pipe;
                cube_pipe.create(rtg);

                uint32_t target_sz = 16;
                GPUFace in_face, out_face;
                in_face.create(rtg, descriptor_pool, cube_pipe, h, w, float_data.data());
                out_face.create(rtg, descriptor_pool, cube_pipe, target_sz * 6, target_sz, nullptr);

                VkCommandBufferBeginInfo begin_info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                };

                VK( vkResetCommandBuffer(cmd, 0) );
                VK( vkBeginCommandBuffer(cmd, &begin_info) );

                vkCmdBindPipeline(
                    cmd, 
                    VK_PIPELINE_BIND_POINT_COMPUTE, 
                    cube_pipe.handle
                );

                std::array<VkDescriptorSet, 2> sets = { in_face.descriptors, out_face.descriptors };
                vkCmdBindDescriptorSets(
                    cmd, 
                    VK_PIPELINE_BIND_POINT_COMPUTE, 
                    cube_pipe.layout, 
                    0, 2, 
                    sets.data(), 
                    0, nullptr
                );

                vkCmdDispatch(
                    cmd, 
                    (target_sz + 7) / 8, 
                    (target_sz * 6 + 7) / 8, 
                    1
                );

                VK( vkEndCommandBuffer(cmd) );

                VkSubmitInfo submit_info{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &cmd
                };

                VK( vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, nullptr) );
                VK( vkDeviceWaitIdle(rtg.device) );

                download_gpu_image(
                    rtg, cmd, rtg.graphics_queue, 
                    out_face.image, target_sz, target_sz * 6, 
                    output_path.c_str()
                );

                in_face.destroy(rtg);
                out_face.destroy(rtg);
                cube_pipe.destroy(rtg);
            }

            if (mode_ggx) {
                GGXPipeline ggx_pipe;
                ggx_pipe.create(rtg);

                Helpers::AllocatedImage in_img = rtg.helpers.create_image(
                    { (uint32_t)w, (uint32_t)h },
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                );
                rtg.helpers.transfer_to_image(float_data.data(), float_data.size() * sizeof(glm::vec4), in_img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                VkSamplerCreateInfo sampler_info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .magFilter = VK_FILTER_LINEAR,
                    .minFilter = VK_FILTER_LINEAR,
                    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                };
                VkSampler sampler;
                VK( vkCreateSampler(rtg.device, &sampler_info, nullptr, &sampler) );

                VkImageViewCreateInfo iv_info{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = in_img.handle,
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = in_img.format,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                };
                VkImageView in_view;
                VK( vkCreateImageView(rtg.device, &iv_info, nullptr, &in_view) );

                VkDescriptorSetAllocateInfo ds_alloc{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &ggx_pipe.set0_env
                };
                VkDescriptorSet set0;
                VK( vkAllocateDescriptorSets(rtg.device, &ds_alloc, &set0) );

                VkDescriptorImageInfo di_info{
                    .sampler = sampler,
                    .imageView = in_view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                VkWriteDescriptorSet ds_write{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = set0,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &di_info
                };
                vkUpdateDescriptorSets(rtg.device, 1, &ds_write, 0, nullptr);

                uint32_t base_sz = (uint32_t)w;
                uint32_t max_mips = 5;

                for (uint32_t m = 1; m <= max_mips; ++m) {
                    uint32_t cur_sz = base_sz >> m;
                    if (cur_sz < 1) break;
                    float roughness = static_cast<float>(m) / static_cast<float>(max_mips);

                    Helpers::AllocatedImage out_img = rtg.helpers.create_image(
                        { cur_sz, cur_sz * 6 },
                        VK_FORMAT_R32G32B32A32_SFLOAT,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    );

                    VkImageViewCreateInfo oiv_info{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .image = out_img.handle,
                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                        .format = out_img.format,
                        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                    };
                    VkImageView out_view;
                    VK( vkCreateImageView(rtg.device, &oiv_info, nullptr, &out_view) );

                    VkDescriptorSetAllocateInfo ds_alloc1{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &ggx_pipe.set1_out
                    };
                    VkDescriptorSet set1;
                    VK( vkAllocateDescriptorSets(rtg.device, &ds_alloc1, &set1) );

                    VkDescriptorImageInfo di_info1{
                        .imageView = out_view,
                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
                    };
                    VkWriteDescriptorSet ds_write1{
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = set1,
                        .dstBinding = 1,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .pImageInfo = &di_info1
                    };
                    vkUpdateDescriptorSets(rtg.device, 1, &ds_write1, 0, nullptr);

                    VkCommandBufferBeginInfo b_info{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                    };
                    VK( vkResetCommandBuffer(cmd, 0) );
                    VK( vkBeginCommandBuffer(cmd, &b_info) );
                    
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .image = out_img.handle,
                        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                    };
                    vkCmdPipelineBarrier(
                        cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                        0, 0, nullptr, 0, nullptr, 1, &barrier
                    );
                    
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ggx_pipe.handle);
                    vkCmdPushConstants(
                        cmd, ggx_pipe.layout, 
                        VK_SHADER_STAGE_COMPUTE_BIT, 
                        0, sizeof(float), &roughness
                    );
                    std::array<VkDescriptorSet, 2> g_sets = { set0, set1 };
                    vkCmdBindDescriptorSets(
                        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                        ggx_pipe.layout, 0, 2, g_sets.data(), 0, nullptr
                    );
                    vkCmdDispatch(cmd, (cur_sz + 7) / 8, (cur_sz * 6 + 7) / 8, 1);
                    VK( vkEndCommandBuffer(cmd) );

                    VkSubmitInfo s_info{
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &cmd
                    };
                    VK( vkQueueSubmit(rtg.graphics_queue, 1, &s_info, nullptr) );
                    VK( vkDeviceWaitIdle(rtg.device) );

                    std::string out_name = output_path;
                    size_t dot_pos = out_name.find_last_of('.');
                    std::string base_name = (dot_pos == std::string::npos) ? out_name : out_name.substr(0, dot_pos);
                    std::string final_path = base_name + "." + std::to_string(m) + ".png";
                    
                    download_gpu_image(
                        rtg, cmd, rtg.graphics_queue, 
                        out_img, cur_sz, cur_sz * 6, 
                        final_path.c_str()
                    );

                    vkDestroyImageView(rtg.device, out_view, nullptr);
                    rtg.helpers.destroy_image(std::move(out_img));
                }

                vkDestroyImageView(rtg.device, in_view, nullptr);
                vkDestroySampler(rtg.device, sampler, nullptr);
                rtg.helpers.destroy_image(std::move(in_img));
                ggx_pipe.destroy(rtg);
            }
        }

        if (mode_lut) {
            LutPipeline lut_pipe;
            lut_pipe.create(rtg);

            uint32_t lut_sz = 512;
            Helpers::AllocatedImage lut_img = rtg.helpers.create_image(
                { lut_sz, lut_sz },
                VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );

            VkImageViewCreateInfo lv_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = lut_img.handle,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = lut_img.format,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };
            VkImageView lut_view;
            VK( vkCreateImageView(rtg.device, &lv_info, nullptr, &lut_view) );

            VkDescriptorSetAllocateInfo ls_alloc{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &lut_pipe.set1_lut
            };
            VkDescriptorSet lut_set;
            VK( vkAllocateDescriptorSets(rtg.device, &ls_alloc, &lut_set) );

            VkDescriptorImageInfo ldi_info{
                .imageView = lut_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            };
            VkWriteDescriptorSet ls_write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = lut_set,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &ldi_info
            };
            vkUpdateDescriptorSets(rtg.device, 1, &ls_write, 0, nullptr);

            VkCommandBufferBeginInfo lb_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            };
            VK( vkResetCommandBuffer(cmd, 0) );
            VK( vkBeginCommandBuffer(cmd, &lb_info) );

            VkImageMemoryBarrier l_bar{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = lut_img.handle,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };
            vkCmdPipelineBarrier(
                cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                0, 0, nullptr, 0, nullptr, 1, &l_bar
            );

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lut_pipe.handle);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                lut_pipe.layout, 0, 1, &lut_set, 0, nullptr
            );
            vkCmdDispatch(cmd, (lut_sz + 7) / 8, (lut_sz + 7) / 8, 1);
            VK( vkEndCommandBuffer(cmd) );

            VkSubmitInfo l_submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd
            };
            VK( vkQueueSubmit(rtg.graphics_queue, 1, &l_submit, nullptr) );
            VK( vkDeviceWaitIdle(rtg.device) );

            std::string base_path = output_path;
            if (base_path.size() > 4) base_path = base_path.substr(0, base_path.size() - 4);
            download_and_save_lut(rtg, cmd, rtg.graphics_queue, lut_img, lut_sz, lut_sz, base_path);

            vkDestroyImageView(rtg.device, lut_view, nullptr);
            rtg.helpers.destroy_image(std::move(lut_img));
            lut_pipe.destroy(rtg);
        }

        vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
        vkDestroyCommandPool(rtg.device, command_pool, nullptr);

    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "Total computation time: " << duration.count() << " ms" << std::endl;

    return 0;
}