#pragma once

#include "../PosColVertex.hpp"
#include "../PosNorTexVertex.hpp"
#include "../a2/PosNorTanTexVertex.hpp"
#include "../libs/S72.hpp"

#include "../RTG.hpp"

#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct S72Loader : RTG::Application {
    S72Loader(RTG &);
    S72Loader(S72Loader const &) = delete; //you shouldn't be copying this object
    ~S72Loader();

    //kept for use in destructor:
    RTG &rtg;

    //--------------------------------------------------------------------
    //Resources that last the lifetime of the application:

    //chosen format for depth buffer:
    VkFormat depth_format{};
    //Render passes describe how pipelines write to images:
    VkRenderPass render_pass = VK_NULL_HANDLE;

    //Pipelines:
    struct BackgroundPipeline {
        struct Push {
            float time;
        };

        VkPipelineLayout layout = VK_NULL_HANDLE;

        VkPipeline handle = VK_NULL_HANDLE;

        void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
        void destroy(RTG &);
    } background_pipeline;

    struct LinesPipeline {
        VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;

        struct Camera {
            glm::mat4 CLIP_FROM_WORLD;
        };
        static_assert(sizeof(Camera) == 16*4, "camera buffer structure is packed");

        VkPipelineLayout layout = VK_NULL_HANDLE;

        using Vertex = PosColVertex;

        VkPipeline handle = VK_NULL_HANDLE;

        void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
        void destroy(RTG &);
    } lines_pipeline;

    struct ObjectsPipeline {
        //VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;
        VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
        VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
        VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;
        VkDescriptorSetLayout set3_Environment = VK_NULL_HANDLE;

        struct World {
            struct { float x, y, z, padding_; } SKY_DIRECTION;
            struct { float r, g, b, padding_; } SKY_ENERGY;
            struct { float x, y, z, padding_; } SUN_DIRECTION;
            struct { float r, g, b, padding_; } SUN_ENERGY;
            struct { float x, y, z, padding_; } EYE;
            struct { float exposure, padding0, padding1, padding2; } EXPOSURE;
            struct { int tone_mapping_mode, padding0, padding1, padding2; } TONEMAPPING;
        };
        static_assert(sizeof(World) == 7*4*4, "World is the expected size.");

        struct Transform {
            glm::mat4 CLIP_FROM_LOCAL;
            glm::mat4 WORLD_FROM_LOCAL;
            glm::mat4 WORLD_FROM_LOCAL_NORMAL;
            struct { int materialType, padding0, padding1, padding2; } MATERIAL_TYPE;
        };
        static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4 + 16, "Transform is the expected size.");

        using Camera = LinesPipeline::Camera;

        VkPipelineLayout layout = VK_NULL_HANDLE;

        using Vertex = PosNorTanTexVertex;

        VkPipeline handle = VK_NULL_HANDLE;

        void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
        void destroy(RTG &);
    } objects_pipeline;

    //pools from which per-workspace things are allocated:
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    //workspaces hold per-render resources:
    struct Workspace {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE; //from the command pool above; reset at the start of every render.

        Helpers::AllocatedBuffer lines_vertices_src;
        Helpers::AllocatedBuffer lines_vertices;

        Helpers::AllocatedBuffer Camera_src;
        Helpers::AllocatedBuffer Camera;
        VkDescriptorSet Camera_descriptors;

        //location for ObjectsPipeline::World data: (streamed to GPU per-frame)
        Helpers::AllocatedBuffer World_src;
        Helpers::AllocatedBuffer World;
        VkDescriptorSet World_descriptors;

        //location for ObjectsPipeline::Transforms data: (streamed to GPU per-frame)
        Helpers::AllocatedBuffer Transforms_src;
        Helpers::AllocatedBuffer Transforms;
        VkDescriptorSet Transforms_descriptors;
    };
    std::vector< Workspace > workspaces;

    //-------------------------------------------------------------------
    //static scene resources:
    struct CullingFrustum {
        float near_right;
        float near_top;
        float near_plane;
        float far_plane;
    } active_frustum;

    struct AABB
    {
        glm::vec3 min;
        glm::vec3 max;
    };

    struct OBB {
        glm::vec3 center = {};
        glm::vec3 extents = {};
        glm::vec3 axes[3] = {};
    };

    Helpers::AllocatedBuffer object_vertices;
    struct ObjectVertices{
        uint32_t first = 0;
        uint32_t count = 0;
        AABB aabb;
    };

    std::vector<Helpers::AllocatedImage> textures;
    std::vector<VkImageView> texture_views;
    VkSampler texture_sampler = VK_NULL_HANDLE;
    VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> texture_descriptors;

    //--------------------------------------------------------------------
    //Resources that change when the swapchain is resized:

    virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

    Helpers::AllocatedImage swapchain_depth_image;
    VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
    std::vector< VkFramebuffer > swapchain_framebuffers;
    //used from on_swapchain and the destructor: (framebuffers are created in on_swapchain)
    void destroy_framebuffers();

    //--------------------------------------------------------------------
    //Resources that change when time passes or the user interacts:

    virtual void update(float dt) override;
    virtual void on_input(InputEvent const &) override;

    //modal action, intercepts inputs:
    std::function< void(InputEvent const &) > action;

    float time = 0.0f;

    //for selecting between cameras:
    enum class CameraMode {
        Scene = 0,
        Free = 1,
    } camera_mode = CameraMode::Scene;

    //used when camera_mode == CameraMode::Free:
    struct OrbitCamera {
        float target_x = 0.0f, target_y = 0.0f, target_z = 0.0f;
        float radius = 2.0f;
        float azimuth = 0.0f;
        float elevation = 0.25f * float(M_PI);

        float fov = 60.0f / 180.0f * float(M_PI);
        float near = 0.1f;
        float far = 1000.0f;
    } free_camera;

    //computed from the current camera (as set by camera_mode) during update():
    glm::mat4 CLIP_FROM_WORLD;

    std::vector<LinesPipeline::Vertex> lines_vertices;

    ObjectsPipeline::World world;
    //--------------------------------------------------------------------
    //Rendering function, uses all the resources above to queue work to draw a frame:

    virtual void render(RTG &, RTG::RenderParams const &) override;

    //---------------------------------------------------------------------
    //Added for a1:
    struct MaterialConstants {
        struct { uint32_t type; int32_t has_albedo_map; int32_t has_normal_map; int32_t has_displacement; } FLAGS;
        struct { float roughness; float metalness; float padding1; float padding2; } PARAMS;
        struct { float r, g, b; float displacement_scale; } ALBEDO;
    };
    static_assert(sizeof(MaterialConstants) == 3 * 4 * 4, "MaterialConstants is the expected size.");

    struct MaterialInstance {
        uint32_t type = 0; //0 = PBR, 1 = lambertian, 2 = mirror, 3 = environment
        int32_t normal_index = 1; //default is textures[1]
        int32_t displacement_idx = 0;
        int32_t albedo_index = 0; //default is textures[0]
        int32_t roughness_index = 2; //default is textures[2]
        int32_t metalness_index = 3; //default is textures[3]

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    };

    struct MeshInstance {
        std::string name;
        ObjectVertices* vertices = nullptr;
        MaterialInstance* material = nullptr;
        glm::mat4 world_from_local = glm::mat4(1.0f);
    };

    struct CameraInstance {
        std::string name;
        S72::Camera::Perspective props;
        glm::mat4 world_from_local = glm::mat4(1.0f);
    };

    struct EnvironmentInstance {
        uint32_t radiance_idx;
        uint32_t lambertian_idx;

        VkDescriptorSet env_descriptors = VK_NULL_HANDLE;
    };

    struct ObjectInstance {
        ObjectVertices vertices;
        ObjectsPipeline::Transform transform;
        MaterialInstance* material = nullptr;
        VkDescriptorSet environment_cube_set = VK_NULL_HANDLE;
    };
    std::vector<ObjectInstance> object_instances;

    S72 scene;
    std::unordered_map<std::string, ObjectVertices> mesh_data;
    std::unordered_map<std::string, MaterialInstance> material_data; 
    std::unordered_map<std::string, CameraInstance> cameras;
    std::unordered_map<std::string, EnvironmentInstance> environment_data;
    std::unordered_map<std::string, uint32_t> loaded_textures;

    std::vector<Helpers::AllocatedImage> textures_cube;
    std::vector<VkImageView> texture_views_cube;

    std::vector<CameraInstance*> camera_list;
    int current_camera_index = 0;

    std::vector<MeshInstance> mesh_instances;
    CameraInstance* active_camera = nullptr;
    bool isDebugMode = false;

    float playback_time = 0.0f;
    void update_animations(float current_time);

    void traverse_scene(S72::Node* node, glm::mat4 const &parent_from_world);
    uint32_t load_texture(std::string path, S72::Texture::Format format);
    uint32_t load_cubemap(std::string path, uint32_t max_lod = 0);
    uint32_t create_1x1_texture(float r, float g, float b, float a, bool is_srgb);

    // SAT test
    // reference: https://bruop.github.io/improved_frustum_culling/
    bool SAT_visibility_test(const CullingFrustum& frustum, const glm::mat4& vs_transform, const AABB& aabb);
};