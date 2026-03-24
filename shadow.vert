#version 450

layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 Tangent;
layout(location=3) in vec2 TexCoord;

struct Transform {
    mat4 CLIP_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL_NORMAL;
    ivec4 MATERIAL_TYPE;
};

layout(set=0, binding=0, std140) readonly buffer Transforms {
    Transform TRANSFORMS[];
};

layout(push_constant) uniform Push {
    mat4 CLIP_FROM_WORLD;
};

void main() {
    vec3 world_pos = (TRANSFORMS[gl_InstanceIndex].WORLD_FROM_LOCAL * vec4(Position, 1.0)).xyz;
    vec4 clip = CLIP_FROM_WORLD * vec4(world_pos, 1.0);
    gl_Position = vec4(clip.xy, clip.z * 0.5 + clip.w * 0.5, clip.w);
}
