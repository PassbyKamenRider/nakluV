#version 450

layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION;
    vec3 SKY_ENERGY;
    vec3 SUN_DIRECTION;
    vec3 SUN_ENERGY;
    vec3 EYE;
    vec3 EXPOSURE;    
    ivec3 TONEMAPPING; 
};

layout(set=2, binding=0) uniform sampler2D TEXTURE;
layout(set=2, binding=1) uniform sampler2D NORMAL;
layout(set=2, binding=2) uniform sampler2D ROUGHNESS;
layout(set=2, binding=3) uniform sampler2D METALNESS;
layout(set=2, binding=4) uniform sampler2D DISPLACEMENT;
layout(set=3, binding=0) uniform samplerCube RADIANCE;
layout(set=3, binding=1) uniform samplerCube IRRADIANCE;
layout(set=3, binding=2) uniform sampler2D BRDF_LUT;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;
layout(location = 4) flat in ivec4 materialType;

layout(location = 0) out vec4 outColor;

vec3 tonemap_reinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

vec3 apply_exposure_and_tonemap(vec3 radiance, float exposure_factor, int mode) {
    vec3 x = radiance * exposure_factor;

    if (mode == 1) {
        return tonemap_reinhard(x);
    }

    return x;
}

void main() {
    vec3 N = normalize(normal);
    vec3 T = normalize(tangent.xyz);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * tangent.w;
    mat3 TBN = mat3(T, B, N);

    vec3 normalSample = texture(NORMAL, texCoord).rgb;
    vec3 n = normalize(TBN * (normalSample * 2.0 - 1.0));
    
    vec3 v = normalize(EYE - position);
    vec3 r = reflect(-v, n);
    float NdotV = max(dot(n, v), 0.001);

    vec3 radiance;
    
    if (materialType.x == 3) {
        //RADIANCE
        radiance = texture(RADIANCE, n).rgb;
    }
    else if (materialType.x == 2) {
        //mirror
        vec3 v = normalize(EYE - position);
        vec3 r = reflect(-v, n);
        radiance = texture(RADIANCE, r).rgb;
    }
    else if (materialType.x == 1) {
        //lambertian
        vec3 albedo = texture(TEXTURE, texCoord).rgb;

        vec3 irradiance = texture(IRRADIANCE, n).rgb;

        vec3 energy = irradiance + SUN_ENERGY * max(dot(n, SUN_DIRECTION), 0.0);
        
        radiance = (albedo / 3.1415926) * energy;
    } else {
        //pbr
        vec3 albedo = texture(TEXTURE, texCoord).rgb;

        vec3 irradiance = texture(IRRADIANCE, n).rgb;

        vec3 energy = irradiance + SUN_ENERGY * max(dot(n, SUN_DIRECTION), 0.0);
        
        radiance = (albedo / 3.1415926) * energy;
    }

    vec3 color = apply_exposure_and_tonemap(
        radiance, 
        EXPOSURE.x, 
        TONEMAPPING.x
    );

    outColor = vec4(color, 1.0);
}