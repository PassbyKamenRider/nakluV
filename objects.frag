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

const float PI = 3.14159265359;

//reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
float GGXdistribution(vec3 N, vec3 H, float a)
{
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / max(denom, 0.000001);
}

float G1_GGX(float NdotX, float alpha2) {
    float denom = NdotX + sqrt(alpha2 + (1.0 - alpha2) * (NdotX * NdotX));
    return 2.0 * NdotX / max(denom, 0.000001);
}

float SmithJoint(vec3 N, vec3 V, vec3 L, float roughness) {
    float alpha2 = pow(roughness, 4.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotL = max(dot(N, L), 0.001);
    
    return G1_GGX(NdotL, alpha2) * G1_GGX(NdotV, alpha2);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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
        //environment
        radiance = textureLod(RADIANCE, n, 0.0).rgb;
    }
    else if (materialType.x == 2) {
        //mirror
        radiance = textureLod(RADIANCE, r, 0.0).rgb;
    }
    else if (materialType.x == 1) {
        //lambertian
        vec3 albedo = texture(TEXTURE, texCoord).rgb;

        vec3 irradiance = texture(IRRADIANCE, n).rgb;

        vec3 direct = (albedo / 3.1415926) * SUN_ENERGY * max(dot(n, SUN_DIRECTION), 0.0);
        vec3 indirect = albedo * irradiance; 
        
        radiance = direct + indirect;
    } else {
        //pbr
        vec3 baseColor = texture(TEXTURE, texCoord).rgb;
        float roughness = texture(ROUGHNESS, texCoord).r;
        float metallic = texture(METALNESS, texCoord).r;
        
        float alpha = roughness * roughness;
        vec3 L = normalize(SUN_DIRECTION);
        vec3 H = normalize(v + L);
        float NdotL = max(dot(n, L), 0.0);
        float VdotH = max(dot(v, H), 0.0);

        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 c_diff = mix(baseColor, vec3(0.0), metallic);

        // direct lighting
        vec3 F_direct = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0); 
        float D = GGXdistribution(n, H, alpha);
        float G = SmithJoint(n, v, L, roughness);
        
        vec3 f_diffuse = (vec3(1.0) - F_direct) * (1.0 / PI) * c_diff;
        vec3 f_specular = (F_direct * D * G) / (4.0 * NdotV * max(NdotL, 0.001) + 0.0001);
        
        vec3 directLight = (f_diffuse + f_specular) * SUN_ENERGY * NdotL;

        // indrect lighting
        vec3 F = fresnelSchlick(NdotV, F0, roughness);

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        
        // diffuse
        vec3 irradiance = texture(IRRADIANCE, n).rgb;
        vec3 diffuse = kD * baseColor * irradiance;

        // specular
        // reference: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
        float MAX_LOD = 5.0; 
        vec3 prefilteredColor = textureLod(RADIANCE, r, roughness * MAX_LOD).rgb;
        vec2 envBRDF = texture(BRDF_LUT, vec2(NdotV, roughness)).rg;
        vec3 specularColor = prefilteredColor * (F * envBRDF.x + envBRDF.y);

        radiance = directLight + diffuse + specularColor;
    }

    vec3 color = apply_exposure_and_tonemap(radiance, EXPOSURE.x, TONEMAPPING.x);

    outColor = vec4(color, 1.0);
}