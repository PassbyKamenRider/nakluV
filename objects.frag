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

struct Light {
    mat4 world_from_local;
    vec4 type_power;    // x=type (1:Sphere, 2:Spot), y=power
    vec4 tint_shadow;   // rgb=tint, w=shadow
    vec4 params;        // x=radius, y=limit, z=cosInner, w=cosOuter
};

struct ShadowLight {
    mat4 world_from_local;
    vec4 type_power;
    vec4 tint_shadow;
    vec4 params;
    mat4 light_VP;
};

layout(set=2, binding=0) uniform sampler2D TEXTURE;
layout(set=2, binding=1) uniform sampler2D NORMAL;
layout(set=2, binding=2) uniform sampler2D ROUGHNESS;
layout(set=2, binding=3) uniform sampler2D METALNESS;
layout(set=2, binding=4) uniform sampler2D DISPLACEMENT;
layout(set=3, binding=0) uniform samplerCube RADIANCE;
layout(set=3, binding=1) uniform samplerCube IRRADIANCE;
layout(set=3, binding=2) uniform sampler2D BRDF_LUT;

layout(set=4, binding=0, std430) readonly buffer LightBuffer {
    Light lights[];
};

layout(set=4, binding=1, std430) readonly buffer ShadowLightBuffer {
    ShadowLight shadowLights[];
};

layout(set=5, binding=0) uniform sampler2DShadow shadowMaps[32];

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;
layout(location = 4) flat in ivec4 materialType;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

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

//reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
float GGXdistribution(vec3 N, vec3 H, float a) {
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

float calcSpotAttenuation(mat4 wfl, vec4 params, vec3 L, float d) {
    float limit = max(params.y, 0.001);
    float window = max(1.0 - pow(d / limit, 4.0), 0.0);
    float att = window / (d * d + 1.0);

    vec3 spotDir = normalize(mat3(wfl) * vec3(0.0, 0.0, -1.0));
    float cosTheta = dot(L, -spotDir);
    float angularAtt = clamp((cosTheta - params.w) / (params.z - params.w), 0.0, 1.0);
    att *= angularAtt;

    return att;
}

float calcLightAttenuation(Light l, vec3 L, float d) {
    float limit = max(l.params.y, 0.001);
    float window = max(1.0 - pow(d / limit, 4.0), 0.0);
    float att = window / (d * d + 1.0);

    if (l.type_power.x == 2.0) {
        vec3 spotDir = normalize(mat3(l.world_from_local) * vec3(0.0, 0.0, -1.0));
        float cosTheta = dot(L, -spotDir);
        float angularAtt = clamp((cosTheta - l.params.w) / (l.params.z - l.params.w), 0.0, 1.0);
        att *= angularAtt;
    }

    return att;
}

float sampleShadow(int idx, vec3 worldPos, mat4 lightVP, float mapSize) {
    vec4 lsPos = lightVP * vec4(worldPos, 1.0);
    if (lsPos.w <= 0.0) return 1.0;

    vec3 proj = lsPos.xyz / lsPos.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    float ref = proj.z * 0.5 + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ref < 0.0 || ref > 1.0)
        return 1.0;

    return texture(shadowMaps[idx], vec3(uv, ref));
}

vec4 shadePBR(vec3 n, vec3 v, vec3 r, float NdotV) {
    vec3 albedo = texture(TEXTURE, texCoord).rgb;
    float roughness = max(texture(ROUGHNESS, texCoord).r, 0.05);
    float metallic = texture(METALNESS, texCoord).r;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // IBL
    vec3 irradiance = textureLod(IRRADIANCE, n, 0.0).rgb;
    vec3 F_ibl = fresnelSchlick(NdotV, F0, roughness);
    vec3 kD_ibl = (vec3(1.0) - F_ibl) * (1.0 - metallic);
    vec3 diffuse_ibl = irradiance * albedo * kD_ibl;
    vec3 prefilteredColor = textureLod(RADIANCE, r, roughness * 5.0).rgb;
    vec2 brdf = texture(BRDF_LUT, vec2(NdotV, roughness)).rg;
    vec3 radiance = diffuse_ibl + prefilteredColor * (F_ibl * brdf.x + brdf.y);

    // Sun
    float sunNdotL = max(dot(n, SUN_DIRECTION), 0.0);
    if (sunNdotL > 0.0) {
        vec3 H = normalize(v + SUN_DIRECTION);
        vec3 F_s = fresnelSchlick(max(dot(H, v), 0.0), F0, roughness);
        vec3 kD_s = (1.0 - F_s) * (1.0 - metallic);
        float D = GGXdistribution(n, H, roughness * roughness);
        float G = SmithJoint(n, v, SUN_DIRECTION, roughness);
        vec3 spec = (D * G * F_s) / (4.0 * NdotV * sunNdotL + 0.0001);
        radiance += (kD_s * albedo / PI + spec) * SUN_ENERGY * sunNdotL;
    }

    // Non-shadow lights
    for (int i = 0; i < lights.length(); i++) {
        Light l = lights[i];
        vec3 lightWorldPos = (l.world_from_local * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec3 toLight = lightWorldPos - position;
        float d = length(toLight);
        vec3 L = normalize(toLight);
        float att = calcLightAttenuation(l, L, d);
        vec3 tint = l.tint_shadow.rgb * l.type_power.y / (4.0 * PI);
        float NdotL = max(dot(n, L), 0.0);

        if (NdotL > 0.0) {
            vec3 L_used = L;
            if (l.type_power.x == 1.0) { // sphere light
                vec3 centerToRay = dot(toLight, r) * r - toLight;
                L_used = normalize(toLight + centerToRay * clamp(l.params.x / max(length(centerToRay), 0.001), 0.0, 1.0));
            }

            vec3 H = normalize(v + L_used);
            vec3 F_l = fresnelSchlick(max(dot(H, v), 0.0), F0, roughness);
            vec3 kD_l = (1.0 - F_l) * (1.0 - metallic);
            float D = GGXdistribution(n, H, roughness * roughness);
            float G = SmithJoint(n, v, L_used, roughness);
            vec3 spec = (D * G * F_l) / (4.0 * NdotV * NdotL + 0.0001);
            radiance += (kD_l * albedo / PI + spec) * tint * att * NdotL;
        }
    }

    // Shadow spot lights
    for (int i = 0; i < shadowLights.length(); i++) {
        ShadowLight sl = shadowLights[i];
        vec3 lightWorldPos = (sl.world_from_local * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec3 toLight = lightWorldPos - position;
        float d = length(toLight);
        vec3 L = normalize(toLight);
        float att = calcSpotAttenuation(sl.world_from_local, sl.params, L, d);
        vec3 tint = sl.tint_shadow.rgb * sl.type_power.y / (4.0 * PI);
        float NdotL = max(dot(n, L), 0.0);

        if (NdotL > 0.0) {
            vec3 H = normalize(v + L);
            vec3 F_l = fresnelSchlick(max(dot(H, v), 0.0), F0, roughness);
            vec3 kD_l = (1.0 - F_l) * (1.0 - metallic);
            float D = GGXdistribution(n, H, roughness * roughness);
            float G = SmithJoint(n, v, L, roughness);
            vec3 spec = (D * G * F_l) / (4.0 * NdotV * NdotL + 0.0001);

            float shadow = sampleShadow(i, position, sl.light_VP, sl.tint_shadow.w);
            radiance += shadow * (kD_l * albedo / PI + spec) * tint * att * NdotL;
        }
    }

    return vec4(radiance, 1.0);
}

vec4 shadeLambertian(vec3 n) {
    vec3 albedo = texture(TEXTURE, texCoord).rgb;

    vec3 radiance = albedo * textureLod(IRRADIANCE, n, 0.0).rgb;

    // Sun
    float sunNdotL = max(dot(n, SUN_DIRECTION), 0.0);
    if (sunNdotL > 0.0) {
        radiance += (albedo / PI) * SUN_ENERGY * sunNdotL;
    }

    // Non-shadow lights
    for (int i = 0; i < lights.length(); i++) {
        Light l = lights[i];
        vec3 lightWorldPos = (l.world_from_local * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec3 toLight = lightWorldPos - position;
        float d = length(toLight);
        vec3 L = normalize(toLight);
        float att = calcLightAttenuation(l, L, d);
        vec3 tint = l.tint_shadow.rgb * l.type_power.y / (4.0 * PI);
        float NdotL = max(dot(n, L), 0.0);

        if (NdotL > 0.0) {
            radiance += (albedo / PI) * tint * att * NdotL;
        }
    }

    // Shadow spot lights
    for (int i = 0; i < shadowLights.length(); i++) {
        ShadowLight sl = shadowLights[i];
        vec3 lightWorldPos = (sl.world_from_local * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec3 toLight = lightWorldPos - position;
        float d = length(toLight);
        vec3 L = normalize(toLight);
        float att = calcSpotAttenuation(sl.world_from_local, sl.params, L, d);
        vec3 tint = sl.tint_shadow.rgb * sl.type_power.y / (4.0 * PI);
        float NdotL = max(dot(n, L), 0.0);

        if (NdotL > 0.0) {
            float shadow = sampleShadow(i, position, sl.light_VP, sl.tint_shadow.w);
            radiance += shadow * (albedo / PI) * tint * att * NdotL;
        }
    }

    return vec4(radiance, 1.0);
}

vec4 shadeMirror(vec3 n, vec3 v) {
    vec3 r = reflect(-v, n);
    vec3 radiance = textureLod(RADIANCE, r, 0.0).rgb;
    return vec4(radiance, 1.0);
}

vec4 shadeEnvironment(vec3 n) {
    vec3 radiance = textureLod(RADIANCE, n, 0.0).rgb;
    return vec4(radiance, 1.0);
}

void main() {
    vec3 N_geom = normalize(normal);
    vec3 v = normalize(EYE - position);

    vec4 computeColor;

    if (materialType.x == 0) { // PBR
        vec3 T = normalize(tangent.xyz);
        T = normalize(T - dot(T, N_geom) * N_geom);
        vec3 B = cross(N_geom, T) * tangent.w;
        mat3 TBN = mat3(T, B, N_geom);
        vec3 normalSample = texture(NORMAL, texCoord).rgb;
        vec3 n;
        if (normalSample == vec3(0.0)) {
            n = N_geom;
        } else {
            n = normalize(TBN * (normalSample * 2.0 - 1.0));
        }
        vec3 r = reflect(-v, n);
        float NdotV = dot(n, v);
        computeColor = shadePBR(n, v, r, NdotV);
    } else if (materialType.x == 1) { // Lambertian
        computeColor = shadeLambertian(N_geom);
    } else if (materialType.x == 2) { // Mirror
        computeColor = shadeMirror(N_geom, v);
    } else { // Environment
        computeColor = shadeEnvironment(N_geom);
    }

    outColor = vec4(apply_exposure_and_tonemap(computeColor.rgb, EXPOSURE.x, TONEMAPPING.x), 1.0);
}
