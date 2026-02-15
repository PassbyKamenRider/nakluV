#version 450

layout(set=0,binding=0,std140) uniform World {
	vec3 SKY_DIRECTION;
	vec3 SKY_ENERGY;
	vec3 SUN_DIRECTION;
	vec3 SUN_ENERGY;
};

layout(set=2, binding=0) uniform sampler2D TEXTURE;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(normal);
    vec3 albedo = texture(TEXTURE, texCoord).rgb;

	vec3 energy =
          SKY_ENERGY * (dot(n, SKY_DIRECTION) * 0.5 + 0.5)
        + SUN_ENERGY * max(dot(n, SUN_DIRECTION), 0.0);
    outColor = vec4(albedo / 3.1415926 * energy, 1.0);
}