#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 position;

layout(push_constant) uniform Push {
    float time;
};

vec2 localUV;

void drawCircle(float radius, vec2 center, vec4 color, float squeeze) {
    float squeezeDirection = (localUV.y < center.y) ? squeeze : -squeeze;
    squeezeDirection *= radius;

    if (distance(localUV, center + vec2(0.0, squeezeDirection)) <= radius) {
        outColor = color;
    }
}

void main() {
    localUV = fract(position * 8.0);
    vec2 center = vec2(0.5);

    outColor = vec4(0.0, 0.0, 0.0, 1.0);

    float blink = pow(abs(sin(time)), 10.0);

    drawCircle(0.4, center, vec4(1.0), blink);
    drawCircle(0.2, center, vec4(position, 0.0, 1.0), blink);
    drawCircle(0.1, center, vec4(0.0, 0.0, 0.0, 1.0), blink);
}