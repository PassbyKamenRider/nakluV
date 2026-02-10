#pragma once

#include <array>
#include <cmath>
#include <cstdint>

using mat4 = std::array<float, 16>;
static_assert(sizeof(mat4) == 16 * 4, "mat4 is exactly 16 32-bit floats.");

using vec4 = std::array<float, 4>;
static_assert(sizeof(vec4) == 4 * 4, "vec4 is exactly 4 32-bit floats.");

inline vec4 operator*(mat4 const &A, vec4 const &b) {
    vec4 ret;
    for (uint32_t r = 0; r < 4; ++r) {
        ret[r] = A[0 * 4 + r] * b[0];
        for (uint32_t k = 1; k < 4; ++k) {
            ret[r] += A[k * 4 + r] * b[k];
        }
    }
    return ret;
}

inline mat4 operator*(mat4 const &A, mat4 const &B) {
    mat4 ret;
    for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
            ret[c * 4 + r] = A[0 * 4 + r] * B[c * 4 + 0];
            for (uint32_t k = 1; k < 4; ++k) {
                ret[c * 4 + r] += A[k * 4 + r] * B[c * 4 + k];
            }
        }
    }
    return ret;
}

inline mat4 perspective(float vfov, float aspect, float near, float far) {
    const float e = 1.0f / std::tan(vfov / 2.0f);
    const float a = aspect;
    const float n = near;
    const float f = far;
    return mat4{
		e/a,  0.0f,                      0.0f, 0.0f,
		0.0f,   -e,                      0.0f, 0.0f,
		0.0f, 0.0f,-0.5f - 0.5f * (f+n)/(f-n),-1.0f,
		0.0f, 0.0f,             - (f*n)/(f-n), 0.0f,
	};
}

inline mat4 look_at(
	float eye_x, float eye_y, float eye_z,
	float target_x, float target_y, float target_z,
	float up_x, float up_y, float up_z ) {
	float in_x = target_x - eye_x;
	float in_y = target_y - eye_y;
	float in_z = target_z - eye_z;

	float inv_in_len = 1.0f / std::sqrt(in_x*in_x + in_y*in_y + in_z*in_z);
	in_x *= inv_in_len;
	in_y *= inv_in_len;
	in_z *= inv_in_len;

	float in_dot_up = in_x*up_x + in_y*up_y +in_z*up_z;
	up_x -= in_dot_up * in_x;
	up_y -= in_dot_up * in_y;
	up_z -= in_dot_up * in_z;

	float inv_up_len = 1.0f / std::sqrt(up_x*up_x + up_y*up_y + up_z*up_z);
	up_x *= inv_up_len;
	up_y *= inv_up_len;
	up_z *= inv_up_len;

	float right_x = in_y*up_z - in_z*up_y;
	float right_y = in_z*up_x - in_x*up_z;
	float right_z = in_x*up_y - in_y*up_x;

	float right_dot_eye = right_x*eye_x + right_y*eye_y + right_z*eye_z;
	float up_dot_eye = up_x*eye_x + up_y*eye_y + up_z*eye_z;
	float in_dot_eye = in_x*eye_x + in_y*eye_y + in_z*eye_z;

	return mat4{
		right_x, up_x, -in_x, 0.0f,
		right_y, up_y, -in_y, 0.0f,
		right_z, up_z, -in_z, 0.0f,
		-right_dot_eye, -up_dot_eye, in_dot_eye, 1.0f,
	};
}

//orbit camera matrix:
// makes a camera-from-world matrix for a camera orbiting target_{x,y,z}
//   at distance radius with angles azimuth and elevation.
// azimuth is counterclockwise angle in the xy plane from the x axis
// elevation is angle up from the xy plane
// both are in radians
inline mat4 orbit(
		float target_x, float target_y, float target_z,
		float azimuth, float elevation, float radius
	) {

	//shorthand for some useful trig values:
	float ca = std::cos(azimuth);
	float sa = std::sin(azimuth);
	float ce = std::cos(elevation);
	float se = std::sin(elevation);

	//camera's right direction is azimuth rotated by 90 degrees:
	float right_x =-sa;
	float right_y = ca;
	float right_z = 0.0f;

	//camera's up direction is elevation rotated 90 degrees:
	// (and points in the same xy direction as azimuth)
	float up_x = -se * ca;
	float up_y = -se * sa;
	float up_z = ce;

	//direction to the camera from the target:
	float out_x = ce * ca;
	float out_y = ce * sa;
	float out_z = se;

	//camera's position:
	float eye_x = target_x + radius * out_x;
	float eye_y = target_y + radius * out_y;
	float eye_z = target_z + radius * out_z;

	//camera's position projected onto the various vectors:
	float right_dot_eye = right_x*eye_x + right_y*eye_y + right_z*eye_z;
	float up_dot_eye    =    up_x*eye_x +    up_y*eye_y +    up_z*eye_z;
	float out_dot_eye   =   out_x*eye_x +   out_y*eye_y +   out_z*eye_z;

	//the final local-from-world transformation (column-major):
	return mat4{
		right_x, up_x, out_x, 0.0f,
		right_y, up_y, out_y, 0.0f,
		right_z, up_z, out_z, 0.0f,
		-right_dot_eye, -up_dot_eye, -out_dot_eye, 1.0f,
	};
}

inline mat4 mat4_translation(float x, float y, float z) {
    return mat4{
        1,0,0,0, 
        0,1,0,0, 
        0,0,1,0, 
        x,y,z,1
    };
}

inline mat4 mat4_scale(float x, float y, float z) {
    return mat4{
        x,0,0,0, 
        0,y,0,0, 
        0,0,z,0, 
        0,0,0,1
    };
}

inline mat4 mat4_rotation(float x, float y, float z, float w) {
    return mat4{
        1.0f - 2.0f*y*y - 2.0f*z*z, 2.0f*x*y + 2.0f*z*w,        2.0f*x*z - 2.0f*y*w,        0.0f,
        2.0f*x*y - 2.0f*z*w,        1.0f - 2.0f*x*x - 2.0f*z*z, 2.0f*y*z + 2.0f*x*w,        0.0f,
        2.0f*x*z + 2.0f*y*w,        2.0f*y*z - 2.0f*x*w,        1.0f - 2.0f*x*x - 2.0f*y*y, 0.0f,
        0.0f,                       0.0f,                       0.0f,                       1.0f
    };
}

inline mat4 mat4_identity() {
    return mat4{
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
}