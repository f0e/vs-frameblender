// the plane kernels, compiled once per ISA level (see kernel_*.cpp and CMakeLists.txt).
//
// the loop order is the whole point. the obvious way - hold N row pointers and reduce across
// them per pixel - only vectorises when N is a compile time constant, which is why upstream
// AverageFrames templates on the source count and caps out at 31. here j is innermost over a
// row sized accumulator instead: one vectorised pass per source frame, nothing depending on N.
// the accumulator is w*4 bytes so it stays in L1, and there is no cap.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace fb {

struct PlaneJob {
    const uint8_t *const *srcs;
    unsigned num_srcs;
    uint8_t *dst;
    unsigned w;
    unsigned h;
    ptrdiff_t stride;
    int32_t bias;    // chroma midpoint, 0 for luma and rgb
    int32_t maxval;
    // reference black and white the curve is anchored to, in sample units. full range is 0 and
    // maxval; limited range footage is 16 and 219 wide at 8 bit, which leaves superwhite above
    // the curve's 1.0 instead of clipped off at it. unused when the curve is off.
    float black;
    float span;
};

// ties to even, matching SSE cvtps_epi32 and upstream AverageFrames.
//
// adding 1.5*2^23 pushes the fraction off the end of the mantissa, which rounds it under the
// current mode (nearest-even), and subtracting it back leaves the rounded value. upstream uses
// lrintf and -fno-math-errno to stop it becoming a libcall, but msvc has no such flag - this
// lowers to two vector adds everywhere. exact for |x| < 2^22, and our accumulators are far
// inside that. do not build this file with -ffast-math, which would fold the pair away.
static inline int32_t round_nearest_i32(float x) {
    const float magic = 12582912.0f; // 1.5f * (1 << 23)
    return static_cast<int32_t>((x + magic) - magic);
}

// every body copies the job out into locals before looping. the stores go through uint8_t* and
// float*, which may alias the struct under char aliasing rules, so reading j.w inside the loop
// forces a reload of the trip count each iteration and blocks vectorisation entirely.

// 8 bit: integer weights and an int32 accumulator. the multiplier is picked in plugin.cpp so
// the accumulator cannot overflow, which leaves enough bits that quantising the weights costs
// well under a tenth of a level.
template <class T>
static inline void blend_int_body(const PlaneJob &j, const int32_t *weights, int32_t scale,
                                  int32_t *__restrict acc) {
    const uint8_t *const *const srcs = j.srcs;
    const unsigned num_srcs = j.num_srcs;
    uint8_t *const dst_base = j.dst;
    const unsigned w = j.w;
    const unsigned h = j.h;
    const ptrdiff_t stride = j.stride;
    const int32_t bias = j.bias;
    const int32_t maxval = j.maxval;
    const float rscale = 1.0f / static_cast<float>(scale);

    for (unsigned y = 0; y < h; ++y) {
        T *__restrict dst = reinterpret_cast<T *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const T *__restrict row =
                reinterpret_cast<const T *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const int32_t wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += (static_cast<int32_t>(row[x]) - bias) * wk;
        }

        for (unsigned x = 0; x < w; ++x) {
            const int32_t r = round_nearest_i32(static_cast<float>(acc[x]) * rscale) + bias;
            dst[x] = static_cast<T>(std::min(std::max(r, 0), maxval));
        }
    }
}

// 16 bit: a float accumulator. an int32 one would only leave ~15 bits for the weights at this
// depth, which is visible; float32's 24 bit mantissa is not, and it vectorises the same width.
template <class T>
static inline void blend_int_via_float_body(const PlaneJob &j, const float *weights, float rscale,
                                            float *__restrict acc) {
    const uint8_t *const *const srcs = j.srcs;
    const unsigned num_srcs = j.num_srcs;
    uint8_t *const dst_base = j.dst;
    const unsigned w = j.w;
    const unsigned h = j.h;
    const ptrdiff_t stride = j.stride;
    const int32_t bias = j.bias;
    const int32_t maxval = j.maxval;
    const float fbias = static_cast<float>(bias);

    for (unsigned y = 0; y < h; ++y) {
        T *__restrict dst = reinterpret_cast<T *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0.0f;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const T *__restrict row =
                reinterpret_cast<const T *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const float wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += (static_cast<float>(row[x]) - fbias) * wk;
        }

        for (unsigned x = 0; x < w; ++x) {
            const int32_t r = round_nearest_i32(acc[x] * rscale) + bias;
            dst[x] = static_cast<T>(std::min(std::max(r, 0), maxval));
        }
    }
}

static inline void blend_float_body(const PlaneJob &j, const float *weights, float rscale,
                                    float *__restrict acc) {
    const uint8_t *const *const srcs = j.srcs;
    const unsigned num_srcs = j.num_srcs;
    uint8_t *const dst_base = j.dst;
    const unsigned w = j.w;
    const unsigned h = j.h;
    const ptrdiff_t stride = j.stride;

    for (unsigned y = 0; y < h; ++y) {
        float *__restrict dst =
            reinterpret_cast<float *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0.0f;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const float *__restrict row =
                reinterpret_cast<const float *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const float wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += row[x] * wk;
        }

        for (unsigned x = 0; x < w; ++x)
            dst[x] = acc[x] * rscale;
    }
}

// squared blend: average in a squared domain instead of a linear one, f^-1(sum w f(v)), per
// channel. a camera integrates light, not the gamma-encoded codes a file stores, so averaging
// the codes directly comes out darker than a real exposure on every moving edge. squaring is a
// close enough stand-in for undoing the transfer - within about 0.02 of a real 2.2 blend
// everywhere - and it keeps the inverse exact and the loop free of anything but a multiply.
//
//   f(v) = v|v|        f^-1(y) = sign(y) * sqrt(|y|)
//
// in normalised units where 0 is reference black and 1 reference white. v|v| rather than v*v so
// sub-black stays ordered and invertible, and superwhite runs past 1.0 rather than clipping.
static inline float squared_fwd(float v) {
    return v * std::fabs(v);
}

static inline float squared_inv(float y) {
    return std::copysign(std::sqrt(std::fabs(y)), y);
}

// integer input, bright blend. samples are normalised to the clip's own black and white on the
// way in - that is what leaves superwhite running past 1.0 instead of clipped at it - and a
// float accumulator carries whatever the curve does at either end. no level table: the curve
// is a few flops, where a table costs a gather in the loop that runs once per source frame.
// bias is not read, because the curve is refused on chroma planes at filter creation.
template <class T>
static inline void blend_squared_int_body(const PlaneJob &j, const float *weights, float rscale,
                                          float *__restrict acc) {
    const uint8_t *const *const srcs = j.srcs;
    const unsigned num_srcs = j.num_srcs;
    uint8_t *const dst_base = j.dst;
    const unsigned w = j.w;
    const unsigned h = j.h;
    const ptrdiff_t stride = j.stride;
    const int32_t maxval = j.maxval;
    const float black = j.black;
    const float span = j.span;
    const float rspan = 1.0f / span;

    for (unsigned y = 0; y < h; ++y) {
        T *__restrict dst = reinterpret_cast<T *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0.0f;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const T *__restrict row =
                reinterpret_cast<const T *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const float wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += squared_fwd((static_cast<float>(row[x]) - black) * rspan) * wk;
        }

        for (unsigned x = 0; x < w; ++x) {
            const int32_t r = round_nearest_i32(black + span * squared_inv(acc[x] * rscale));
            dst[x] = static_cast<T>(std::min(std::max(r, 0), maxval));
        }
    }
}

static inline void blend_squared_float_body(const PlaneJob &j, const float *weights, float rscale,
                                            float *__restrict acc) {
    const uint8_t *const *const srcs = j.srcs;
    const unsigned num_srcs = j.num_srcs;
    uint8_t *const dst_base = j.dst;
    const unsigned w = j.w;
    const unsigned h = j.h;
    const ptrdiff_t stride = j.stride;
    const float black = j.black;
    const float span = j.span;
    const float rspan = 1.0f / span;

    for (unsigned y = 0; y < h; ++y) {
        float *__restrict dst =
            reinterpret_cast<float *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0.0f;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const float *__restrict row =
                reinterpret_cast<const float *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const float wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += squared_fwd((row[x] - black) * rspan) * wk;
        }

        for (unsigned x = 0; x < w; ++x)
            dst[x] = black + span * squared_inv(acc[x] * rscale);
    }
}

// one set of these per ISA translation unit
#define FB_DECLARE_KERNELS(suffix)                                                                 \
    void blend_u8_##suffix(const PlaneJob &, const int32_t *weights, int32_t scale, int32_t *acc); \
    void blend_u16_##suffix(const PlaneJob &, const float *weights, float rscale, float *acc);     \
    void blend_f32_##suffix(const PlaneJob &, const float *weights, float rscale, float *acc);     \
    void blend_squared_u8_##suffix(const PlaneJob &, const float *weights, float rscale,           \
                                   float *acc);                                                    \
    void blend_squared_u16_##suffix(const PlaneJob &, const float *weights, float rscale,          \
                                    float *acc);                                                   \
    void blend_squared_f32_##suffix(const PlaneJob &, const float *weights, float rscale,          \
                                    float *acc);

FB_DECLARE_KERNELS(baseline)
FB_DECLARE_KERNELS(avx2)
FB_DECLARE_KERNELS(avx512)

#define FB_DEFINE_KERNELS(suffix)                                                                  \
    void blend_u8_##suffix(const PlaneJob &j, const int32_t *w, int32_t s, int32_t *acc) {         \
        blend_int_body<uint8_t>(j, w, s, acc);                                                     \
    }                                                                                              \
    void blend_u16_##suffix(const PlaneJob &j, const float *w, float rs, float *acc) {             \
        blend_int_via_float_body<uint16_t>(j, w, rs, acc);                                         \
    }                                                                                              \
    void blend_f32_##suffix(const PlaneJob &j, const float *w, float rs, float *acc) {             \
        blend_float_body(j, w, rs, acc);                                                           \
    }                                                                                              \
    void blend_squared_u8_##suffix(const PlaneJob &j, const float *w, float rs, float *acc) {      \
        blend_squared_int_body<uint8_t>(j, w, rs, acc);                                            \
    }                                                                                              \
    void blend_squared_u16_##suffix(const PlaneJob &j, const float *w, float rs, float *acc) {     \
        blend_squared_int_body<uint16_t>(j, w, rs, acc);                                           \
    }                                                                                              \
    void blend_squared_f32_##suffix(const PlaneJob &j, const float *w, float rs, float *acc) {     \
        blend_squared_float_body(j, w, rs, acc);                                                   \
    }
} // namespace fb
