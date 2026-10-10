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
#include <type_traits>

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

// squared blend straight from yuv, so it doesn't need a conversion to rgb and back around it. each
// source pixel goes to rgb in float, through the curve, into three accumulators, and the result
// comes back to yuv on the way out. subsampled chroma is taken as mpeg-2 sited (left, and centred
// vertically) and brought up to the luma's size bilinearly. only what the curve changes goes back
// down through a [1 2 1] / box filter, on top of a plain blend of the chroma at its own size, so
// the resampling doesn't soften chroma the curve leaves alone (a still frame comes back exact)
struct YuvJob {
    const uint8_t *const *srcs[3];
    unsigned num_srcs;
    uint8_t *dst[3];
    unsigned w;
    unsigned h;
    unsigned ssw, ssh;
    ptrdiff_t stride[3];
    int32_t maxval;
    float black;       // luma reference black and white, in sample units
    float span;
    float mid;         // chroma zero and the distance to its extremes
    float cspan;
    float kr, kb;      // matrix luma coefficients
};

// floats of scratch blend_squared_yuv needs for a row of `w`
inline size_t yuv_scratch(unsigned w) {
    return static_cast<size_t>(w) * 20;
}

template <class T>
static inline float yuv_load(const uint8_t *row, unsigned x) {
    return static_cast<float>(reinterpret_cast<const T *>(row)[x]);
}

template <class T>
static inline void yuv_store(uint8_t *row, unsigned x, float v, int32_t maxval) {
    if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float *>(row)[x] = v;
    } else {
        const int32_t r = round_nearest_i32(v);
        reinterpret_cast<T *>(row)[x] = static_cast<T>(std::min(std::max(r, 0), maxval));
    }
}

// one chroma row normalised and at the luma's width. `near` and `far` are the chroma rows either
// side of the luma row
template <class T>
static inline void yuv_chroma_row(const uint8_t *near, const uint8_t *far, float far_weight,
                                  unsigned cw, bool ssw, float mid, float rcspan,
                                  float *__restrict row, float *__restrict up) {
    const float near_weight = 1.0f - far_weight;
    for (unsigned i = 0; i < cw; ++i)
        row[i] = (near_weight * yuv_load<T>(near, i) + far_weight * yuv_load<T>(far, i) - mid) * rcspan;

    if (!ssw)
        return;
    for (unsigned i = 0; i + 1 < cw; ++i) {
        up[2 * i] = row[i];
        up[2 * i + 1] = 0.5f * (row[i] + row[i + 1]);
    }
    up[2 * (cw - 1)] = up[2 * (cw - 1) + 1] = row[cw - 1];
}

template <class T>
static inline void blend_squared_yuv_body(const YuvJob &j, const float *weights, float rscale,
                                          float *__restrict scratch) {
    const unsigned num_srcs = j.num_srcs;
    const unsigned w = j.w;
    const bool ssw = j.ssw != 0, ssh = j.ssh != 0;
    const unsigned rows = ssh ? 2 : 1;
    const unsigned cw = w >> j.ssw;
    const unsigned ch = j.h >> j.ssh;
    const int32_t maxval = j.maxval;
    const float black = j.black, span = j.span, rspan = 1.0f / span;
    const float mid = j.mid, cspan = j.cspan, rcspan = 1.0f / cspan;
    const float kr = j.kr, kb = j.kb, kg = 1.0f - kr - kb;
    const float rv = 2.0f * (1.0f - kr), bu = 2.0f * (1.0f - kb);
    const float gv = rv * kr / kg, gu = bu * kb / kg;
    const ptrdiff_t s1 = j.stride[1], s2 = j.stride[2];

    float *const acc = scratch;                  // r, g, b for each luma row of the group
    float *const row_u = acc + 3 * 2 * w;
    float *const row_v = row_u + w;
    float *const up_u = row_v + w;
    float *const up_v = up_u + w;
    float *const out_u = up_v + w;               // each luma row's chroma before it's shrunk
    float *const out_v = out_u + 2 * w;
    float *const lin_up_u = out_v + 2 * w;       // the plain blend of the upsampled chroma
    float *const lin_up_v = lin_up_u + 2 * w;
    float *const lin_u = lin_up_v + 2 * w;       // and of the chroma at its own size
    float *const lin_v = lin_u + w;
    const bool sub = ssw || ssh;

    for (unsigned cy = 0; cy < ch; ++cy) {
        for (unsigned x = 0; x < 3 * rows * w; ++x)
            acc[x] = 0.0f;
        if (sub) {
            for (unsigned x = 0; x < 2 * w; ++x)
                lin_up_u[x] = lin_up_v[x] = 0.0f;
            for (unsigned i = 0; i < cw; ++i)
                lin_u[i] = lin_v[i] = 0.0f;
        }

        for (unsigned k = 0; k < num_srcs; ++k) {
            const float wk = weights[k];
            const uint8_t *const cu = j.srcs[1][k];
            const uint8_t *const cv = j.srcs[2][k];

            if (sub) {
                const uint8_t *nu = cu + cy * s1, *nv = cv + cy * s2;
                for (unsigned i = 0; i < cw; ++i) {
                    lin_u[i] += (yuv_load<T>(nu, i) - mid) * rcspan * wk;
                    lin_v[i] += (yuv_load<T>(nv, i) - mid) * rcspan * wk;
                }
            }

            for (unsigned r = 0; r < rows; ++r) {
                const unsigned far = !ssh ? cy : (r == 0 ? (cy ? cy - 1 : 0) : std::min(cy + 1, ch - 1));
                const float far_weight = ssh ? 0.25f : 0.0f;
                yuv_chroma_row<T>(cu + cy * s1, cu + far * s1, far_weight, cw, ssw, mid, rcspan, row_u, up_u);
                yuv_chroma_row<T>(cv + cy * s2, cv + far * s2, far_weight, cw, ssw, mid, rcspan, row_v, up_v);
                const float *__restrict pu = ssw ? up_u : row_u;
                const float *__restrict pv = ssw ? up_v : row_v;

                const uint8_t *ry = j.srcs[0][k] + static_cast<ptrdiff_t>(cy * rows + r) * j.stride[0];
                float *__restrict ar = acc + 3 * r * w;
                float *__restrict ag = ar + w;
                float *__restrict ab = ag + w;
                if (sub) {
                    float *__restrict lu = lin_up_u + r * w;
                    float *__restrict lv = lin_up_v + r * w;
                    for (unsigned x = 0; x < w; ++x) {
                        lu[x] += pu[x] * wk;
                        lv[x] += pv[x] * wk;
                    }
                }
                for (unsigned x = 0; x < w; ++x) {
                    const float l = (yuv_load<T>(ry, x) - black) * rspan;
                    ar[x] += squared_fwd(l + rv * pv[x]) * wk;
                    ag[x] += squared_fwd(l - gv * pv[x] - gu * pu[x]) * wk;
                    ab[x] += squared_fwd(l + bu * pu[x]) * wk;
                }
            }
        }

        for (unsigned r = 0; r < rows; ++r) {
            const float *ar = acc + 3 * r * w;
            const float *ag = ar + w;
            const float *ab = ag + w;
            uint8_t *dy = j.dst[0] + static_cast<ptrdiff_t>(cy * rows + r) * j.stride[0];
            for (unsigned x = 0; x < w; ++x) {
                const float red = squared_inv(ar[x] * rscale);
                const float green = squared_inv(ag[x] * rscale);
                const float blue = squared_inv(ab[x] * rscale);
                const float l = kr * red + kg * green + kb * blue;
                yuv_store<T>(dy, x, black + span * l, maxval);
                out_u[r * w + x] = (blue - l) / bu;
                out_v[r * w + x] = (red - l) / rv;
                if (sub) {
                    out_u[r * w + x] -= lin_up_u[r * w + x] * rscale;
                    out_v[r * w + x] -= lin_up_v[r * w + x] * rscale;
                }
            }
        }

        uint8_t *du = j.dst[1] + static_cast<ptrdiff_t>(cy) * s1;
        uint8_t *dv = j.dst[2] + static_cast<ptrdiff_t>(cy) * s2;
        const float rrows = 1.0f / static_cast<float>(rows);
        auto column = [&](const float *q, unsigned x) {
            return ssh ? (q[x] + q[w + x]) * rrows : q[x];
        };
        for (unsigned i = 0; i < cw; ++i) {
            float u, v;
            if (ssw) {
                const unsigned x = 2 * i, left = x ? x - 1 : 0, right = std::min(x + 1, w - 1);
                u = 0.25f * column(out_u, left) + 0.5f * column(out_u, x) + 0.25f * column(out_u, right);
                v = 0.25f * column(out_v, left) + 0.5f * column(out_v, x) + 0.25f * column(out_v, right);
            } else {
                u = column(out_u, i);
                v = column(out_v, i);
            }
            if (sub) {
                u += lin_u[i] * rscale;
                v += lin_v[i] * rscale;
            }
            yuv_store<T>(du, i, mid + cspan * u, maxval);
            yuv_store<T>(dv, i, mid + cspan * v, maxval);
        }
    }
}

// one set of these per ISA translation unit
#define FB_DECLARE_KERNELS(suffix)                                                                 \
    void blend_squared_yuv_u8_##suffix(const YuvJob &, const float *weights, float rscale,         \
                                       float *acc);                                                \
    void blend_squared_yuv_u16_##suffix(const YuvJob &, const float *weights, float rscale,        \
                                        float *acc);                                               \
    void blend_squared_yuv_f32_##suffix(const YuvJob &, const float *weights, float rscale,        \
                                        float *acc);                                               \
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
    }                                                                                              \
    void blend_squared_yuv_u8_##suffix(const YuvJob &j, const float *w, float rs, float *acc) {    \
        blend_squared_yuv_body<uint8_t>(j, w, rs, acc);                                            \
    }                                                                                              \
    void blend_squared_yuv_u16_##suffix(const YuvJob &j, const float *w, float rs, float *acc) {   \
        blend_squared_yuv_body<uint16_t>(j, w, rs, acc);                                           \
    }                                                                                              \
    void blend_squared_yuv_f32_##suffix(const YuvJob &j, const float *w, float rs, float *acc) {   \
        blend_squared_yuv_body<float>(j, w, rs, acc);                                              \
    }
} // namespace fb
