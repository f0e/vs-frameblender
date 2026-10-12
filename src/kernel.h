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

// curved blend: average light instead of the gamma-encoded codes a file stores, f^-1(sum w f(v)),
// per channel. a camera integrates light, so averaging the codes directly comes out darker than a
// real exposure on every moving edge. f undoes the clip's transfer curve.
//
// in normalised units where 0 is reference black and 1 reference white. every curve is mirrored
// below black (f(-v) = -f(v)) so sub-black stays ordered and invertible, and superwhite runs past
// 1.0 rather than clipping.

// pow and friends without libm calls, so the loops still vectorise. log2 reduces the mantissa to
// [sqrt(1/2), sqrt(2)) and uses the atanh series, exp2 a taylor series about the middle of its
// range. both are within a few 1e-7 relative, below a 16 bit level after the curve
static inline float bits_float(int32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof(f));
    return f;
}

static inline int32_t float_bits(float f) {
    int32_t i;
    std::memcpy(&i, &f, sizeof(i));
    return i;
}

// x > 0, normal
static inline float fast_log2(float x) {
    const int32_t bits = float_bits(x);
    float e = static_cast<float>(((bits >> 23) & 0xff) - 127);
    float m = bits_float((bits & 0x007fffff) | 0x3f800000);
    const bool high = m > 1.41421356f;
    m = high ? m * 0.5f : m;
    e = high ? e + 1.0f : e;
    const float t = (m - 1.0f) / (m + 1.0f);
    const float t2 = t * t;
    const float series = t * (2.0f + t2 * (2.0f / 3.0f + t2 * (2.0f / 5.0f + t2 * (2.0f / 7.0f))));
    return e + series * 1.44269504f;
}

static inline float fast_exp2(float y) {
    y = std::min(std::max(y, -126.0f), 127.0f);
    int32_t i = static_cast<int32_t>(y);
    i -= y < static_cast<float>(i) ? 1 : 0;
    const float x = (y - static_cast<float>(i) - 0.5f) * 0.693147181f;
    const float series =
        1.0f + x * (1.0f + x * (0.5f + x * (1.0f / 6.0f + x * (1.0f / 24.0f + x * (1.0f / 120.0f + x * (1.0f / 720.0f))))));
    return series * 1.41421356f * bits_float((i + 127) << 23);
}

// x >= 0
static inline float fast_pow(float x, float p) {
    return x > 0.0f ? fast_exp2(p * fast_log2(x)) : 0.0f;
}

namespace {

// squaring is a close enough stand-in for undoing a ~2.2 curve - within about 0.02 everywhere -
// and it keeps the inverse exact and the loop free of anything but a multiply
struct SquareCurve {
    float fwd(float v) const { return v * std::fabs(v); }
    float inv(float y) const { return std::copysign(std::sqrt(std::fabs(y)), y); }
};

// smpte st 2084, light normalised so 1 is 10000 nits
struct PqCurve {
    static constexpr float m1 = 2610.0f / 16384.0f;
    static constexpr float m2 = 2523.0f / 4096.0f * 128.0f;
    static constexpr float c1 = 3424.0f / 4096.0f;
    static constexpr float c2 = 2413.0f / 4096.0f * 32.0f;
    static constexpr float c3 = 2392.0f / 4096.0f * 32.0f;
    float fwd(float v) const {
        const float p = fast_pow(std::fabs(v), 1.0f / m2);
        const float r = fast_pow(std::max(p - c1, 0.0f) / (c2 - c3 * p), 1.0f / m1);
        return std::copysign(r, v);
    }
    float inv(float y) const {
        const float p = fast_pow(std::fabs(y), m1);
        return std::copysign(fast_pow((c1 + c2 * p) / (1.0f + c3 * p), m2), y);
    }
};

// bt.2100 hlg, blended in scene light (the inverse oetf), so there's no display to assume
struct HlgCurve {
    static constexpr float a = 0.17883277f;
    static constexpr float b = 0.28466892f;
    static constexpr float c = 0.55991073f;
    float fwd(float v) const {
        const float e = std::fabs(v);
        const float r = e <= 0.5f ? e * e * (1.0f / 3.0f)
                                  : (fast_exp2((e - c) * (1.0f / a) * 1.44269504f) + b) * (1.0f / 12.0f);
        return std::copysign(r, v);
    }
    float inv(float y) const {
        const float l = std::fabs(y);
        const float r = l <= 1.0f / 12.0f ? std::sqrt(3.0f * l)
                                          : a * fast_log2(std::max(12.0f * l - b, 1e-30f)) * 0.693147181f + c;
        return std::copysign(r, y);
    }
};

} // namespace

// integer input, curved blend. samples are normalised to the clip's own black and white on the
// way in - that is what leaves superwhite running past 1.0 instead of clipped at it - and a
// float accumulator carries whatever the curve does at either end. no level table: the curve
// is a few flops, where a table costs a gather in the loop that runs once per source frame.
// bias is not read, because the curve is refused on chroma planes at filter creation.
template <class T, class Curve>
static inline void blend_curve_int_body(const PlaneJob &j, const float *weights, float rscale,
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
    const Curve curve{};

    for (unsigned y = 0; y < h; ++y) {
        T *__restrict dst = reinterpret_cast<T *>(dst_base + static_cast<ptrdiff_t>(y) * stride);

        for (unsigned x = 0; x < w; ++x)
            acc[x] = 0.0f;

        for (unsigned k = 0; k < num_srcs; ++k) {
            const T *__restrict row =
                reinterpret_cast<const T *>(srcs[k] + static_cast<ptrdiff_t>(y) * stride);
            const float wk = weights[k];
            for (unsigned x = 0; x < w; ++x)
                acc[x] += curve.fwd((static_cast<float>(row[x]) - black) * rspan) * wk;
        }

        for (unsigned x = 0; x < w; ++x) {
            const int32_t r = round_nearest_i32(black + span * curve.inv(acc[x] * rscale));
            dst[x] = static_cast<T>(std::min(std::max(r, 0), maxval));
        }
    }
}

template <class Curve>
static inline void blend_curve_float_body(const PlaneJob &j, const float *weights, float rscale,
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
    const Curve curve{};

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
                acc[x] += curve.fwd((row[x] - black) * rspan) * wk;
        }

        for (unsigned x = 0; x < w; ++x)
            dst[x] = black + span * curve.inv(acc[x] * rscale);
    }
}

// curved blend straight from yuv, so it doesn't need a conversion to rgb and back around it. each
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

// floats of scratch blend_curve_yuv needs for a row of `w`
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

template <class T, class Curve>
static inline void blend_curve_yuv_body(const YuvJob &j, const float *weights, float rscale,
                                        float *__restrict scratch) {
    const Curve curve{};
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
                    ar[x] += curve.fwd(l + rv * pv[x]) * wk;
                    ag[x] += curve.fwd(l - gv * pv[x] - gu * pu[x]) * wk;
                    ab[x] += curve.fwd(l + bu * pu[x]) * wk;
                }
            }
        }

        for (unsigned r = 0; r < rows; ++r) {
            const float *ar = acc + 3 * r * w;
            const float *ag = ar + w;
            const float *ab = ag + w;
            uint8_t *dy = j.dst[0] + static_cast<ptrdiff_t>(cy * rows + r) * j.stride[0];
            for (unsigned x = 0; x < w; ++x) {
                const float red = curve.inv(ar[x] * rscale);
                const float green = curve.inv(ag[x] * rscale);
                const float blue = curve.inv(ab[x] * rscale);
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

enum Transfer { TransferSquare, TransferPq, TransferHlg, TransferCount };

using PlaneFn = void (*)(const PlaneJob &, const float *weights, float rscale, float *acc);
using YuvFn = void (*)(const YuvJob &, const float *weights, float rscale, float *acc);

// every kernel for one ISA level. the curved ones are indexed by Transfer
struct Kernels {
    void (*blend_u8)(const PlaneJob &, const int32_t *weights, int32_t scale, int32_t *acc);
    PlaneFn blend_u16;
    PlaneFn blend_f32;
    PlaneFn curve_u8[TransferCount];
    PlaneFn curve_u16[TransferCount];
    PlaneFn curve_f32[TransferCount];
    YuvFn curve_yuv_u8[TransferCount];
    YuvFn curve_yuv_u16[TransferCount];
    YuvFn curve_yuv_f32[TransferCount];
};

const Kernels &kernels_baseline();
const Kernels &kernels_avx2();
const Kernels &kernels_avx512();

// static, and the curves are in an unnamed namespace, so each ISA translation unit keeps its own
// copies compiled with its own flags instead of the linker merging them into one
template <class T>
static void blend_int_kernel(const PlaneJob &j, const int32_t *w, int32_t s, int32_t *acc) {
    blend_int_body<T>(j, w, s, acc);
}
template <class T>
static void blend_int_via_float_kernel(const PlaneJob &j, const float *w, float rs, float *acc) {
    blend_int_via_float_body<T>(j, w, rs, acc);
}
static void blend_float_kernel(const PlaneJob &j, const float *w, float rs, float *acc) {
    blend_float_body(j, w, rs, acc);
}
template <class T, class Curve>
static void curve_int_kernel(const PlaneJob &j, const float *w, float rs, float *acc) {
    blend_curve_int_body<T, Curve>(j, w, rs, acc);
}
template <class Curve>
static void curve_float_kernel(const PlaneJob &j, const float *w, float rs, float *acc) {
    blend_curve_float_body<Curve>(j, w, rs, acc);
}
template <class T, class Curve>
static void curve_yuv_kernel(const YuvJob &j, const float *w, float rs, float *acc) {
    blend_curve_yuv_body<T, Curve>(j, w, rs, acc);
}

#define FB_CURVES(kernel, ...)                                                                     \
    {                                                                                              \
        kernel<__VA_ARGS__ SquareCurve>, kernel<__VA_ARGS__ PqCurve>, kernel<__VA_ARGS__ HlgCurve> \
    }

#define FB_DEFINE_KERNELS(suffix)                                                                  \
    const Kernels &kernels_##suffix() {                                                            \
        static const Kernels k = {                                                                 \
            blend_int_kernel<uint8_t>,                                                             \
            blend_int_via_float_kernel<uint16_t>,                                                  \
            blend_float_kernel,                                                                    \
            FB_CURVES(curve_int_kernel, uint8_t, ),                                                \
            FB_CURVES(curve_int_kernel, uint16_t, ),                                               \
            FB_CURVES(curve_float_kernel, ),                                                       \
            FB_CURVES(curve_yuv_kernel, uint8_t, ),                                                \
            FB_CURVES(curve_yuv_kernel, uint16_t, ),                                               \
            FB_CURVES(curve_yuv_kernel, float, ),                                                  \
        };                                                                                         \
        return k;                                                                                  \
    }
} // namespace fb
