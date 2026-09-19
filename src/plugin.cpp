// frameblender.FrameBlend - temporal weighted average over one clip, no weight cap.
//
// the plane kernels live in kernel.h, compiled once per ISA level into kernel_*.cpp and picked
// by cpuid at load. see README.md for why the loop order is what it is.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FB_X86 1
#endif

// msvc and clang-cl take the cpuid intrinsics from intrin.h, which gcc does not have. gcc and
// clang elsewhere have __builtin_cpu_supports instead, which does the xgetbv OS-state check
// itself. aarch64 needs neither: neon is the baseline, so the baseline kernel is already
// vectorised and there is nothing to dispatch between.
#if defined(FB_X86) && defined(_MSC_VER)
#include <intrin.h>
#endif

#include <VSHelper4.h>
#include <VapourSynth4.h>

#include "kernel.h"

namespace {

using fb::PlaneJob;

enum class Isa { Baseline, Avx2, Avx512 };

Isa detect_isa() {
#if defined(FB_X86) && defined(_MSC_VER)
    int regs[4] = { 0, 0, 0, 0 };
    __cpuid(regs, 0);
    const int maxLeaf = regs[0];
    if (maxLeaf < 7)
        return Isa::Baseline;

    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    if (!osxsave)
        return Isa::Baseline;

    // the cpu can have the instructions while the os has not enabled the register state
    const unsigned long long xcr0 = _xgetbv(0);
    const bool ymmState = (xcr0 & 0x6) == 0x6;         // SSE + AVX state
    const bool zmmState = (xcr0 & 0xE6) == 0xE6;       // + opmask, ZMM_Hi256, Hi16_ZMM
    if (!ymmState)
        return Isa::Baseline;

    __cpuidex(regs, 7, 0);
    const bool avx2 = (regs[1] & (1 << 5)) != 0;
    const bool avx512f = (regs[1] & (1 << 16)) != 0;
    const bool avx512bw = (regs[1] & (1 << 30)) != 0;
    const bool avx512vl = (regs[1] & (1u << 31)) != 0;

    if (zmmState && avx512f && avx512bw && avx512vl)
        return Isa::Avx512;
    if (avx2)
        return Isa::Avx2;
    return Isa::Baseline;

#elif defined(FB_X86)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl"))
        return Isa::Avx512;
    if (__builtin_cpu_supports("avx2"))
        return Isa::Avx2;
    return Isa::Baseline;

#else
    return Isa::Baseline;
#endif
}

const Isa g_isa = detect_isa();

void blend_u8(const PlaneJob &j, const int32_t *w, int32_t s, int32_t *acc) {
    switch (g_isa) {
    case Isa::Avx512: fb::blend_u8_avx512(j, w, s, acc); return;
    case Isa::Avx2:   fb::blend_u8_avx2(j, w, s, acc); return;
    default:          fb::blend_u8_baseline(j, w, s, acc); return;
    }
}
void blend_u16(const PlaneJob &j, const float *w, float rs, float *acc) {
    switch (g_isa) {
    case Isa::Avx512: fb::blend_u16_avx512(j, w, rs, acc); return;
    case Isa::Avx2:   fb::blend_u16_avx2(j, w, rs, acc); return;
    default:          fb::blend_u16_baseline(j, w, rs, acc); return;
    }
}
void blend_f32(const PlaneJob &j, const float *w, float rs, float *acc) {
    switch (g_isa) {
    case Isa::Avx512: fb::blend_f32_avx512(j, w, rs, acc); return;
    case Isa::Avx2:   fb::blend_f32_avx2(j, w, rs, acc); return;
    default:          fb::blend_f32_baseline(j, w, rs, acc); return;
    }
}

struct BlendData {
    VSNode *node = nullptr;
    VSVideoInfo vi{};
    std::vector<int32_t> iweights;   // 8 bit path
    std::vector<float> fweights;     // 16 bit and float paths
    int32_t iscale = 1;
    float frscale = 1.0f;
    bool process[3] = { true, true, true };
};

const VSFrame *VS_CC blendGetFrame(int n, int activationReason, void *instanceData,
                                   void **frameData, VSFrameContext *frameCtx, VSCore *core,
                                   const VSAPI *vsapi) {
    BlendData *d = static_cast<BlendData *>(instanceData);
    const int count = static_cast<int>(d->fweights.size());
    const int radius = count / 2;
    const int last = d->vi.numFrames - 1;

    if (activationReason == arInitial) {
        for (int k = 0; k < count; ++k)
            vsapi->requestFrameFilter(std::min(std::max(n - radius + k, 0), last), d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    std::vector<const VSFrame *> frames(count);
    for (int k = 0; k < count; ++k)
        frames[k] = vsapi->getFrameFilter(std::min(std::max(n - radius + k, 0), last), d->node, frameCtx);

    const VSFrame *center = frames[radius];
    const VSVideoFormat *fi = vsapi->getVideoFrameFormat(center);

    const int pl[] = { 0, 1, 2 };
    const VSFrame *fr[] = {
        d->process[0] ? nullptr : center,
        d->process[1] ? nullptr : center,
        d->process[2] ? nullptr : center,
    };
    VSFrame *dst = vsapi->newVideoFrame2(fi, vsapi->getFrameWidth(center, 0),
                                         vsapi->getFrameHeight(center, 0), fr, pl, center, core);

    std::vector<const uint8_t *> srcs(count);
    std::vector<int32_t> iacc;
    std::vector<float> facc;

    for (int plane = 0; plane < fi->numPlanes; ++plane) {
        if (!d->process[plane])
            continue;

        for (int k = 0; k < count; ++k)
            srcs[k] = vsapi->getReadPtr(frames[k], plane);

        const bool chroma = (plane == 1 || plane == 2) && fi->colorFamily == cfYUV;

        PlaneJob job{};
        job.srcs = srcs.data();
        job.num_srcs = static_cast<unsigned>(count);
        job.dst = vsapi->getWritePtr(dst, plane);
        job.w = static_cast<unsigned>(vsapi->getFrameWidth(center, plane));
        job.h = static_cast<unsigned>(vsapi->getFrameHeight(center, plane));
        job.stride = vsapi->getStride(center, plane);
        job.bias = 0;
        job.maxval = 0;

        if (fi->sampleType == stInteger) {
            job.maxval = (1 << fi->bitsPerSample) - 1;
            job.bias = chroma ? (1 << (fi->bitsPerSample - 1)) : 0;
            if (fi->bytesPerSample == 1) {
                if (iacc.size() < job.w) iacc.resize(job.w);
                blend_u8(job, d->iweights.data(), d->iscale, iacc.data());
            } else {
                if (facc.size() < job.w) facc.resize(job.w);
                blend_u16(job, d->fweights.data(), d->frscale, facc.data());
            }
        } else {
            if (facc.size() < job.w) facc.resize(job.w);
            blend_f32(job, d->fweights.data(), d->frscale, facc.data());
        }
    }

    for (int k = 0; k < count; ++k)
        vsapi->freeFrame(frames[k]);
    return dst;
}

void VS_CC blendFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    BlendData *d = static_cast<BlendData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC blendCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core,
                       const VSAPI *vsapi) {
    std::unique_ptr<BlendData> d(new BlendData);
    int err = 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);

    auto fail = [&](const std::string &msg) {
        vsapi->mapSetError(out, ("FrameBlend: " + msg).c_str());
        vsapi->freeNode(d->node);
    };

    if (!vsh::isConstantVideoFormat(&d->vi))
        return fail("clip must have a constant format and dimensions");
    if (d->vi.format.sampleType == stInteger && d->vi.format.bytesPerSample > 2)
        return fail("only 8-16 bit integer and 32 bit float are supported");

    const int numWeights = vsapi->mapNumElements(in, "weights");
    if (numWeights < 1)
        return fail("at least one weight is required");
    if (numWeights % 2 == 0)
        return fail("an odd number of weights is required");

    d->fweights.resize(numWeights);
    double sum = 0.0, sumabs = 0.0;
    for (int i = 0; i < numWeights; ++i) {
        const double v = vsapi->mapGetFloat(in, "weights", i, nullptr);
        d->fweights[i] = static_cast<float>(v);
        sum += v;
        sumabs += std::abs(v);
    }

    double scale = vsapi->mapGetFloat(in, "scale", 0, &err);
    if (err)
        scale = sum;
    if (scale == 0.0)
        return fail("weights sum to zero - pass a non-zero `scale` if that is intended");
    if (sumabs == 0.0)
        return fail("all weights are zero");

    d->frscale = static_cast<float>(1.0 / scale);

    // 8 bit path quantises the weights. pick the largest multiplier that still cannot overflow
    // the int32 accumulator, which leaves the rounding error far below one level.
    if (d->vi.format.sampleType == stInteger && d->vi.format.bytesPerSample == 1) {
        const double headroom = 2147483647.0 * 0.9;
        double mult = headroom / (255.0 * sumabs);
        mult = std::min(mult, 4194304.0);
        if (mult < 1.0)
            return fail("weights are too large to quantise - normalise them first");

        d->iweights.resize(numWeights);
        for (int i = 0; i < numWeights; ++i)
            d->iweights[i] = static_cast<int32_t>(std::lround(d->fweights[i] * mult));
        d->iscale = static_cast<int32_t>(std::lround(scale * mult));
        if (d->iscale == 0)
            return fail("scale underflowed after quantisation");
    }

    const int numPlanes = vsapi->mapNumElements(in, "planes");
    if (numPlanes > 0) {
        for (bool &p : d->process)
            p = false;
        for (int i = 0; i < numPlanes; ++i) {
            const int p = vsh::int64ToIntS(vsapi->mapGetInt(in, "planes", i, nullptr));
            if (p < 0 || p >= d->vi.format.numPlanes)
                return fail("plane index out of range");
            d->process[p] = true;
        }
    }

    VSFilterDependency deps[] = { { d->node, rpGeneral } };
    vsapi->createVideoFilter(out, "FrameBlend", &d->vi, blendGetFrame, blendFree, fmParallel,
                             deps, 1, d.get(), core);
    d.release();
}

void VS_CC isaInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    const char *name = g_isa == Isa::Avx512 ? "avx512" : (g_isa == Isa::Avx2 ? "avx2" : "baseline");
    vsapi->mapSetData(out, "isa", name, -1, dtUtf8, maReplace);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.f0e.frameblender", "frameblender", "Frame blender",
                         VS_MAKE_VERSION(2, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("FrameBlend",
                             "clip:vnode;weights:float[];scale:float:opt;planes:int[]:opt;",
                             "clip:vnode;", blendCreate, nullptr, plugin);
    vspapi->registerFunction("Isa", "", "isa:data;", isaInfo, nullptr, plugin);
}
