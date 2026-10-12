// frameblender.FrameBlend - temporal weighted average over one clip, no weight cap.
//
// the plane kernels live in kernel.h, compiled once per ISA level into kernel_*.cpp and picked
// by cpuid at load. see README.md for why the loop order is what it is.

#include <algorithm>
#include <array>
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

const fb::Kernels &g_kernels = g_isa == Isa::Avx512 ? fb::kernels_avx512()
                               : g_isa == Isa::Avx2 ? fb::kernels_avx2()
                                                    : fb::kernels_baseline();

struct BlendData {
    VSNode *node = nullptr;
    VSVideoInfo vi{};
    std::vector<int32_t> iweights;   // 8 bit path
    std::vector<float> fweights;     // 16 bit and float paths
    int32_t iscale = 1;
    float frscale = 1.0f;
    bool process[3] = { true, true, true };
    bool use_curve = false;
    int transfer = fb::TransferSquare;
    float black = 0.0f;           // reference black and white the curve is anchored to
    float span = 1.0f;
    bool yuv = false;             // curved blend done in rgb from yuv, see YuvJob
    float mid = 0.0f;
    float cspan = 1.0f;
    float kr = 0.0f, kb = 0.0f;
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

    // a frame repeated in the window (a held frame, a clamped edge, a clip stretched by repeating
    // frames) is read once with the weights summed. integer weights add exactly. compared by their
    // pixels' address, since a filter that only touches props hands back a new frame sharing them
    using Pixels = std::array<const uint8_t *, 3>;
    std::vector<const VSFrame *> unique;
    std::vector<Pixels> pixels;
    std::vector<float> fweights;
    std::vector<int32_t> iweights;
    unique.reserve(count);
    pixels.reserve(count);
    for (int k = 0; k < count; ++k) {
        Pixels data{};
        for (int plane = 0; plane < fi->numPlanes; ++plane)
            data[plane] = vsapi->getReadPtr(frames[k], plane);
        const auto at = std::find(pixels.begin(), pixels.end(), data);
        if (at == pixels.end()) {
            unique.push_back(frames[k]);
            pixels.push_back(data);
            fweights.push_back(d->fweights[k]);
            if (!d->iweights.empty())
                iweights.push_back(d->iweights[k]);
            continue;
        }
        const size_t i = static_cast<size_t>(at - pixels.begin());
        fweights[i] += d->fweights[k];
        if (!d->iweights.empty())
            iweights[i] += d->iweights[k];
    }
    const int sources = static_cast<int>(unique.size());

    const int pl[] = { 0, 1, 2 };
    const VSFrame *fr[] = {
        d->process[0] ? nullptr : center,
        d->process[1] ? nullptr : center,
        d->process[2] ? nullptr : center,
    };
    VSFrame *dst = vsapi->newVideoFrame2(fi, vsapi->getFrameWidth(center, 0),
                                         vsapi->getFrameHeight(center, 0), fr, pl, center, core);

    std::vector<const uint8_t *> srcs(sources);
    std::vector<int32_t> iacc;
    std::vector<float> facc;

    if (d->yuv) {
        std::vector<const uint8_t *> plane_srcs[3];
        fb::YuvJob job{};
        for (int plane = 0; plane < 3; ++plane) {
            plane_srcs[plane].resize(sources);
            for (int k = 0; k < sources; ++k)
                plane_srcs[plane][k] = vsapi->getReadPtr(unique[k], plane);
            job.srcs[plane] = plane_srcs[plane].data();
            job.dst[plane] = vsapi->getWritePtr(dst, plane);
            job.stride[plane] = vsapi->getStride(center, plane);
        }
        job.num_srcs = static_cast<unsigned>(sources);
        job.w = static_cast<unsigned>(vsapi->getFrameWidth(center, 0));
        job.h = static_cast<unsigned>(vsapi->getFrameHeight(center, 0));
        job.ssw = static_cast<unsigned>(fi->subSamplingW);
        job.ssh = static_cast<unsigned>(fi->subSamplingH);
        job.maxval = fi->sampleType == stInteger ? (1 << fi->bitsPerSample) - 1 : 0;
        job.black = d->black;
        job.span = d->span;
        job.mid = d->mid;
        job.cspan = d->cspan;
        job.kr = d->kr;
        job.kb = d->kb;
        facc.resize(fb::yuv_scratch(job.w));
        const fb::YuvFn blend = fi->sampleType == stFloat ? g_kernels.curve_yuv_f32[d->transfer]
                                : fi->bytesPerSample == 1 ? g_kernels.curve_yuv_u8[d->transfer]
                                                          : g_kernels.curve_yuv_u16[d->transfer];
        blend(job, fweights.data(), d->frscale, facc.data());

        for (int k = 0; k < count; ++k)
            vsapi->freeFrame(frames[k]);
        return dst;
    }

    for (int plane = 0; plane < fi->numPlanes; ++plane) {
        if (!d->process[plane])
            continue;

        for (int k = 0; k < sources; ++k)
            srcs[k] = vsapi->getReadPtr(unique[k], plane);

        const bool chroma = (plane == 1 || plane == 2) && fi->colorFamily == cfYUV;

        PlaneJob job{};
        job.srcs = srcs.data();
        job.num_srcs = static_cast<unsigned>(sources);
        job.dst = vsapi->getWritePtr(dst, plane);
        job.w = static_cast<unsigned>(vsapi->getFrameWidth(center, plane));
        job.h = static_cast<unsigned>(vsapi->getFrameHeight(center, plane));
        job.stride = vsapi->getStride(center, plane);
        job.bias = 0;
        job.maxval = 0;
        job.black = d->black;
        job.span = d->span;

        if (fi->sampleType == stInteger) {
            job.maxval = (1 << fi->bitsPerSample) - 1;
            job.bias = chroma ? (1 << (fi->bitsPerSample - 1)) : 0;
            if (facc.size() < job.w) facc.resize(job.w);
            if (d->use_curve) {
                if (fi->bytesPerSample == 1)
                    g_kernels.curve_u8[d->transfer](job, fweights.data(), d->frscale, facc.data());
                else
                    g_kernels.curve_u16[d->transfer](job, fweights.data(), d->frscale, facc.data());
            } else if (fi->bytesPerSample == 1) {
                if (iacc.size() < job.w) iacc.resize(job.w);
                g_kernels.blend_u8(job, iweights.data(), d->iscale, iacc.data());
            } else {
                g_kernels.blend_u16(job, fweights.data(), d->frscale, facc.data());
            }
        } else {
            if (facc.size() < job.w) facc.resize(job.w);
            if (d->use_curve)
                g_kernels.curve_f32[d->transfer](job, fweights.data(), d->frscale, facc.data());
            else
                g_kernels.blend_f32(job, fweights.data(), d->frscale, facc.data());
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

    // squared=True predates transfer, and is the same as transfer="square"
    int squared_err = 0;
    const bool squared = vsapi->mapGetInt(in, "squared", 0, &squared_err) && !squared_err;

    int transfer_err = 0;
    const char *transfer = vsapi->mapGetData(in, "transfer", 0, &transfer_err);
    const bool use_curve = squared || !transfer_err;
    if (!transfer_err) {
        const std::string name = transfer;
        if (name == "square") d->transfer = fb::TransferSquare;
        else if (name == "pq") d->transfer = fb::TransferPq;
        else if (name == "hlg") d->transfer = fb::TransferHlg;
        else return fail("transfer must be one of square, pq and hlg");
    }

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
    if (!use_curve && d->vi.format.sampleType == stInteger && d->vi.format.bytesPerSample == 1) {
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



    d->use_curve = use_curve;
    if (d->use_curve) {
        const VSVideoFormat &f = d->vi.format;

        // a curve about the chroma midpoint has no meaning - the emphasis has to be applied per
        // colour channel, or bright saturated things come out bright and washed out. so with all
        // of a yuv clip's planes it runs in rgb, converting with the clip's _Matrix
        const bool chroma = f.colorFamily == cfYUV && (d->process[1] || d->process[2]);
        if (chroma && !(d->process[0] && d->process[1] && d->process[2]))
            return fail("the curved blend needs every plane of a yuv clip, or planes=[0]");
        if (chroma && (f.subSamplingW > 1 || f.subSamplingH > 1))
            return fail("the curved blend handles 4:4:4, 4:2:2 and 4:2:0 chroma - convert to one of those or rgb");
        d->yuv = chroma;

        // the curve is anchored to the clip's own black and white, taken from _ColorRange rather
        // than made the caller's problem. limited range sits at 16..235, and expanding it to fill
        // 0..255 first is what throws superwhite away - anchored here, samples above white just
        // run past the curve's 1.0, which a float accumulator handles fine
        char msg[256] = { 0 };
        const VSFrame *first = vsapi->getFrame(0, d->node, msg, sizeof(msg));
        if (!first)
            return fail(std::string("could not read the first frame for _ColorRange: ") + msg);

        int range_err = 0;
        const int64_t range =
            vsapi->mapGetInt(vsapi->getFramePropertiesRO(first), "_ColorRange", 0, &range_err);
        const bool limited = !range_err && range == 1;

        int matrix_err = 0;
        const int64_t matrix =
            vsapi->mapGetInt(vsapi->getFramePropertiesRO(first), "_Matrix", 0, &matrix_err);
        vsapi->freeFrame(first);

        if (d->yuv) {
            // bt.709, bt.470bg / smpte 170m (601) and bt.2020 non-constant luminance
            if (matrix_err) return fail("the curved blend on yuv needs _Matrix set");
            if (matrix == 1) { d->kr = 0.2126f; d->kb = 0.0722f; }
            else if (matrix == 5 || matrix == 6) { d->kr = 0.299f; d->kb = 0.114f; }
            else if (matrix == 9) { d->kr = 0.2627f; d->kb = 0.0593f; }
            else return fail("the curved blend on yuv supports _Matrix 1, 5, 6 and 9");
        }

        // float is already normalised so black is 0 and white is 1 whichever range it came from,
        // and anything the source had above white simply sits above 1
        if (f.sampleType == stFloat || !limited) {
            d->black = 0.0f;
            d->span = f.sampleType == stFloat ? 1.0f
                                              : static_cast<float>((1 << f.bitsPerSample) - 1);
        } else {
            const int shift = f.bitsPerSample - 8;
            d->black = static_cast<float>(16 << shift);
            d->span = static_cast<float>(219 << shift);
        }

        if (f.sampleType == stFloat) {
            d->mid = 0.0f;
            d->cspan = 1.0f;
        } else {
            d->mid = static_cast<float>(1 << (f.bitsPerSample - 1));
            d->cspan = limited ? static_cast<float>(224 << (f.bitsPerSample - 8)) : d->span;
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
                         VS_MAKE_VERSION(3, 2), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("FrameBlend",
                             "clip:vnode;weights:float[];scale:float:opt;planes:int[]:opt;"
                             "squared:int:opt;transfer:data:opt;",
                             "clip:vnode;", blendCreate, nullptr, plugin);
    vspapi->registerFunction("Isa", "", "isa:data;", isaInfo, nullptr, plugin);
}
