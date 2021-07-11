#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <VapourSynth.h>
#include <VSHelper.h>

static inline void getPlanesArg(const VSMap* in, bool* process, const VSAPI* vsapi) {
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

        if (o < 0 || o >= 3)
            throw std::runtime_error("plane index out of range");

        if (process[o])
            throw std::runtime_error("plane specified twice");

        process[o] = true;
    }
}

namespace {
    typedef struct {
        VSNodeRef* node;
        VSVideoInfo vi;
        std::vector<float> weightPercents;
        bool process[3];
    } FrameBlendData;
}

static void VS_CC frameBlendInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi) {
    FrameBlendData* d = static_cast<FrameBlendData*>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

template <typename T>
static void frameBlend(const FrameBlendData* d, const VSFrameRef* const* srcs, VSFrameRef* dst, int plane, const VSAPI* vsapi) {
    int stride = vsapi->getStride(dst, plane) / sizeof(T);
    int width = vsapi->getFrameWidth(dst, plane);
    int height = vsapi->getFrameHeight(dst, plane);

    const T* srcpp[128];
    const size_t numSrcs = d->weightPercents.size();

    std::transform(srcs, srcs + numSrcs, srcpp, [=](const VSFrameRef* f) {
        return reinterpret_cast<const T*>(vsapi->getReadPtr(f, plane));
     });

    T* VS_RESTRICT dstp = reinterpret_cast<T*>(vsapi->getWritePtr(dst, plane));

    unsigned maxVal = (1U << d->vi.format->bitsPerSample) - 1;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float acc = 0;

            for (size_t i = 0; i < numSrcs; ++i) {
                T val = srcpp[i][w];
                acc += val * d->weightPercents[i];
            }

            int actualAcc = std::clamp(int(acc), 0, int(maxVal));
            dstp[w] = static_cast<T>(actualAcc);
        }

        std::transform(srcpp, srcpp + numSrcs, srcpp, [=](const T* ptr) { return ptr + stride; });
        dstp += stride;
    }
}

static const VSFrameRef* VS_CC frameBlendGetFrame(int n, int activationReason, void** instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    FrameBlendData* d = static_cast<FrameBlendData*>(*instanceData);

    const int half = int(d->weightPercents.size() / 2);

    bool clamp = (n > INT_MAX - 1 - half);
    int lastframe = clamp ? INT_MAX - 1 : n + half;

    if (activationReason == arInitial) {
        // request all the frames we'll need
        for (int i = std::max(0, n - half); i <= lastframe; i++)
            vsapi->requestFrameFilter(i, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        // get this frame's frames to be blended
        std::vector<const VSFrameRef*> frames(d->weightPercents.size());

        int fn = n - half;
        for (size_t i = 0; i < d->weightPercents.size(); i++) {
            frames[i] = vsapi->getFrameFilter(std::max(0, fn), d->node, frameCtx);
            if (fn < INT_MAX - 1)
                fn++;
        }

        const VSFrameRef* center = frames[frames.size() / 2];
        const VSFormat* fi = vsapi->getFrameFormat(center);

        const int pl[] = { 0, 1, 2 };
        const VSFrameRef* fr[] = {
            d->process[0] ? nullptr : center,
            d->process[1] ? nullptr : center,
            d->process[2] ? nullptr : center
        };

        VSFrameRef* dst;
        dst = vsapi->newVideoFrame2(fi, vsapi->getFrameWidth(center, 0), vsapi->getFrameHeight(center, 0), fr, pl, center, core);

        for (int plane = 0; plane < fi->numPlanes; plane++) {
            if (d->process[plane]) {
                if (fi->bytesPerSample == 1)
                    frameBlend<uint8_t>(d, frames.data(), dst, plane, vsapi);
                else if (fi->bytesPerSample == 2)
                    frameBlend<uint16_t>(d, frames.data(), dst, plane, vsapi);
                else {
                    vsapi->logMessage(mtFatal, "msg tekno and tell him to fix :alien:");
                    return nullptr;
                }
            }
        }

        for (auto iter : frames)
            vsapi->freeFrame(iter);

        return dst;
    }

    return nullptr;
}

static void VS_CC frameBlendFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    FrameBlendData* d = static_cast<FrameBlendData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC frameBlendCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    std::unique_ptr<FrameBlendData> d(new FrameBlendData());
    int numWeights = vsapi->propNumElements(in, "weights");
    int err;

    try {
        if ((numWeights % 2) != 1)
            throw std::runtime_error("Number of weights must be odd");
        
        // get clip and clip video info
        d->node = vsapi->propGetNode(in, "clip", 0, &err);
        d->vi = *vsapi->getVideoInfo(d->node);

        // get weights
        float totalWeights = 0.f;
        for (int i = 0; i < numWeights; i++)
            totalWeights += vsapi->propGetFloat(in, "weights", i, 0);

        // scale weights
        for (int i = 0; i < numWeights; i++)
            d->weightPercents.push_back(vsapi->propGetFloat(in, "weights", i, 0) / totalWeights);

        // logging
        std::string weightStr;
        for (auto& weight : d->weightPercents) {
            if (weightStr != "")
                weightStr += ", ";

            weightStr += std::to_string(weight);
        }

        vsapi->logMessage(mtDebug, ("Frame blending with weights [" + weightStr + "]").c_str());

        getPlanesArg(in, d->process, vsapi);
    }
    catch (const std::runtime_error& e) {
        vsapi->freeNode(d->node);
        vsapi->setError(out, (std::string("FrameBlend: ") + e.what()).c_str());
        return;
    }

    vsapi->createFilter(in, out, "FrameBlend", frameBlendInit, frameBlendGetFrame, frameBlendFree, fmParallel, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin) {
    configFunc("com.vapoursynth.frameblender", "frameblender", "Frame blender", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("FrameBlend", "clip:clip;weights:float[]", frameBlendCreate, 0, plugin);
}