# vs-frameblender

This is a shamelessly AI-written modification of the original [AverageFrames](https://github.com/vapoursynth/vapoursynth/blob/master/src/core/averageframesfilter.cpp) that removes the cap of 31 weights, and also attempts to improve performance.

---

Temporal weighted average for VapourSynth, with no cap on the number of weights.

`std.AverageFrames` is limited to 31 weights. This does the same job without that limit, and is
faster than every alternative once the window gets large.

```python
core.frameblender.FrameBlend(clip, weights=[...], scale=None, planes=None)
```

- `weights` — one float per frame in the window, odd count, centred on the current frame.
  Any scale; they do not need to be normalised.
- `scale` — divisor. Defaults to the sum of the weights, which is what you want for an average.
- `planes` — which planes to process. Defaults to all.

Frames past the start and end of the clip are clamped to the first/last frame.
8–16 bit integer and 32 bit float are supported.

`core.frameblender.Isa()` reports which kernel the CPU dispatch selected (`baseline`, `avx2` or
`avx512`) — useful for checking the dispatch is doing what you expect.

## Why it's fast

The obvious way to write this is to hold N row pointers and reduce across them for each pixel.
That only vectorises when N is a compile-time constant, which is exactly why upstream's
`AverageFrames` templates on the source count and stops at 31 — past that it falls back to a
runtime loop that doesn't vectorise at all and is ~24× slower.

This swaps the loops. The pixel loop is innermost over a row-sized `int32` accumulator, so each
source frame is one clean vectorised pass and nothing depends on N being known at compile time:

```
for each row:
    zero acc[0..w)
    for k in 0..num_srcs:
        for x in 0..w:  acc[x] += (src[k][x] - bias) * weights[k]
    for x in 0..w:      dst[x] = clamp(round(acc[x] / scale) + bias)
```

The accumulator is `w * 4` bytes, so it stays in L1. Removing the cap costs nothing — it is
_faster_ than the templated path at every window size worth using, because that version's
register pressure grows with N while this one's doesn't.

Measured at 1080p yuv420p8, output frames/sec, on a 9800X3D:

| weights | akarin 1.5.0 | llvmexpr | std.AverageFrames (R72) | **frameblender** |
| ------- | ------------ | -------- | ----------------------- | ---------------- |
| 21      | 752          | 762      | 755                     | **783**          |
| 31      | 586          | 599      | 567                     | **739**          |
| 81      | 265          | 362      | cap                     | **495**          |

Frames are requested with a stride, which is how the consumer of this actually pulls them —
benchmarking consecutive frames instead makes the windows overlap, keeps everything in L3, and
measures a different bottleneck.

## Accuracy

Validated against a float64 reference. Max error is **≤1 LSB** on yuv420p8, yuv420p10 and
rgb48, and **bit-exact** on RGBS and on 8-bit at ≥21 weights — the same accuracy as
`akarin.Expr` and `llvmexpr.Expr`.

8-bit uses an int32 accumulator with the weights quantised by the largest multiplier that
cannot overflow it, which puts the quantisation error far below one level. 16-bit uses a float
accumulator instead: int32 would only leave ~15 bits for the weights at that depth, which is
visible.

## Building

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Produces `frameblender.dll` / `libframeblender.so`. The VapourSynth 4 headers are vendored in
`include/`, so there is nothing else to fetch.

Do not add `-ffast-math`. The rounding in `kernel.h` relies on an add/subtract pair that
fast-math would fold away.

## Platforms

**x86-64 (Windows, Linux, Intel macOS).** The kernel is compiled three times — baseline, AVX2,
AVX-512 — into separate object files with different `-march`/`/arch` flags, and the right one is
picked by CPUID at load. Same shape upstream VapourSynth uses for its own filters, and it
matters: the baseline kernel is ~25% slower than AVX2 at 21 weights and **2.4× slower at 81**.

CPU feature detection uses `__cpuid`/`_xgetbv` under MSVC and clang-cl, and
`__builtin_cpu_supports` under GCC and clang elsewhere — `<intrin.h>` does not exist on GCC.

**aarch64 (Apple Silicon, ARM Linux).** NEON is the baseline, so there is nothing to dispatch
between: CMake skips the x86 flags, all three variants build identically, and detection always
returns `baseline`. The kernel is plain C++ with no intrinsics, so it auto-vectorises to NEON on
its own. Slightly redundant (three identical copies of a small function) but correct.

`core.frameblender.Isa()` tells you which one is live.

> Only the Windows x86-64 build has been run and benchmarked. The Linux and aarch64 paths are
> compile-checked for the preprocessor branches but otherwise untested — CI is the real check.

## Notes

- Requires VapourSynth API 4 (R55+). The previous version of this plugin used API 3; the
  namespace and function name are unchanged, but the `log` argument is gone and `scale` and
  `planes` are new.
- The old version truncated instead of rounding, so output was up to one level darker.
