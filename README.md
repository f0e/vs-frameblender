# vs-frameblender

This is a shamelessly AI-written modification of the original [AverageFrames](https://github.com/vapoursynth/vapoursynth/blob/master/src/core/averageframesfilter.cpp) that removes the cap of 31 weights, adds a squared blend option, and attempts to improve performance.

---

Temporal weighted average for VapourSynth, with no cap on the number of weights.

`std.AverageFrames` is limited to 31 weights. This does the same job without that limit, and is
faster than every alternative once the window gets large.

```python
core.frameblender.FrameBlend(clip, weights=[...], scale=None, planes=None, squared=False)
```

- `weights` — one float per frame in the window, odd count, centred on the current frame.
  Any scale; they do not need to be normalised.
- `scale` — divisor. Defaults to the sum of the weights, which is what you want for an average.
- `planes` — which planes to process. Defaults to all.
- `squared` — average in a squared domain rather than a linear one, so the blend adds up light
  rather than gamma-encoded codes. `False` (the default) dispatches to the plain kernels and
  costs nothing. RGB and GRAY only; see below.

Frames past the start and end of the clip are clamped to the first/last frame.
8–16 bit integer and 32 bit float are supported.

`core.frameblender.Isa()` reports which kernel the CPU dispatch selected (`baseline`, `avx2` or
`avx512`) — useful for checking the dispatch is doing what you expect.

## The squared blend

With `squared=True` the blend runs in a squared domain instead of a linear one:

```
dst = f^-1( sum w_i * f(src_i) / scale )

f(v) = v|v|        f^-1(y) = sign(y) * sqrt(|y|)
```

in normalised units where 0 is reference black and 1 reference white.

The numbers in a video file are not brightness — they sit on a gamma curve, where code 128 is
about a fifth of the light of code 255, not half. Averaging them directly does not give the
average of what they represent, the same way averaging decibels does not give the average sound
power. A real camera integrates light over the shutter, so a linear average of encoded codes
comes out darker than a real exposure on every moving edge.

Squaring first is a close enough stand-in for undoing the transfer: against a true 2.2 blend the
worst case is ~0.02 in normalised units, and it costs one multiply instead of a `pow`. A static
image is returned exactly, since `f^-1(f(v)) = v`.

`v|v|` rather than `v*v` so sub-black stays ordered and invertible, and superwhite runs past 1.0
rather than folding back.

### Where black and white are

The curve has to know where black and white are. That comes from the clip's own `_ColorRange` on
the first frame — 0 and the format maximum for full range, 16 and 235 at 8 bit for limited.
Float is already normalised either way, so it is always 0 and 1.

This matters more than it sounds. Converting limited range footage to full range first, so the
curve can assume the whole container, looks equivalent and is not: it clips every sample above
white on the way in, and those are exactly the samples a brightness curve exists to protect.
Anchored to the real white instead, samples past it come out above 1.0, which the float
accumulator carries without complaint, and the inverse puts them back where they were. Hand the
filter a limited range clip and leave it limited.

Below black the curve is simply linear, so there is nothing to floor.

To anchor somewhere else, set `_ColorRange` on the clip — there is no separate argument for it.

### Notes

It only applies to RGB and GRAY. A curve about the chroma midpoint has no meaning, so
`squared=True` on a YUV clip is an error unless `planes` excludes chroma — and squaring luma
alone desaturates, because the chroma still averages linearly and a bright saturated thing comes
out bright and washed out. Convert to RGB.

The squared blend is not free — the accumulate loop is where all the work is, and it gains a
normalise, a multiply and an fma per sample:

| weights | linear | squared |
| ------- | ------ | ------- |
| 21      | 619    | 138     |
| 81      | 179    | 71      |

720p RGB24 in and out, output frames/sec, best of 4 on a 9800X3D. Against the power mean this
replaced, measured the same way on the same machine: **2.4× at 21 weights and 3.8× at 81** (58
and 18 fps), all of it from dropping the two tables.

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

With the squared blend, validated the same way against a float64 reference: **bit-exact on
RGB24 and RGB48** and ~3e-8 on RGBS. A static frame comes back unchanged to 0 levels. The
accumulator is float on the squared path at all depths, not the int32 one the linear 8 bit path
uses, so samples above white have somewhere to go.

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
  namespace and function name are unchanged, but the `log` argument is gone and `scale`,
  `planes` and `squared` are new.
- v3 replaced the `gamma` argument with `squared`. A power mean can only reweight the frames in
  the window, so past about `gamma=3` it hands one frame nearly the whole pixel and the output
  stops being a blur at all. Squaring is the transfer function the codes were encoded with, not
  a gain, so it corrects the average without trading blur away for brightness - and is faster.
- The old version truncated instead of rounding, so output was up to one level darker.
