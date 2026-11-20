# Divergences from the vendor kernels

Every re-implemented or transformed kernel is logged here with its reason and the argument for why it preserves
results (or an honest note that it may not). The vendor's Vulkan SPIR-V passes (12 `AIFF*_cs`) and all weights run
**unchanged**; the items below are the pieces that had no RDNA3-runnable vendor form.

> **Correction to D1 (2026-11-19, VERIFIED from the readable parts of `Kernels_Hawi/*.cl`):** the Adreno requant is
> `f = mad.f32(scale, float(acc), bias_f)`, clamp to `[min,max]` (kernel args), **`rndaz`** (round, ties away from zero), convert
> to int — *not* `convert_char_sat` truncation. For `INV_POW2` it is an integer `mad.s24(scale_i, acc, bias_i)` followed by
> taking the high half (`>>16`, floor). The SR conv (`conv_int8.comp`) still uses truncation and `ReLU = max(f,0)`; it must be redone
> against this (and the offset-domain finding in D2) before SR output can be trusted. FG already uses the corrected semantics.

## D1 — int8 convolution (`kernels/conv_int8.comp`)  [reconstructed; SUPERSEDED for rounding, see correction above]
**Why:** ANF's convolutions exist only as Adreno machine code (QASM inline asm in `Kernels_Hawi/*.cl`, precompiled
Adreno OpenCL binaries, and ARM data-graph caches). None runs on RDNA3. The network *weights, biases, per-channel
scales and architecture* are used **unchanged** (`vendor/assets/modelzoo/...`); only the conv *execution* is ours.

**Semantics reproduced (from `modelconfig.json`, `*.scales.json`, and the readable C in the `.cl` kernels):**
- int8 × int8 → **int32** accumulation. Integer addition is associative, so the result is independent of summation
  order: bit-identical regardless of tiling/subgroup. (VERIFIED by construction.)
- Requant: `f = float(acc + bias_i32) * scale[oc]`, then ReLU `max(f,0)`, then convert to int8 with
  **round-toward-zero + saturate** to [-128,127] (OpenCL `convert_char_sat`, which the Adreno kernel uses via
  `run_activation` + store).
- `scale[oc] = weights_scales[oc]` from `*.scales.json`. For the **`invpow2`** model variant these are exact powers
  of two (2^-6, 2^-7, 2^-8), so the float multiply is **exact** (no rounding) — the whole layer is then bit-exact on
  any IEEE-754 device. For the default model the scales are arbitrary floats; the multiply rounds once (RNE), matched
  between our CPU reference and GPU.

**Preservation argument:** The only hardware-defined step is the single `float(acc)*scale` multiply and the
float→int8 conversion. Both are IEEE-754 / fixed-semantics and identical between our CPU reference and the GPU
(VERIFIED bit-exact, `tests/conv_test.cpp`, all SR layers). Versus Adreno: with `invpow2` the multiply is exact and
only the convert rounding could differ; we use round-toward-zero to match OpenCL `convert_char`. **ASSUMED** that
Adreno's store uses `convert_char_sat` (round-toward-zero); unverifiable without a device. Risk is at most ±1 LSB on
rare exact-half cases, and zero for the invpow2 model except at the convert step.

**Validation:** `tests/conv_test.cpp` — GPU kernel vs independent CPU reference, bit-exact (0 mismatches) on real
extracted weights for cnn1 (20→32 s2), cnn2-0, cnn2-1 (32→32 s1), cnn3 (32→64 s1), both invpow2 and float-scale.

## D2 — FG int8 convolution + heads (`kernels/fg_conv_i8.comp`, `fg_head_lr.comp`, `fg_head_qr.comp`)  [reconstructed]
**Why:** same as D1 — FG convs exist only as Adreno asm / precompiled binaries. Weights, biases, scales and topology are the vendor's.
**Semantics, each VERIFIED against the readable parts of the vendor kernels or the data:**
- *Activation domain.* Hidden activations are `uint8 - 128` stored as int8. Evidence: layer-0 `bias*scale == -128` for every channel;
  layers >=1 have `bias ~ 128*sum(w)` (corr 0.97-1.0, slope 104-150) = input zero-point folded into the bias; `Conv2D_*` use pad `0x80808080`.
  So **ReLU is the clamp at -128** (`LO=-128, HI=127`), and **out-of-bounds taps read -128, not 0** — except the first layer (`ZERO_PADDING`, pad 0,
  input is true signed int8) and the fp16 head (real zero). Getting these two wrong made the nets output garbage; with them, static frames
  reproduce the input at 57 dB.
- *Requant:* `scale[oc] = weights_scales*input_activation_scale/output_activation_scale`, `bias_f = float(bias_i32)*scale`,
  `f = fma(scale, float(acc), bias_f)`, clamp, **round half away from zero** (`rndaz`; ASSUMED = OpenCL `round`, ties only), int8 store.
  Whether the host passes exactly `bias_i32*scale` as the float bias (vs rounding differently) is ASSUMED. fma vs mul+add rounding on Adreno unknown.
- *LR `conv_out`* (int8 -> fp16): `relu((acc+bias)*ws*ias)` stored as fp16, **round-to-nearest-even implemented explicitly** (RADV `packHalf2x16` truncates).
- *LR head* (`conv_out_flow_mask`, fp16): fp16 weights are *already dequantised* (`w = q*ws`, VERIFIED `max|w|/ws = 127`), fp32 accumulate,
  `+ fp16 bias`, **no scale, no ReLU** (matches `requant_N20` in `Conv2D_64_3_3_20_Relu_mixed.cl`, which only does `add.f bias`). Channels 0-15 = flow,
  depth-to-space x2 in **DCR** order (`c = (dy*2+dx)*4 + k`, from the asm store offsets); channel 16 = mask plane at net resolution; 17-19 unused (zero weights).
  Accumulation order differs from Adreno `pkmad.f16`; each fp16 product is exact in fp32, so only the add order can differ (<= 1 fp16 ulp).
- Weight layout repacked `[oc][ic][3][3] -> [oc][ky][kx][ic]` (layout of the files VERIFIED by tap-correlation); pure permutation.
**Validation:** `tests/fgnet_test.cpp` — GPU == independent CPU reference, bit-exact on all 14 layers of both nets (real weights), 3 runs deterministic, validation-clean.

## D3 — QR head output quantisation  [reconstructed, partly ASSUMED]
QR `conv_out_flow_mask` stores uint8 (`modeBias=128` into the vendor `QRUnpack`, so no shift). VERIFIED: the *bias already carries the output zero-point*
(adding `+offset` again shifted every flow by `offset*scale*512 = 6 px`; removing it makes zero residual map to `q == offset`). The `max/min` clamp
the host passes is not recoverable: we clamp to `[zp, 255]` (ReLU in the quantised domain, as `modelconfig.json` says `relu`); `ANF_QR_RELU=0` clamps to `[0,255]`.
On the test scene both give identical PSNR. ASSUMED.

## D4 — FG pipeline wiring (`runtime/fg.cpp`)  [reconstructed host logic; the 8 vendor shaders run unchanged]
The vendor SPIR-V is unchanged. The *host* logic that ANF runs around it is ours:
- **Resolutions (ASSUMED, consistent with every shader):** PREPROCESS = ceil(FULL/2); QR = PREPROCESS padded to x4; LR = ceil(QR/4) padded to x4
  (pyramid shader: 4x4 QR texels per LR texel; QRBlend samples flow at `pix/4` and mask at `pix/8` of FULL).
- Push constants/spec constants/format of every intermediate come from the reflected shader interfaces; quant scales/offsets from `network_io_encodings.json` of the
  prod-8 model dirs. `jitter_*=0`, `use_norm_depth=0`, `invert_depth` configurable, `discontinuityThresholdSq=0` (value ANF uses unknown).
- Pre-frame state: packed-MV buffers and fallback stats zeroed; depth-stats accumulators initialised (`min=+inf` bits, `max=0`).
- History (prev/curr ping-pong for QR/LR YUV, LR depth, QR depth pyramid; prior colour copy). First frame: previous := current.
- `AIFFQRBlendV2FINT` chosen over `QRBlendV2`; `AIFFLRUnpackV2CL` (fp16 input) over `LRUnpackV2` (uint16-quantised data-graph encoding).
- **Fallback semantics (VERIFIED from the shader):** the blend outputs the *current frame unmodified* when `luma_sum > 8*LRW*(LRH-1)` (mean |dluma| > 8/255
  per LR texel: a scene-cut detector) or when > 20% of LR texels have |MV| > 32 px. Fast motion therefore bypasses FG by design.
**Validation:** `tests/fg_test.cpp` on the SDK's own hero render translated by a known sub-pixel vector: deterministic across reset+replay; identity at zero motion
(57 dB); at (2, 0.7) px/frame FG is 31.7 dB vs the analytic mid-frame against 31.5 dB for plain averaging and 27.2 dB for repeat-current. NOT device-validated.

## D5 — SR convolutions (`runtime/sr.cpp` driving `kernels/fg_conv_i8.comp`)  [reconstructed; supersedes D1 for SR]
The SR convs now use the FG conv kernel with the semantics verified in D2: `-128` activation domain (hidden layers carry the `-128` output zero-point in the
folded bias: residual `(bias - 128*sum(w))*ws` = -123 / -106 for cnn2-0 / cnn2-1, vs -128 in FG), pad `0x80` (verified in `Conv2D_20_3_3_32_Jittered_Relu_int8.cl`),
clamp `[-128,127]`, round-half-away. `cnn3` is plain signed int8 (residual -28: no output zero-point); the postprocess adds 128 (`raw = q + 128`).
SR scales files only carry `weights_scales` (`ias = oas = 1`), so `scale = ws`, `bias_f = bias_i32*ws`. The INV_POW2 vendor path is an integer `mad.s24` then `>>16`
(floor) with a host-built bias; we model it as round-half-up on the exact value (ASSUMED: the host adds the `0x8000` rounding term). `conv_int8.comp`/`model_test` (D1) are
**obsolete** and kept only for history (they used truncation and `ReLU=max(f,0)`, both wrong).

## D6 — SR pre/post-processing (`kernels/sr_pre.comp`, `kernels/sr_post.comp`)  [reconstructed from readable asm]
Source: `preprocess_qmfnet_v2.cl` (+ `_v2_1`), `postprocess_qmfnet_v2_uint8.cl`, and the inverse-tonemap block of `Conv2D_32_3_3_64_Jittered_Relu_Blend_D2S_qmfnet_v2_int8.cl`.
VERIFIED from those sources:
- Model input, 20 ch, `MODEL_INPUT_STRIDE_BYTES = 20` variant: words = `[R,G,B,depth]`, `[feature x4]`, `[prev R x4]`, `[prev G x4]`, `[prev B x4]` (the 4 = the 2x2 HR pixels of an LR pixel), each
  `clamp(rndaz(v*255),0,255)` stored as `u8 - 128` (non-`UINT8_PACKING`); pad words `0x80808080`.
- Colour: `s = sqrt(c)`, `t = s / (max(s) + pre_exposure)` (reversible tonemap); depth: 3x3 *max* (reverse-Z, strict `>`, out-of-frame -1) selects the MV; `FINAL_MV_INPUT` so the MV is used
  directly: `fm = mv * (W*-1, H*+1)`, previous-frame samples at `2*uv + 2*fm + {0.5,1.5}` (bilinear, `CLK_ADDRESS_CLAMP` = border 0) from the HR tonemapped history and the HR feature history.
- Net output: `DepthToSpaceDCR(4)`: `raw[(y/4, x/4)][((y%4)*4+x%4)*4 + {0,1,2,3}] = {offset_x, offset_y, alpha, feature}`, each `/255`; candidate = bilinear LR colour at
  `floor((p-0.5)/2 - (jx,-jy)) + offset`; `out = alpha*warped_prev + (1-alpha)*candidate` (tonemapped space); inverse tonemap `(t/(1-max t))^2`.
ASSUMED / ours: `pre_exposure = 1` (the vendor's inverse implies 1); upper clamp of `t` before the inverse = 0.9995 (the vendor passes `min/max` as kernel args); jitter quantisation
(8x8 phase grid, `index = iy*8+ix`, quantised offset = cell centre) — the phase table is not recoverable without the host code; depth convention reverse-Z (option `invertDepth`);
fp32 arithmetic where the vendor uses fp16 (`mad.f16`) in the blend. History is zero on reset. Not device-validated.
**Validation (no ground truth exists):** `tools/sr_demo.py` renders a static-camera Halton-jittered sequence from the SDK hero image and compares with the HR source:
alignment is exact (best global shift 0,0), **jitter sign convention (x right, y up) is the unique best of the four** (31.4 vs 26-30 dB), PSNR rises monotonically with frames
(30.0 -> 31.5 dB at 320x180; 32.0 -> 33.2 dB at 640x360 vs 28.4 dB for bicubic of the same LR frames). Deterministic, validation-clean (`tests/check_sr.sh`).

## D7 — performance kernels (no change in results)  [VERIFIED bit-identical]
`kernels/conv_wmma.comp` computes the same convolutions as `fg_conv_i8.comp` as an implicit GEMM on the RDNA3 matrix units (`VK_KHR_cooperative_matrix`, int8 x int8 -> int32, 16x16x16):
A (im2col rows) is staged through LDS with the vendor's out-of-bounds value (`PADV`), B is the repacked weights loaded column-major, the epilogue (fma, clamp, rndaz, int8/fp16/uint8 D2S
store) is the same per-element code. int32 accumulation is order-independent, so the results are identical; verified by `tests/fgnet_test` (all layers incl. the QR head, GPU == CPU reference) and
by `cmp` of the SR output against the pre-optimisation baseline (also for odd sizes, and with `ANF_NO_WMMA=1` which selects the scalar fallback). The LR head still uses plain fp32 sums in the old order (one
thread per channel instead of per position: same arithmetic). `sr_pre` uses an LDS tile for the 3x3 depth/MV search; the warped previous colour is no longer written to a 29 MB image but re-derived in
`sr_post` with the identical `precise` expressions. Pipelines pin subgroup size 32; if the device refuses the cooperative-matrix pipeline the scalar kernels are used.
