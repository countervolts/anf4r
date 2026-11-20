# Architecture

anf4r reimplements the *host side* and the *convolution kernels* of ANF's two techniques on RDNA3. Everything below is what the code does; the label
**[V]** = verified here (bit-exact test or measured), **[R]** = reconstructed from observed behaviour and consistent with all checks, **[A]** = assumed (see `divergences.md`).

## Networks
- **Quantisation domain [V].** Activations are `uint8 − 128` held as int8; ReLU is the clamp at −128; out-of-bounds taps read −128 (0 for the first layer's true-int8 input).
  Per-layer: `scale[oc] = weight_scale · input_scale / output_scale`, `bias_f = bias_i32 · scale`, `q = clamp(round_half_away(fma(scale, acc, bias_f)), lo, hi)` with int32 accumulation. The folded
  biases already contain the zero-point terms.
- **Super resolution (QMFNet V2) [R].** `cnn1` (20→32, stride 2, jitter-specific weights) → `cnn2-0` → `cnn2-1` (32→32) → `cnn3` (32→64, jitter-specific). 64 weight sets per jitter layer = an 8×8 sub-pixel
  phase grid (index `iy*8+ix`, quantised offset = cell centre; the layout was selected by measurement, other layouts and Halton tables were worse). Output = depth-to-space 4 (DCR): per HR pixel `{offset_x, offset_y, alpha, feature}`.
- **Frame generation [V]/[R].** Two flow networks, 14 convs each: LR (32→96…→64 fp16 head 64→20) and QR (32→32…→20). Head channels: 16 flow (depth-to-space ×2, DCR) + 1 mask + 3 unused.

## Pipelines
**SR** (`runtime/sr.cpp`): `sr_pre` (unpack + tonemap `t = √c / (max√c + e)`; 3×3 max-depth MV pick; warp of the previous HR history; packs the 20-channel int8 input
`[R G B depth][feature×4][prevR×4][prevG×4][prevB×4]`) → 4 convolutions → `sr_post` (offset-warped candidate sample of the LR colour, `α·history + (1−α)·candidate`, inverse tonemap `(t·e/(1−max t))²`).
History: tonemapped colour + feature per HR pixel (ping-pong). Optional `sr_stab` output stabiliser (off by default; divergences D8) and `sr_resize` (bicubic) for ratios ≠ 2.

**FG** (`runtime/fg.cpp`): eight Qualcomm SPIR-V passes run **unchanged** (PreprocessFused, UnpackMV, PyramidDownscaleFused, MetaAndLRPack, LRUnpack(CL), LRUpscaleAndQRPack, QRUnpack, QRBlend(FINT)) around our LR/QR convolution
networks. Resolutions: FULL → QR = FULL/2 → LR = QR/4. The blend returns the current frame when the summed luma difference is large (scene-cut guard, part of the vendor shader).

**Convolutions** (`kernels/conv_wmma.comp`): implicit GEMM on `VK_KHR_cooperative_matrix` int8 16×16×16 tiles, one wave32 = 32 pixels × 32 channels, A staged through LDS with the padding value, weights loaded column-major,
epilogue per element. `fg_conv_i8.comp` is the scalar reference/fallback; `tests/fgnet_test` proves the GPU path equals an independent CPU implementation on all layers.

## Runtime
`runtime/vk.*` (context, or adopt a caller's device), `pass.*` (compute pipeline from SPIR-V + interface, 8-deep descriptor ring), `vkfn.*` (function table for use inside a game's process),
`fgnet.*`/`fg.*`/`sr.*` (pipelines), `anf4r.h`/`anf4r_api.cpp` (C API; recording into the caller's command buffer). Frame capture for debugging: create `<data>/dump.txt` containing a frame count.

## Data layout
```
<data>/shaders/*.spv            built from kernels/ (cmake target anf_kernels)
<data>/spirv/AIFF*_cs.bin       the frame-generation compute shaders (SPIR-V) from your ANF data
<data>/modelzoo/aiff-sr-v2s-prod-7p1-invpow2/{cnn1,cnn2-0,cnn2-1,cnn3}/<phase>/<layer>.{weight_shape-OC-IC-3-3_int8.raw,bias_shape-OC_int32.raw,scales.json}
<data>/modelzoo/aiff-frc-{lr,qr}-v2-prod-8/<layer>/jitter0/{weights.raw,bias.raw,scales.json}  +  network_io_encodings.json
<data>/config.txt               optional: stabilize = <0..1>, pre_exposure = <float>
```