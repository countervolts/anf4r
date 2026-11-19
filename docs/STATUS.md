# anf4r status — verified / reconstructed / blocked

Updated 2026-11-19. Tags: **VERIFIED** (seen here), **ASSUMED** (inference), **BLOCKED** (needs a device or more work).

## Verified working
1. **Decryptor** — loader at `libanf.so:0x1c3904` reversed: `zlib(data) XOR keystream`, keystream = filename stem
   repeated. `tools/extract_assets.py` extracts all 1278 assets to exact sizes, 0 failures. VERIFIED
2. **Runtime runs vendor SPIR-V** — `AIFFPrepareInputs_cs` builds + dispatches on RX 7700 XT, validation-clean. VERIFIED
3. **int8 conv bit-exact** — `kernels/conv_int8.comp` == CPU reference on real weights, all SR layers. VERIFIED
4. **Full SR network** — QMFNet V2 chain bit-exact to CPU and deterministic across runs and jitter phases. VERIFIED
5. **Plan generator** — `tools/prepare_sr.py` builds per-resolution SR plans (per-jitter weights for cnn1/cnn3,
   shared cnn2-0/cnn2-1). VERIFIED
6. **Dispatch facts** — SR convs dispatch via OpenCL; FG uses the 12 SPIR-V passes in order UnpackMV → PreprocessFused
   → PyramidDownscale → MetaAndLRPack → [LR conv] → LRUnpack/LRUpscaleAndQRPack → [QR conv] → QRUnpack → QRBlend.
   From the full decompile (`build/ghidra/all.c`). VERIFIED

## Reconstructed (runs, correctness unverifiable without a device)
- **int8 conv requant rounding** — assumed OpenCL `convert_char_sat` (round-toward-zero). For the `invpow2` model the
  scale multiply is exact, so only this rounding could differ from Adreno. ASSUMED, logged in divergences.md D1.
- **SR input packing (20 channels)** — Adreno-asm only (`preprocess.cl`: quantized depth, feature fibers). The known
  channels are color (3), depth (1), motion (2), warped previous color; the full 20-channel composition and exact
  quantization are not recoverable with confidence. BLOCKED on a device or a dynamic Adreno trace.
- **SR output** — `cnn3` is a `ConvBlend` producing 64 channels → depth-to-space ×2 + blend with the upsampled input.
  The blend math is Adreno-asm (`postprocess*.cl`). BLOCKED similarly.

## Blocked / not done
- **End-to-end SR image correctness** — needs the real preprocess/postprocess (above). Without ground truth it cannot
  be validated even if reconstructed. A Snapdragon 8 Elite Gen 6+ device, or running `libanf.so` under qemu-aarch64
  with a recording OpenCL/Vulkan stub, would unblock this (the latter is set up-able; the SDK obfuscation is already
  broken so the kernels are known).
- **Frame Generation pipeline** — the faithful path (vendor SPIR-V glue unchanged + our conv). Pass order known; the
  exact tensor pack/unpack byte layout is in the `MetaAndLRPack`/`*Unpack` SPIR-V (readable, no asm) and still needs
  transcribing. Convs (aiff-frc-lr/qr) re-implementable with the existing conv kernel. NOT YET WIRED.
- **OptiScaler + vkd3d-proton integration** — C API (`runtime/anf4r.h`) defined; backend hook and patch series
  (follow m4r `patches/optiscaler/`, pinned commit fd3a45a) NOT YET BUILT.

## Recommended next steps (in order)
1. Wire **FG** end-to-end (vendor SPIR-V + conv) — the most faithful, fully portable result; validate by determinism.
2. If a Snapdragon device becomes available: build the Android capture tool, dump per-frame SR inputs/outputs, and
   reconstruct+validate SR preprocess/postprocess bit-for-bit. Otherwise do it via a qemu trace of `libanf.so`.
3. OptiScaler backend + patches; D3D12 harness under Proton; measure frame time at 1440p.
