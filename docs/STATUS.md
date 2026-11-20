# anf4r status

Updated 2026-11-19. Tags: **VERIFIED** (seen here), **ASSUMED** (inference), **BLOCKED** (needs a device or more work).
No Adreno device exists, so nothing here is *device-validated*; the validation we have is CPU-reference equality, determinism,
Vulkan-validation cleanliness and quality metrics against ground truth we can synthesise.

## Working (run `tests/check_fg.sh`, `tests/check_sr.sh`)
1. **Asset decryptor** `tools/extract_assets.py` — all 1278 assets. VERIFIED
2. **Runtime** `runtime/{vk,pass}.cpp` — Vulkan context + SPIR-V pass runner; vendor shaders run unchanged on RADV. VERIFIED
3. **Frame Generation, end to end** `runtime/fg.cpp`, `runtime/fgnet.cpp`, `kernels/fg_*.comp` — Qualcomm's 8 SPIR-V passes unchanged + our LR/QR conv nets with Qualcomm's
   weights. LR/QR nets bit-exact to a CPU reference (all 14 layers each), deterministic, validation-clean. At zero motion output == input (57 dB); on a known-translation test
   scene it beats plain averaging. 5.7 ms/frame at 1080p, 10 ms at 1440p (naive kernels). VERIFIED (self-consistency) / ASSUMED (host wiring, see divergences D3, D4)
4. **Super Resolution, end to end** `runtime/sr.cpp`, `kernels/sr_{pre,post}.comp` — reconstructed from the readable parts of the vendor kernels (divergences D5, D6).
   Temporal accumulation works: +4.7 dB over bicubic after 16 frames at 640x360, exact alignment, unique jitter-sign optimum. 1.8 ms/frame at 640x360. VERIFIED (self-consistency) / ASSUMED (see D6)
5. **qemu trace** of `libanf.so` — chip gate and requirements recovered; deadlocks after `CreateTechnique` (no longer needed: the kernels were readable).

6. **OptiScaler + vkd3d-proton (SR)** — `patches/optiscaler/0001-anf-upscaler.patch` (patch only), `anf4r.dll` (MinGW), `tests/run_optiscaler.sh`; in m4r's D3D12 harness under Proton:
   26.76 dB vs reference (bilinear 21.89), 2.7 ms/frame at 640x360->1280x720. See docs/optiscaler.md. VERIFIED in the harness; not tried in a game.

## Not done
- **FG through OptiScaler**: C API stubs only (needs FULL-size depth/MV resampling and a present-time hook like m4r's patch 0002).
- **Per-frame descriptor sets**: `Pass` rewrites one descriptor set per `record()`; fine for submit-and-wait, must become per-frame ring for in-flight frames.
- **Performance**: convs are one-thread-per-output scalar loops; dot4/WMMA tiling would be several times faster.
- **Unknown host constants** (cannot be recovered without the host code or a device): SR jitter->phase mapping, SR `min/max`, QR head `min/max`, FG `discontinuityThresholdSq`, FG/SR depth conventions.
- Old `conv_int8.comp` / `model_test` (SR v0) are obsolete (see divergences D1/D5).
