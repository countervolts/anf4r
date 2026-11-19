# anf4r — Qualcomm Adreno Neural Fusion on Radeon (RDNA3)

Runs Qualcomm's **Adreno™ Neural Fusion** (ANF) networks on AMD RDNA3 under Linux. Qualcomm's real quantized
weights and their Vulkan SPIR-V data-path shaders are used **unchanged**; the convolutions — which Qualcomm ships
only as Adreno machine code — are re-implemented as int8 SPIR-V with identical integer arithmetic.

Sibling projects: [d4r](https://github.com/countervolts/d4r) (DLSS), [m4r](https://github.com/countervolts/m4r)
(MetalFX), [x4r](https://github.com/countervolts/x4r) (XeSS). **Private, research only. Not affiliated with Qualcomm.**

## What works today (verified on an RX 7700 XT, RADV / Mesa 26.2)

| | |
|---|---|
| Asset extraction | All 1278 embedded ANF assets (SPIR-V, OpenCL, weights, pipeline caches) extracted from `libanf.so`. The "encryption" is XOR-with-filename + zlib; `tools/extract_assets.py` reproduces it exactly. |
| Vendor SPIR-V on RADV | Qualcomm's own `AIFF*_cs` compute shaders build and dispatch on RDNA3, validation-clean (`tests/smoke_pass.cpp`). |
| int8 convolution | Re-implemented conv (`kernels/conv_int8.comp`) is **bit-exact** to an independent CPU reference on Qualcomm's real weights, all SR layers (`tests/conv_test.cpp`). |
| Full SR network | QMFNet V2 (cnn1→cnn2-0→cnn2-1→cnn3) runs end-to-end on the GPU, **bit-exact to CPU and deterministic** across runs and jitter phases (`tests/model_test.cpp`). |

See [docs/STATUS.md](docs/STATUS.md) for exactly what is verified, reconstructed, and blocked, and
[docs/findings.md](docs/findings.md) / [docs/divergences.md](docs/divergences.md) for the detail.

## The honest caveat

ANF's convolutions and SR input/output packing exist in the SDK only as **Adreno machine code** (QASM inline asm,
precompiled Adreno OpenCL, or ARM data-graph). There is **no Snapdragon device** here for ground truth, so:
- The conv network is validated by *determinism* and *bit-exact agreement with an independent CPU reference* — not
  against a real Adreno run.
- SR's 20-channel input packing and output depth-to-space/blend are Adreno-asm only; a faithful reconstruction
  cannot be validated without a device. The network core is proven; the full in-game SR pipeline is **not** yet a
  drop-in replacement. Frame Generation, whose glue is portable vendor SPIR-V, is the more faithful target.

## Build & test

```bash
cmake -S . -B build/cm -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build/cm
glslc --target-env=vulkan1.3 -O kernels/conv_int8.comp -o build/conv_int8.spv
python3 tools/extract_assets.py vendor/anf-sdk/lib/arm64-v8a/libanf.so vendor/assets   # one-time
./build/cm/smoke_pass        # vendor SPIR-V runs on RADV
./build/cm/conv_test         # conv bit-exact vs CPU
./build/cm/model_test        # full SR net bit-exact + deterministic
python3 tools/prepare_sr.py --out build/plans/sr-2560x1440-1280x720 --in 1280x720
```

## Layout
- `runtime/` — Vulkan context, SPIR-V pass runner, QMFNet model runner, C API (`anf4r.h`).
- `kernels/` — our SPIR-V (conv; SR/FG glue uses Qualcomm's unchanged).
- `tools/` — `extract_assets.py` (decryptor), `spv_reflect.py`, `prepare_sr.py`.
- `tests/` — smoke, conv, model.
- `vendor/` — Qualcomm SDK + extracted assets (private; not for redistribution).

## License
GPL-3.0-only for anf4r's own code. Qualcomm's `libanf.so`, the extracted assets and anything derived from them are
**not** covered and must not be redistributed.
