# anf4r — Adreno Neural Fusion super resolution and frame generation on AMD RDNA3

anf4r runs the networks of Qualcomm's **Adreno™ Neural Fusion (ANF)** — a 2× temporal super-resolution network and an
optical-flow frame-generation pair — on AMD RDNA3 GPUs through Vulkan, and plugs them into games via
[OptiScaler](https://github.com/optiscaler/OptiScaler) (D3D12 through vkd3d-proton / Proton).

It is an independent research project. **Not affiliated with or endorsed by Qualcomm.** It contains no Qualcomm code, weights or binaries:
you supply the ANF model data yourself, see [NOTICE.md](NOTICE.md) before using or contributing.

## Status (RX 7700 XT, RADV / Mesa 26.2; see [docs/STATUS.md](docs/STATUS.md))

| | |
|---|---|
| Frame generation | Runs end to end. Qualcomm's SPIR-V passes unchanged plus our LR/QR convolution networks. Networks are **bit-exact** to an independent CPU reference and deterministic. 1.1 ms at 1080p, 1.6 ms at 1440p. |
| Super resolution | Runs end to end (reconstructed pre/post-processing around the networks). Temporal accumulation verified on synthetic sequences; 1.2 ms at 1280×720 input. Any scale 1×–3× (network runs at its native 2×, bicubic resize). |
| Performance kernels | int8 convolutions on the RDNA3 matrix units (`VK_KHR_cooperative_matrix`), **bit-identical** to the scalar kernels (7.7→1.2 ms SR, 9.9→1.6 ms FG). Scalar fallback included. |
| OptiScaler | `patches/optiscaler/` adds `Dx12Upscaler=anf` and `FGOutput=anf`; works in the D3D12 test harness under Proton and has been tried in one game. |
| Not device-validated | No Adreno device was available. Reconstructed host-side constants are listed in [docs/divergences.md](docs/divergences.md); quality in real games is not on par with a verified port and some settings were tuned by eye. |

## How it works
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): network topology, tensor layouts, the quantisation domain, each pipeline stage, and
which parts are Qualcomm's unchanged shaders versus our reconstructions.

## Build
Requirements: a C++17 compiler, CMake ≥ 3.20, Ninja, Vulkan headers/loader, `glslc` (shaderc), Python 3 with numpy and Pillow.
Windows DLL for OptiScaler: MinGW-w64; OptiScaler itself: clang-cl + xwin (see [docs/optiscaler.md](docs/optiscaler.md)).

```bash
cmake -S . -B build/cm -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build/cm        # kernels -> build/*.spv, tests, tools
scripts/setup_vulkan_headers.sh && cmake -S . -B build/win -G Ninja -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/mingw-w64.cmake \
    -DCMAKE_BUILD_TYPE=Release && ninja -C build/win                                    # build/win/anf4r.dll
```

## Model data
anf4r needs the ANF model data from **your own copy** of the ANF SDK (`libanf.so`): weights, quantisation scales and the frame-generation SPIR-V passes,
laid out as `modelzoo/`, `spirv/` (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#data-layout)). `scripts/install_anf.py` assembles the folder next to a game.
This repository does not provide the data and does not provide a way to obtain it; see [NOTICE.md](NOTICE.md).

## Test
```bash
tests/check_fg.sh          # FG nets == CPU reference, pipeline deterministic + validation-clean   (needs the model data)
tests/check_sr.sh          # SR deterministic + validation-clean on a synthetic jittered sequence  (needs the model data)
build/cm/profile sr 1280 720 ; build/cm/profile fg 2560 1440     # per-pass GPU timing
tests/run_optiscaler.sh ...                                      # OptiScaler + D3D12 harness under Proton (needs m4r's harness)
```
Without a real image the tests use a procedural one (`tools/testimage.py`; `HERO_IMAGE=<file>` to use your own).

## Layout
`runtime/` Vulkan context, pass runner, SR/FG pipelines, C API (`anf4r.h`) · `kernels/` our GLSL (matrix-unit convs, SR pre/post, heads, resize, stabiliser) ·
`patches/optiscaler/` OptiScaler patches (the OptiScaler tree itself is not included) · `tools/` profiler, offline runner, capture analysis · `tests/` · `docs/`.

## Credits and license
GPL-3.0-only ([LICENSE](LICENSE)). The OptiScaler integration follows [m4r](https://github.com/countervolts/m4r)'s approach and derives its patches from it; OptiScaler is GPL-3.0.
Qualcomm, Adreno and Snapdragon are trademarks of Qualcomm Incorporated.
