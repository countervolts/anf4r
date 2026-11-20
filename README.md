# anf4r — Adreno Neural Fusion super resolution and frame generation on AMD RDNA3

anf4r runs the networks of Qualcomm's **Adreno™ Neural Fusion (ANF)** — a 2× temporal super-resolution network and an
optical-flow frame-generation pair — on AMD RDNA3 GPUs through Vulkan, and plugs them into games via
[OptiScaler](https://github.com/optiscaler/OptiScaler) (D3D12 through vkd3d-proton / Proton).

## How it works
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): network topology, tensor layouts, the quantisation domain, each pipeline stage, and
which parts are Qualcomm's unchanged shaders versus our reconstructions.

## Model data
anf4r needs the ANF model data from **your own copy** of the ANF SDK (`libanf.so`): weights, quantisation scales and the frame-generation SPIR-V passes,
laid out as `modelzoo/`, `spirv/` (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#data-layout)). `scripts/install_anf.py` assembles the folder next to a game.

## Benchmarks
GPU time per frame at **2560×1440 output** on an RX 7700 XT (RDNA3, RADV / Mesa 26.2), measured with timestamp queries around every pass (`build/cm/profile sr|fg W H [OUT_W OUT_H]`), mean of 20 frames,
output bit-identical to the scalar reference kernels. Mode ratios as in OptiScaler. The network always runs at its native 2× on the render-size input, so Quality, Balanced and Ultra Performance include
a bicubic resize to the output size (0.1–0.3 ms).

| | Quality (1.5×) | Balanced (1.7×) | Performance (2×) | Ultra Performance (3×) |
|---|---|---|---|---|
| Render size | 1706×960 | 1505×847 | 1280×720 | 853×480 |
| **Super resolution** | 2.41 ms | 1.89 ms | 1.19 ms | 0.63 ms |

**Frame generation** (one generated frame, independent of the upscaler mode): **1.61 ms** at 2560×1440.

## GPU support
The fast path needs `VK_KHR_cooperative_matrix` with int8×int8→int32 16×16×16 subgroup matrices and a pinnable subgroup size of 32. anf4r checks this on the device at start-up (nothing is keyed to a GPU name) and logs what it chose to `anf4r.log` next to the data folder.

| GPU | Path | Status |
|---|---|---|
| RDNA3 (RX 7000: gfx1100 / 1101 / 1102, RDNA3.5 iGPUs) | matrix units | **Tested on an RX 7700 XT (gfx1101) only.** The other RDNA3 parts expose the same shapes and should take the same path. |
| RDNA4 (RX 9000: gfx1200 / 1201) | matrix units | **Untested** (no hardware here). The same cooperative-matrix shapes are exposed by recent Mesa/RADV. Since the best kernel layout may differ from RDNA3, anf4r times three bit-identical kernel variants per convolution shape on the actual GPU the first time it runs and keeps the fastest (cached in `shaders/tune.txt`, delete it to re-tune). If you have one, please send the `anf4r.log` and `tune.txt`. |
| RDNA2 and older, other vendors | scalar fallback | Slower (roughly 5–8× on the convolutions). Verified bit-exact against the CPU reference on a CPU Vulkan driver (llvmpipe); SR output there matches RADV to about 75 dB PSNR, not bit for bit, because texture filtering is hardware-defined. |

`ANF_NO_WMMA=1` forces the scalar path, `ANF_NO_TUNE=1` skips tuning and `ANF_WMMA_VARIANT=<0-2>` forces a variant (all produce identical results); `ANF_DEVICE=<name substring>` selects the device for the tests and tools.

## Credits and license
GPL-3.0-only ([LICENSE](LICENSE)). The OptiScaler integration follows [m4r](https://github.com/countervolts/m4r)'s approach and derives its patches from it; OptiScaler is GPL-3.0.
Qualcomm, Adreno and Snapdragon are trademarks of Qualcomm Incorporated.
