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
This repository does not provide the data and does not provide a way to obtain it; see [NOTICE.md](NOTICE.md).

## Benchmarks
GPU time per frame on an RX 7700 XT (RDNA3, RADV / Mesa 26.2), measured with timestamp queries around every pass (`build/cm/profile sr|fg W H [OUT_W OUT_H]`), mean of 20 frames, output bit-identical to the scalar reference kernels.
Mode ratios as in OptiScaler (Quality 1.5×, Balanced 1.7×, Performance 2×, Ultra Performance 3×). The network always runs at its native 2× on the render-size input, so Quality, Balanced and
Ultra Performance include a bicubic resize to the output size (0.1–0.3 ms at 1440p) and Quality/Balanced run the network on a larger input than Performance does.

**Super resolution** (ms; input → output):

| Output | Quality | Balanced | Performance | Ultra Performance |
|---|---|---|---|---|
| 1920×1080 | 1.33 (1280×720) | 1.03 (1129×635) | 0.63 (960×540) | 0.28 (640×360) |
| 2560×1440 | 2.41 (1706×960) | 1.89 (1505×847) | 1.19 (1280×720) | 0.63 (853×480) |
| 3840×2160 | 5.56 (2560×1440) | 4.35 (2258×1270) | 2.77 (1920×1080) | 1.50 (1280×720) |

**Frame generation** (ms per generated frame, independent of the upscaler mode):

| Frame size | 1920×1080 | 2560×1440 | 3840×2160 |
|---|---|---|---|
| FG | 1.10 | 1.61 | 3.60 |

Not included: OptiScaler's own work, the copy/blit into the game's images, and the optional output stabiliser (about +0.2 ms at 1280×720 input). Before the matrix-unit kernels the same
Performance-at-1440p case took 7.7 ms and FG at 1440p 9.9 ms ([docs/STATUS.md](docs/STATUS.md)).

## Credits and license
GPL-3.0-only ([LICENSE](LICENSE)). The OptiScaler integration follows [m4r](https://github.com/countervolts/m4r)'s approach and derives its patches from it; OptiScaler is GPL-3.0.
Qualcomm, Adreno and Snapdragon are trademarks of Qualcomm Incorporated.
