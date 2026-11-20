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
GPU time per frame at **2560×1440 output** on an RX 7700 XT (RDNA3, RADV / Mesa 26.2), measured with timestamp queries around every pass (`build/cm/profile sr|fg W H [OUT_W OUT_H]`), mean of 20 frames,
output bit-identical to the scalar reference kernels. Mode ratios as in OptiScaler. The network always runs at its native 2× on the render-size input, so Quality, Balanced and Ultra Performance include
a bicubic resize to the output size (0.1–0.3 ms).

| | Quality (1.5×) | Balanced (1.7×) | Performance (2×) | Ultra Performance (3×) |
|---|---|---|---|---|
| Render size | 1706×960 | 1505×847 | 1280×720 | 853×480 |
| **Super resolution** | 2.41 ms | 1.89 ms | 1.19 ms | 0.63 ms |

**Frame generation** (one generated frame, independent of the upscaler mode): **1.61 ms** at 2560×1440.

Not included: OptiScaler's own work, the copy/blit into the game's images, and the optional output stabiliser (about +0.2 ms at 1280×720 input). Before the matrix-unit kernels the same
Performance case took 7.7 ms and FG 9.9 ms ([docs/STATUS.md](docs/STATUS.md)).

## Credits and license
GPL-3.0-only ([LICENSE](LICENSE)). The OptiScaler integration follows [m4r](https://github.com/countervolts/m4r)'s approach and derives its patches from it; OptiScaler is GPL-3.0.
Qualcomm, Adreno and Snapdragon are trademarks of Qualcomm Incorporated.
