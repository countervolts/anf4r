# ANF in OptiScaler

The repo carries **only a patch** (`patches/optiscaler/0001-anf-upscaler.patch` and `0002-anf-frame-generation.patch`), not OptiScaler. It is made against OptiScaler commit `fd3a45a`
("Display asi's version") and adds `Upscaler::ANF` (`Dx12Upscaler=anf`), an NGX D3D12 feature built on vkd3d-proton's Vulkan interop, a `[ANF] FlipJitter`
ini key, the dropdown entry and a loader for `anf4r\\anf4r.dll`. It is derived mechanically from m4r's MetalFX patch by `scripts/make_optiscaler_patch.py`
(rename MetalFX/m4r -> ANF/anf4r, data dir `anf4r\\data`, swap in `runtime/anf4r.h`), so it tracks m4r's integration approach.
`optiscaler/clang-cl.patch` (from m4r) is only needed to build OptiScaler with clang-cl on Linux.

## Build and test (Linux host, Proton)
```bash
git clone https://github.com/optiscaler/OptiScaler external/optiscaler && git -C external/optiscaler checkout fd3a45a   # + its submodules
git -C external/optiscaler apply $PWD/patches/optiscaler/0001-anf-upscaler.patch $PWD/optiscaler/clang-cl.patch
cmake -S optiscaler -B build/opti -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/clang-cl-xwin.cmake && ninja -C build/opti
cmake -S . -B build/win -G Ninja -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/mingw-w64.cmake -DCMAKE_BUILD_TYPE=Release && ninja -C build/win   # anf4r.dll
python3 scripts/install_anf.py --out /path/to/game/dir          # anf4r\\anf4r.dll + data (shaders, vendor SPIR-V, weights) from your own libanf.so
tests/run_optiscaler.sh $PROTON_DIR build/opti/OptiScaler.dll <fixture dir> 8 640 360 1280 720     # uses m4r's D3D12 harness + fixtures
```
(Case-insensitive includes need the symlinks from `optiscaler/casefix.py` on Linux.)

## What happens
OptiScaler gives the feature the game's D3D12 resources; they become `VkImage`s via `ID3D12DXVKInteropDevice1`, and `anf4r_upscale` records the SR passes
into the game's own command list. `anf4r.dll` takes every Vulkan entry point from the game's device (`runtime/vkfn.h`), keeps its own history and
intermediates, and blits its RGBA16F result into the game's output image. Descriptor sets are a ring of 8 per pass; each frame starts with a barrier against the previous one.
Jitter: the NGX/image-coordinate sample offset (y down) is converted to ANF's y-up `jitterOffset` inside `anf4r_upscale`. MV: `previous - current` pixels are negated to ANF's convention.

## Status (2026-11-19)
| | |
|---|---|
| SR through OptiScaler under Proton (GE-Proton10-34, RX 7700 XT) | **works** in m4r's harness: 640x360 -> 1280x720, 8 frames, **26.76 dB** vs the supersampled reference (bilinear 21.89 dB; with `FlipJitter=true` 21.23 dB, so the conversion is right), 2.7 ms per frame including submit and wait. Not tried in a game. |
| Frame generation (`FGOutput=anf`, `FGInput=upscaler`) | **works in the harness** (`ANF_FG=1 tests/run_optiscaler.sh`, 20 frames, swapchain): from the first active frame every present becomes two, the real frame is put back bit-exactly, generated frames are 29.5 dB vs the true in-between frame (repeat-current 27.8 dB, plain average 30.3 dB; hero fixture `tools/make_hero_fixture.py`, 2 px/frame pan). `anf4r_framegen_capture` copies the backbuffer, up-samples OptiScaler's render-size depth/MV to FULL size (`kernels/fg_resample.comp`), runs `FrameGen`, blits the result into the backbuffer; `restore` writes the real frame back. The scene-cut guard of the vendor blend (luma difference) makes fast motion pass through unmodified. |
| Device features | the game's device must have `shaderInt8` + `storageBuffer8BitAccess` + `bufferDeviceAddress`; vkd3d-proton under RADV provided them. |
Known limits: one SR context per size pair, display-resolution MVs unsupported (as in m4r), vkd3d-proton only, depth assumed reverse-Z unless `DepthInverted()` says otherwise.

## Scaling modes
The ANF network is fixed 2x; every other ratio (DLAA 1x, Quality 1.5x, Balanced 1.7x, Ultra Performance 3x, any custom ratio) runs the network at 2x on the render-size input and
resamples the result to the requested size with a Catmull-Rom bicubic pass (`kernels/sr_resize.comp`). Harness, 640x360 input, 20 frames (`tests/check_ratios.sh`), PSNR vs the
reference at the output size / bilinear of the same input: 1x 35.6/30.8 dB, 1.5x 33.7/29.5, 1.7x 33.4/29.3, 2x 32.7/29.0, 3x 33.0/29.2.

## Ghost of Tsushima (installed 2026-11-19, not yet played)
`dxgi.dll` is this repo's patched OptiScaler (ANF backend); `anf4r/` (dll + data, 18 MB) sits next to the game executable; `OptiScaler.ini` has
`Dx12Upscaler = anf`, `FGOutput = anf`, `[ANF] FlipJitter = true` (GoT's raw NGX jitter sign is inverted, as measured with m4r). The previous `dxgi.dll` and ini are in
`anf4r-backup-20261119/` (`restore.sh` puts them back and removes `anf4r/`); the m4r folder is untouched. Check `OptiScaler.log` for
`InitInternal ANF: ready, WxH -> WxH` (SR; any quality mode works: the network runs at its native 2x and is resampled to the output size) and `Activate ANF FG` (FG). Untested in the real game:
display-resolution motion vectors (OptiScaler warns), the game's depth/UI handling, and whether GoT's vkd3d-proton device exposes the int8 features anf4r needs (the harness's does).
