# ANF in OptiScaler

The repo carries **only a patch** (`patches/optiscaler/0001-anf-upscaler.patch`), not OptiScaler. It is made against OptiScaler commit `fd3a45a`
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
| Frame generation | **not wired** (`anf4r_interpolate*` / `framegen_*` return an error). `FrameGen` exists and works standalone; it needs FULL-size depth/MV (ANF requirement) while OptiScaler supplies render-size ones, so an up-sampling step is needed. |
| Device features | the game's device must have `shaderInt8` + `storageBuffer8BitAccess` + `bufferDeviceAddress`; vkd3d-proton under RADV provided them. |
Known limits: one SR context per size pair, display-resolution MVs unsupported (as in m4r), vkd3d-proton only, depth assumed reverse-Z unless `DepthInverted()` says otherwise.
