# Status

Tags: **VERIFIED** (checked by a test or measurement here), **ASSUMED** (inference), **OPEN**. No Adreno device was available, so nothing is device-validated; validation is
CPU-reference equality, determinism, Vulkan validation cleanliness, and quality metrics on synthetic and captured data.

## Working
- **Runtime**: Vulkan context / adopt a caller's device, SPIR-V pass runner with a descriptor ring, function table for in-process use. VERIFIED
- **Frame generation**: Qualcomm's eight SPIR-V passes unchanged + our LR/QR convolution networks. Networks bit-exact to an independent CPU reference (all layers), deterministic,
  validation-clean (`tests/check_fg.sh`). At zero motion output == input. VERIFIED. Host wiring (resolutions, history, constants) reconstructed, ASSUMED where listed in `divergences.md`.
- **Super resolution**: reconstructed pre/post-processing around the networks; temporal accumulation and alignment verified on synthetic jittered sequences, jitter conventions chosen by measurement
  on captured game frames (`tests/check_sr.sh`, `tools/dump_experiments.py`). Any scale 1×–3×. VERIFIED (self-consistency) / ASSUMED (see D3–D6).
- **Matrix-unit kernels**: cooperative-matrix int8 convolutions, bit-identical to the scalar kernels (`ANF_NO_WMMA=1` selects the scalar path). VERIFIED.
- **OptiScaler (D3D12 via vkd3d-proton)**: SR and FG patches; works in the test harness and has been run in one game. Not broadly tested.

## Performance (RX 7700 XT, GPU timestamps, `build/cm/profile sr|fg W H`)
| | ms |
|---|---|
| SR, input 853×480 / 1280×720 / 1706×960 | 0.47 / 1.19 / 2.2 |
| FG, 1920×1080 / 2560×1440 | 1.10 / 1.62 |
Not included: the bicubic resize and copies of the OptiScaler path. Per-stage numbers: see git history of this file / run the profiler.

## Open
- Host constants that cannot be recovered without a device: SR jitter→weight-set mapping (chosen by measurement), SR/QR-head clamps, the FG discontinuity threshold, depth conventions.
- SR quality in motion on real game content is acceptable at best; an optional motion-aware output stabiliser exists but is off (D8).
- FG: layer fusion could save about 0.25 ms; the vendor shaders (about 0.5 ms at 1440p) cannot be changed.
- Only RDNA3 (gfx1101) was tested; the scalar fallback should run on any Vulkan 1.3 device with int8 storage but was not tested elsewhere.
