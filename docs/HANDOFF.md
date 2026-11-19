# anf4r — handoff prompt for the next agent

You are taking over **anf4r**: a private research project that runs Qualcomm's **Adreno Neural Fusion (ANF)**
super-resolution and frame-generation networks on an AMD RDNA3 GPU under Linux. Read this whole file, then read
`docs/findings.md`, `docs/divergences.md` and `docs/STATUS.md` before writing code. Everything below is verified
unless marked ASSUMED/BLOCKED.

## Mandate and rules
- The owner (Ayo) has granted **full executive control**: do not stop to ask clarifying questions; make the call,
  log it, keep building. The repo is **private** and nothing is published; vendor-derived data is committed on purpose.
- **Commits: no attribution.** Plain messages, no "Claude"/Co-Authored-By trailer. Author is the repo's git identity.
- **Honesty is the hard rule.** Never silently substitute an approximation for a vendor kernel. Every re-implemented
  or transformed kernel goes in `docs/divergences.md` with a preservation argument or an honest "may not match" note.
- Reverse engineering `libanf.so` is authorized (static, decompile, qemu trace, capability spoofing). Defeating the
  asset obfuscation was authorized and is already done.
- Goal, in priority order: a **working SR upscaler** on RDNA3 using Qualcomm's real weights, integrated into
  OptiScaler; then Frame Generation. "Working" is constrained by the ground-truth problem below — read it.

## Environment (all installed and verified on this machine)
- GPU: RX 7700 XT (RDNA3, gfx1101), RADV / Mesa 26.2, Vulkan 1.4. CPU Ryzen 9 5900XT, 32 GB. Arch Linux, linux-lts.
- Tools: `cmake`, `ninja`, `clang`, `glslc`, `spirv-val/dis`, Vulkan validation layers; `qemu-aarch64` (11.1);
  Android NDK at `/opt/android-ndk`; Ghidra at `/opt/ghidra` (use pyghidra: `build/venv/bin/python` has it);
  `lpunpack`/`simg2img`; `paru` for AUR. No Snapdragon device exists (this is the key constraint).
- Repo: `~/Desktop/anf` (remote `github.com/countervolts/anf4r`, private). Sibling projects to mirror for the
  OptiScaler/vkd3d path: `~/Desktop/d4r` (DLSS), `~/Desktop/m4r` (MetalFX — closest template), `~/Desktop/x4r` (XeSS).
  Reuse m4r's OptiScaler patch approach (pinned commit `fd3a45a`) and `runtime/vkhip` if you need native code objects.

## What ANF actually is (verified)
- `vendor/anf-sdk/` is the SDK (pinned commit in `vendor/anf-sdk/PINNED_COMMIT`). `libanf.so` SHA-256 in findings.md.
- The network is **QMFNet V2**, a tiny int8 CNN. SR: `cnn1`(20→32, stride2) → `cnn2-0` → `cnn2-1` (32→32) →
  `cnn3`(32→64, ConvBlend), ReLU, depth-to-space ×2. **Per-jitter weights for cnn1 and cnn3 (64 phases); cnn2-0/1
  shared.** FG: two branches (`aiff-frc-lr`/`aiff-frc-qr`), 14 convs each (32→96→…→64→20 / 32→…→20).
- Three GPU backends in the binary, selected per technique/chip: **(1) Adreno OpenCL** (inline QASM — NOT portable),
  **(2) ARM data-graph** (`vkCmdDispatchDataGraphARM` — not on RADV), **(3) Vulkan SPIR-V** (the 12 `AIFF*_cs`
  shaders — **portable, run on RADV unchanged**). SR's compute path is OpenCL; FG's glue is the 12 SPIR-V shaders.

## What is DONE and verified (don't redo)
1. **Decryptor** `tools/extract_assets.py`: ANF's "encryption" is `zlib(data) XOR keystream`, keystream = each file's
   own name-stem repeated. Extracts all 1278 assets exactly to `vendor/assets/`. (Loader at `libanf.so:0x1c3904`.)
2. **Runtime** `runtime/vk.cpp` + `runtime/pass.cpp`: headless Vulkan 1.3 context (+`adopt()` an existing device for
   OptiScaler) and a `Pass` that builds a compute pipeline from a SPIR-V module using a reflected interface
   (`tools/spv_reflect.py` → `build/reflect/*.json`). Vendor shader runs validation-clean (`tests/smoke_pass.cpp`).
3. **int8 conv** `kernels/conv_int8.comp`: bit-exact to a CPU reference on real weights, all SR layers
   (`tests/conv_test.cpp`). int32 accumulation (order-exact) + fp32 requant `(acc+bias)*scale[oc]`, ReLU,
   round-to-zero+saturate to int8. The `invpow2` SR model makes the scale multiply exact. See divergences.md D1.
4. **Full SR network** `runtime/model.cpp`: QMFNet runs end-to-end on the GPU, **bit-exact to CPU + deterministic**
   across runs and jitter phases (`tests/model_test.cpp`, `tests/check_determinism.sh`).
5. **Plan generator** `tools/prepare_sr.py`: builds per-resolution SR plans (per-jitter weights + manifest).
6. **C API** `runtime/anf4r.h` (mirrors m4r's), for the OptiScaler backend.
7. **qemu trace** `tools/qemu-trace/` (cl_rec.c, vk_rec.c, sysprop.c, harness.cpp, run.sh) + bionic at
   `vendor/bionic/` (linker64 + libs from an arm64 system image). `bash tools/qemu-trace/run.sh` runs `libanf.so`
   under qemu with recording shims. Verified: `IsTechniqueSupported(SR)`=SUCCESS, SR requirements captured, chip gate
   understood (report `CL_DEVICE_NAME`="hawi" via env `ANF_CHIP` to pass it).
8. Full decompile at `build/ghidra/all.c` (regenerate with `build/ghidra/decompile_pyghidra.py`; not committed).

## What is BLOCKED and why (read before planning)
- **No Adreno ground truth.** No device, and `libanf.so`'s SR convs/preprocess are Adreno machine code the qemu shims
  can only *record*, not *execute*. So you can recover the SR dispatch *recipe* but never a numeric reference output
  on this PC. Determinism + CPU-reference equality are the only validation available for reconstructed SR.
- **qemu trace deadlocks** right after `CreateTechnique` (worker-thread sync qemu-user timing doesn't satisfy; 0% CPU,
  not a crash). Getting the complete live graph would need the deadlock broken (real arm64 Linux/Android, or RE the
  sync) — but see the ground-truth point: it still wouldn't execute the Adreno kernels.
- **SR input packing (the last gap):** 20-channel int8 tensor, `MODEL_INPUT_STRIDE_BYTES`≈28 B/pixel (7 words),
  `-DDEPTH_FP32`/`-DUINT8_PACKING` variants; channels = current color(3)+depth(1)+MV(2)+warped-prev-color+recurrent
  prev-features. Exact per-channel byte order is still in Adreno asm (`Kernels_Hawi/preprocess_qmfnet_v2.cl`).

## Your next tasks (pick up here; all are "keep building", no need to ask)
1. **FG end-to-end (most faithful, fully buildable, do this first).** Wire the 12 vendor SPIR-V passes **unchanged**
   through the `Pass` runtime in the order in findings.md (UnpackMV → PreprocessFused → PyramidDownscale →
   MetaAndLRPack → [LR conv] → LRUnpack/LRUpscaleAndQRPack → [QR conv] → QRUnpack → QRBlend), with the FG convs
   (`aiff-frc-lr/qr`) on the existing conv kernel. Transcribe the tensor pack/unpack byte layout from the pack/unpack
   SPIR-V (readable — `spirv-dis`, no asm). Validate by determinism + per-kernel CPU reference. Add a `tools/prepare_fg.py`.
2. **SR reconstruction.** Build SR preprocess (pack 20ch from the recovered layout) + postprocess (D2S ×2 + blend) as
   new SPIR-V, driven by the model core, in `anf4r_api.cpp`. Label every reconstructed kernel in divergences.md. Make
   an offline PNG→PNG runner so output is inspectable. It will run deterministically; it is NOT device-validated.
3. **OptiScaler + vkd3d-proton integration.** Follow `m4r/docs/optiscaler.md` and `m4r/patches/optiscaler/`
   (pinned `fd3a45a`): add an `anf` backend (`Dx12Upscaler=anf`), use the external-memory requirement
   (`VK_KHR_external_memory_fd`, captured from the trace) for D3D12↔Vulkan interop, build the D3D12 test harness under
   Proton as m4r does, measure frame time at 1440p.
4. **Optional:** break the qemu deadlock (interpose pthread/futex, or qemu-system + real Android) only if you need the
   complete live dispatch graph; weigh it against the ground-truth ceiling.

## How to build and check
```bash
cmake -S . -B build/cm -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build/cm
glslc --target-env=vulkan1.3 -O kernels/conv_int8.comp -o build/conv_int8.spv
./build/cm/smoke_pass && ./build/cm/conv_test && ./build/cm/model_test && ./tests/check_determinism.sh
bash tools/qemu-trace/run.sh            # ANF_CHIP=hawi to pass the chip gate
VAL=1 ./build/cm/<test>                 # run under Vulkan validation
```
Commit small, push often, keep `docs/findings.md` / `docs/divergences.md` current, tag claims VERIFIED/SOURCED/ASSUMED.
