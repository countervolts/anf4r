# anf4r findings

Tags: **VERIFIED** = seen directly in this repo's work; **SOURCED** = from a cited document; **ASSUMED** = inference, not yet checked.
Vendor-derived data (binary, extracted tables, strings, traces) lives only in `vendor/` (gitignored). This file
describes structure and names; it contains no Qualcomm code or weights.

## Phase 0: environment (2026-11-19)

### Host
- GPU0 AMD Radeon RX 7700 XT (RADV NAVI32), Mesa 26.2.7-arch1.1, Vulkan 1.4.354. GPU1 llvmpipe (LLVM 23.1.4). VERIFIED (`vulkaninfo`)
- Kernel 6.18.63-2-lts. VERIFIED
- OpenCL: ICD loader installed, only vendor `rusticl.icd`; `clinfo` not installed, so no device has been tested yet. VERIFIED
- Missing tools: radare2, Ghidra, qemu-aarch64, hipcc. VERIFIED

### RADV properties relevant to determinism (GPU0)
- `VK_KHR_shader_float_controls` rev 4, `VK_KHR_shader_float_controls2`. VERIFIED
  - denorm and rounding-mode independence: `32_BIT_ONLY` (fp32 can be set separately; fp16 and fp64 share one setting). VERIFIED
  - SignedZeroInfNanPreserve fp16/32/64: true. DenormPreserve and DenormFlushToZero fp16/32/64: true. RTE and RTZ fp16/32/64: true. VERIFIED
- Subgroups: default 64; `VK_EXT_subgroup_size_control` with min 32 and max 64, required size settable in 4 stages. VERIFIED
- `VK_KHR_cooperative_matrix` rev 2, `cooperativeMatrix = true`. Shapes and types not listed by `vulkaninfo`; still to be queried with `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`. VERIFIED (presence only)
- int8 support: `shaderInt8`, `storageBuffer8BitAccess`, `VK_KHR_shader_integer_dot_product` (8-bit signed and 4x8 packed signed: accelerated). VERIFIED
- External memory: `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`. VERIFIED
- Not present: `VK_ARM_tensors`, `VK_ARM_data_graph`, all `VK_QCOM_*`. VERIFIED

### Sibling projects (local checkouts on this machine)
- m4r at `0eef8f5`, d4r at `793a100`, x4r at `8256b5f`. Read so far: m4r README, `docs/optiscaler.md`, `docs/air-translation.md`, `runtime/plan.h`, `runtime/m4r.h`, `runtime/vkhip/vkhip.h`, `scripts/apply_optiscaler_patches.sh`. VERIFIED
- From m4r we reuse the C API shape (caller's VkDevice, recording into the caller's command buffer), plan directories per resolution pair, the OptiScaler patch series at `fd3a45a`, the D3D12 harness under Proton, and the determinism and replay tests. vkhip (HIP code objects in Vulkan) is probably **not** needed: ANF's Vulkan kernels are SPIR-V. ASSUMED

## ANF SDK
- Repo pinned at `2f29921da1453acd2fa428cfa05db15e9ef39a85` (2026-10-20, "docs: add Chinese translations and mip bias guidance"). VERIFIED
- `libanf.so` SHA-256 `284e8a1e0ec0a572eb56b1d06c778def9db0cd09e4ea20564caaa9612089384c`, 9,016,640 bytes. VERIFIED
- Headers read in full: `anf.h`, `anf_types.h`, `anf_types_vk.h`, `anf_sr.h`, `anf_fg.h`; `INTEGRATION-GUIDE.md` and README read in full. VERIFIED

### API facts not in earlier research (all VERIFIED from headers or guide)
- Function table `AnfFunctions` has 10 entries: QueryAnfVersion, CreateInstance, DestroyInstance, SetInstanceClientAPIAdapterInfo,
  IsTechniqueSupported, QueryTechniqueRequirements, QueryTechniqueResourceRequirements, CreateTechnique, DestroyTechnique, DispatchTechnique.
- `AnfClientResource` is documented as "VkImage, VkBuffer, **VkTensorARM**".
- `AnfFormat`: R8G8B8A8_UNORM, B10G11R11_UFLOAT, D32_FLOAT, D24S8_UNORM, R16G16_FLOAT, R16G16B16A16_FLOAT.
- `AnfModelWeightsInfo { pWeightsPath }` in the CreateTechnique pNext chain: weights can be overridden from a file ("fine tuned weights").
- **FG supports only `ANF_TECHNIQUE_CREATE_FLAG_DISPATCH_IMMEDIATE`** (ANF submits to its own queues using semaphores). SR supports immediate and recorded mode.
  The binary's error text says recorded ("indirect") dispatch is unsupported "on this driver for this technique", so recorded SR may also depend on the device. VERIFIED (string), ASSUMED (meaning)
- Recorded mode may use `VK_QCOM_tile_memory_heap`. An OpenCL loading error is a documented failure mode, so ANF uses OpenCL on device.
- Debug overlay (8 modes) is part of SR.

## Phase 1: static analysis of libanf.so

### ELF (VERIFIED, readelf)
- AArch64 shared object for Android API 26, NDK r29 (clang 21, +pgo +bolt +lto), stripped, BIND_NOW, 16 KiB segment alignment.
- NEEDED: liblog, libz, libandroid, libm, libdl, libc. **libvulkan and libOpenCL are not linked; both are dlopen'ed** (strings `libvulkan.so`, `libOpenCL.so`, `libOpenCL_system.so`).
- One export: `GetAnfFunctions`.
- Notable imports: `uncompress`, `inflateEnd` (zlib); `AHardwareBuffer_release`; `__system_property_get/find/read`; `dlopen/dlsym`; `syscall`; `getauxval`. No crypto library imported (any cipher is in-binary).
- Sections: `.text` 1.25 MiB, `.rodata` 187 KiB, **`.data` 6.76 MiB** (file offset 0x1d6408..0x898b30).

### Embedded asset table (VERIFIED, `tools/elfmap.py` and inline scripts; table dump in `vendor/filetable.tsv`)
- `.data.rel.ro` 0x1d2100..0x1d98d0: 1278 entries of 24 bytes: `{char* name; u32 offset; u32 stored_size; u64 size}`.
- Offsets are relative to the start of `.data`; entries are contiguous with no gaps and cover all of `.data` except its last 0x11a bytes.
- Totals: 7,087,630 bytes stored, 13,587,393 bytes unpacked.
- Groups: `modelzoo/` 871, `graphPipeline/` 270, `kernel_bins/` 87, `Kernels_Hawi/` 38, `spirv/` 12.

| group | contents (from names) |
|---|---|
| `spirv/` | 12 Vulkan compute shaders `AIFF*_cs.bin` (DebugOverlay, LRUnpackV2(CL), LRUpscaleAndQRPackV2, MetaAndLRPackV2, PrepareInputs, PreprocessFusedV2, PyramidDownscaleFusedV2, QRBlendV2(FINT), QRUnpackV2, UnpackMVV2) |
| `Kernels_Hawi/` | 38 OpenCL C sources: int8 Conv2D variants (Relu, Jitter, Persistent, Blend, residual, D2S), `Conv_MxM.cl`, pre/postprocess for "qmfnet_v2" |
| `kernel_bins/{hawi,kaanapali,maili}/` | precompiled OpenCL binaries per SoC (Adreno ISA, not usable on AMD) for SR prod-5, prod-7p1, prod-7p1-invpow2 and FG prod-8 |
| `modelzoo/` | weights: SR `aiff-sr-v2s-prod-7p1` (int8 weights, int32 biases, **one set per jitter index**, e.g. `cnn1/<n>/cnn1.weight_shape-32-20-3-3_int8.raw`); FG `aiff-frc-lr-v2-prod-8` and `aiff-frc-qr-v2-prod-8` (conv0_0..conv5_1, conv_out, conv_out_flow_mask) |
| `graphPipeline/` | SR `aiff-sr-v2s-prod-5`: Vulkan pipeline caches per input size and SoC (`pipeline_cache_1600x768_hawi.bin`, ... sizes 1024x448, 1088x512, 1344x640, 1600x768 and generic) and per-jitter weights `wts/*_jidx<n>.bin`; FG `framegen_v2p0` 1080p landscape caches and `network_io_encodings.json` |

### Assets are obfuscated or encrypted: stop condition (VERIFIED evidence, ASSUMED interpretation)
- Shannon entropy of 8.00 bits/byte over the whole `.data` payload.
- No SPIR-V magic anywhere in the file; no valid zlib or raw-deflate stream at any entry offset.
- 32-byte bias files are stored as 40 bytes: an 8-byte header or prefix, and no compression.
- `conv_bias1_jidx0` and `conv_bias1_jidx1` share the same first 8 bytes and differ only in a few bits after them.
  All 12 `spirv/` entries start with `39 d5`. This pattern fits a position-dependent XOR keystream reused for every
  entry, applied after an 8-byte header and zlib compression (`uncompress` is imported). ASSUMED
- No standard cipher constants (AES S-box or T-tables, ChaCha sigma, SHA-256, MD5) are present in the file.
- Per the task brief, Phase 1 stops here. I have not attempted any decryption.

### Device and SoC gating (VERIFIED strings; logic not yet read)
- GPU names `Adreno (TM) 840/845/850`; SoC codenames pakala, kaanapali, glymur, hawi, trenton, maili.
- System properties `aiff.debug.device_id_override`, `aiff.upscale_override_weights_path`, `aiff.fg_override_weights_path`,
  `aiff.upscale_debug_dump_tensors`, `debug.aiff.plugins.enable`; dump path `/data/local/tmp/`.
- "persistent_waves" kernels need the device to be known in order to choose the wave count.
- OpenCL must match the Vulkan device UUID ("Could not find an OpenCL platform + device that ... matches the Vulkan device UUID").

### Compute backends present (VERIFIED from imported names and strings)
1. **Vulkan ARM data graph**: `vkCreateDataGraphPipelinesARM`, `vkCreateDataGraphPipelineSessionARM`, `vkCmdDispatchDataGraphARM`,
   `vkCreateTensorARM`, `vkBindTensorMemoryARM`, `vkCmdCopyTensorARM`, `vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM`,
   with the `graphPipeline/` caches. Neither extension exists on RADV.
2. **OpenCL** (Qualcomm extensions): recordable queues (`clNewRecordingQCOM`, `clEnqueueRecordingQCOM`), `clEnqueueNDRangeKernelWithPropertiesQCOM`,
   `clImportMemoryARM`, `cl_khr_external_memory` and `cl_khr_external_semaphore` interop with Vulkan, build option `-qcom-accelerate-16-bit`,
   compile-time tiling defines (`TILESIZE_M/N/K`, `FILTER_*`, `IC`, `OC`, `DILATION`, `STRIDE`). Programs are built from source **or** from binary.
3. **Vulkan compute**: the 12 `spirv/` shaders; the Vulkan imports are the standard compute set plus `vkCmdImportCommandsQCOM`.
- Which backend runs for which technique, model and SoC is not known yet. ASSUMED: SR prod-7p1 and FG prod-8 use OpenCL; SR prod-5 uses the data graph.

### Early divergence notes (ASSUMED until the kernels are read)
- The int8 convolutions accumulate in int32, which is exact in any order. The requantization step (float scale or power-of-two shift; there is an `invpow2` model variant) decides whether results can be bit-identical.
- On a real device, ANF may run the precompiled `kernel_bins` or data-graph caches rather than the `.cl` sources. Then "same kernels" means the same source, not the same binary.

## Phase 1 (cont.) + Phase 2: assets recovered (2026-11-19)

### Obfuscation fully reversed — NOT cryptographic (VERIFIED)
Decompiled the loader at `libanf.so:0x1c3904` (the function that calls `uncompress@plt` at `0x1c3b18`):
- **key** = the asset's filename stem: `basename(name)` with the last `.ext` removed (e.g. `conv_bias1_jidx0`, `AIFFDebugOverlay`).
- **stage 1** = repeating XOR: `stored[i] ^ key[i % len(key)]` for `i` in `[0, stored_size)` (loop at `0x1c3a68`, `eor w10,w11,w10`; key index wraps via `csinc`).
- **stage 2** = `zlib.decompress(stage1)`, expected length = the entry's `size` field.
- Blob base = start of `.data` (GOT slot `0x1d9ca8` relocates to `.data` va `0x1de408`); table offsets are `.data`-relative.
- No device binding, no checksum, no standard cipher. The only "protection" is XOR-with-filename + zlib.

`tools/extract_assets.py` reproduces this and **extracts all 1278 assets, each to its exact `size`, 0 failures** (`vendor/assets/`, 17 MB). This replaces the planned qemu dynamic capture for getting the kernels and weights; dynamic tracing is now only needed for the per-frame **dispatch graph** (host logic), not for the assets.

### What the assets are (VERIFIED by inspection)
- **`spirv/` — 12 Vulkan compute shaders, all `spirv-val` clean (vulkan1.3).** Capabilities are only `Shader`, `StorageImageExtendedFormats`, `ImageQuery`. **No QCOM extensions, no cooperative matrix, no tensor/data-graph ops. These run on RADV unchanged.** They are the glue passes: PrepareInputs, PreprocessFusedV2, PyramidDownscaleFusedV2, UnpackMVV2, Meta/LR pack+unpack, QR pack/unpack, QRBlendV2(/FINT), LRUpscaleAndQRPack, DebugOverlay. LocalSize 16×16 (overlay 32×32); 4–12 descriptor bindings each.
- **`Kernels_Hawi/` — 38 OpenCL C conv kernels; 37 of 38 use Qualcomm inline assembly** (`#pragma OPENCL EXTENSION cl_qcom_inline_asm`, ~41 `__asm` blocks each: hand-written Adreno QASM MxM inner loops). Only `Conv_MxM.cl`-style scaffolding is portable; the compute is Adreno ISA. **These do NOT run on RADV or rusticl.**
- **`kernel_bins/{hawi,kaanapali,maili}/` — precompiled Adreno OpenCL binaries** (opaque `data`). Adreno-only.
- **`graphPipeline/` — ARM data-graph (TOSA/ML) pipeline caches**, e.g. header names `QMFNet_v2_stride...`. Compiled for Adreno's ML path. Adreno-only.
- **`modelzoo/` — the weights, portable and standard:** int8 conv weights `[OC,IC,KH,KW]`, int32 biases, per-layer `scales.json` (`weights_scales[]`). The net is **QMFNet_v2**.

### Network architecture (VERIFIED from shapes)
- **SR (`aiff-sr-v2s-prod-7p1`)**: `cnn1 [32,20,3,3]` → `cnn2-0 [32,32,3,3]` → `cnn2-1 [32,32,3,3]` → `cnn3 [64,32,3,3]`, ReLU, then depth-to-space ×2 (D2S) + blend. **64 jitter-phase weight sets** per layer (indices 0..63). An `invpow2` variant exists (inverse-power-of-two requant). `prod-5` is the older data-graph model.
- **FG (`aiff-frc-{lr,qr}-v2-prod-8`)**: conv0_0..conv5_1 (12 convs) + conv_out + conv_out_flow_mask; LR (low-res flow) and QR branches; `network_io_encodings.json` gives input/flow/mask quant scale+offset.
- All convolutions are int8×int8→int32 (exact under any summation order). Only **requantization** (int32→int8 via float `scale`, or inverse-pow2 shift) is rounding-sensitive; `scales.json` and the kernel's requant code define it. This is the one place to pin for bit-exactness.

### Consequence for "bit-identical" (the honest statement)
- The **portable** parts (12 SPIR-V passes, all weights, quant scales, architecture) are exactly Qualcomm's and run on RADV unchanged.
- The **convolution kernels** (the heavy compute) exist only as Adreno QASM / Adreno OpenCL binaries / Adreno ML caches. **No form runs on RDNA3.** Running "Qualcomm's own conv kernels unchanged" is impossible on this GPU.
- To run the network on RDNA3 we must re-express conv2d+ReLU+requant+D2S ourselves (as WMMA/dot-product or plain int8 compute), matching Qualcomm's integer arithmetic and requant rounding. This is a kernel re-implementation with identical math, not a retrain or approximation — but it is NOT the vendor's own kernel binary, and every such kernel must be logged in `docs/divergences.md`. This is the Phase 3 decision for Ayo.

## Phase 3: backend reality and the plan (2026-11-19)

Reflected all 12 SPIR-V passes (`tools/spv_reflect.py` → `build/reflect/*.json`) and read the OpenCL SR kernels. Decisive finding on which ANF path can run on RADV:

- **The 12 `AIFF*_cs` SPIR-V shaders are the Frame-Generation (FRC) Vulkan data path.** Binding/spec names show a two-branch LR→QR flow+mask network with prev/curr-frame blending: `MetaAndLRPackV2` (pack LR net input) → [LR conv net] → `LRUnpackV2`/`LRUpscaleAndQRPackV2` (unpack flow+mask, upscale, pack QR net input) → [QR conv net] → `QRUnpackV2` → `QRBlendV2(FINT)` (blend `texPriorFrame`/`texCurrFrame` by flow+mask). `PreprocessFusedV2`/`PyramidDownscaleFusedV2`/`PrepareInputs`/`UnpackMVV2` build the depth pyramid, dilated/projected MV and depth stats. These are **plain Vulkan compute, run on RADV unchanged.** Resolution is supplied via specialization constants (`AIFF_*_WIDTH/HEIGHT`, quant scales/offsets); per-dispatch params via a push-constant struct.
- **SR (`aiff-sr-v2s`) has no portable Vulkan glue.** Its Vulkan form is the ARM data-graph (`graphPipeline/upscale/...`, `vkCmdDispatchDataGraphARM` — not on RADV); its OpenCL form (`Kernels_Hawi/*.cl`) is Adreno machine code wrapped in inline asm (`ldib.untyped...`, `getone`, LM layout) — not reconstructable with confidence. Only the SR **weights** (int8 QMFNet, per-jitter) and quant scales are portable.

### Decision (executive control, Ayo away)
1. **Shared runtime + int8 conv engine + CPU reference first** (needed by both techniques; certain value; self-validatable).
2. **Frame Generation = primary faithful target**: run the 12 vendor SPIR-V passes **unchanged** + re-implemented FG convs (aiff-frc-lr/qr int8). Most faithful to "run Qualcomm's own kernels" and fully portable.
3. **Super Resolution = reconstruction target**: same runtime, Qualcomm's SR int8 weights unchanged, with our own SPIR-V pack/conv/unpack/D2S matching the documented quantization (`scales.json`, `network_io_encodings.json`). Logged per-kernel in `docs/divergences.md`. This is the "upscaler" Ayo asked for, delivered honestly as weights-unchanged + reconstructed kernels.
4. Integration via OptiScaler + vkd3d-proton, following d4r/m4r.
5. Validation without a device: determinism (repeat-run bit equality), CPU↔GPU conv equality, and a catalogued divergence list. No Adreno ground truth exists (Ayo has no device).

Exact per-frame dispatch order and push-constant values come from the full Ghidra decompile (running) and, if ambiguous, a qemu trace of `libanf.so`.
