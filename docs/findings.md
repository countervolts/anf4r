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
