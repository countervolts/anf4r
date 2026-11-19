# anf4r qemu trace

Runs Qualcomm's `libanf.so` under qemu-aarch64 against recording OpenCL/Vulkan shims and Android bionic, to
recover the exact SR/FG dispatch graph, backend choice, kernel build defines, argument layout and tensor strides
without a Snapdragon device. The shims spoof an Adreno device so `IsTechniqueSupported` passes (authorized: the
goal is to observe ANF's own setup, nothing is redistributed).

Pieces:
- `cl_rec.c`  — recording libOpenCL.so: answers platform/device/context/program/kernel/enqueue calls, logs
  `clBuildProgram` options, `clCreateKernel` names, every `clSetKernelArg`, NDRange sizes, buffer writes.
- `vk_rec.c`  — recording libvulkan.so: `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr` return logged thunks;
  physical-device queries report an Adreno with the extensions ANF needs; shader modules dumped.
- `harness.c` — dlopens libanf.so, GetAnfFunctions, creates an instance, registers fake Vulkan handles, queries
  techniques, creates SR (then FG), and dispatches with dummy resources of the documented formats/sizes.
- `run.sh`    — builds everything with the NDK and runs under qemu-aarch64 with bionic from an arm64 system image.

Output: `trace.log` (ordered call log) and `dump/` (SPIR-V modules, kernel sources, arg blobs).
