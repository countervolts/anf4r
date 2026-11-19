# Divergences from the vendor kernels

Every re-implemented or transformed kernel is logged here with its reason and the argument for why it preserves
results (or an honest note that it may not). The vendor's Vulkan SPIR-V passes (12 `AIFF*_cs`) and all weights run
**unchanged**; the items below are the pieces that had no RDNA3-runnable vendor form.

## D1 — int8 convolution (`kernels/conv_int8.comp`)  [reconstructed]
**Why:** ANF's convolutions exist only as Adreno machine code (QASM inline asm in `Kernels_Hawi/*.cl`, precompiled
Adreno OpenCL binaries, and ARM data-graph caches). None runs on RDNA3. The network *weights, biases, per-channel
scales and architecture* are used **unchanged** (`vendor/assets/modelzoo/...`); only the conv *execution* is ours.

**Semantics reproduced (from `modelconfig.json`, `*.scales.json`, and the readable C in the `.cl` kernels):**
- int8 × int8 → **int32** accumulation. Integer addition is associative, so the result is independent of summation
  order: bit-identical regardless of tiling/subgroup. (VERIFIED by construction.)
- Requant: `f = float(acc + bias_i32) * scale[oc]`, then ReLU `max(f,0)`, then convert to int8 with
  **round-toward-zero + saturate** to [-128,127] (OpenCL `convert_char_sat`, which the Adreno kernel uses via
  `run_activation` + store).
- `scale[oc] = weights_scales[oc]` from `*.scales.json`. For the **`invpow2`** model variant these are exact powers
  of two (2^-6, 2^-7, 2^-8), so the float multiply is **exact** (no rounding) — the whole layer is then bit-exact on
  any IEEE-754 device. For the default model the scales are arbitrary floats; the multiply rounds once (RNE), matched
  between our CPU reference and GPU.

**Preservation argument:** The only hardware-defined step is the single `float(acc)*scale` multiply and the
float→int8 conversion. Both are IEEE-754 / fixed-semantics and identical between our CPU reference and the GPU
(VERIFIED bit-exact, `tests/conv_test.cpp`, all SR layers). Versus Adreno: with `invpow2` the multiply is exact and
only the convert rounding could differ; we use round-toward-zero to match OpenCL `convert_char`. **ASSUMED** that
Adreno's store uses `convert_char_sat` (round-toward-zero); unverifiable without a device. Risk is at most ±1 LSB on
rare exact-half cases, and zero for the invpow2 model except at the convert step.

**Validation:** `tests/conv_test.cpp` — GPU kernel vs independent CPU reference, bit-exact (0 mismatches) on real
extracted weights for cnn1 (20→32 s2), cnn2-0, cnn2-1 (32→32 s1), cnn3 (32→64 s1), both invpow2 and float-scale.
