// anf4r: builds the matrix-unit convolution pass, choosing among bit-identical kernel variants by timing them on the actual GPU (cached per GPU and shape).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "pass.h"
#include <string>
#include <vector>

namespace anf4r {

struct WmmaShape {
    int inW, inH, icw, instride, oc, stride, outW, outH, mode, lo, hi, padv, outOff;
    std::vector<SpecConst> extra;   // additional specialisation constants (head modes)
};

// The pass has 8 storage-buffer bindings (in, weights, bias, scale, out8, out16, outFlow, outMask). Grid: x = ((outW+31)/32) * pass.lx, y = outH, z = oc/32.
// tune = false (or ANF_NO_TUNE=1) uses the default variant; ANF_WMMA_VARIANT=<n> forces a variant. cacheDir: where tune.txt is kept ("" = none).
bool buildWmmaConv(Context& ctx, Pass& out, const std::vector<uint32_t>& spv, const WmmaShape& s, const std::string& cacheDir, bool tune = true);
inline void wmmaGrid(const Pass& p, const WmmaShape& s, uint32_t& gx, uint32_t& gy, uint32_t& gz) { gx = ((s.outW + 31) / 32) * p.lx; gy = s.outH; gz = s.oc / 32; }

} // namespace anf4r
