// anf4r: matrix-unit convolution variants + start-up timing. SPDX-License-Identifier: GPL-3.0-only
#include "wmma.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace anf4r {

struct Variant { int style, waves; };
static const Variant kVariants[] = {{0, 1}, {1, 1}, {1, 3}};
static const int kNumVariants = 3;

static bool buildVariant(Context& ctx, Pass& p, const std::vector<uint32_t>& spv, const WmmaShape& s, int v) {
    std::vector<BindDesc> b; for (uint32_t i = 0; i < 8; ++i) b.push_back({i, BindKind::StorageBuffer, "b"});
    std::vector<SpecConst> sp = {{0,s.inW},{1,s.inH},{2,s.icw},{3,s.instride},{4,s.oc},{5,s.stride},{6,s.outW},{7,s.outH},{8,s.mode},{9,s.lo},{10,s.hi},{11,s.padv},{12,s.outOff},
                                 {18, 32 * kVariants[v].waves}, {19, kVariants[v].style}};
    for (auto& e : s.extra) sp.push_back(e);
    p.requiredSubgroup = 32;
    return p.build(ctx, spv, b, sp, 0, 32 * kVariants[v].waves, 1, 1, "main");
}

static std::map<std::string, int>& cache(const std::string& dir, bool& loaded) {
    static std::map<std::string, int> m;
    if (!loaded) {
        loaded = true; std::ifstream f(dir + "/tune.txt"); std::string line;
        while (std::getline(f, line)) { auto e = line.rfind('='); if (e != std::string::npos) m[line.substr(0, e)] = atoi(line.c_str() + e + 1); }
    }
    return m;
}

// times variant v: 8 dispatches back to back (two submits, best of); returns ms per dispatch or -1
static double timeVariant(Context& ctx, const std::vector<uint32_t>& spv, const WmmaShape& s, int v, VkQueryPool qp, double period, const Buffer* bufs) {
    Pass p; if (!buildVariant(ctx, p, spv, s, v)) { ctx.error.clear(); p.destroy(ctx); return -1; }
    for (uint32_t i = 0; i < 8; ++i) p.bindBuffer(i, bufs[i < 4 ? i : (i == 4 || i == 5 ? 4 : 5)].buffer);   // in, w, b, s, out (aliased), flow/mask dummies
    uint32_t gx, gy, gz; wmmaGrid(p, s, gx, gy, gz); double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        VkCommandBuffer cb = ctx.begin(); vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        for (int i = 0; i < 8; ++i) { p.record(ctx, cb, gx, gy, gz, nullptr, 0); ctx.barrierAll(cb); }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        if (!ctx.submit(cb)) { p.destroy(ctx); return -1; }
        uint64_t ts[2]; if (vkGetQueryPoolResults(ctx.device, qp, 0, 2, 16, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) { p.destroy(ctx); return -1; }
        if (rep > 0) best = std::min(best, (ts[1] - ts[0]) * period * 1e-6 / 8);   // first run is a warm-up
    }
    p.destroy(ctx); return best;
}

bool buildWmmaConv(Context& ctx, Pass& out, const std::vector<uint32_t>& spv, const WmmaShape& s, const std::string& cacheDir, bool tune) {
    int pick = 0;
    if (const char* f = getenv("ANF_WMMA_VARIANT")) pick = std::min(std::max(atoi(f), 0), kNumVariants - 1);
    else if (tune && !getenv("ANF_NO_TUNE") && s.mode != 2) {
        std::ostringstream k; k << ctx.deviceName << "|ic" << s.icw << " oc" << s.oc << " s" << s.stride << " m" << s.mode << " px" << (int)std::log2((double)s.outW * s.outH);
        bool loaded = (cacheDir.empty()); auto& m = cache(cacheDir, loaded); auto it = m.find(k.str());
        if (it != m.end()) pick = it->second;
        else {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 2; VkQueryPool qp = VK_NULL_HANDLE;
            if (vkCreateQueryPool(ctx.device, &qi, nullptr, &qp) == VK_SUCCESS) {
                VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(ctx.physical, &pr);
                Buffer bufs[6]; size_t sz[6] = {(size_t)s.inW * s.inH * s.instride, (size_t)s.oc * 9 * s.icw, (size_t)s.oc * 4, (size_t)s.oc * 4, (size_t)s.outW * s.outH * s.oc * 2 + 64, 64};
                bool ok = true; for (int i = 0; i < 6; ++i) ok &= ctx.createBuffer(bufs[i], std::max<size_t>(sz[i], 64), false);
                if (ok) {
                    VkCommandBuffer cb = ctx.begin(); for (auto& b : bufs) vkCmdFillBuffer(cb, b.buffer, 0, VK_WHOLE_SIZE, 0x01010101u); ctx.submit(cb);
                    double t[kNumVariants]; double bestT = 1e30;
                    for (int v = 0; v < kNumVariants; ++v) { t[v] = timeVariant(ctx, spv, s, v, qp, pr.limits.timestampPeriod, bufs); if (t[v] > 0) bestT = std::min(bestT, t[v]); }
                    pick = 0; if (t[0] > 0 && bestT < 1e29) { double bt = t[0]; for (int v = 1; v < kNumVariants; ++v) if (t[v] > 0 && t[v] < bt * 0.97) { bt = t[v]; pick = v; } }   // keep the default unless clearly faster
                    m[k.str()] = pick;
                    if (!cacheDir.empty()) std::ofstream(cacheDir + "/tune.txt", std::ios::app) << k.str() << "=" << pick << "\n";
                    ok = true;
                }
                for (auto& b : bufs) if (b.buffer) ctx.destroyBuffer(b);
                vkDestroyQueryPool(ctx.device, qp, nullptr);
            }
        }
    }
    return buildVariant(ctx, out, spv, s, pick);
}

} // namespace anf4r
