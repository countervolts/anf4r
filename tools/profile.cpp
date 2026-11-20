// anf4r: per-pass GPU timing (timestamp queries) of the SR and FG pipelines. usage: profile sr W H [OUT_W OUT_H] | profile fg W H   (W,H = SR input size / FG frame size;
// with OUT_W/OUT_H the bicubic resize to the requested output size, used for ratios other than 2x, is timed as well)
#include "../runtime/fg.h"
#include "../runtime/sr.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
using namespace anf4r;

static VkQueryPool qp; static std::vector<const Pass*> order; static std::vector<std::pair<std::string, int>> marks; static Context* gctx; static uint32_t nq;
static void hook(VkCommandBuffer cb, const Pass* p, bool begin) {
    vkCmdWriteTimestamp(cb, begin ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, nq);
    marks.push_back({begin ? "" : p->label, (int)nq}); ++nq;
}
int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: profile sr|fg W H\n"); return 1; }
    bool sr = !strcmp(argv[1], "sr"); int W = atoi(argv[2]), H = atoi(argv[3]);
    Context ctx; if (!ctx.create(false)) return 1; gctx = &ctx;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 512; vkCreateQueryPool(ctx.device, &qi, nullptr, &qp);
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(ctx.physical, &pr); double ns = pr.limits.timestampPeriod;
    std::map<std::string, double> sum; std::vector<std::string> keys; double total = 0; const int N = 20, WARM = 5;
    SuperRes S; FrameGen F; Texture color, depth, mv, out, rsz; Pass resizePass; int OW = argc > 5 ? atoi(argv[4]) : 0, OH = argc > 5 ? atoi(argv[5]) : 0; bool doResize = sr && OW && (OW != 2 * W || OH != 2 * H);
    if (sr) {
        SrConfig c; c.lrW = W; c.lrH = H; if (!S.init(ctx, c)) { printf("init %s\n", ctx.error.c_str()); return 1; }
        ctx.createTexture(color, VK_FORMAT_R16G16B16A16_SFLOAT, W, H); ctx.createTexture(depth, VK_FORMAT_R32_SFLOAT, W, H); ctx.createTexture(mv, VK_FORMAT_R16G16_SFLOAT, W, H); ctx.createTexture(out, VK_FORMAT_R16G16B16A16_SFLOAT, 2 * W, 2 * H);
    } else {
        FgConfig c; c.width = W; c.height = H; if (!F.init(ctx, c)) { printf("init %s\n", ctx.error.c_str()); return 1; }
        ctx.createTexture(color, VK_FORMAT_R8G8B8A8_UNORM, W, H); ctx.createTexture(depth, VK_FORMAT_R32_SFLOAT, W, H); ctx.createTexture(mv, VK_FORMAT_R16G16_SFLOAT, W, H); ctx.createTexture(out, VK_FORMAT_R8G8B8A8_UNORM, W, H);
    }
    if (doResize) {
        ctx.createTexture(rsz, VK_FORMAT_R16G16B16A16_SFLOAT, OW, OH);
        auto code = readSpirv("build/sr_resize.spv"); resizePass.label = "sr_resize (bicubic)";
        if (code.empty() || !resizePass.build(ctx, code, {{0, BindKind::SampledImage, "s"}, {1, BindKind::StorageImage, "d"}}, {}, 16, 8, 8, 1, "main")) { printf("resize pass: %s\n", ctx.error.c_str()); return 1; }
    }
    { VkCommandBuffer cb = ctx.begin(); for (Texture* t : {&color, &depth, &mv, &out}) ctx.toLayout(cb, *t, VK_IMAGE_LAYOUT_GENERAL); if (doResize) ctx.toLayout(cb, rsz, VK_IMAGE_LAYOUT_GENERAL); ctx.submit(cb); }
    g_passHook = hook;
    for (int f = 0; f < N + WARM; ++f) {
        nq = 0; marks.clear();
        VkCommandBuffer cb = ctx.begin(); vkCmdResetQueryPool(cb, qp, 0, 512);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, nq); int t0 = nq++;
        if (sr) S.record(ctx, cb, color, depth, mv, out, 0.1f * (f % 5), 0.f); else F.record(ctx, cb, color, depth, mv, out);
        if (doResize) { struct { int32_t sw, sh, dw, dh; } pc{2 * W, 2 * H, OW, OH}; resizePass.bindImage(0, out.view); resizePass.bindImage(1, rsz.view); resizePass.record(ctx, cb, OW, OH, 1, &pc, sizeof pc); ctx.barrierAll(cb); }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, nq); int t1 = nq++;
        ctx.submit(cb);
        std::vector<uint64_t> ts(nq); vkGetQueryPoolResults(ctx.device, qp, 0, nq, nq * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (f < WARM) continue;
        total += (ts[t1] - ts[t0]) * ns * 1e-6;
        int beginIdx = -1;
        for (auto& [label, idx] : marks) { if (label.empty()) { beginIdx = idx; continue; }
            if (!sum.count(label)) keys.push_back(label); sum[label] += (ts[idx] - ts[beginIdx]) * ns * 1e-6; }
    }
    printf("%s, %dx%d (%s), mean of %d frames, GPU timestamps\n", sr ? "SR" : "FG", W, H, pr.deviceName, N);
    double conv = 0, other = 0; for (auto& k : keys) { double v = sum[k] / N; printf("  %-26s %7.3f ms\n", k.c_str(), v); (k.find("conv") != std::string::npos || k.find("head") != std::string::npos ? conv : other) += v; }
    printf("  -- convs+heads %.3f ms, other passes %.3f ms, whole record() %.3f ms (incl. fills/copies/barriers)\n", conv, other, total / N);
    return 0;
}
