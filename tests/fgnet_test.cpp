// anf4r: FG LR/QR networks: GPU vs CPU reference (bit-exact), determinism. Real prod-8 weights.
#include "../runtime/fgnet.h"
#include <cstdio>
#include <cstring>
#include <random>
using namespace anf4r;

static std::vector<uint8_t> readback(Context& ctx, const Buffer& b, size_t n) {
    Buffer h; ctx.createBuffer(h, n, true);
    VkCommandBuffer cb = ctx.begin(); VkBufferCopy c{0, 0, n}; vkCmdCopyBuffer(cb, b.buffer, h.buffer, 1, &c); ctx.submit(cb);
    std::vector<uint8_t> v((uint8_t*)h.map, (uint8_t*)h.map + n); ctx.destroyBuffer(h); return v;
}
static int cmp(const char* what, const void* a, const void* b, size_t n) {
    size_t bad = 0, first = ~0ul; for (size_t i = 0; i < n; ++i) if (((uint8_t*)a)[i] != ((uint8_t*)b)[i]) { if (first == ~0ul) first = i; ++bad; }
    printf("  %-22s %8zu bytes  mismatches=%zu%s\n", what, n, bad, bad ? "  <-- FAIL" : ""); if (bad) printf("    first at byte %zu\n", first); return bad ? 1 : 0;
}

static int runNet(Context& ctx, bool lr, int W, int H, const char* root) {
    std::string dir = std::string(root) + (lr ? "/aiff-frc-lr-v2-prod-8" : "/aiff-frc-qr-v2-prod-8"), err;
    FgNetModel m; if (!m.load(dir, lr, err)) { printf("load: %s\n", err.c_str()); return 1; }
    printf("== %s net, input %dx%d (enc in %.5f/%.0f flow %.2e/%.0f mask %.2e/%.0f)\n", lr ? "LR" : "QR", W, H,
           m.enc.inScale, m.enc.inOffset, m.enc.flowScale, m.enc.flowOffset, m.enc.maskScale, m.enc.maskOffset);
    std::mt19937 rng(7); std::vector<int8_t> in((size_t)W * H * 32);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (int8_t)((int)(rng() % 61) - 30);
    // CPU chain
    std::vector<std::vector<uint8_t>> cpuAct; std::vector<int8_t> cur = in; int cw = W, ch = H; std::vector<uint16_t> curH;
    size_t nl = m.hidden.size();
    for (size_t i = 0; i < nl; ++i) { int ow, oh;
        if (lr && i == nl - 1) { curH = fgConvI8ToHalfCPU(m.hidden[i], cur, cw, ch, ow, oh); cpuAct.emplace_back((uint8_t*)curH.data(), (uint8_t*)curH.data() + curH.size() * 2); }
        else { cur = fgConvI8CPU(m.hidden[i], cur, cw, ch, ow, oh, 0, 127); cpuAct.emplace_back((uint8_t*)cur.data(), (uint8_t*)cur.data() + cur.size()); }
        cw = ow; ch = oh; }
    std::vector<uint8_t> cflow, cmask;
    if (lr) { std::vector<uint16_t> f, mk; fgHeadLRCPU(m.head, curH, cw, ch, f, mk); cflow.assign((uint8_t*)f.data(), (uint8_t*)f.data() + f.size() * 2); cmask.assign((uint8_t*)mk.data(), (uint8_t*)mk.data() + (size_t)cw * ch * 2); }
    else fgHeadQRCPU(m.head, cur, cw, ch, (int)m.enc.flowOffset, (int)m.enc.maskOffset, cflow, cmask);
    // GPU
    FgNet net; if (!net.build(ctx, m, W, H, "build")) { printf("build: %s\n", ctx.error.c_str()); return 1; }
    Buffer inb; ctx.createBuffer(inb, in.size(), true); memcpy(inb.map, in.data(), in.size());
    int fails = 0; std::vector<uint8_t> prevFlow, prevMask;
    for (int run = 0; run < 3; ++run) {
        VkCommandBuffer cb = ctx.begin(); net.record(ctx, cb, inb.buffer); if (!ctx.submit(cb)) { printf("submit failed\n"); return 1; }
        auto gflow = readback(ctx, net.flow, cflow.size()), gmask = readback(ctx, net.mask, cmask.size());
        if (run == 0) {
            for (size_t i = 0; i < nl; ++i) { auto g = readback(ctx, net.activation(i), cpuAct[i].size()); char nm[40]; snprintf(nm, 40, "%s", m.hidden[i].name.c_str());
                if (getenv("DBG") && lr && i == nl - 1) for (int k = 0, shown = 0; k < (int)g.size() / 2 && shown < 6; ++k) { uint16_t a, b; memcpy(&a, &g[k*2], 2); memcpy(&b, &cpuAct[i][k*2], 2); if (a != b) { printf("    [%d] gpu %04x (%g) cpu %04x (%g)\n", k, a, halfToFloat(a), b, halfToFloat(b)); ++shown; } } fails += cmp(nm, g.data(), cpuAct[i].data(), g.size()); }
            fails += cmp("head flow", gflow.data(), cflow.data(), cflow.size()); fails += cmp("head mask", gmask.data(), cmask.data(), cmask.size());
            size_t nz = 0; for (auto b : gflow) nz += b != 0; printf("  flow nonzero bytes: %zu / %zu\n", nz, gflow.size());
        } else { fails += (gflow != prevFlow) + (gmask != prevMask); printf("  run %d deterministic vs run %d: %s\n", run, run - 1, (gflow == prevFlow && gmask == prevMask) ? "yes" : "NO"); }
        prevFlow = gflow; prevMask = gmask;
    }
    net.destroy(ctx); ctx.destroyBuffer(inb); return fails;
}

int main(int argc, char** argv) {
    const char* root = argc > 1 ? argv[1] : "vendor/assets/modelzoo";
    Context ctx; if (!ctx.create(getenv("VAL") != nullptr)) { printf("ctx: %s\n", ctx.error.c_str()); return 1; }
    int f = 0; f += runNet(ctx, true, 70, 50, root); f += runNet(ctx, false, 66, 38, root);
    printf(f ? "FGNET FAIL (%d)\n" : "FGNET OK: GPU == CPU reference, deterministic\n", f); ctx.destroy(); return f != 0;
}
