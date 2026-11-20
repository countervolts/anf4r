// anf4r: offline SR runner. Reads a raw stream, runs the SR pipeline frame by frame, writes raw HR float32 RGB.
// Input  : int32 lrW, lrH, nFrames; per frame: float jx, jy; float colour[lrW*lrH*3] (linear HDR);
//          float depth[lrW*lrH] (reverse-Z); float mv[lrW*lrH*2] (uv units, jitter-free).
// Output : per frame float rgb[2lrW*2lrH*3] (linear). Usage: sr_run in.bin out.bin [modelDir]
#include "../runtime/sr.h"
#include <chrono>
#include <cstdio>
#include <cstring>
using namespace anf4r;

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: sr_run in.bin out.bin [modelDir]\n"); return 1; }
    FILE* fi = fopen(argv[1], "rb"); FILE* fo = fopen(argv[2], "wb"); if (!fi || !fo) { printf("io error\n"); return 1; }
    int32_t hdr[3]; if (fread(hdr, 4, 3, fi) != 3) return 1; int W = hdr[0], H = hdr[1], N = hdr[2];
    Context ctx; if (!ctx.create(getenv("VAL") != nullptr)) { printf("ctx: %s\n", ctx.error.c_str()); return 1; }
    SrConfig cfg; cfg.lrW = W; cfg.lrH = H; if (argc > 3) cfg.modelDir = argv[3];
    SuperRes sr; if (!sr.init(ctx, cfg)) { printf("init: %s\n", ctx.error.c_str()); return 1; }
    Texture color, depth, mv, out;
    ctx.createTexture(color, VK_FORMAT_R16G16B16A16_SFLOAT, W, H); ctx.createTexture(depth, VK_FORMAT_R32_SFLOAT, W, H);
    ctx.createTexture(mv, VK_FORMAT_R16G16_SFLOAT, W, H); ctx.createTexture(out, VK_FORMAT_R16G16B16A16_SFLOAT, 2 * W, 2 * H);
    { VkCommandBuffer cb = ctx.begin(); ctx.toLayout(cb, out, VK_IMAGE_LAYOUT_GENERAL); ctx.submit(cb); }
    size_t n = (size_t)W * H; std::vector<float> col(n * 3), dep(n), mvf(n * 2); std::vector<uint16_t> c16(n * 4), m16(n * 2), o16((size_t)4 * n * 4);
    double totalMs = 0; int saveLast = getenv("ANF_SAVE_LAST") ? atoi(getenv("ANF_SAVE_LAST")) : N;   // only write the last K frames
    for (int f = 0; f < N; ++f) {
        float j[2]; if (fread(j, 4, 2, fi) != 2 || fread(col.data(), 4, col.size(), fi) != col.size() || fread(dep.data(), 4, n, fi) != n || fread(mvf.data(), 4, mvf.size(), fi) != mvf.size()) { printf("short read at frame %d\n", f); return 1; }
        for (size_t i = 0; i < n; ++i) { for (int c = 0; c < 3; ++c) c16[i * 4 + c] = floatToHalf(col[i * 3 + c]); c16[i * 4 + 3] = floatToHalf(1.f); m16[i * 2] = floatToHalf(mvf[i * 2]); m16[i * 2 + 1] = floatToHalf(mvf[i * 2 + 1]); }
        ctx.upload(color, c16.data(), c16.size() * 2); ctx.upload(depth, dep.data(), n * 4); ctx.upload(mv, m16.data(), m16.size() * 2);
        auto t0 = std::chrono::steady_clock::now();
        VkCommandBuffer cb = ctx.begin(); sr.record(ctx, cb, color, depth, mv, out, j[0], j[1]); if (!ctx.submit(cb)) { printf("submit failed\n"); return 1; }
        totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ctx.download(out, o16.data(), o16.size() * 2);
        std::vector<float> o((size_t)4 * n * 3); for (size_t i = 0; i < (size_t)4 * n; ++i) for (int c = 0; c < 3; ++c) o[i * 3 + c] = halfToFloat(o16[i * 4 + c]);
        if (f >= N - saveLast) fwrite(o.data(), 4, o.size(), fo);
    }
    if (getenv("DBG")) {   // statistics of the decoded net output (raw = int8+128): offset_x, offset_y, alpha, feature per HR sub-position
        size_t bytes = (size_t)sr.netW * sr.netH * 64; Buffer h; ctx.createBuffer(h, bytes, true); VkCommandBuffer cb = ctx.begin(); VkBufferCopy c{0, 0, bytes}; vkCmdCopyBuffer(cb, sr.act[3].buffer, h.buffer, 1, &c); ctx.submit(cb);
        const int8_t* q = (const int8_t*)h.map; double acc[16][4] = {}; size_t cnt = (size_t)sr.netW * sr.netH;
        for (size_t i = 0; i < cnt; ++i) for (int sp = 0; sp < 16; ++sp) for (int k = 0; k < 4; ++k) acc[sp][k] += (q[i * 64 + sp * 4 + k] + 128) / 255.0;
        printf("net output means per sub-position (sy,sx): offx offy alpha feature\n"); for (int sp = 0; sp < 16; ++sp) printf("  (%d,%d) %.3f %.3f %.3f %.3f\n", sp / 4, sp % 4, acc[sp][0] / cnt, acc[sp][1] / cnt, acc[sp][2] / cnt, acc[sp][3] / cnt);
        ctx.destroyBuffer(h);
    }
    printf("sr_run: %d frames %dx%d -> %dx%d, %.2f ms/frame (record+submit+wait)\n", N, W, H, 2 * W, 2 * H, totalMs / N);
    fclose(fo); fclose(fi); sr.destroy(ctx); for (Texture* t : {&color, &depth, &mv, &out}) ctx.destroyTexture(*t); ctx.destroy(); return 0;
}
