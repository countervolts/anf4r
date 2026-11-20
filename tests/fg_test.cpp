// anf4r: FG end-to-end on a synthetic translating scene: vendor SPIR-V passes + our LR/QR nets (real prod-8 weights).
// Checks: runs validation-clean, deterministic across resets, and interpolation quality vs the analytic mid frame.
#include "../runtime/fg.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
using namespace anf4r;

struct ImgScene {   // real render (raw RGB file) translated with bilinear sampling; origin offset so shifts stay in-bounds
    int W, H, IW, IH, ox, oy; std::vector<uint8_t> px;
    ImgScene(const char* path, int iw, int ih, int w, int h) : W(w), H(h), IW(iw), IH(ih) { FILE* f = fopen(path, "rb"); px.resize((size_t)iw * ih * 3); if (!f || fread(px.data(), 1, px.size(), f) != px.size()) { printf("cannot read %s\n", path); exit(1); } fclose(f); ox = (iw - w) / 2; oy = (ih - h) / 2; }
    float at(int c, int x, int y) const { x = std::min(std::max(x, 0), IW - 1); y = std::min(std::max(y, 0), IH - 1); return px[((size_t)y * IW + x) * 3 + c] / 255.f; }
    void render(float sx, float sy, std::vector<uint8_t>& out) const {
        out.resize((size_t)W * H * 4);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { float X = ox + x - sx, Y = oy + y - sy; int ix = (int)std::floor(X), iy = (int)std::floor(Y); float tx = X - ix, ty = Y - iy;
            for (int c = 0; c < 3; ++c) { float v = (at(c, ix, iy) * (1 - tx) + at(c, ix + 1, iy) * tx) * (1 - ty) + (at(c, ix, iy + 1) * (1 - tx) + at(c, ix + 1, iy + 1) * tx) * ty; out[((size_t)y * W + x) * 4 + c] = (uint8_t)std::lround(v * 255.f); }
            out[((size_t)y * W + x) * 4 + 3] = 255; }
    }
};
struct Scene {   // fractal value-noise "terrain" + hard-edged boxes, translated by (sx,sy) pixels with bilinear sampling
    int W, H; static const int N = 512; std::vector<float> g[3];
    Scene(int w, int h) : W(w), H(h) { std::mt19937 r(5); std::uniform_real_distribution<float> u(0, 1);
        for (int c = 0; c < 3; ++c) { g[c].assign(N * N, 0.f); float amp = 1.f, tot = 0.f;
            for (int o = 0; o < 6; ++o) { int cell = 64 >> o; if (cell < 2) cell = 2; int gn = N / cell + 2; std::vector<float> lat((size_t)gn * gn); for (auto& v : lat) v = u(r);
                for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) { float fx = (float)x / cell, fy = (float)y / cell; int ix = (int)fx, iy = (int)fy; float tx = fx - ix, ty = fy - iy; tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
                    float a = lat[iy * gn + ix], b = lat[iy * gn + ix + 1], cc = lat[(iy + 1) * gn + ix], d = lat[(iy + 1) * gn + ix + 1]; g[c][y * N + x] += amp * ((a + (b - a) * tx) * (1 - ty) + (cc + (d - cc) * tx) * ty); }
                tot += amp; amp *= 0.55f; }
            for (auto& v : g[c]) v /= tot; } }
    float samp(int c, float x, float y) const { x = std::fmod(x, (float)N); if (x < 0) x += N; y = std::fmod(y, (float)N); if (y < 0) y += N; int ix = (int)x, iy = (int)y; float tx = x - ix, ty = y - iy; int ix1 = (ix + 1) % N, iy1 = (iy + 1) % N;
        return (g[c][iy * N + ix] * (1 - tx) + g[c][iy * N + ix1] * tx) * (1 - ty) + (g[c][iy1 * N + ix] * (1 - tx) + g[c][iy1 * N + ix1] * tx) * ty; }
    void render(float sx, float sy, std::vector<uint8_t>& out) const {
        out.resize((size_t)W * H * 4);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float X = x - sx, Y = y - sy, c[3]; for (int k = 0; k < 3; ++k) c[k] = 0.15f + 0.8f * samp(k, X, Y);
            for (int b = 0; b < 5; ++b) { float bx = 20 + b * 53, by = 18 + (b * 47) % (H - 50); if (X >= bx && X < bx + 26 && Y >= by && Y < by + 20) { c[b % 3] = 0.95f; c[(b + 1) % 3] = 0.12f; c[(b + 2) % 3] = 0.3f; } }
            for (int k = 0; k < 3; ++k) out[((size_t)y * W + x) * 4 + k] = (uint8_t)std::lround(std::fmin(std::fmax(c[k], 0.f), 1.f) * 255.f);
            out[((size_t)y * W + x) * 4 + 3] = 255; }
    }
};
static double psnr(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int W, int H, int margin) {
    double se = 0; size_t n = 0; for (int y = margin; y < H - margin; ++y) for (int x = margin; x < W - margin; ++x) for (int k = 0; k < 3; ++k) { double d = (double)a[((size_t)y * W + x) * 4 + k] - b[((size_t)y * W + x) * 4 + k]; se += d * d; ++n; }
    return se == 0 ? 99.0 : 10 * std::log10(255.0 * 255.0 / (se / n)); }
static void ppm(const char* p, const std::vector<uint8_t>& im, int W, int H) { FILE* f = fopen(p, "wb"); if (!f) return; fprintf(f, "P6\n%d %d\n255\n", W, H); for (size_t i = 0; i < (size_t)W * H; ++i) fwrite(&im[i * 4], 1, 3, f); fclose(f); }

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 320, H = argc > 2 ? atoi(argv[2]) : 192; float sx = argc > 3 ? atof(argv[3]) : 6.0f, sy = argc > 4 ? atof(argv[4]) : 2.0f;
    Context ctx; if (!ctx.create(getenv("VAL") != nullptr)) { printf("ctx: %s\n", ctx.error.c_str()); return 1; }
    FgConfig cfg; cfg.width = W; cfg.height = H;
    FrameGen fg; if (!fg.init(ctx, cfg)) { printf("init: %s\n", ctx.error.c_str()); return 1; }
    printf("FG %dx%d: pre %dx%d qr %dx%d lr %dx%d | lr net flow %dx%d mask %dx%d | qr net flow %dx%d mask %dx%d\n", W, H, fg.sz.preW, fg.sz.preH, fg.sz.qrW, fg.sz.qrH,
        fg.sz.lrW, fg.sz.lrH, fg.sz.lrNet.flowW, fg.sz.lrNet.flowH, fg.sz.lrNet.nw, fg.sz.lrNet.nh, fg.sz.qrNet.flowW, fg.sz.qrNet.flowH, fg.sz.qrNet.nw, fg.sz.qrNet.nh);
    Texture color, depth, motion, out;
    ctx.createTexture(color, VK_FORMAT_R8G8B8A8_UNORM, W, H); ctx.createTexture(depth, VK_FORMAT_R32_SFLOAT, W, H);
    ctx.createTexture(motion, VK_FORMAT_R16G16_SFLOAT, W, H); ctx.createTexture(out, VK_FORMAT_R8G8B8A8_UNORM, W, H);
    { VkCommandBuffer cb = ctx.begin(); ctx.toLayout(cb, out, VK_IMAGE_LAYOUT_GENERAL); ctx.submit(cb); }
    const char* hero = getenv("ANF_HERO"); struct Any { const Scene* a; const ImgScene* b; void render(float x, float y, std::vector<uint8_t>& o) const { a ? a->render(x, y, o) : b->render(x, y, o); } };
    Scene* sc0 = hero ? nullptr : new Scene(W, H); ImgScene* sc1 = hero ? new ImgScene(hero, 2869, 2008, W, H) : nullptr; Any sc{sc0, sc1};
    std::vector<float> d((size_t)W * H, 0.5f); std::vector<uint16_t> mv((size_t)W * H * 2);
    for (size_t i = 0; i < (size_t)W * H; ++i) { mv[i * 2] = floatToHalf(-sx / W); mv[i * 2 + 1] = floatToHalf(-sy / H); }   // uv displacement prev-curr
    ctx.upload(depth, d.data(), d.size() * 4); ctx.upload(motion, mv.data(), mv.size() * 2);
    std::vector<std::vector<uint8_t>> outs[2];
    for (int pass = 0; pass < 2; ++pass) {
        fg.reset();
        for (int k = 0; k < 4; ++k) {
            std::vector<uint8_t> cur; sc.render(sx * k, sy * k, cur); ctx.upload(color, cur.data(), cur.size());
            VkCommandBuffer cb = ctx.begin(); fg.record(ctx, cb, color, depth, motion, out); if (!ctx.submit(cb)) { printf("submit failed\n"); return 1; }
            if (getenv("DBG")) { Buffer h; ctx.createBuffer(h, 16, true); VkCommandBuffer c2 = ctx.begin(); VkBufferCopy cc{0, 0, 16}; vkCmdCopyBuffer(c2, fg.fallback.buffer, h.buffer, 1, &cc); ctx.submit(c2); uint32_t* f = (uint32_t*)h.map; printf("  pass %d frame %d: luma_sum %u oos %u (thr %d)\n", pass, k, f[0], f[1], 8 * fg.sz.lrW * (fg.sz.lrH - 1)); ctx.destroyBuffer(h); }
            std::vector<uint8_t> o(cur.size()); ctx.download(out, o.data(), o.size()); outs[pass].push_back(o);
        }
    }
    if (getenv("DBG")) {
        auto rb = [&](const Buffer& b, size_t n) { Buffer h; ctx.createBuffer(h, n, true); VkCommandBuffer cb = ctx.begin(); VkBufferCopy c{0, 0, n}; vkCmdCopyBuffer(cb, b.buffer, h.buffer, 1, &c); ctx.submit(cb);
            std::vector<uint8_t> v((uint8_t*)h.map, (uint8_t*)h.map + n); ctx.destroyBuffer(h); return v; };
        auto fb = rb(fg.fallback, 16); uint32_t* f = (uint32_t*)fb.data(); int N = fg.sz.lrW * (fg.sz.lrH - 1);
        printf("fallback stats: luma_sum=%u (need > %d) oos=%u (need <= %d)\n", f[0], 8 * N, f[1], N * 20 / 100);
        auto ds = rb(fg.depthStats, 16); printf("depthStats: %08x %08x %08x %08x\n", ((uint32_t*)ds.data())[0], ((uint32_t*)ds.data())[1], ((uint32_t*)ds.data())[2], ((uint32_t*)ds.data())[3]);
        auto px = rb(fg.packedX, (size_t)fg.sz.preW * fg.sz.preH * 4); size_t nz = 0; for (size_t i = 0; i < px.size() / 4; ++i) nz += ((uint32_t*)px.data())[i] != 0; printf("packed_mv_x nonzero: %zu / %zu\n", nz, px.size() / 4);
        std::vector<uint8_t> t((size_t)fg.sz.preW * fg.sz.preH * 4);
        ctx.download(fg.fwdMV, t.data(), t.size()); int16_t* m = (int16_t*)t.data(); size_t c0 = ((size_t)fg.sz.preH / 2 * fg.sz.preW + fg.sz.preW / 2) * 2; printf("fwdMV centre (1/128 px): %d %d\n", m[c0], m[c0 + 1]);
        std::vector<uint8_t> t1((size_t)fg.sz.preW * fg.sz.preH); ctx.download(fg.maskRes, t1.data(), t1.size()); printf("maskRes centre %u, depthRes centre %u\n", t1[c0 / 2], 0u);
        std::vector<uint8_t> lf((size_t)fg.sz.lrNet.flowW * fg.sz.lrNet.flowH * 8); ctx.download(fg.lrFlow, lf.data(), lf.size()); uint16_t* h = (uint16_t*)lf.data();
        size_t o = ((size_t)fg.sz.lrNet.flowH / 2 * fg.sz.lrNet.flowW + fg.sz.lrNet.flowW / 2) * 4; printf("LR flow centre: %g %g %g %g\n", halfToFloat(h[o]), halfToFloat(h[o+1]), halfToFloat(h[o+2]), halfToFloat(h[o+3]));
    }
    if (getenv("DBG")) {
        auto stats = [&](const char* n, Texture& t, int comps, bool half) { size_t px = (size_t)t.width * t.height; std::vector<uint8_t> raw(px * comps * (half ? 2 : 4)); ctx.download(t, raw.data(), raw.size());
            for (int c = 0; c < comps; ++c) { double mn = 1e30, mx = -1e30, sum = 0; for (size_t i = 0; i < px; ++i) { float v; if (half) { uint16_t h; memcpy(&h, &raw[(i * comps + c) * 2], 2); v = halfToFloat(h); } else memcpy(&v, &raw[(i * comps + c) * 4], 4); mn = std::fmin(mn, v); mx = std::fmax(mx, v); sum += v; }
                printf("  %-8s ch%d min %9.4f max %9.4f mean %9.4f\n", n, c, mn, mx, sum / px); } };
        { size_t px = (size_t)fg.lrFlow.width * fg.lrFlow.height; std::vector<uint8_t> raw(px * 8); ctx.download(fg.lrFlow, raw.data(), raw.size()); uint16_t* h = (uint16_t*)raw.data();
            for (int c = 0; c < 4; ++c) { std::vector<float> v; for (size_t i = 0; i < px; ++i) v.push_back(halfToFloat(h[i * 4 + c])); std::sort(v.begin(), v.end()); printf("  lrFlow ch%d median %8.3f  p10 %8.3f p90 %8.3f\n", c, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10]); } }
        stats("lrFlow", fg.lrFlow, 4, true); stats("lrMask", fg.lrMask, 1, false); stats("qrFlow", fg.qrFlow, 4, true); stats("qrMask", fg.qrMask, 1, true);
    }
    if (getenv("DBG")) {
        auto rb2 = [&](const Buffer& b, size_t n) { Buffer h; ctx.createBuffer(h, n, true); VkCommandBuffer cb = ctx.begin(); VkBufferCopy c{0, 0, n}; vkCmdCopyBuffer(cb, b.buffer, h.buffer, 1, &c); ctx.submit(cb);
            std::vector<int8_t> v((int8_t*)h.map, (int8_t*)h.map + n); ctx.destroyBuffer(h); return v; };
        auto dump = [&](const char* nm, const std::vector<int8_t>& v, int ch) { size_t z = 0, sat = 0; double sum = 0; for (auto x : v) { z += x == 0; sat += x == 127; sum += x; }
            printf("  %-12s n=%zu zero %.1f%% sat127 %.1f%% mean %.2f\n", nm, v.size(), 100.0 * z / v.size(), 100.0 * sat / v.size(), sum / v.size()); (void)ch; };
        auto in = rb2(fg.netInLR, (size_t)fg.sz.lrW * fg.sz.lrH * 32); dump("LR input", in, 32); { FILE* f = fopen("build/dbg_lr_in.bin", "wb"); fwrite(in.data(), 1, in.size(), f); fclose(f); printf("  (LR input %dx%d saved)\n", fg.sz.lrW, fg.sz.lrH); }
        for (int c = 0; c < 32; ++c) { double sm = 0, mn = 1e9, mx = -1e9; for (size_t i = c; i < in.size(); i += 32) { sm += in[i]; mn = std::min<double>(mn, in[i]); mx = std::max<double>(mx, in[i]); } printf("   in ch%02d mean %7.2f min %4.0f max %4.0f\n", c, sm / (in.size() / 32), mn, mx); }
        { auto a = rb2(fg.lrNet.activation(12), fg.lrNet.activation(12).size); size_t n = a.size() / 2, nz = 0; double sm = 0, mx = 0; for (size_t i = 0; i < n; ++i) { uint16_t h; memcpy(&h, &a[i * 2], 2); float v = halfToFloat(h); nz += v != 0; sm += v; mx = std::max<double>(mx, v); }
          printf("  LR conv_out (fp16 real): n=%zu nonzero %.1f%% mean %.5f max %.4f\n", n, 100.0 * nz / n, sm / n, mx); }
        for (size_t i = 0; i + 1 < 13; ++i) { size_t n = fg.lrNet.activation(i).size; auto a = rb2(fg.lrNet.activation(i), n); char nm[16]; snprintf(nm, 16, "LR act%zu", i); dump(nm, a, 0); }
    }
    if (getenv("BENCH")) { auto t0 = std::chrono::steady_clock::now(); int N = 30; for (int i = 0; i < N; ++i) { VkCommandBuffer cb = ctx.begin(); fg.record(ctx, cb, color, depth, motion, out); ctx.submit(cb); }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N; printf("BENCH: %.2f ms/frame (record+submit+wait, %dx%d)\n", ms, W, H); }
    bool det = outs[0] == outs[1]; printf("deterministic across reset+replay: %s\n", det ? "yes" : "NO");
    system("mkdir -p build/fg_out");
    for (int k = 1; k < 4; ++k) {
        std::vector<uint8_t> prev, cur, gt, avg; sc.render(sx * (k - 1), sy * (k - 1), prev); sc.render(sx * k, sy * k, cur); sc.render(sx * (k - 0.5f), sy * (k - 0.5f), gt);
        avg.resize(cur.size()); for (size_t i = 0; i < cur.size(); ++i) avg[i] = (uint8_t)((prev[i] + cur[i] + 1) / 2);
        { size_t diff = 0; double md = 0; for (size_t i = 0; i < cur.size(); ++i) { int d = std::abs((int)outs[0][k][i] - (int)cur[i]); diff += d != 0; md += d; } size_t dp = 0; for (size_t i = 0; i < prev.size(); ++i) dp += outs[0][k][i] != prev[i];
          printf("frame %d: output differs from current in %.1f%% of bytes (mean |d| %.2f), from previous in %.1f%%\n", k, 100.0 * diff / cur.size(), md / cur.size(), 100.0 * dp / prev.size()); }
        printf("frame %d vs analytic mid: FG %.2f dB | plain average %.2f dB | repeat-current %.2f dB\n", k, psnr(outs[0][k], gt, W, H, 12), psnr(avg, gt, W, H, 12), psnr(cur, gt, W, H, 12));
        char n[64]; snprintf(n, 64, "build/fg_out/fg_%d.ppm", k); ppm(n, outs[0][k], W, H); snprintf(n, 64, "build/fg_out/gt_%d.ppm", k); ppm(n, gt, W, H);
    }
    fg.destroy(ctx); for (Texture* t : {&color, &depth, &motion, &out}) ctx.destroyTexture(*t); ctx.destroy(); return det ? 0 : 1;
}
