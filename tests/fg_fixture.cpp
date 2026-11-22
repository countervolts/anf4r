// anf4r: frame generation through the C API (anf4r_interpolate) on an m4r-style FG fixture (tests/scene_fg.py layout:
// color-NN.rgba16f at output size, depth-NN.r32f and mv-NN.rg16f at input size, input pixels, previous - current; reversed depth).
// Frame t >= 1 writes <out>/interp-NN.rgba16f, the frame between t-1 and t: m4r's tests/compare_fg.py scores them.
// usage: fg_fixture <data dir> <fixture dir> <frames> <out dir> [ow oh iw ih]
// SPDX-License-Identifier: GPL-3.0-only
#include "../runtime/vk.h"
#include "../runtime/anf4r.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
using namespace anf4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 5) { std::printf("usage: fg_fixture <data dir> <fixture dir> <frames> <out dir> [ow oh iw ih]\n"); return 2; }
    const std::string data = argv[1], fx = argv[2], out = argv[4];
    const int frames = std::atoi(argv[3]);
    const uint32_t ow = argc > 8 ? std::atoi(argv[5]) : 1280, oh = argc > 8 ? std::atoi(argv[6]) : 720;
    const uint32_t iw = argc > 8 ? std::atoi(argv[7]) : 640, ih = argc > 8 ? std::atoi(argv[8]) : 360;
    Context ctx;
    if (!ctx.create()) { std::printf("ctx: %s\n", ctx.error.c_str()); return 1; }
    anf4r_context* a = nullptr;
#ifdef ANF4R_HIPVK
    PFN_vkGetInstanceProcAddr gipa = nullptr;
#else
    PFN_vkGetInstanceProcAddr gipa = vkGetInstanceProcAddr;
#endif
    if (anf4r_create(&a, gipa, ctx.instance, ctx.physical, ctx.device, ctx.queueFamily, data.c_str()) != 0) { std::printf("anf4r_create failed\n"); return 1; }
    if (anf4r_interpolate_init(a, ow, oh, iw, ih) != 0) { std::printf("init: %s\n", anf4r_last_error(a)); return 1; }
    Texture color[2], depth, motion, output;
    for (auto& c : color) ctx.createTexture(c, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
    ctx.createTexture(depth, VK_FORMAT_R32_SFLOAT, iw, ih); ctx.createTexture(motion, VK_FORMAT_R16G16_SFLOAT, iw, ih);
    ctx.createTexture(output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
    { VkCommandBuffer cb = ctx.begin(); ctx.toLayout(cb, output, VK_IMAGE_LAYOUT_GENERAL); ctx.submit(cb); }
    auto img = [](const Texture& t) { return anf4r_image{t.image, t.format, t.width, t.height, VK_IMAGE_LAYOUT_GENERAL}; };
    std::vector<uint8_t> o(size_t(ow) * oh * 8);
    for (int t = 0; t < frames; ++t) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "/color-%02d.rgba16f", t); auto c = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/mv-%02d.rg16f", t); auto m = readAll(fx + nm);
        if (c.size() != o.size() || d.size() != size_t(iw) * ih * 4 || m.size() != size_t(iw) * ih * 4) { std::printf("frame %d: fixture size\n", t); return 1; }
        Texture& cur = color[t & 1];
        ctx.upload(cur, c.data(), c.size()); ctx.upload(depth, d.data(), d.size()); ctx.upload(motion, m.data(), m.size());
        anf4r_interpolate_desc in{};
        in.color = img(cur); in.prev_color = img(color[(t + 1) & 1]); in.depth = img(depth); in.motion = img(motion); in.output = img(output);
        in.motion_scale_x = in.motion_scale_y = 1.0f; in.delta_time = 1.0f / 60.0f; in.near_plane = 0.1f; in.far_plane = 1000.0f;
        in.depth_reversed = 1; in.reset = t == 0;
        VkCommandBuffer cb = ctx.begin();
        if (anf4r_interpolate(a, cb, &in) != 0) { std::printf("frame %d: %s\n", t, anf4r_last_error(a)); return 1; }
        ctx.submit(cb);
        if (t == 0) continue;
        ctx.download(output, o.data(), o.size());
        std::snprintf(nm, sizeof nm, "/interp-%02d.rgba16f", t);
        std::ofstream(out + nm, std::ios::binary).write(reinterpret_cast<char*>(o.data()), o.size());
    }
    anf4r_destroy(a);
    std::printf("ok\n");
    return 0;
}
