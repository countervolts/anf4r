// anf4r: C API over SuperRes / FrameGen. SPDX-License-Identifier: GPL-3.0-only
#include "vkfn.h"
#include "anf4r.h"
#ifdef ANF4R_HIPVK
#include "pass.h"
#include "hipvk/hipvk.h"
#endif
#include "fg.h"
#include "sr.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>

using namespace anf4r;

static void dumpFlush(struct anf4r_context* c);
static void logLine(const std::string& dir, const std::string& m) { std::ofstream(dir + "/anf4r.log", std::ios::app) << m << "\n"; }
struct anf4r_context {
    Context ctx;
    std::string dataDir, error;
    std::unique_ptr<SuperRes> sr; int srW = 0, srH = 0; uint32_t outW = 0, outH = 0;   // requested SR output size (any ratio)
    Texture srOut, srFinal;                          // internal HR RGBA16F target (2x), and its bicubic resize to outW x outH when ratio != 2
    Pass srResize;
    std::map<std::pair<VkImage, VkFormat>, Texture> views;   // image views of the caller's images (cached)
    // frame generation
    std::unique_ptr<FrameGen> fg; int fgW = 0, fgH = 0; uint64_t fgFrames = 0;
    Texture fgColor, fgDepth, fgMotion, fgOut, real; VkFormat realFmt = VK_FORMAT_UNDEFINED;
    Pass resample; bool generated = false;
    // input capture (debug): create <dataDir>/dump.txt containing a frame count N; the next N SR frames are written to <dataDir>/dump/
    int dumpLeft = 0, dumpIdx = 0, upscaleCalls = 0; bool dumpPending = false; std::string dumpMeta;
    Buffer dumpBuf[4]; VkDeviceSize dumpSize[4] = {0, 0, 0, 0}; bool dumpOut = false;
};

static const Texture& view(anf4r_context* c, const anf4r_image& im) {
    auto key = std::make_pair(im.image, im.format);
    auto it = c->views.find(key);
    if (it != c->views.end() && it->second.width == im.width && it->second.height == im.height) return it->second;
    if (it != c->views.end()) { c->ctx.destroyTexture(it->second); c->views.erase(it); }
    Texture t; t.owned = false; t.image = im.image;
    c->ctx.wrap(t, im.image, im.format, im.width, im.height);
    t.owned = false;
    return c->views.emplace(key, t).first->second;
}
static void toGeneral(anf4r_context* c, VkCommandBuffer cb, const anf4r_image& im, Texture& t, VkImageLayout to) {
    t.layout = im.layout; c->ctx.toLayout(cb, t, to);
}

int anf4r_create(anf4r_context** out, PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t family, const char* dataDir) {
    auto* c = new anf4r_context(); c->dataDir = dataDir ? dataDir : ".";
    if (!anf4r::loadVulkan(gipa, inst, dev)) { delete c; return -1; }
#ifdef ANF4R_HIPVK
    anf4r::g_hipDir = c->dataDir + "/hip";   // the passes' code objects (scripts/build_hip.py, built by the installer for this GPU)
    hipvk::setKernelDir(anf4r::g_hipDir);    // and hipvk's format-converting blit (hipvk_blit.hsaco)
#endif
    if (!c->ctx.adopt(inst, phys, dev, family)) { delete c; return -2; }
    { std::ifstream t(c->dataDir + "/anf4r.log", std::ios::ate); if (t && t.tellg() > 65536) { t.close(); std::remove((c->dataDir + "/anf4r.log").c_str()); } }   // append; keep it small
    logLine(c->dataDir, "GPU: " + c->ctx.deviceName + " | int8 cooperative matrices: " + (c->ctx.coopInt8 ? "yes" : "no") + " | subgroup 32: " + (c->ctx.subgroup32 ? "yes" : "no") +
            " | convolution path: " + (c->ctx.wmmaOk() ? "matrix units (cooperative matrix)" : "scalar fallback"));
    *out = c; return 0;
}
void anf4r_destroy(anf4r_context* c) {
    if (!c) return;
    vkDeviceWaitIdle(c->ctx.device);
    dumpFlush(c); for (auto& b : c->dumpBuf) if (b.buffer) c->ctx.destroyBuffer(b);
    if (c->sr) c->sr->destroy(c->ctx);
    if (c->fg) { c->fg->destroy(c->ctx); c->resample.destroy(c->ctx); }
    for (Texture* t : {&c->fgColor, &c->fgDepth, &c->fgMotion, &c->fgOut, &c->real}) c->ctx.destroyTexture(*t);
    c->ctx.destroyTexture(c->srOut); c->ctx.destroyTexture(c->srFinal); c->srResize.destroy(c->ctx);
    for (auto& [k, t] : c->views) c->ctx.destroyTexture(t);
    c->ctx.destroy(); delete c;
}
const char* anf4r_last_error(anf4r_context* c) { return c ? (c->error.empty() ? c->ctx.error.c_str() : c->error.c_str()) : "null context"; }

int anf4r_upscale_init(anf4r_context* c, uint32_t ow, uint32_t oh, uint32_t iw, uint32_t ih) {
    c->error.clear();
    // The network is fixed 2x: for other ratios it runs at 2x and the result is resampled (linear) to ow x oh.
    c->outW = ow; c->outH = oh;
    if (c->sr && c->srW == (int)iw && c->srH == (int)ih) return 0;
    vkDeviceWaitIdle(c->ctx.device);
    if (c->sr) { c->sr->destroy(c->ctx); c->sr.reset(); c->ctx.destroyTexture(c->srOut); c->ctx.destroyTexture(c->srFinal); c->srResize.destroy(c->ctx); c->srResize = Pass(); }
    SrConfig cfg; cfg.lrW = iw; cfg.lrH = ih;
    { std::ifstream cf(c->dataDir + "/config.txt"); std::string line;   // optional: key = value lines (stabilize, pre_exposure); see docs/divergences.md D8
      while (std::getline(cf, line)) { auto e = line.find('='); if (e == std::string::npos || line[0] == '#') continue;
        std::string k = line.substr(0, e), v = line.substr(e + 1); k.erase(k.find_last_not_of(" \t\r") + 1); k.erase(0, k.find_first_not_of(" \t"));
        if (k == "stabilize") cfg.stabilize = (float)atof(v.c_str()); else if (k == "pre_exposure") cfg.preExposure = (float)atof(v.c_str()); } } cfg.modelDir = c->dataDir + "/modelzoo/aiff-sr-v2s-prod-7p1-invpow2"; cfg.shaderDir = c->dataDir + "/shaders";
    c->sr = std::make_unique<SuperRes>();
    if (!c->sr->init(c->ctx, cfg)) { c->error = c->ctx.error; c->sr.reset(); return 2; }
    if (!c->ctx.createTexture(c->srOut, VK_FORMAT_R16G16B16A16_SFLOAT, 2 * iw, 2 * ih)) { c->error = "output allocation"; return 3; }
    if (ow != 2 * iw || oh != 2 * ih) {
        auto code = readSpirv(c->dataDir + "/shaders/sr_resize.spv");
        if (code.empty()) { c->error = "missing sr_resize.spv"; return 4; }
        if (!c->ctx.createTexture(c->srFinal, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh)) { c->error = "resize target allocation"; return 3; }
        if (!c->srResize.build(c->ctx, code, {{0, BindKind::SampledImage, "s"}, {1, BindKind::StorageImage, "d"}}, {}, 16, 8, 8, 1, "main")) { c->error = c->ctx.error; return 5; }
    }
    { VkCommandBuffer cb = c->ctx.begin(); c->ctx.toLayout(cb, c->srOut, VK_IMAGE_LAYOUT_GENERAL); if (c->srFinal.image) c->ctx.toLayout(cb, c->srFinal, VK_IMAGE_LAYOUT_GENERAL); c->ctx.submit(cb); }
    logLine(c->dataDir, "SR " + std::to_string(iw) + "x" + std::to_string(ih) + " -> " + std::to_string(ow) + "x" + std::to_string(oh) + ": convolutions on " + (c->sr->usesWmma() ? "matrix units" : "scalar kernels"));
    c->srW = iw; c->srH = ih; return 0;
}

// ---- input capture ----
static void dumpFlush(anf4r_context* c) {
    if (!c->dumpPending) return;
    vkDeviceWaitIdle(c->ctx.device);
    static const char* nm[4] = {"color", "depth", "motion", "out"};
    for (int i = 0; i < (c->dumpOut ? 4 : 3); ++i) {
        char f[512]; snprintf(f, sizeof f, "%s/dump/frame_%03d.%s.bin", c->dataDir.c_str(), c->dumpIdx, nm[i]);
        std::ofstream o(f, std::ios::binary); o.write((const char*)c->dumpBuf[i].map, c->dumpSize[i]);
    }
    std::ofstream m(c->dataDir + "/dump/manifest.txt", std::ios::app); m << c->dumpMeta << "\n";
    ++c->dumpIdx; c->dumpPending = false; c->dumpOut = false;
}
static void dumpMaybeStart(anf4r_context* c) {
    if (c->dumpLeft > 0 || (++c->upscaleCalls % 30) != 1) return;
    std::ifstream t(c->dataDir + "/dump.txt"); int n = 0; if (!(t >> n) || n <= 0) return; t.close();
    std::remove((c->dataDir + "/dump.txt").c_str());
    #ifdef _WIN32
    std::string cmd = "mkdir \"" + c->dataDir + "\\dump\" 2>nul"; system(cmd.c_str());
    #else
    std::string cmd = "mkdir -p \"" + c->dataDir + "/dump\""; if (system(cmd.c_str())) {}
    #endif
    std::remove((c->dataDir + "/dump/manifest.txt").c_str());
    c->dumpLeft = n; c->dumpIdx = 0;
}
static void dumpRecord(anf4r_context* c, VkCommandBuffer cb, const anf4r_upscale_desc* d, float jx, float jy) {
    dumpFlush(c);
    if (c->dumpLeft <= 0) return;
    const anf4r_image* im[3] = {&d->color, &d->depth, &d->motion};
    for (int i = 0; i < 3; ++i) {
        VkFormat f_ = im[i]->format;
        uint32_t bpp = (f_ == VK_FORMAT_D32_SFLOAT_S8_UINT || f_ == VK_FORMAT_X8_D24_UNORM_PACK32 || f_ == VK_FORMAT_D24_UNORM_S8_UINT || f_ == VK_FORMAT_D32_SFLOAT) ? 4 : f_ == VK_FORMAT_D16_UNORM ? 2 :
                       (f_ == VK_FORMAT_R16G16_UNORM || f_ == VK_FORMAT_R16G16_SNORM) ? 4 : f_ == VK_FORMAT_R16G16B16A16_UNORM ? 8 : formatBytes(f_);
        if (!bpp) { std::ofstream e(c->dataDir + "/dump/error.txt", std::ios::app); e << "unsupported format " << (int)f_ << " for input " << i << "\n"; c->dumpLeft = 0; return; }
        VkDeviceSize need = (VkDeviceSize)im[i]->width * im[i]->height * bpp;
        if (need > c->dumpSize[i] || !c->dumpBuf[i].buffer) { if (c->dumpBuf[i].buffer) c->ctx.destroyBuffer(c->dumpBuf[i]); c->ctx.createBuffer(c->dumpBuf[i], need, true); }
        c->dumpSize[i] = need;
        bool depthFmt = im[i]->format == VK_FORMAT_D32_SFLOAT || im[i]->format == VK_FORMAT_D24_UNORM_S8_UINT || im[i]->format == VK_FORMAT_D32_SFLOAT_S8_UINT || im[i]->format == VK_FORMAT_X8_D24_UNORM_PACK32 || im[i]->format == VK_FORMAT_D16_UNORM;
        VkBufferImageCopy r{}; r.imageSubresource = {(VkImageAspectFlags)(depthFmt ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1}; r.imageExtent = {im[i]->width, im[i]->height, 1};
        vkCmdCopyImageToBuffer(cb, im[i]->image, VK_IMAGE_LAYOUT_GENERAL, c->dumpBuf[i].buffer, 1, &r);
    }
    char m[512]; snprintf(m, sizeof m, "%d %u %u %d %u %u %d %u %u %d %u %u %g %g %g %g %g %g %d %d %u %u", c->dumpIdx,
        d->color.width, d->color.height, (int)d->color.format, d->depth.width, d->depth.height, (int)d->depth.format, d->motion.width, d->motion.height, (int)d->motion.format,
        d->content_width, d->content_height, d->jitter_x, d->jitter_y, d->motion_scale_x, d->motion_scale_y, jx, jy, d->reset, d->depth_reversed, d->output.width, d->output.height);
    c->dumpMeta = m; c->dumpPending = true; --c->dumpLeft; c->dumpOut = true;
}
// the live SR result (internal RGBA16F, 2x input size), recorded after the SR passes
static void dumpOutput(anf4r_context* c, VkCommandBuffer cb, uint32_t w, uint32_t h) {
    if (!c->dumpOut) return;
    VkDeviceSize need = (VkDeviceSize)w * h * 8;
    if (need > c->dumpSize[3] || !c->dumpBuf[3].buffer) { if (c->dumpBuf[3].buffer) c->ctx.destroyBuffer(c->dumpBuf[3]); c->ctx.createBuffer(c->dumpBuf[3], need, true); }
    c->dumpSize[3] = need;
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cb, c->srOut.image, VK_IMAGE_LAYOUT_GENERAL, c->dumpBuf[3].buffer, 1, &r);
}

int anf4r_upscale(anf4r_context* c, VkCommandBuffer cb, const anf4r_upscale_desc* d) {
    c->error.clear(); dumpMaybeStart(c);
    uint32_t iw = d->content_width ? d->content_width : d->color.width, ih = d->content_height ? d->content_height : d->color.height;
    if (!c->sr || c->srW != (int)iw || c->srH != (int)ih) { int r = anf4r_upscale_init(c, c->outW ? c->outW : 2 * iw, c->outH ? c->outH : 2 * ih, iw, ih); if (r) return r; }
    if (d->reset) c->sr->reset();
    Texture col = view(c, d->color), dep = view(c, d->depth), mv = view(c, d->motion);
    Texture* tex[3] = {&col, &dep, &mv}; const anf4r_image* im[3] = {&d->color, &d->depth, &d->motion};
    for (int i = 0; i < 3; ++i) toGeneral(c, cb, *im[i], *tex[i], VK_IMAGE_LAYOUT_GENERAL);
    // MV: texture value * scale = pixels (previous - current)  ->  ANF wants (current - previous) in LR pixels
    c->sr->mvScale[0] = -d->motion_scale_x; c->sr->mvScale[1] = -d->motion_scale_y;
    c->sr->setDepthInverted(!d->depth_reversed);
    dumpRecord(c, cb, d, d->jitter_x, -d->jitter_y);
    // desc jitter = sample offset in image coordinates (x right, y down); ANF's jitterOffset has y up
    c->sr->record(c->ctx, cb, col, dep, mv, c->srOut, d->jitter_x, -d->jitter_y);
    dumpOutput(c, cb, 2 * iw, 2 * ih);
    // convert/copy the internal RGBA16F result into the caller's output image
    Texture out = view(c, d->output); out.layout = d->output.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : d->output.layout;
    VkImageLayout back = out.layout;
    c->ctx.toLayout(cb, out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit b{}; b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    uint32_t ow = std::min<uint32_t>(c->outW ? c->outW : 2 * iw, d->output.width), oh = std::min<uint32_t>(c->outH ? c->outH : 2 * ih, d->output.height);
    const Texture* src = &c->srOut;
    if (c->srFinal.image) {   // ratio != 2: bicubic resize of the native 2x result to the requested size, then a same-size copy/convert
        struct { int32_t sw, sh, dw, dh; } pc{(int32_t)(2 * iw), (int32_t)(2 * ih), (int32_t)c->srFinal.width, (int32_t)c->srFinal.height};
        c->srResize.bindImage(0, c->srOut.view); c->srResize.bindImage(1, c->srFinal.view);
        c->srResize.record(c->ctx, cb, c->srFinal.width, c->srFinal.height, 1, &pc, sizeof pc); c->ctx.barrierAll(cb);
        src = &c->srFinal;
    }
    b.srcOffsets[1] = {(int32_t)std::min<uint32_t>(src->width, ow), (int32_t)std::min<uint32_t>(src->height, oh), 1}; b.dstOffsets[1] = b.srcOffsets[1];
    vkCmdBlitImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
    c->ctx.toLayout(cb, out, back);
    for (int i = 0; i < 3; ++i) { VkImageLayout orig = im[i]->layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : im[i]->layout; c->ctx.toLayout(cb, *tex[i], orig); }
    return 0;
}


// ---------------- frame generation ----------------
static Texture ext(const anf4r_image& im) {   // caller's image as a layout-tracking handle (no view)
    Texture t; t.owned = false; t.image = im.image; t.format = im.format; t.width = im.width; t.height = im.height;
    t.layout = im.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : im.layout; return t;
}
// copy (same format) or blit (format conversion) between two images; layouts of external images are restored afterwards
static void xfer(anf4r_context* c, VkCommandBuffer cb, Texture src, bool srcExt, Texture dst, bool dstExt, bool blit) {
    VkImageLayout so = src.layout, dO = dst.layout, sl = VK_IMAGE_LAYOUT_GENERAL, dl = VK_IMAGE_LAYOUT_GENERAL;
    if (srcExt) { c->ctx.toLayout(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); sl = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; }
    if (dstExt) { c->ctx.toLayout(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL); dl = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; }
    uint32_t w = std::min(src.width, dst.width), h = std::min(src.height, dst.height);
    if (blit) {
        VkImageBlit b{}; b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1}; b.dstOffsets[1] = b.srcOffsets[1];
        vkCmdBlitImage(cb, src.image, sl, dst.image, dl, 1, &b, VK_FILTER_NEAREST);
    } else {
        VkImageCopy r{}; r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.extent = {w, h, 1};
        vkCmdCopyImage(cb, src.image, sl, dst.image, dl, 1, &r);
    }
    if (srcExt) c->ctx.toLayout(cb, src, so);
    if (dstExt) c->ctx.toLayout(cb, dst, dO);
}

static int ensureFg(anf4r_context* c, uint32_t w, uint32_t h, VkFormat bbFormat) {
    if (c->fg && c->fgW == (int)w && c->fgH == (int)h && c->realFmt == bbFormat) return 0;
    vkDeviceWaitIdle(c->ctx.device);
    if (c->fg) { c->fg->destroy(c->ctx); c->fg.reset(); c->resample.destroy(c->ctx); c->resample = Pass(); }
    for (Texture* t : {&c->fgColor, &c->fgDepth, &c->fgMotion, &c->fgOut, &c->real}) c->ctx.destroyTexture(*t);
    FgConfig cfg; cfg.width = w; cfg.height = h; cfg.normalizeDepth = true;
    cfg.spirvDir = c->dataDir + "/spirv"; cfg.modelDir = c->dataDir + "/modelzoo"; cfg.shaderDir = c->dataDir + "/shaders";
    c->fg = std::make_unique<FrameGen>();
    if (!c->fg->init(c->ctx, cfg)) { c->error = c->ctx.error; c->fg.reset(); return 1; }
    bool ok = c->ctx.createTexture(c->fgColor, VK_FORMAT_R8G8B8A8_UNORM, w, h) && c->ctx.createTexture(c->fgOut, VK_FORMAT_R8G8B8A8_UNORM, w, h) &&
              c->ctx.createTexture(c->fgDepth, VK_FORMAT_R32_SFLOAT, w, h) && c->ctx.createTexture(c->fgMotion, VK_FORMAT_R16G16_SFLOAT, w, h) &&
              c->ctx.createTexture(c->real, bbFormat, w, h);
    if (!ok) { c->error = "FG allocation failed"; return 2; }
    { VkCommandBuffer cb = c->ctx.begin(); for (Texture* t : {&c->fgColor, &c->fgOut, &c->fgDepth, &c->fgMotion, &c->real}) c->ctx.toLayout(cb, *t, VK_IMAGE_LAYOUT_GENERAL); c->ctx.submit(cb); }
    auto code = readSpirv(c->dataDir + "/shaders/fg_resample.spv");
    if (code.empty()) { c->error = "missing fg_resample.spv"; return 3; }
    const BindKind SI = BindKind::SampledImage, ST = BindKind::StorageImage;
    if (!c->resample.build(c->ctx, code, {{0, SI, "d"}, {1, SI, "m"}, {2, ST, "od"}, {3, ST, "om"}}, {}, 24, 8, 8, 1, "main")) { c->error = c->ctx.error; return 4; }
    logLine(c->dataDir, "FG " + std::to_string(w) + "x" + std::to_string(h) + ": convolutions on " + (c->fg->lrNet.usedWmma ? "matrix units" : "scalar kernels"));
    c->fgW = w; c->fgH = h; c->realFmt = bbFormat; c->fgFrames = 0; return 0;
}

// runs the FG pipeline on (current colour already in fgColor); result in fgOut
static void fgRun(anf4r_context* c, VkCommandBuffer cb, const anf4r_image& depth, const anf4r_image& motion, float msx, float msy, bool depthRev, bool reset) {
    Texture dep = view(c, depth), mv = view(c, motion), dt = dep, mt = mv;
    dt.layout = depth.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : depth.layout; mt.layout = motion.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : motion.layout;
    VkImageLayout dl = dt.layout, ml = mt.layout;
    c->ctx.toLayout(cb, dt, VK_IMAGE_LAYOUT_GENERAL); c->ctx.toLayout(cb, mt, VK_IMAGE_LAYOUT_GENERAL);
    struct { float sx, sy, rw, rh; int32_t ow, oh; } pc{msx, msy, (float)depth.width, (float)depth.height, c->fgW, c->fgH};
    c->resample.bindImage(0, dt.view); c->resample.bindImage(1, mt.view); c->resample.bindImage(2, c->fgDepth.view); c->resample.bindImage(3, c->fgMotion.view);
    c->resample.record(c->ctx, cb, c->fgW, c->fgH, 1, &pc, sizeof pc); c->ctx.barrierAll(cb);
    c->ctx.toLayout(cb, dt, dl); c->ctx.toLayout(cb, mt, ml);
    if (reset) c->fg->reset();
    c->fg->setDepthInverted(!depthRev);
    c->fg->record(c->ctx, cb, c->fgColor, c->fgDepth, c->fgMotion, c->fgOut);
}

int anf4r_interpolate_init(anf4r_context* c, uint32_t w, uint32_t h, uint32_t, uint32_t) { c->error.clear(); return ensureFg(c, w, h, VK_FORMAT_R8G8B8A8_UNORM); }

int anf4r_interpolate(anf4r_context* c, VkCommandBuffer cb, const anf4r_interpolate_desc* d) {
    c->error.clear();
    if (int r = ensureFg(c, d->color.width, d->color.height, VK_FORMAT_R8G8B8A8_UNORM)) return -r;
    xfer(c, cb, ext(d->color), true, c->fgColor, false, true);
    fgRun(c, cb, d->depth, d->motion, d->motion_scale_x, d->motion_scale_y, d->depth_reversed != 0, d->reset != 0);
    xfer(c, cb, c->fgOut, false, ext(d->output), true, true);
    return 0;
}

int anf4r_framegen_capture(anf4r_context* c, VkCommandBuffer cb, const anf4r_framegen_desc* d) {
    c->error.clear(); c->generated = false;
    const anf4r_image& bb = d->backbuffer;
    if (int r = ensureFg(c, bb.width, bb.height, bb.format)) return -r;
    Texture bbt = ext(bb);
    xfer(c, cb, bbt, true, c->real, false, false);                    // keep the real frame (native format) to put back
    xfer(c, cb, bbt, true, c->fgColor, false, true);                  // current colour for the network
    fgRun(c, cb, d->depth, d->motion, d->motion_scale_x, d->motion_scale_y, d->depth_reversed != 0, d->reset != 0);
    bool first = c->fgFrames++ == 0 || d->reset;
    if (first) return 0;                                              // no previous frame yet: nothing generated
    xfer(c, cb, c->fgOut, false, bbt, true, true);
    c->generated = true; return 1;
}

int anf4r_framegen_restore(anf4r_context* c, VkCommandBuffer cb, const anf4r_image* backbuffer) {
    c->error.clear();
    if (!c->fg || c->real.image == VK_NULL_HANDLE) { c->error = "restore without capture"; return -1; }
    xfer(c, cb, c->real, false, ext(*backbuffer), true, false);
    return 0;
}
