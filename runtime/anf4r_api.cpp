// anf4r: C API over SuperRes / FrameGen. SPDX-License-Identifier: GPL-3.0-only
#include "vkfn.h"
#include "anf4r.h"
#include "fg.h"
#include "sr.h"
#include <map>
#include <memory>

using namespace anf4r;

struct anf4r_context {
    Context ctx;
    std::string dataDir, error;
    std::unique_ptr<SuperRes> sr; int srW = 0, srH = 0;
    Texture srOut;                                   // internal HR RGBA16F target
    std::map<std::pair<VkImage, VkFormat>, Texture> views;   // image views of the caller's images (cached)
    // frame generation
    std::unique_ptr<FrameGen> fg; int fgW = 0, fgH = 0; uint64_t fgFrames = 0;
    Texture fgColor, fgDepth, fgMotion, fgOut, real; VkFormat realFmt = VK_FORMAT_UNDEFINED;
    Pass resample; bool generated = false;
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
    if (!c->ctx.adopt(inst, phys, dev, family)) { delete c; return -2; }
    *out = c; return 0;
}
void anf4r_destroy(anf4r_context* c) {
    if (!c) return;
    vkDeviceWaitIdle(c->ctx.device);
    if (c->sr) c->sr->destroy(c->ctx);
    if (c->fg) { c->fg->destroy(c->ctx); c->resample.destroy(c->ctx); }
    for (Texture* t : {&c->fgColor, &c->fgDepth, &c->fgMotion, &c->fgOut, &c->real}) c->ctx.destroyTexture(*t);
    c->ctx.destroyTexture(c->srOut);
    for (auto& [k, t] : c->views) c->ctx.destroyTexture(t);
    c->ctx.destroy(); delete c;
}
const char* anf4r_last_error(anf4r_context* c) { return c ? (c->error.empty() ? c->ctx.error.c_str() : c->error.c_str()) : "null context"; }

int anf4r_upscale_init(anf4r_context* c, uint32_t ow, uint32_t oh, uint32_t iw, uint32_t ih) {
    c->error.clear();
    if (ow != 2 * iw || oh != 2 * ih) { c->error = "ANF super resolution is a fixed 2.0x per dimension"; return 1; }
    if (c->sr && c->srW == (int)iw && c->srH == (int)ih) return 0;
    vkDeviceWaitIdle(c->ctx.device);
    if (c->sr) { c->sr->destroy(c->ctx); c->sr.reset(); c->ctx.destroyTexture(c->srOut); }
    SrConfig cfg; cfg.lrW = iw; cfg.lrH = ih; cfg.modelDir = c->dataDir + "/modelzoo/aiff-sr-v2s-prod-7p1-invpow2"; cfg.shaderDir = c->dataDir + "/shaders";
    c->sr = std::make_unique<SuperRes>();
    if (!c->sr->init(c->ctx, cfg)) { c->error = c->ctx.error; c->sr.reset(); return 2; }
    if (!c->ctx.createTexture(c->srOut, VK_FORMAT_R16G16B16A16_SFLOAT, 2 * iw, 2 * ih)) { c->error = "output allocation"; return 3; }
    { VkCommandBuffer cb = c->ctx.begin(); c->ctx.toLayout(cb, c->srOut, VK_IMAGE_LAYOUT_GENERAL); c->ctx.submit(cb); }
    c->srW = iw; c->srH = ih; return 0;
}

int anf4r_upscale(anf4r_context* c, VkCommandBuffer cb, const anf4r_upscale_desc* d) {
    c->error.clear();
    uint32_t iw = d->content_width ? d->content_width : d->color.width, ih = d->content_height ? d->content_height : d->color.height;
    if (!c->sr || c->srW != (int)iw || c->srH != (int)ih) { int r = anf4r_upscale_init(c, 2 * iw, 2 * ih, iw, ih); if (r) return r; }
    if (d->reset) c->sr->reset();
    Texture col = view(c, d->color), dep = view(c, d->depth), mv = view(c, d->motion);
    Texture* tex[3] = {&col, &dep, &mv}; const anf4r_image* im[3] = {&d->color, &d->depth, &d->motion};
    for (int i = 0; i < 3; ++i) toGeneral(c, cb, *im[i], *tex[i], VK_IMAGE_LAYOUT_GENERAL);
    // MV: texture value * scale = pixels (previous - current)  ->  ANF wants (current - previous) in LR pixels
    c->sr->mvScale[0] = -d->motion_scale_x; c->sr->mvScale[1] = -d->motion_scale_y;
    c->sr->setDepthInverted(!d->depth_reversed);
    // desc jitter = sample offset in image coordinates (x right, y down); ANF's jitterOffset has y up
    c->sr->record(c->ctx, cb, col, dep, mv, c->srOut, d->jitter_x, -d->jitter_y);
    // convert/copy the internal RGBA16F result into the caller's output image
    Texture out = view(c, d->output); out.layout = d->output.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : d->output.layout;
    VkImageLayout back = out.layout;
    c->ctx.toLayout(cb, out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit b{}; b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {(int32_t)(2 * iw), (int32_t)(2 * ih), 1}; b.dstOffsets[1] = b.srcOffsets[1];
    vkCmdBlitImage(cb, c->srOut.image, VK_IMAGE_LAYOUT_GENERAL, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
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
