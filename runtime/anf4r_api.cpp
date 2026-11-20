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

int anf4r_interpolate_init(anf4r_context* c, uint32_t, uint32_t, uint32_t, uint32_t) { c->error = "anf4r: frame generation through this API is not implemented yet"; return 1; }
int anf4r_interpolate(anf4r_context* c, VkCommandBuffer, const anf4r_interpolate_desc*) { c->error = "anf4r: frame generation through this API is not implemented yet"; return -1; }
int anf4r_framegen_capture(anf4r_context* c, VkCommandBuffer, const anf4r_framegen_desc*) { c->error = "anf4r: frame generation through this API is not implemented yet"; return -1; }
int anf4r_framegen_restore(anf4r_context* c, VkCommandBuffer, const anf4r_image*) { c->error = "anf4r: frame generation through this API is not implemented yet"; return -1; }
