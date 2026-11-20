// anf4r: SR pipeline. GPL-3.0-only
#include "sr.h"
#include <cmath>
#include <cstring>

namespace anf4r {

static SpecConst S(uint32_t id, int v) { return {id, v}; }
static SpecConst SF(uint32_t id, float f) { int32_t v; memcpy(&v, &f, 4); return {id, v}; }

static bool loadLayer(const std::string& dir, const char* name, int ic, int oc, int icw, std::vector<int8_t>& w, std::vector<float>& scale, std::vector<float>& biasf, std::string& err) {
    char wn[320], bn[320], sn[320];
    snprintf(wn, 320, "%s/%s.weight_shape-%d-%d-3-3_int8.raw", dir.c_str(), name, oc, ic);
    snprintf(bn, 320, "%s/%s.bias_shape-%d_int32.raw", dir.c_str(), name, oc);
    snprintf(sn, 320, "%s/%s.scales.json", dir.c_str(), name);
    auto wr = readAllBytes(wn), br = readAllBytes(bn); auto ws = jsonNumbers(readTextFile(sn), "weights_scales");
    if (wr.size() != (size_t)oc * ic * 9 || br.size() != (size_t)oc * 4 || (int)ws.size() != oc) { err = std::string("bad layer files in ") + dir + "/" + name; return false; }
    w.assign((size_t)oc * 9 * icw, 0);   // repack [oc][ic][3][3] -> [oc][ky][kx][icw] (channels zero padded to icw)
    for (int o = 0; o < oc; ++o) for (int c = 0; c < ic; ++c) for (int k = 0; k < 9; ++k) w[((size_t)o * 9 + k) * icw + c] = (int8_t)wr[((size_t)o * ic + c) * 9 + k];
    scale = ws; biasf.resize(oc);
    for (int o = 0; o < oc; ++o) { int32_t b; memcpy(&b, &br[o * 4], 4); biasf[o] = (float)b * scale[o]; }   // bias is in accumulator units
    return true;
}
static void up(Context& c, Buffer& b, const void* d, size_t n) { c.createBuffer(b, n, true); memcpy(b.map, d, n); }

int SuperRes::jitterPhase(float jx, float jy, float* qx, float* qy) const {
    // ASSUMED: 8x8 phase grid over [-0.5,0.5]^2 (64 weight sets per layer), index = iy*8 + ix.
    int ix = std::min(7, std::max(0, (int)std::floor((jx + 0.5f) * 8.f))), iy = std::min(7, std::max(0, (int)std::floor((jy + 0.5f) * 8.f)));
    if (qx) *qx = (ix + 0.5f) / 8.f - 0.5f; if (qy) *qy = (iy + 0.5f) / 8.f - 0.5f;
    static int mode = getenv("ANF_PHASE_MODE") ? atoi(getenv("ANF_PHASE_MODE")) : 0;   // debug: alternative index layouts
    if (mode >= 8) {   // debug: weight sets placed on a Halton(2,3) sequence, nearest set to the requested jitter
        auto halton = [](int i, int b) { float f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; };
        int off = (mode == 8 || mode == 10) ? 1 : 0; bool swap = mode >= 10; int best = 0; float bd = 1e9f, bx = 0, by = 0;
        for (int i = 0; i < 64; ++i) { float hx = halton(i + off, swap ? 3 : 2) - 0.5f, hy = halton(i + off, swap ? 2 : 3) - 0.5f; float d = (hx - jx) * (hx - jx) + (hy - jy) * (hy - jy); if (d < bd) { bd = d; best = i; bx = hx; by = hy; } }
        if (qx) *qx = bx; if (qy) *qy = by; return best;
    }
    int fx = 7 - ix, fy = 7 - iy;
    switch (mode) { case 1: return ix * 8 + iy; case 2: return iy * 8 + fx; case 3: return fy * 8 + ix; case 4: return fy * 8 + fx;
                    case 5: return ix * 8 + fy; case 6: return fx * 8 + iy; case 7: return fx * 8 + fy; default: return iy * 8 + ix; }
}

bool SuperRes::init(Context& ctx, const SrConfig& cfg) {
    cfg_ = cfg; if (getenv("ANF_STAB")) cfg_.stabilize = (float)atof(getenv("ANF_STAB"));
    if (getenv("ANF_PREEXP")) cfg_.preExposure = (float)atof(getenv("ANF_PREEXP"));
    mvScale[0] = (float)cfg.lrW; mvScale[1] = (float)cfg.lrH; hrW = cfg.lrW * 2; hrH = cfg.lrH * 2; netW = (cfg.lrW + 1) / 2; netH = (cfg.lrH + 1) / 2;
    const SrLayer L[4] = {{"cnn1", 20, 32, 2}, {"cnn2-0", 32, 32, 1}, {"cnn2-1", 32, 32, 1}, {"cnn3", 32, 64, 1}};
    std::string err;
    for (int t = 0; t < 2; ++t) {   // per-jitter layers: cnn1 (t=0) and cnn3 (t=1)
        const SrLayer& l = L[t ? 3 : 0]; phaseW_[t].resize(64);
        for (int ph = 0; ph < 64; ++ph) {
            std::vector<int8_t> w; std::vector<float> s, b;
            if (!loadLayer(cfg.modelDir + "/" + l.name + "/" + std::to_string(ph), l.name.c_str(), l.ic, l.oc, l.ic == 20 ? 32 : l.ic, w, s, b, err)) { ctx.error = err; return false; }
            up(ctx, phaseW_[t][ph].w, w.data(), w.size()); up(ctx, phaseW_[t][ph].b, b.data(), b.size() * 4); up(ctx, phaseW_[t][ph].s, s.data(), s.size() * 4);
        }
    }
    for (int i = 0; i < 2; ++i) {   // shared layers (single weight set in dir 0)
        const SrLayer& l = L[1 + i]; std::vector<int8_t> w; std::vector<float> s, b;
        if (!loadLayer(cfg.modelDir + "/" + l.name + "/0", l.name.c_str(), l.ic, l.oc, l.ic == 20 ? 32 : l.ic, w, s, b, err)) { ctx.error = err; return false; }
        up(ctx, shared_[i].w, w.data(), w.size()); up(ctx, shared_[i].b, b.data(), b.size() * 4); up(ctx, shared_[i].s, s.data(), s.size() * 4);
    }
    bool ok = ctx.createBuffer(modelIn, (size_t)cfg.lrW * cfg.lrH * 32, false);
    act.resize(4);
    for (int i = 0; i < 4; ++i) ok &= ctx.createBuffer(act[i], (size_t)netW * netH * L[i].oc, false);
    ok &= ctx.createTexture(colorLR, VK_FORMAT_R16G16B16A16_SFLOAT, cfg.lrW, cfg.lrH) && ctx.createTexture(prevWarped, VK_FORMAT_R32G32_SFLOAT, cfg.lrW, cfg.lrH);
    if (cfg_.stabilize > 0) for (int i = 0; i < 2; ++i) ok &= ctx.createTexture(stabHist[i], VK_FORMAT_R16G16B16A16_SFLOAT, hrW, hrH);
    for (int i = 0; i < 2; ++i) ok &= ctx.createTexture(histColor[i], VK_FORMAT_R16G16B16A16_SFLOAT, hrW, hrH) && ctx.createTexture(histFeat[i], VK_FORMAT_R32_SFLOAT, hrW, hrH);
    if (!ok) { ctx.error = "SR allocation failed"; return false; }
    { VkCommandBuffer cb = ctx.begin(); for (Texture* t : {&colorLR, &prevWarped, &histColor[0], &histColor[1], &histFeat[0], &histFeat[1], &stabHist[0], &stabHist[1]}) if (t->image) ctx.toLayout(cb, *t, VK_IMAGE_LAYOUT_GENERAL); ctx.submit(cb); }

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; si.maxLod = 0;
    vkCreateSampler(ctx.device, &si, nullptr, &sBorder_);
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; vkCreateSampler(ctx.device, &si, nullptr, &sEdge_);

    auto B = [](uint32_t b, BindKind k, const char* n) { return BindDesc{b, k, n}; };
    const BindKind SI = BindKind::SampledImage, ST = BindKind::StorageImage, SB = BindKind::StorageBuffer, SM = BindKind::Sampler;
    auto pre = readSpirv(cfg.shaderDir + "/sr_pre.spv"), post = readSpirv(cfg.shaderDir + "/sr_post.spv"), cv = readSpirv(cfg.shaderDir + "/fg_conv_i8.spv"), wmma = readSpirv(cfg.shaderDir + "/conv_wmma.spv");
    if (pre.empty() || post.empty() || cv.empty()) { ctx.error = "missing SR spv in " + cfg.shaderDir; return false; }
    if (!pre_.build(ctx, pre, {B(0,SI,"c"),B(1,SI,"d"),B(2,SI,"mv"),B(3,SI,"pc"),B(4,SI,"pf"),B(5,SM,"s"),B(6,SB,"mi"),B(7,ST,"clr"),B(8,ST,"mv2")},
                    {S(0,cfg.lrW),S(1,cfg.lrH)}, 24, 8, 8, 1, "main")) return false;
    pre_.bindSampler(5, sBorder_); pre_.label = "sr_pre";
    if (!post_.build(ctx, post, {B(0,SI,"c"),B(1,SI,"mv2"),B(2,SB,"net"),B(3,SM,"s"),B(4,ST,"hist"),B(5,ST,"feat"),B(6,ST,"out"),B(7,SI,"pc"),B(8,SM,"sb")},
                     {S(0,cfg.lrW),S(1,cfg.lrH),S(2,netW),S(4,cfg_.stabilize > 0 ? 1 : 0)}, 20, 8, 8, 1, "main")) return false;
    post_.bindSampler(3, sEdge_); post_.bindSampler(8, sBorder_); post_.label = "sr_post";
    if (cfg_.stabilize > 0) {
        auto stab = readSpirv(cfg.shaderDir + "/sr_stab.spv");
        if (stab.empty()) { ctx.error = "missing sr_stab.spv"; return false; }
        if (!stab_.build(ctx, stab, {B(0,SI,"cur"),B(1,SI,"mv"),B(2,SI,"prev"),B(3,SM,"s"),B(4,ST,"stab"),B(5,ST,"out")}, {}, 16, 8, 8, 1, "main")) return false;
        stab_.bindSampler(3, sBorder_); stab_.label = "sr_stab";
    }
    int cw = cfg.lrW, ch = cfg.lrH;
    for (int i = 0; i < 4; ++i) {
        int ow = (cw - 1) / L[i].stride + 1, oh = (ch - 1) / L[i].stride + 1;
        const int icw = L[i].ic == 20 ? 32 : L[i].ic;
        std::vector<BindDesc> cb6 = {B(0,SB,"in"),B(1,SB,"w"),B(2,SB,"b"),B(3,SB,"s"),B(4,SB,"o"),B(5,SB,"o8")};
        bool wm = !wmma.empty() && !getenv("ANF_NO_WMMA");
        if (wm) {
            conv_[i].requiredSubgroup = 32;
            std::vector<BindDesc> cb8 = cb6; cb8.push_back(B(6,SB,"f")); cb8.push_back(B(7,SB,"m"));
            if (!conv_[i].build(ctx, wmma, cb8, {S(0,cw),S(1,ch),S(2,icw),S(3,icw),S(4,L[i].oc),S(5,L[i].stride),S(6,ow),S(7,oh),S(8,0),S(9,-128),S(10,127),S(11,-128),S(12,0)}, 0, 32, 1, 1, "main")) { wm = false; ctx.error.clear(); conv_[i] = Pass(); }
        }
        convWmma_[i] = wm;
        if (!wm && !conv_[i].build(ctx, cv, cb6, {S(0,cw),S(1,ch),S(2,icw),S(3,L[i].oc),S(4,L[i].stride),S(5,ow),S(6,oh),S(7,0),S(8,-128),S(9,127),S(10,-128),S(11,0),S(12,icw)}, 0, 8, 8, 1, "main")) return false;
        conv_[i].label = std::string("sr_") + L[i].name;
        conv_[i].bindBuffer(4, act[i].buffer); conv_[i].bindBuffer(5, act[i].buffer);
        if (convWmma_[i]) { conv_[i].bindBuffer(6, act[i].buffer); conv_[i].bindBuffer(7, act[i].buffer); }
        cw = ow; ch = oh;
    }
    return true;
}

bool SuperRes::record(Context& ctx, VkCommandBuffer cb, const Texture& color, const Texture& depth, const Texture& motion, const Texture& output, float jx, float jy) {
    const int c = frame_ & 1, p = c ^ 1;
    ctx.barrierAll(cb);   // order against the previous frame's commands (submission order)
    if (needClear_) {
        VkClearColorValue z{}; VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (Texture* t : {&histColor[0], &histColor[1], &histFeat[0], &histFeat[1], &stabHist[0], &stabHist[1]}) if (t->image) vkCmdClearColorImage(cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
        ctx.barrierAll(cb); needClear_ = false; frame_ = 0;
    }
    float qx, qy; int ph = jitterPhase(jx, jy, &qx, &qy);
    // 1. preprocess (grid = LR)
    pre_.bindImage(0, color.view); pre_.bindImage(1, depth.view); pre_.bindImage(2, motion.view); pre_.bindImage(3, histColor[p].view); pre_.bindImage(4, histFeat[p].view);
    pre_.bindBuffer(6, modelIn.buffer); pre_.bindImage(7, colorLR.view); pre_.bindImage(8, prevWarped.view);
    struct { float pre, pad0, mvx, mvy; int inv; float pad1; } pcp{cfg_.preExposure, 0.f, mvScale[0], mvScale[1], cfg_.invertDepth ? 1 : 0, 0.f};   // std430 push layout (vec2 @ 8)
    pre_.record(ctx, cb, ((cfg_.lrW + 7) / 8) * 8, ((cfg_.lrH + 7) / 8) * 8, 1, &pcp, 24); ctx.barrierAll(cb);
    // 2. network: cnn1(phase) -> cnn2-0 -> cnn2-1 -> cnn3(phase)
    const LW* lw[4] = {&phaseW_[0][ph], &shared_[0], &shared_[1], &phaseW_[1][ph]};
    const int ocs[4] = {32, 32, 32, 64};
    VkBuffer in = modelIn.buffer; int cw = cfg_.lrW, ch = cfg_.lrH;
    for (int i = 0; i < 4; ++i) {
        int ow = (cw - 1) / (i == 0 ? 2 : 1) + 1, oh = (ch - 1) / (i == 0 ? 2 : 1) + 1;
        conv_[i].bindBuffer(0, in); conv_[i].bindBuffer(1, lw[i]->w.buffer); conv_[i].bindBuffer(2, lw[i]->b.buffer); conv_[i].bindBuffer(3, lw[i]->s.buffer);
        if (convWmma_[i]) conv_[i].record(ctx, cb, ((ow + 31) / 32) * 32, oh, ocs[i] / 32, nullptr, 0); else conv_[i].record(ctx, cb, ow, oh, ocs[i], nullptr, 0); ctx.barrierAll(cb);
        in = act[i].buffer; cw = ow; ch = oh;
    }
    // 3. postprocess (grid = HR)
    struct { float jx, jy; int w, h; float e; } pc{qx, qy, hrW, hrH, cfg_.preExposure};
    post_.bindImage(0, colorLR.view); post_.bindImage(1, prevWarped.view); post_.bindImage(7, histColor[p].view); post_.bindBuffer(2, act[3].buffer);
    post_.bindImage(4, histColor[c].view); post_.bindImage(5, histFeat[c].view); post_.bindImage(6, output.view);
    post_.record(ctx, cb, hrW, hrH, 1, &pc, sizeof pc); ctx.barrierAll(cb);
    if (cfg_.stabilize > 0) {
        struct { int w, h; float beta, e; } sp{hrW, hrH, frame_ == 0 ? 0.0f : cfg_.stabilize, cfg_.preExposure};
        stab_.bindImage(0, histColor[c].view); stab_.bindImage(1, prevWarped.view); stab_.bindImage(2, stabHist[p].view); stab_.bindImage(4, stabHist[c].view); stab_.bindImage(5, output.view);
        stab_.record(ctx, cb, hrW, hrH, 1, &sp, sizeof sp); ctx.barrierAll(cb);
    }
    ++frame_;
    return true;
}

void SuperRes::destroy(Context& ctx) {
    pre_.destroy(ctx); post_.destroy(ctx); stab_.destroy(ctx); for (auto& p : conv_) p.destroy(ctx);
    for (int t = 0; t < 2; ++t) { for (auto& l : phaseW_[t]) { ctx.destroyBuffer(l.w); ctx.destroyBuffer(l.b); ctx.destroyBuffer(l.s); } phaseW_[t].clear(); }
    for (auto& l : shared_) { ctx.destroyBuffer(l.w); ctx.destroyBuffer(l.b); ctx.destroyBuffer(l.s); }
    ctx.destroyBuffer(modelIn); for (auto& b : act) ctx.destroyBuffer(b); act.clear();
    for (Texture* t : {&colorLR, &prevWarped, &histColor[0], &histColor[1], &histFeat[0], &histFeat[1], &stabHist[0], &stabHist[1]}) ctx.destroyTexture(*t);
    if (sBorder_) vkDestroySampler(ctx.device, sBorder_, nullptr); if (sEdge_) vkDestroySampler(ctx.device, sEdge_, nullptr);
}

} // namespace anf4r
