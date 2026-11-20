// anf4r: FG pipeline wiring. GPL-3.0-only
#include "fg.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace anf4r {

static SpecConst S(uint32_t id, int v) { return {id, v}; }
static SpecConst SF(uint32_t id, float f) { int32_t v; memcpy(&v, &f, 4); return {id, v}; }
static int up4(int v) { return (v + 3) & ~3; }

bool FrameGen::init(Context& ctx, const FgConfig& cfg) {
    cfg_ = cfg; std::string err;
    sz.fullW = cfg.width; sz.fullH = cfg.height;
    sz.preW = (cfg.width + 1) / 2; sz.preH = (cfg.height + 1) / 2;       // PREPROCESS res = FULL/2 (ASSUMED; see divergences D5)
    sz.qrW = up4(sz.preW); sz.qrH = up4(sz.preH);                        // QR grid, padded to x4 so the nets' D2S dims line up
    sz.lrW = up4((sz.qrW + 3) / 4); sz.lrH = up4((sz.qrH + 3) / 4);      // LR = QR/4 (pyramid shader: 4x4 QR texels per LR texel)
    if (!lrModel_.load(cfg.modelDir + "/aiff-frc-lr-v2-prod-8", true, err) || !qrModel_.load(cfg.modelDir + "/aiff-frc-qr-v2-prod-8", false, err)) { ctx.error = err; return false; }
    if (!lrNet.build(ctx, lrModel_, sz.lrW, sz.lrH, cfg.shaderDir)) return false;
    if (!qrNet.build(ctx, qrModel_, sz.qrW, sz.qrH, cfg.shaderDir)) return false;
    sz.lrNet = lrNet.dims; sz.qrNet = qrNet.dims;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
    vkCreateSampler(ctx.device, &si, nullptr, &samp_);

    // ---- resources ----
    auto tex = [&](Texture& t, VkFormat f, int w, int h) { return ctx.createTexture(t, f, w, h); };
    bool ok = true;
    ok &= tex(dilatedMV, VK_FORMAT_R16G16_SFLOAT, sz.preW, sz.preH);
    ok &= tex(fwdMV, VK_FORMAT_R16G16_SINT, sz.preW, sz.preH); ok &= tex(bwdMV, VK_FORMAT_R16G16_SINT, sz.preW, sz.preH);
    ok &= tex(depthRes, VK_FORMAT_R8_UINT, sz.preW, sz.preH); ok &= tex(maskRes, VK_FORMAT_R8_UINT, sz.preW, sz.preH);
    for (int i = 0; i < 2; ++i) {
        ok &= tex(qrYUV[i], VK_FORMAT_R8G8B8A8_UNORM, sz.qrW, sz.qrH); ok &= tex(lrYUV[i], VK_FORMAT_R8G8B8A8_UNORM, sz.lrW, sz.lrH);
        ok &= tex(depthLR[i], VK_FORMAT_R8_UNORM, sz.lrW, sz.lrH); ok &= tex(depthPyr[i], VK_FORMAT_R8_UNORM, sz.qrW, sz.qrH);
    }
    ok &= tex(prior, VK_FORMAT_R8G8B8A8_UNORM, cfg.width, cfg.height);
    ok &= tex(lrFlow, VK_FORMAT_R16G16B16A16_SFLOAT, sz.lrNet.flowW, sz.lrNet.flowH); ok &= tex(lrMask, VK_FORMAT_R32_SFLOAT, sz.lrNet.nw, sz.lrNet.nh);
    ok &= tex(qrFlow, VK_FORMAT_R16G16B16A16_SFLOAT, sz.qrNet.flowW, sz.qrNet.flowH); ok &= tex(qrMask, VK_FORMAT_R16_SFLOAT, sz.qrNet.nw, sz.qrNet.nh);
    ok &= ctx.createBuffer(netInLR, (size_t)sz.lrW * sz.lrH * 32, false) && ctx.createBuffer(netInQR, (size_t)sz.qrW * sz.qrH * 32, false);
    ok &= ctx.createBuffer(packedX, (size_t)sz.preW * sz.preH * 4, false) && ctx.createBuffer(packedY, (size_t)sz.preW * sz.preH * 4, false);
    ok &= ctx.createBuffer(depthStats, 16, false) && ctx.createBuffer(fallback, 16, false);
    if (!ok) { ctx.error = "FG resource allocation failed"; return false; }
    { VkCommandBuffer cb = ctx.begin();
      for (Texture* t : {&dilatedMV, &fwdMV, &bwdMV, &depthRes, &maskRes, &prior, &lrFlow, &lrMask, &qrFlow, &qrMask}) ctx.toLayout(cb, *t, VK_IMAGE_LAYOUT_GENERAL);
      for (int i = 0; i < 2; ++i) for (Texture* t : {&qrYUV[i], &lrYUV[i], &depthLR[i], &depthPyr[i]}) ctx.toLayout(cb, *t, VK_IMAGE_LAYOUT_GENERAL);
      ctx.submit(cb); }

    // ---- vendor passes (SPIR-V unchanged) ----
    auto sp = [&](const char* n) { return readSpirv(cfg.spirvDir + "/" + n + "_cs.bin"); };
    auto B = [](uint32_t b, BindKind k, const char* n) { return BindDesc{b, k, n}; };
    const BindKind SI = BindKind::SampledImage, ST = BindKind::StorageImage, SB = BindKind::StorageBuffer, SM = BindKind::Sampler;
    auto build = [&](Pass& p, const char* n, std::vector<BindDesc> b, std::vector<SpecConst> s, uint32_t push) {
        auto code = sp(n); if (code.empty()) { ctx.error = std::string("missing spirv ") + n; return false; }
        if (!p.build(ctx, code, b, s, push, 16, 16, 1, "main_cs")) { ctx.error = std::string(n) + ": " + ctx.error; return false; }
        if (p.hasBinding(128)) p.bindSampler(128, samp_);
        return true;
    };
    const int PW = sz.preW, PH = sz.preH, FW = sz.fullW, FH = sz.fullH, QW = sz.qrW, QH = sz.qrH, LW = sz.lrW, LH = sz.lrH;
    ok = build(preprocess_, "AIFFPreprocessFusedV2",
        {B(0,SI,"texRawMotion"),B(1,SI,"texRawDepth"),B(384,ST,"outputDepthPyramidQR"),B(385,ST,"outputDilatedMV"),B(386,SB,"px"),B(387,SB,"py"),B(388,SB,"depthStats")},
        {S(0,FW),S(1,FH),S(2,PW),S(3,PH),S(12,QW),S(13,QH)}, 24)
      && build(unpackMV_, "AIFFUnpackMVV2",
        {B(0,SI,"texCollocatedMVProj"),B(1,SB,"packed_x"),B(2,SB,"packed_y"),B(384,ST,"fwd"),B(385,ST,"bwd"),B(386,ST,"depth_res"),B(387,ST,"mask_res")},
        {S(2,PW),S(3,PH)}, 0)
      && build(pyramid_, "AIFFPyramidDownscaleFusedV2",
        {B(0,SI,"inputTexture"),B(1,SI,"texDepthPyramidQR"),B(128,SM,"samplerLinear"),B(384,ST,"outputQRYUV"),B(385,ST,"outputLRYUV"),B(386,ST,"outputDepthLR")},
        {S(0,FW),S(1,FH),S(2,PW),S(3,PH),S(6,LW),S(7,LH),S(12,QW),S(13,QH)}, 0)
      && build(metaLR_, "AIFFMetaAndLRPackV2",
        {B(0,SI,"texSrcDepth"),B(1,SI,"texSrcMask"),B(2,SI,"texSrcMVFwd"),B(3,SI,"texSrcMVBwd"),B(4,SI,"texLRPrevYUV"),B(5,SI,"texLRCurrYUV"),B(6,SI,"texDepthLRPrev"),B(7,SI,"texDepthLRCurr"),B(128,SM,"samplerLinear"),B(384,SB,"bufNetworkInputLR"),B(385,SB,"bufFallbackStats")},
        {S(2,PW),S(3,PH),S(6,LW),S(7,LH),SF(22,lrModel_.enc.inScale),SF(23,lrModel_.enc.inOffset)}, 4)
      && build(lrUnpack_, "AIFFLRUnpackV2CL",
        {B(0,SB,"bufFlowLR"),B(1,SB,"bufMaskLR"),B(384,ST,"texLRFlowUnpacked"),B(385,ST,"texLRMaskUnpacked")},
        {S(8,sz.lrNet.flowW),S(9,sz.lrNet.flowH),S(10,sz.lrNet.nw),S(11,sz.lrNet.nh)}, 0)
      && build(upscaleQR_, "AIFFLRUpscaleAndQRPackV2",
        {B(0,SI,"texPrevYUV"),B(1,SI,"texCurrYUV"),B(2,SI,"texHoleMask"),B(3,SI,"texMVFwd"),B(4,SI,"texMVBwd"),B(5,SI,"texDepth"),B(6,SI,"texLRFlowUnpacked"),B(7,SI,"texLRMaskUnpacked"),B(8,SI,"texDepthPyramidQRPrev"),B(9,SI,"texDepthPyramidQRCurr"),B(128,SM,"samplerLinear"),B(384,SB,"bufNetworkInputQR")},
        {S(2,PW),S(3,PH),S(12,QW),S(13,QH),SF(28,qrModel_.enc.inScale),SF(29,qrModel_.enc.inOffset)}, 4)
      && build(qrUnpack_, "AIFFQRUnpackV2",
        {B(0,SI,"texLRFlowUnpacked"),B(1,SI,"texLRMaskUnpacked"),B(2,SB,"bufQRFlow"),B(3,SB,"bufQRMask"),B(128,SM,"samplerLinear"),B(384,ST,"texQRFlowUnpacked"),B(385,ST,"texQRMaskUnpacked")},
        {S(8,sz.lrNet.flowW),S(9,sz.lrNet.flowH),S(10,sz.lrNet.nw),S(11,sz.lrNet.nh),S(14,sz.qrNet.flowW),S(16,sz.qrNet.flowW),S(17,sz.qrNet.flowH),
         S(18,sz.qrNet.nw),S(20,sz.qrNet.nw),S(21,sz.qrNet.nh),
         SF(30,qrModel_.enc.flowScale),SF(31,qrModel_.enc.flowOffset),SF(32,qrModel_.enc.maskScale),SF(33,qrModel_.enc.maskOffset)}, 4)
      && build(blend_, "AIFFQRBlendV2FINT",
        {B(0,SI,"texPriorFrame"),B(1,SI,"texCurrFrame"),B(2,SI,"texQRFlowUnpacked"),B(3,SI,"texQRMaskUnpacked"),B(4,SB,"bufFallbackStats"),B(128,SM,"samplerLinear"),B(384,ST,"texOutput")},
        {S(0,FW),S(1,FH),S(6,LW),S(7,LH),S(16,sz.qrNet.flowW),S(17,sz.qrNet.flowH),S(20,sz.qrNet.nw),S(21,sz.qrNet.nh)}, 4);
    return ok;
}

bool FrameGen::record(Context& ctx, VkCommandBuffer cb, const Texture& color, const Texture& depth, const Texture& motion, const Texture& output) {
    const int c = frame_ & 1, p = c ^ 1; const bool first = frame_ == 0;
    const int PW = sz.preW, PH = sz.preH;
    ctx.barrierAll(cb);   // order against the previous frame's commands (submission order)
    // per-frame state
    vkCmdFillBuffer(cb, packedX.buffer, 0, VK_WHOLE_SIZE, 0); vkCmdFillBuffer(cb, packedY.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cb, fallback.buffer, 0, VK_WHOLE_SIZE, 0);
    // depthStats: [0..1] = previous frame's min/max (read by the norm-depth path), [2..3] = this frame's accumulators (min=+inf bits, max=0)
    uint32_t ds[4] = {0, 0x3f800000u, 0x7f800000u, 0};
    if (frame_ == 0) vkCmdUpdateBuffer(cb, depthStats.buffer, 0, 16, ds); else vkCmdUpdateBuffer(cb, depthStats.buffer, 8, 8, ds + 2);
    ctx.barrierAll(cb);

    // 1. PreprocessFused (grid = QR)
    struct { float jp[2], jc[2]; uint32_t norm, inv; } pc1{{cfg_.jitterPrev[0], cfg_.jitterPrev[1]}, {cfg_.jitterCurr[0], cfg_.jitterCurr[1]}, cfg_.normalizeDepth ? 1u : 0u, cfg_.invertDepth ? 1u : 0u};
    preprocess_.bindImage(0, motion.view); preprocess_.bindImage(1, depth.view); preprocess_.bindImage(384, depthPyr[c].view); preprocess_.bindImage(385, dilatedMV.view);
    preprocess_.bindBuffer(386, packedX.buffer); preprocess_.bindBuffer(387, packedY.buffer); preprocess_.bindBuffer(388, depthStats.buffer);
    preprocess_.record(ctx, cb, sz.qrW, sz.qrH, 1, &pc1, sizeof pc1); ctx.barrierAll(cb);
    // 2. UnpackMV (grid = PREPROCESS)
    unpackMV_.bindImage(0, dilatedMV.view); unpackMV_.bindBuffer(1, packedX.buffer); unpackMV_.bindBuffer(2, packedY.buffer);
    unpackMV_.bindImage(384, fwdMV.view); unpackMV_.bindImage(385, bwdMV.view); unpackMV_.bindImage(386, depthRes.view); unpackMV_.bindImage(387, maskRes.view);
    unpackMV_.record(ctx, cb, PW, PH, 1, nullptr, 0); ctx.barrierAll(cb);
    // 3. PyramidDownscale (grid = QR): colour -> QR/LR YUV + LR depth
    pyramid_.bindImage(0, color.view); pyramid_.bindImage(1, depthPyr[c].view);
    pyramid_.bindImage(384, qrYUV[c].view); pyramid_.bindImage(385, lrYUV[c].view); pyramid_.bindImage(386, depthLR[c].view);
    pyramid_.record(ctx, cb, sz.qrW, sz.qrH, 1, nullptr, 0); ctx.barrierAll(cb);
    // history for the very first frame: previous := current
    if (first) {
        auto cp = [&](Texture& s, Texture& d) { VkImageCopy r{}; r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.extent = {s.width, s.height, 1};
            vkCmdCopyImage(cb, s.image, VK_IMAGE_LAYOUT_GENERAL, d.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r); };
        cp(qrYUV[c], qrYUV[p]); cp(lrYUV[c], lrYUV[p]); cp(depthLR[c], depthLR[p]); cp(depthPyr[c], depthPyr[p]);
        cp(const_cast<Texture&>(color), prior); ctx.barrierAll(cb);
    }
    // 4. MetaAndLRPack (grid = LR): network input (signed int8: modeBias 0)
    int modeSigned = 0, modeUnsigned = 128;
    metaLR_.bindImage(0, depthRes.view); metaLR_.bindImage(1, maskRes.view); metaLR_.bindImage(2, fwdMV.view); metaLR_.bindImage(3, bwdMV.view);
    metaLR_.bindImage(4, lrYUV[p].view); metaLR_.bindImage(5, lrYUV[c].view); metaLR_.bindImage(6, depthLR[p].view); metaLR_.bindImage(7, depthLR[c].view);
    metaLR_.bindBuffer(384, netInLR.buffer); metaLR_.bindBuffer(385, fallback.buffer);
    metaLR_.record(ctx, cb, sz.lrW, sz.lrH, 1, &modeSigned, 4); ctx.barrierAll(cb);
    // 5. LR network, then LRUnpack (CL variant: fp16 in)
    lrNet.record(ctx, cb, netInLR.buffer);
    lrUnpack_.bindBuffer(0, lrNet.flow.buffer); lrUnpack_.bindBuffer(1, lrNet.mask.buffer); lrUnpack_.bindImage(384, lrFlow.view); lrUnpack_.bindImage(385, lrMask.view);
    lrUnpack_.record(ctx, cb, sz.lrNet.flowW, sz.lrNet.flowH, 1, nullptr, 0); ctx.barrierAll(cb);
    // 6. LRUpscaleAndQRPack (grid = QR)
    upscaleQR_.bindImage(0, qrYUV[p].view); upscaleQR_.bindImage(1, qrYUV[c].view); upscaleQR_.bindImage(2, maskRes.view);
    upscaleQR_.bindImage(3, fwdMV.view); upscaleQR_.bindImage(4, bwdMV.view); upscaleQR_.bindImage(5, depthRes.view);
    upscaleQR_.bindImage(6, lrFlow.view); upscaleQR_.bindImage(7, lrMask.view); upscaleQR_.bindImage(8, depthPyr[p].view); upscaleQR_.bindImage(9, depthPyr[c].view);
    upscaleQR_.bindBuffer(384, netInQR.buffer);
    upscaleQR_.record(ctx, cb, sz.qrW, sz.qrH, 1, &modeSigned, 4); ctx.barrierAll(cb);
    // 7. QR network, QRUnpack (uint8 outputs: modeBias 128 -> no shift)
    qrNet.record(ctx, cb, netInQR.buffer);
    if (getenv("ANF_QR_ZERO")) {   // debug: neutralise the QR residual (u8 == zero-point) to isolate the LR flow
        uint32_t zf = (uint32_t)qrModel_.enc.flowOffset * 0x01010101u, zm = (uint32_t)qrModel_.enc.maskOffset * 0x01010101u;
        vkCmdFillBuffer(cb, qrNet.flow.buffer, 0, VK_WHOLE_SIZE, zf); vkCmdFillBuffer(cb, qrNet.mask.buffer, 0, VK_WHOLE_SIZE, zm); ctx.barrierAll(cb);
    }
    qrUnpack_.bindImage(0, lrFlow.view); qrUnpack_.bindImage(1, lrMask.view); qrUnpack_.bindBuffer(2, qrNet.flow.buffer); qrUnpack_.bindBuffer(3, qrNet.mask.buffer);
    qrUnpack_.bindImage(384, qrFlow.view); qrUnpack_.bindImage(385, qrMask.view);
    qrUnpack_.record(ctx, cb, sz.qrNet.flowW, sz.qrNet.flowH, 1, &modeUnsigned, 4); ctx.barrierAll(cb);
    // 8. QRBlend (grid = FULL)
    blend_.bindImage(0, prior.view); blend_.bindImage(1, color.view); blend_.bindImage(2, qrFlow.view); blend_.bindImage(3, qrMask.view);
    blend_.bindBuffer(4, fallback.buffer); blend_.bindImage(384, output.view);
    blend_.record(ctx, cb, sz.fullW, sz.fullH, 1, &cfg_.discontinuityThresholdSq, 4); ctx.barrierAll(cb);
    // history: this colour becomes next frame's prior
    { VkImageCopy r{}; r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.extent = {color.width, color.height, 1};
      vkCmdCopyImage(cb, color.image, VK_IMAGE_LAYOUT_GENERAL, prior.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r); ctx.barrierAll(cb); }
    { VkBufferCopy bc{8, 0, 8}; vkCmdCopyBuffer(cb, depthStats.buffer, depthStats.buffer, 1, &bc); ctx.barrierAll(cb); }   // this frame's min/max -> next frame's normalisation
    ++frame_;
    return true;
}

void FrameGen::destroy(Context& ctx) {
    for (Pass* p : {&preprocess_, &unpackMV_, &pyramid_, &metaLR_, &lrUnpack_, &upscaleQR_, &qrUnpack_, &blend_}) p->destroy(ctx);
    for (Texture* t : {&dilatedMV, &fwdMV, &bwdMV, &depthRes, &maskRes, &lrFlow, &lrMask, &qrFlow, &qrMask, &prior}) ctx.destroyTexture(*t);
    for (int i = 0; i < 2; ++i) for (Texture* t : {&qrYUV[i], &lrYUV[i], &depthLR[i], &depthPyr[i]}) ctx.destroyTexture(*t);
    for (Buffer* b : {&netInLR, &netInQR, &packedX, &packedY, &depthStats, &fallback}) ctx.destroyBuffer(*b);
    lrNet.destroy(ctx); qrNet.destroy(ctx);
    if (samp_) vkDestroySampler(ctx.device, samp_, nullptr);
}

} // namespace anf4r
