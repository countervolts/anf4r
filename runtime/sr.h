// anf4r: Super Resolution (QMFNet V2, 2x). Reconstructed pipeline around Qualcomm's weights. GPL-3.0-only
#pragma once
#include "fgnet.h"

namespace anf4r {

struct SrConfig {
    int lrW = 0, lrH = 0;                           // input (render) size; output is 2x
    std::string modelDir  = "vendor/assets/modelzoo/aiff-sr-v2s-prod-7p1-invpow2";
    std::string shaderDir = "build";
    float preExposure = 1.0f;                       // tonemap constant (the inverse in sr_post assumes 1.0)
    bool invertDepth = false;                       // true: input depth is forward-Z (smaller = nearer)
};

struct SrLayer { std::string name; int ic, oc, stride; };

class SuperRes {
public:
    bool init(Context& ctx, const SrConfig& cfg);
    void destroy(Context& ctx);
    void reset() { needClear_ = true; }
    void setDepthInverted(bool inv) { cfg_.invertDepth = inv; }
    // color: LR linear HDR (sampled), depth: LR fp32 reverse-Z, motion: LR rg (uv units, jitter-free),
    // output: HR RGBA16F storage image (2*lrW x 2*lrH). All GENERAL layout. jitter in [-0.5,0.5] LR px (ANF convention).
    bool record(Context& ctx, VkCommandBuffer cb, const Texture& color, const Texture& depth, const Texture& motion,
                const Texture& output, float jitterX, float jitterY);
    float mvScale[2] = {0, 0};                      // set before record(): motion texture value -> LR pixels (curr - prev); default = uv units (lrW, lrH)
    int netW = 0, netH = 0, hrW = 0, hrH = 0;
    int jitterPhase(float jx, float jy, float* qx, float* qy) const;
    Buffer modelIn;
    std::vector<Buffer> act;                        // cnn1, cnn2-0, cnn2-1, cnn3 outputs (int8 = uint8-128)
    Texture colorLR, prevWarped, histColor[2], histFeat[2];
private:
    SrConfig cfg_;
    struct LW { Buffer w, b, s; };
    std::vector<LW> phaseW_[2];                     // [0] cnn1 per phase, [1] cnn3 per phase
    LW shared_[2];                                  // cnn2-0, cnn2-1
    Pass pre_, post_, conv_[4]; bool convWmma_[4] = {};
    VkSampler sBorder_ = VK_NULL_HANDLE, sEdge_ = VK_NULL_HANDLE;
    bool needClear_ = true; uint32_t frame_ = 0;
};

} // namespace anf4r
