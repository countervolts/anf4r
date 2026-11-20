// anf4r: Frame Generation pipeline. Runs Qualcomm's own 10 SPIR-V passes (unchanged) around our LR/QR conv nets.
// GPL-3.0-only
#pragma once
#include "fgnet.h"

namespace anf4r {

struct FgConfig {
    int width = 0, height = 0;                       // FULL resolution (input color/depth/motion + output)
    std::string spirvDir  = "vendor/assets/spirv";   // vendor AIFF*_cs.bin
    std::string modelDir  = "vendor/assets/modelzoo";// aiff-frc-{lr,qr}-v2-prod-8
    std::string shaderDir = "build";                 // our fg_*.spv
    float discontinuityThresholdSq = 0.0f;           // QRBlendV2FINT push constant (0 = disabled); value used by ANF unknown
    bool invertDepth = false;                        // PreprocessFused push: depth is reverse-Z -> remap (D4)
    bool normalizeDepth = false;
    float jitterPrev[2] = {0, 0}, jitterCurr[2] = {0, 0};
};

struct FgSizes {
    int fullW, fullH, preW, preH, qrW, qrH, lrW, lrH;
    FgNetDims lrNet, qrNet;
};

class FrameGen {
public:
    bool init(Context& ctx, const FgConfig& cfg);
    void destroy(Context& ctx);
    void reset() { frame_ = 0; }
    void setDepthInverted(bool v) { cfg_.invertDepth = v; }
    // Records the full interpolation of (previous frame, this frame) into cb. color: RGBA8, depth: R32F, motion: RG16F,
    // output: RGBA8 storage image. All must be in GENERAL layout; sizes == cfg.width x cfg.height.
    bool record(Context& ctx, VkCommandBuffer cb, const Texture& color, const Texture& depth, const Texture& motion, const Texture& output);
    FgSizes sz{};
    // Intermediates exposed for tests/debug.
    Texture dilatedMV, fwdMV, bwdMV, depthRes, maskRes, lrFlow, lrMask, qrFlow, qrMask;
    Texture qrYUV[2], lrYUV[2], depthLR[2], depthPyr[2], prior;
    Buffer netInLR, netInQR, packedX, packedY, depthStats, fallback;
    FgNet lrNet, qrNet;
private:
    FgConfig cfg_;
    FgNetModel lrModel_, qrModel_;
    Pass preprocess_, unpackMV_, pyramid_, metaLR_, lrUnpack_, upscaleQR_, qrUnpack_, blend_;
    VkSampler samp_ = VK_NULL_HANDLE;
    uint64_t frame_ = 0;
};

} // namespace anf4r
