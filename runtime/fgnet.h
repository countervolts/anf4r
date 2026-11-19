// anf4r: Frame-Generation LR / QR flow networks (aiff-frc-lr/qr-v2-prod-8) on the GPU + CPU references. GPL-3.0-only
#pragma once
#include "pass.h"
#include <string>
#include <vector>

namespace anf4r {

struct FgLayer {
    std::string name;
    int ic = 0, oc = 0, stride = 1;
    std::vector<int8_t>   w8;      // [oc][ky][kx][ic]  (int8 layers)
    std::vector<uint16_t> w16;     // [oc][ky][kx][ic]  (fp16 LR head)
    std::vector<float>    scale;   // ws*ias/oas per output channel (int8 layers)
    std::vector<float>    biasf;   // float(bias_i32)*scale (int8 layers); fp16 bias widened (LR head)
};

struct FgIoEnc { float inScale = 1, inOffset = 0, flowScale = 1, flowOffset = 0, maskScale = 1, maskOffset = 0; };

struct FgNetModel {
    bool isLR = true;
    std::vector<FgLayer> hidden;   // conv0_0 .. conv5_1 (+ conv_out for both)
    FgLayer head;                  // conv_out_flow_mask
    FgIoEnc enc;
    bool load(const std::string& modelDir, bool isLR, std::string& err);   // modelDir = .../aiff-frc-{lr,qr}-v2-prod-8
};

// Output geometry for an input of (W,H) (stride-2 x2, then depth-to-space x2 for flow).
struct FgNetDims { int nw, nh;  /* net output res = ceil(ceil(W/2)/2) */  int flowW, flowH;  /* 2*nw, 2*nh */ };
FgNetDims fgNetDims(int W, int H);

class FgNet {
public:
    bool build(Context& ctx, const FgNetModel& m, int W, int H, const std::string& spvDir);
    void destroy(Context& ctx);
    void record(Context& ctx, VkCommandBuffer cb, VkBuffer input);   // input: int8 NHWC [H][W][32]
    FgNetDims dims;
    const Buffer& activation(size_t i) const { return act_[i]; }
    Buffer flow, mask;   // LR: fp16 [flowH][flowW][4] / fp16 [nh][nw];  QR: uint8 [flowH][flowW][4] / uint8 [nh][nw]
private:
    const FgNetModel* m_ = nullptr;
    std::vector<Pass> passes_;
    std::vector<Buffer> wbuf_, bbuf_, sbuf_, act_;
    struct Step { int pass, gx, gy, gz; };
    std::vector<Step> steps_;
    std::vector<int> inputOf_;   // activation index feeding each pass (-1 = network input)
    std::vector<VkBuffer> bound_;
};

// CPU references (bit-exact intent). int8 in [H][W][IC] -> int8 out.
std::vector<int8_t> fgConvI8CPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int& OW, int& OH, int lo, int hi);
std::vector<uint16_t> fgConvI8ToHalfCPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int& OW, int& OH);
void fgHeadLRCPU(const FgLayer& L, const std::vector<uint16_t>& in, int W, int H, std::vector<uint16_t>& flow, std::vector<uint16_t>& mask);
void fgHeadQRCPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int flowZp, int maskZp, std::vector<uint8_t>& flow, std::vector<uint8_t>& mask);
float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);   // round-to-nearest-even

} // namespace anf4r
