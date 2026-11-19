// anf4r: a quantized conv network (QMFNet SR / FRC) run as a chain of int8 conv passes. GPL-3.0-only
#pragma once
#include "pass.h"
#include <string>
#include <vector>

namespace anf4r {

struct ConvLayer {
    std::string name;
    int ic, oc, stride, pad = 1;
    bool relu = true;
    std::vector<int8_t>  weights;   // [oc][ic][3][3]
    std::vector<int32_t> bias;      // [oc]
    std::vector<float>   scale;     // [oc]  (weights_scales)
};

struct Model {
    std::vector<ConvLayer> layers;
    int inChannels = 0;
    // Load a modelzoo dir (e.g. .../aiff-sr-v2s-prod-7p1-invpow2) for a given jitter phase.
    bool loadSR(const std::string& modelDir, int jitterPhase, std::string& err);
};

// Runs the model on the GPU: input int8 NHWC [H][W][inChannels] -> output int8 NHWC of the last layer.
class ModelRunner {
public:
    bool build(Context& ctx, const Model& m, const std::vector<uint32_t>& convSpirv);
    void destroy(Context& ctx);
    // in/out are device buffers; W,H are the INPUT dims. Intermediate buffers are owned here.
    bool run(Context& ctx, VkCommandBuffer cb, VkBuffer in, int W, int H, VkBuffer& outBuf, int& outW, int& outH);
    void destroyScratch(Context& ctx);
private:
    const Model* model_ = nullptr;
    std::vector<Pass> passes_;
    std::vector<Buffer> wbuf_, bbuf_, sbuf_, act_;   // weights/bias/scale per layer, activations between layers
};

// CPU reference for the whole chain (used by tests). Returns last layer output.
std::vector<int8_t> runModelCPU(const Model& m, const std::vector<int8_t>& in, int W, int H, int& outW, int& outH);

} // namespace anf4r
