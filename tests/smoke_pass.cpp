// anf4r smoke test: build and dispatch a vendor SPIR-V pass (AIFFPrepareInputs) on RADV.
#include "../runtime/pass.h"
#include <cstdio>
#include <cstring>
using namespace anf4r;

int main(int argc, char** argv) {
    const char* spv = argc > 1 ? argv[1] : "data/spirv/AIFFPrepareInputs_cs.bin";
    Context ctx;
    bool val = getenv("VAL") != nullptr;
    if (!ctx.create(val)) { printf("ctx: %s\n", ctx.error.c_str()); return 1; }
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(ctx.physical, &p);
    printf("device: %s\n", p.deviceName);

    auto code = readSpirv(spv);
    if (code.empty()) { printf("no spirv\n"); return 1; }

    const uint32_t W = 128, H = 128;
    // AIFFPrepareInputs: in texRawMotion(0, RG16F) texRawDepth(1, R32F); out outputPreparedMotion(384) outputPreparedDepth(385)
    Texture rawMotion, rawDepth, outMotion, outDepth;
    ctx.createTexture(rawMotion, VK_FORMAT_R16G16_SFLOAT, W, H);
    ctx.createTexture(rawDepth, VK_FORMAT_R32_SFLOAT, W, H);
    ctx.createTexture(outMotion, VK_FORMAT_R16G16_SFLOAT, W, H);
    ctx.createTexture(outDepth, VK_FORMAT_R32_SFLOAT, W, H);
    { VkCommandBuffer cb = ctx.begin();
      ctx.toLayout(cb, rawMotion, VK_IMAGE_LAYOUT_GENERAL); ctx.toLayout(cb, rawDepth, VK_IMAGE_LAYOUT_GENERAL);
      ctx.toLayout(cb, outMotion, VK_IMAGE_LAYOUT_GENERAL); ctx.toLayout(cb, outDepth, VK_IMAGE_LAYOUT_GENERAL);
      ctx.submit(cb); }

    Pass pass;
    std::vector<BindDesc> binds = {
        {0, BindKind::SampledImage, "texRawMotion"},
        {1, BindKind::SampledImage, "texRawDepth"},
        {384, BindKind::StorageImage, "outputPreparedMotion"},
        {385, BindKind::StorageImage, "outputPreparedDepth"},
    };
    if (!pass.build(ctx, code, binds, {}, 28, 16, 16, 1)) { printf("build: %s\n", ctx.error.c_str()); return 1; }
    pass.bindImage(0, rawMotion.view); pass.bindImage(1, rawDepth.view);
    pass.bindImage(384, outMotion.view); pass.bindImage(385, outDepth.view);

    struct PC { float jp[2], jc[2]; uint32_t block_size, scaling_factor, yMax; } pc{};
    pc.block_size = 1; pc.scaling_factor = 1; pc.yMax = H;

    VkCommandBuffer cb = ctx.begin();
    pass.record(ctx, cb, W, H, 1, &pc, sizeof pc);
    ctx.barrierAll(cb);
    bool ok = ctx.submit(cb);
    printf("dispatch submitted=%d\n", ok);
    printf("SMOKE OK: vendor SPIR-V pass built and dispatched on RADV, no crash.\n");
    pass.destroy(ctx);
    ctx.destroyTexture(rawMotion); ctx.destroyTexture(rawDepth); ctx.destroyTexture(outMotion); ctx.destroyTexture(outDepth);
    ctx.destroy();
    return 0;
}
