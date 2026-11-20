// anf4r: one vendor (or reconstructed) SPIR-V compute pass as a Vulkan pipeline. SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "vk.h"
#include <map>
#include <string>
#include <vector>

namespace anf4r {

enum class BindKind { SampledImage, StorageImage, Sampler, StorageBuffer, UniformBuffer };

struct BindDesc { uint32_t binding; BindKind kind; std::string name; };

struct SpecConst { uint32_t id; int32_t value; };  // all ANF spec constants are int/uint or bool-as-int

// A compute pass built from a SPIR-V module + its reflected interface.
class Pass {
public:
    bool build(Context& ctx, const std::vector<uint32_t>& spirv, const std::vector<BindDesc>& binds,
               const std::vector<SpecConst>& specs, uint32_t pushBytes,
               uint32_t lx, uint32_t ly, uint32_t lz, const char* entry = "main_cs");
    void destroy(Context& ctx);

    // Per-dispatch binding by binding-number. Provide exactly the resources the layout declares.
    void bindImage(uint32_t binding, VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
    void bindSampler(uint32_t binding, VkSampler s);
    void bindBuffer(uint32_t binding, VkBuffer buf, VkDeviceSize off = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    // Writes descriptors, binds pipeline+set, pushes constants, dispatches ceil(w/lx) x ceil(h/ly).
    void record(Context& ctx, VkCommandBuffer cb, uint32_t gridW, uint32_t gridH, uint32_t gridZ,
                const void* push, uint32_t pushBytes);

    std::string label;   // for profiling
    bool hasBinding(uint32_t b) const { return kindOf_.count(b) != 0; }
    uint32_t lx = 1, ly = 1, lz = 1;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
private:
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    static constexpr uint32_t kRing = 8;          // descriptor sets cycled per record() so frames in flight never share one
    VkDescriptorSet sets_[kRing] = {};
    uint32_t cur_ = 0;
    std::vector<BindDesc> binds_;
    std::map<uint32_t, VkDescriptorImageInfo> imgInfo_;
    std::map<uint32_t, VkDescriptorImageInfo> sampInfo_;
    std::map<uint32_t, VkDescriptorBufferInfo> bufInfo_;
    std::map<uint32_t, BindKind> kindOf_;
};

std::vector<uint32_t> readSpirv(const std::string& path);
// Optional profiling hook: called with begin=true before and begin=false after each recorded dispatch.
extern void (*g_passHook)(VkCommandBuffer, const Pass*, bool begin);

} // namespace anf4r
