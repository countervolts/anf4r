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
    uint32_t requiredSubgroup = 0;   // set before build(): pin the subgroup size (cooperative-matrix kernels need 32)
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
#ifdef ANF4R_HIPVK
    std::vector<int32_t> spec_;   // the specialisation constants by id: the HIP ports read them from the argument block
    uint32_t specSet_[2] = {};
#endif
};

#ifdef ANF4R_HIPVK
// Native Windows (hipvk): a pass is the code object of the kernel's OpenCL C port (kernels/hip/<name>.cl, built by
// scripts/build_hip.py into <g_hipDir>/<name>.hsaco). Its one kernel argument is the address of this block, in a host-visible ring:
struct HipArgs {
    uint64_t slot[64];   // by binding (hipSlot): buffer address (+ offset) / texture record address (sampled descriptor at +0, storage at
                         // +32) / sampler descriptor address
    int32_t spec[64];    // specialisation constants by id
    uint32_t specSet[2]; // bit per id: set by the host (otherwise a port uses the GLSL default, as Vulkan does)
    uint32_t pad[2];
    uint32_t push[32];   // push constants
    uint32_t grid[4];    // the dispatch's thread grid (gridW, gridH, gridZ)
};
// Qualcomm's passes number bindings by HLSL register class (t0.. at 0, s0.. at 128, u0.. at 384): 16 slots per class of 128
inline int hipSlot(uint32_t binding) { return (binding & 127) < 16 && binding < 512 ? int((binding >> 7) * 16 + (binding & 15)) : -1; }
extern std::string g_hipDir;   // where readSpirv finds the code objects (default "build/hip")
#endif

// The SPIR-V words of `path`; with ANF4R_HIPVK the bytes of the code object <g_hipDir>/<basename without .spv / .bin>.hsaco.
std::vector<uint32_t> readSpirv(const std::string& path);
// Optional profiling hook: called with begin=true before and begin=false after each recorded dispatch.
extern void (*g_passHook)(VkCommandBuffer, const Pass*, bool begin);

} // namespace anf4r
