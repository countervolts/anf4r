// anf4r: compute pass implementation. SPDX-License-Identifier: GPL-3.0-only
#include "pass.h"
#include <fstream>
#include <cstring>
#ifdef ANF4R_HIPVK
#include "hipvk/hipvk.h"
#include <mutex>
#endif

namespace anf4r {

void (*g_passHook)(VkCommandBuffer, const Pass*, bool) = nullptr;

#ifdef ANF4R_HIPVK
std::string g_hipDir = "build/hip";
#endif

std::vector<uint32_t> readSpirv(const std::string& spirvPath) {
#ifdef ANF4R_HIPVK
    std::string name = spirvPath.substr(spirvPath.find_last_of("/\\") + 1);
    name = name.substr(0, name.find_last_of('.'));
    const std::string path = g_hipDir + "/" + name + ".hsaco";
#else
    const std::string& path = spirvPath;
#endif
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t n = f.tellg(); f.seekg(0);
    std::vector<uint32_t> v((n + 3) / 4);   // a code object's size need not be a multiple of 4
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

static VkDescriptorType vkType(BindKind k) {
    switch (k) {
    case BindKind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case BindKind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case BindKind::Sampler:      return VK_DESCRIPTOR_TYPE_SAMPLER;
    case BindKind::StorageBuffer:return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case BindKind::UniformBuffer:return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

#ifdef ANF4R_HIPVK
namespace {
// Argument blocks and texture/sampler records of recorded dispatches: a host-visible ring (hipvk uploads what the CPU wrote at the
// next submission, in stream order, so a slot can be rewritten once the dispatch that read it was submitted)
struct ArgRing {
    std::mutex m;
    Buffer buf;
    size_t head = 0;
    PFN_vkGetDescriptorEXT getDescriptor = nullptr;
    static constexpr size_t kBytes = 16u << 20;
    bool init(Context& ctx) {
        if (buf.buffer) return true;
        getDescriptor = reinterpret_cast<PFN_vkGetDescriptorEXT>(hipvk::procAddr("vkGetDescriptorEXT"));
        return getDescriptor && ctx.createBuffer(buf, kBytes, true);
    }
    size_t alloc(size_t n) {
        n = (n + 255) & ~size_t(255);
        if (head + n > kBytes) head = 0;
        const size_t at = head;
        head += n;
        return at;
    }
} g_ring;
}

bool Pass::build(Context& ctx, const std::vector<uint32_t>& code, const std::vector<BindDesc>& binds,
                 const std::vector<SpecConst>& specs, uint32_t, uint32_t lxx, uint32_t lyy, uint32_t lzz, const char* entry) {
    lx = lxx; ly = lyy; lz = lzz; binds_ = binds;
    for (auto& b : binds) {
        if (hipSlot(b.binding) < 0) { ctx.error = "HIP pass: binding " + std::to_string(b.binding) + " has no argument slot"; return false; }
        kindOf_[b.binding] = b.kind;
    }
    spec_.assign(64, 0);
    for (auto& s : specs) {
        if (s.id >= 64) { ctx.error = "HIP pass: specialisation constant " + std::to_string(s.id) + " >= 64"; return false; }
        spec_[s.id] = s.value;
        specSet_[s.id >> 5] |= 1u << (s.id & 31);
    }
    if (code.empty()) { ctx.error = std::string("HIP pass: no code object (scripts/build_hip.py) in ") + g_hipDir; return false; }
    {
        std::lock_guard<std::mutex> lock(g_ring.m);
        if (!g_ring.init(ctx)) { ctx.error = "HIP pass: argument ring"; return false; }
    }
    const std::vector<uint8_t> elf(reinterpret_cast<const uint8_t*>(code.data()), reinterpret_cast<const uint8_t*>(code.data()) + code.size() * 4);
    const uint32_t tg[3] = {lx, ly, lz};
    std::string why;
    pipeline_ = hipvk::createKernel(elf, std::string("anf_") + entry, hipvk::KernelAbi::ArgPointer, tg, 8, 0, {}, why);   // OpenCL has no kernel "main"
    if (!pipeline_) { ctx.error = "HIP pass " + label + " (" + entry + "): " + why; return false; }
    return true;
}

void Pass::bindImage(uint32_t b, VkImageView v, VkImageLayout l) { imgInfo_[b] = {VK_NULL_HANDLE, v, l}; }
void Pass::bindSampler(uint32_t b, VkSampler s) { sampInfo_[b] = {s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}; }
void Pass::bindBuffer(uint32_t b, VkBuffer buf, VkDeviceSize off, VkDeviceSize range) { bufInfo_[b] = {buf, off, range}; }

void Pass::record(Context& ctx, VkCommandBuffer cb, uint32_t gw, uint32_t gh, uint32_t gz, const void* push, uint32_t pushBytes) {
    size_t images = 0;
    for (auto& b : binds_) images += b.kind == BindKind::SampledImage || b.kind == BindKind::StorageImage || b.kind == BindKind::Sampler;
    std::lock_guard<std::mutex> lock(g_ring.m);
    const size_t at = g_ring.alloc(sizeof(HipArgs) + 80 * images);
    uint8_t* base = static_cast<uint8_t*>(g_ring.buf.map) + at;
    HipArgs& a = *reinterpret_cast<HipArgs*>(base);
    a = HipArgs{};
    size_t rec = sizeof(HipArgs);
    for (auto& b : binds_) {
        if (b.kind == BindKind::StorageBuffer || b.kind == BindKind::UniformBuffer) {
            const VkDescriptorBufferInfo& bi = bufInfo_[b.binding];
            VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            ai.buffer = bi.buffer;
            a.slot[hipSlot(b.binding)] = vkGetBufferDeviceAddress(ctx.device, &ai) + bi.offset;
            continue;
        }
        VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
        if (b.kind == BindKind::Sampler) {
            gi.type = VK_DESCRIPTOR_TYPE_SAMPLER;
            gi.data.pSampler = &sampInfo_[b.binding].sampler;
            g_ring.getDescriptor(ctx.device, &gi, 16, base + rec);
        } else {
            gi.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            gi.data.pSampledImage = &imgInfo_[b.binding];
            g_ring.getDescriptor(ctx.device, &gi, 32, base + rec);
            std::memcpy(base + rec + 32, base + rec, 32);   // hipvk's storage descriptor is the sampled one
            uint32_t* size = reinterpret_cast<uint32_t*>(base + rec + 64);   // textureSize
            if (!hipvk::imageViewSize(imgInfo_[b.binding].imageView, size[0], size[1])) size[0] = size[1] = 0;
        }
        a.slot[hipSlot(b.binding)] = g_ring.buf.address + at + rec;
        rec += 80;
    }
    std::memcpy(a.spec, spec_.data(), sizeof a.spec);
    a.specSet[0] = specSet_[0]; a.specSet[1] = specSet_[1];
    if (push && pushBytes) std::memcpy(a.push, push, std::min<size_t>(pushBytes, sizeof a.push));
    a.grid[0] = gw; a.grid[1] = gh; a.grid[2] = gz;
    const uint64_t addr = g_ring.buf.address + at;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdPushConstants(cb, VK_NULL_HANDLE, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &addr);
    if (g_passHook) g_passHook(cb, this, true);
    vkCmdDispatch(cb, (gw + lx - 1) / lx, (gh + ly - 1) / ly, (gz + lz - 1) / lz);
    if (g_passHook) g_passHook(cb, this, false);
}

void Pass::destroy(Context& ctx) {
    if (pipeline_) vkDestroyPipeline(ctx.device, pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
}
#else
bool Pass::build(Context& ctx, const std::vector<uint32_t>& spirv, const std::vector<BindDesc>& binds,
                 const std::vector<SpecConst>& specs, uint32_t pushBytes,
                 uint32_t lxx, uint32_t lyy, uint32_t lzz, const char* entry) {
    lx = lxx; ly = lyy; lz = lzz; binds_ = binds;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size() * 4; smi.pCode = spirv.data();
    if (vkCreateShaderModule(ctx.device, &smi, nullptr, &module_) != VK_SUCCESS) { ctx.error = "shaderModule"; return false; }

    std::vector<VkDescriptorSetLayoutBinding> lb;
    std::map<VkDescriptorType, uint32_t> counts;
    for (auto& b : binds) {
        kindOf_[b.binding] = b.kind;
        VkDescriptorSetLayoutBinding x{};
        x.binding = b.binding; x.descriptorType = vkType(b.kind); x.descriptorCount = 1;
        x.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        lb.push_back(x); counts[vkType(b.kind)]++;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = lb.size(); lci.pBindings = lb.data();
    if (vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &setLayout) != VK_SUCCESS) { ctx.error = "setLayout"; return false; }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes ? pushBytes : 4};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &setLayout;
    if (pushBytes) { pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr; }
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeLayout_) != VK_SUCCESS) { ctx.error = "pipeLayout"; return false; }

    std::vector<VkSpecializationMapEntry> sm;
    std::vector<int32_t> sv;
    for (auto& s : specs) { sm.push_back({s.id, (uint32_t)(sv.size()*4), 4}); sv.push_back(s.value); }
    VkSpecializationInfo si{}; si.mapEntryCount = sm.size(); si.pMapEntries = sm.data();
    si.dataSize = sv.size()*4; si.pData = sv.data();

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = module_; cpi.stage.pName = entry;
    if (!specs.empty()) cpi.stage.pSpecializationInfo = &si;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (requiredSubgroup) { rss.requiredSubgroupSize = requiredSubgroup; cpi.stage.pNext = &rss; }
    cpi.layout = pipeLayout_;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline_) != VK_SUCCESS) {
        ctx.error = "computePipeline"; return false;
    }

    std::vector<VkDescriptorPoolSize> ps;
    for (auto& [t, c] : counts) ps.push_back({t, c * kRing});
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kRing; dpi.poolSizeCount = ps.size(); dpi.pPoolSizes = ps.data();
    if (vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &pool_) != VK_SUCCESS) { ctx.error = "descPool"; return false; }
    VkDescriptorSetLayout layouts[kRing]; for (auto& l : layouts) l = setLayout;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_; dai.descriptorSetCount = kRing; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(ctx.device, &dai, sets_) != VK_SUCCESS) { ctx.error = "descSet"; return false; }
    return true;
}

void Pass::bindImage(uint32_t b, VkImageView v, VkImageLayout l) { imgInfo_[b] = {VK_NULL_HANDLE, v, l}; }
void Pass::bindSampler(uint32_t b, VkSampler s) { sampInfo_[b] = {s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}; }
void Pass::bindBuffer(uint32_t b, VkBuffer buf, VkDeviceSize off, VkDeviceSize range) { bufInfo_[b] = {buf, off, range}; }

void Pass::record(Context& ctx, VkCommandBuffer cb, uint32_t gw, uint32_t gh, uint32_t gz,
                  const void* push, uint32_t pushBytes) {
    VkDescriptorSet set = sets_[cur_++ % kRing];
    std::vector<VkWriteDescriptorSet> w;
    for (auto& b : binds_) {
        VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        x.dstSet = set; x.dstBinding = b.binding; x.descriptorCount = 1; x.descriptorType = vkType(b.kind);
        if (b.kind == BindKind::StorageBuffer || b.kind == BindKind::UniformBuffer) x.pBufferInfo = &bufInfo_[b.binding];
        else if (b.kind == BindKind::Sampler) x.pImageInfo = &sampInfo_[b.binding];
        else x.pImageInfo = &imgInfo_[b.binding];
        w.push_back(x);
    }
    vkUpdateDescriptorSets(ctx.device, w.size(), w.data(), 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &set, 0, nullptr);
    if (push && pushBytes) vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
    if (g_passHook) g_passHook(cb, this, true);
    vkCmdDispatch(cb, (gw + lx - 1) / lx, (gh + ly - 1) / ly, (gz + lz - 1) / lz);
    if (g_passHook) g_passHook(cb, this, false);
}

void Pass::destroy(Context& ctx) {
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    if (pipeline_) vkDestroyPipeline(ctx.device, pipeline_, nullptr);
    if (pipeLayout_) vkDestroyPipelineLayout(ctx.device, pipeLayout_, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(ctx.device, setLayout, nullptr);
    if (module_) vkDestroyShaderModule(ctx.device, module_, nullptr);
}
#endif

} // namespace anf4r
