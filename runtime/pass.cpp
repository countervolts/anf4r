// anf4r: compute pass implementation. SPDX-License-Identifier: GPL-3.0-only
#include "pass.h"
#include <fstream>
#include <cstring>

namespace anf4r {

void (*g_passHook)(VkCommandBuffer, const Pass*, bool) = nullptr;

std::vector<uint32_t> readSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t n = f.tellg(); f.seekg(0);
    std::vector<uint32_t> v(n / 4);
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

} // namespace anf4r
