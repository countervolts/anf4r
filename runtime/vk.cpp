// anf4r: Vulkan context implementation. SPDX-License-Identifier: GPL-3.0-only
#include "vk.h"
#include <cstring>

namespace anf4r {

uint32_t formatBytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT: return 4;
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: return 2;
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
    default: return 0;
    }
}

static bool hasStencil(VkFormat f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT; }
static bool isDepth(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D16_UNORM ||
           f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32;
}

uint32_t Context::memoryType(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}

bool Context::create(bool validation) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    app.pApplicationName = "anf4r";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    if (validation) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = layers; }
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) { error = "vkCreateInstance"; return false; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(instance, &n, devs.data());
    physical = VK_NULL_HANDLE;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { physical = d; break; }
        if (!physical) physical = d;
    }
    if (!physical) { error = "no physical device"; return false; }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, qf.data());
    queueFamily = ~0u;
    for (uint32_t i = 0; i < qn; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    if (queueFamily == ~0u) { error = "no compute queue"; return false; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE; f12.storageBuffer8BitAccess = VK_TRUE; f12.shaderInt8 = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE; f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.pNext = &f12; f13.computeFullSubgroups = VK_TRUE; f13.subgroupSizeControl = VK_TRUE; f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; feat.pNext = &f13;
    feat.features.shaderInt16 = VK_TRUE; feat.features.shaderStorageImageExtendedFormats = VK_TRUE;
    const char* exts[] = {};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &feat; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 0; dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) { error = "vkCreateDevice"; return false; }
    ownsDevice = true;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(device, &pci, nullptr, &pool) != VK_SUCCESS) { error = "command pool"; return false; }
    return true;
}

bool Context::adopt(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t family) {
    instance = inst; physical = phys; device = dev; queueFamily = family; ownsDevice = false;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(device, &pci, nullptr, &pool) != VK_SUCCESS) { error = "command pool"; return false; }
    return true;
}

void Context::destroy() {
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (ownsDevice && device) vkDestroyDevice(device, nullptr);
    if (ownsDevice && instance) vkDestroyInstance(instance, nullptr);
    *this = Context{};
}

bool Context::createBuffer(Buffer& b, VkDeviceSize size, bool hostVisible, VkBufferUsageFlags extra) {
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extra;
    if (vkCreateBuffer(device, &bci, nullptr, &b.buffer) != VK_SUCCESS) { error = "createBuffer"; return false; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b.buffer, &mr);
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext = &fi;
    ai.allocationSize = mr.size;
    VkMemoryPropertyFlags want = hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    ai.memoryTypeIndex = memoryType(mr.memoryTypeBits, want);
    if (ai.memoryTypeIndex == ~0u) ai.memoryTypeIndex = memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (vkAllocateMemory(device, &ai, nullptr, &b.memory) != VK_SUCCESS) { error = "allocateMemory(buf)"; return false; }
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; dai.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device, &dai);
    if (hostVisible) vkMapMemory(device, b.memory, 0, size, 0, &b.map);
    return true;
}

void Context::destroyBuffer(Buffer& b) {
    if (b.map) vkUnmapMemory(device, b.memory);
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = Buffer{};
}

bool Context::createTexture(Texture& t, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags extra) {
    t.format = fmt; t.width = w; t.height = h; t.owned = true; t.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt; ici.extent = {w, h, 1};
    ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &t.image) != VK_SUCCESS) { error = "createImage"; return false; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(device, t.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &ai, nullptr, &t.memory) != VK_SUCCESS) { error = "allocateMemory(img)"; return false; }
    vkBindImageMemory(device, t.image, t.memory, 0);
    return wrap(t, t.image, fmt, w, h) ? true : (error = "view", false);
}

bool Context::wrap(Texture& t, VkImage image, VkFormat fmt, uint32_t w, uint32_t h) {
    bool ownedSave = t.owned;
    if (t.image != image) { t.owned = false; }
    t.image = image; t.format = fmt; t.width = w; t.height = h;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
    vci.subresourceRange.aspectMask = isDepth(fmt) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1; vci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &vci, nullptr, &t.view) != VK_SUCCESS) return false;
    if (t.image == image && ownedSave) t.owned = ownedSave;
    return true;
}

void Context::destroyTexture(Texture& t) {
    if (t.view) vkDestroyImageView(device, t.view, nullptr);
    if (t.owned && t.image) vkDestroyImage(device, t.image, nullptr);
    if (t.owned && t.memory) vkFreeMemory(device, t.memory, nullptr);
    t = Texture{};
}

VkCommandBuffer Context::begin() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

bool Context::submit(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(device, &fci, nullptr, &fence);
    VkResult r = vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return r == VK_SUCCESS;
}

void Context::barrierAll(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Context::toLayout(VkCommandBuffer cb, Texture& t, VkImageLayout to) {
    if (t.layout == to) return;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = t.layout; b.newLayout = to; b.image = t.image;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange.aspectMask = isDepth(t.format) ? (hasStencil(t.format) ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    t.layout = to;
}

bool Context::upload(Texture& t, const void* data, size_t bytes) {
    Buffer stg; if (!createBuffer(stg, bytes, true)) return false;
    std::memcpy(stg.map, data, bytes);
    VkCommandBuffer cb = begin();
    toLayout(cb, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{}; c.imageSubresource.aspectMask = isDepth(t.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.layerCount = 1; c.imageExtent = {t.width, t.height, 1};
    vkCmdCopyBufferToImage(cb, stg.buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    toLayout(cb, t, VK_IMAGE_LAYOUT_GENERAL);
    bool ok = submit(cb);
    destroyBuffer(stg);
    return ok;
}

bool Context::download(Texture& t, void* data, size_t bytes) {
    Buffer stg; if (!createBuffer(stg, bytes, true)) return false;
    VkCommandBuffer cb = begin();
    VkImageLayout save = t.layout;
    toLayout(cb, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy c{}; c.imageSubresource.aspectMask = isDepth(t.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    c.imageSubresource.layerCount = 1; c.imageExtent = {t.width, t.height, 1};
    vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stg.buffer, 1, &c);
    if (save != VK_IMAGE_LAYOUT_UNDEFINED) toLayout(cb, t, save);
    bool ok = submit(cb);
    std::memcpy(data, stg.map, bytes);
    destroyBuffer(stg);
    return ok;
}

} // namespace anf4r
