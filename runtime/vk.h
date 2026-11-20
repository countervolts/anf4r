// anf4r: minimal Vulkan 1.3 compute context (headless harness + adopt-an-existing-device for OptiScaler).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "vkfn.h"
#include <cstdint>
#include <string>
#include <vector>

namespace anf4r {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* map = nullptr;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;   // GENERAL-layout 2D view (color aspect, or depth aspect for depth fmts)
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool owned = true;
};

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    bool ownsDevice = false;
    // capabilities, filled by create()/adopt(): the matrix-unit convolution needs int8 16x16x16 subgroup cooperative matrices and a pinnable subgroup size of 32
    std::string deviceName; bool coopInt8 = false, subgroup32 = false;
    bool wmmaOk() const { return coopInt8 && subgroup32; }
    void probe();
    std::string error;

    bool create(bool validation = false);     // own instance+device (harness)
    bool adopt(VkInstance, VkPhysicalDevice, VkDevice, uint32_t family);
    void destroy();

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) const;
    bool createBuffer(Buffer&, VkDeviceSize, bool hostVisible, VkBufferUsageFlags extra = 0);
    void destroyBuffer(Buffer&);
    bool createTexture(Texture&, VkFormat, uint32_t w, uint32_t h, VkImageUsageFlags extra = 0);
    bool wrap(Texture&, VkImage, VkFormat, uint32_t w, uint32_t h);  // image owned elsewhere
    void destroyTexture(Texture&);

    VkCommandBuffer begin();
    bool submit(VkCommandBuffer);                 // submit + wait idle
    void barrierAll(VkCommandBuffer);             // compute+transfer RW -> compute+transfer RW
    void toLayout(VkCommandBuffer, Texture&, VkImageLayout);
    bool upload(Texture&, const void* data, size_t bytes);   // via staging; leaves image GENERAL
    bool download(Texture&, void* data, size_t bytes);
};

uint32_t formatBytes(VkFormat);

} // namespace anf4r
