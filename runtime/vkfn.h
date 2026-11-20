// anf4r: Vulkan entry points. On Linux (harness/tests) the loader's prototypes are used; with ANF4R_DYNAMIC_VK (the Windows
// DLL inside a game) every function is a pointer resolved from the game's own instance/device. SPDX-License-Identifier: GPL-3.0-only
#pragma once
#if defined(ANF4R_DYNAMIC_VK) && !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define ANF4R_VK_INSTANCE_FUNCS(X) X(vkCreateDevice) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkDestroyInstance) X(vkGetDeviceProcAddr)
#define ANF4R_VK_DEVICE_FUNCS(X) X(vkAllocateCommandBuffers) X(vkAllocateDescriptorSets) X(vkAllocateMemory) X(vkBeginCommandBuffer) \
    X(vkBindBufferMemory) X(vkBindImageMemory) X(vkCmdBindDescriptorSets) X(vkCmdBindPipeline) X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkCmdCopyBuffer) \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImage) X(vkCmdCopyImageToBuffer) X(vkCmdDispatch) X(vkCmdFillBuffer) X(vkCmdPipelineBarrier) \
    X(vkCmdPushConstants) X(vkCmdUpdateBuffer) X(vkCreateBuffer) X(vkCreateCommandPool) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) \
    X(vkCreateDescriptorSetLayout) X(vkCreateFence) X(vkCreateImage) X(vkCreateImageView) X(vkCreatePipelineLayout) X(vkCreateSampler) \
    X(vkCreateShaderModule) X(vkDestroyBuffer) X(vkDestroyCommandPool) X(vkDestroyDescriptorPool) X(vkDestroyDescriptorSetLayout) \
    X(vkDestroyDevice) X(vkDestroyFence) X(vkDestroyImage) X(vkDestroyImageView) X(vkDestroyPipeline) X(vkDestroyPipelineLayout) \
    X(vkDestroySampler) X(vkDestroyShaderModule) X(vkDeviceWaitIdle) X(vkEndCommandBuffer) X(vkFreeCommandBuffers) X(vkFreeMemory) \
    X(vkGetBufferDeviceAddress) X(vkGetBufferMemoryRequirements) X(vkGetDeviceQueue) X(vkGetImageMemoryRequirements) X(vkMapMemory) \
    X(vkQueueSubmit) X(vkUnmapMemory) X(vkUpdateDescriptorSets) X(vkWaitForFences)

#ifdef ANF4R_DYNAMIC_VK
#define ANF4R_VK_DECLARE(f) extern PFN_##f f;
extern PFN_vkCreateInstance vkCreateInstance;
ANF4R_VK_INSTANCE_FUNCS(ANF4R_VK_DECLARE)
ANF4R_VK_DEVICE_FUNCS(ANF4R_VK_DECLARE)
#undef ANF4R_VK_DECLARE
#endif

namespace anf4r {
// Loads the table (no-op without ANF4R_DYNAMIC_VK). Returns false if a function is missing.
bool loadVulkan(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device);
}
