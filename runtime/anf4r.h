/* anf4r: Qualcomm Adreno Neural Fusion (ANF) super-resolution and frame-generation for Vulkan callers on RDNA3.
 *
 * The caller owns the Vulkan device; anf4r records its passes into the caller's command buffer (SR) or submits
 * to a supplied queue (FG). Qualcomm's real network weights and their 12 Vulkan SPIR-V data-path shaders run
 * unchanged; the convolutions (Adreno-only in the SDK) are re-implemented as int8 SPIR-V with identical integer
 * arithmetic (see docs/divergences.md). Plans built by tools/prepare_*.py are read from <data_dir>.
 * SPDX-License-Identifier: GPL-3.0-only */
#pragma once
#include <stdint.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#ifdef _WIN32
#define ANF4R_API __declspec(dllexport)
#else
#define ANF4R_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anf4r_context anf4r_context;

typedef struct anf4r_image {
    VkImage image;          /* VK_NULL_HANDLE = not provided */
    VkFormat format;
    uint32_t width, height;
    VkImageLayout layout;   /* layout when the commands run; anf4r transitions to GENERAL and back */
} anf4r_image;

/* Super resolution. ANF SR is fixed 2.0x per dimension (ANF_SR_QUALITY_MODE_PERFORMANCE). */
typedef struct anf4r_sr_desc {
    anf4r_image color, depth, motion, output;   /* color/depth/motion at input size; output at 2x */
    uint32_t content_width, content_height;      /* rendered area of the inputs */
    float jitter_x, jitter_y;                    /* sub-pixel offset, components in [-0.5, 0.5] */
    float motion_scale_x, motion_scale_y;        /* motion texel -> pixels (previous - current) */
    int reset, depth_reversed;
} anf4r_sr_desc;

ANF4R_API int  anf4r_create(anf4r_context** out, PFN_vkGetInstanceProcAddr gipa, VkInstance, VkPhysicalDevice,
                            VkDevice, uint32_t queue_family, const char* data_dir);
ANF4R_API void anf4r_destroy(anf4r_context*);
ANF4R_API int  anf4r_sr_init(anf4r_context*, uint32_t out_w, uint32_t out_h, uint32_t in_w, uint32_t in_h);
ANF4R_API int  anf4r_sr(anf4r_context*, VkCommandBuffer, const anf4r_sr_desc*);
ANF4R_API const char* anf4r_last_error(anf4r_context*);

#ifdef __cplusplus
}
#endif
