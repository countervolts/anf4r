/* anf4r: Qualcomm Adreno Neural Fusion super resolution (2x) and frame generation on non-Adreno Vulkan devices.
 * Shape-compatible with m4r's API so the OptiScaler backend is a rename of m4r's. The caller owns the Vulkan device; anf4r records
 * its passes into the caller's command buffer. Images are passed as VkImage + format + size + current layout; anf4r moves them to
 * GENERAL (or a transfer layout) around its passes and back. Data (<data_dir>): shaders/*.spv (our kernels), spirv/*.bin (vendor FG shaders),
 * modelzoo/ (vendor weights) -- produced by scripts/install_anf.py.
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
    VkImageLayout layout;   /* layout when the commands run; VK_IMAGE_LAYOUT_UNDEFINED is treated as GENERAL */
} anf4r_image;

typedef struct anf4r_upscale_desc {
    anf4r_image color, depth, motion, exposure /* unused by ANF */, output;
    uint32_t content_width, content_height;  /* rendered area of color/depth/motion (top-left aligned) */
    float jitter_x, jitter_y;                /* sub-pixel sample offset the frame was rendered with (pixels, image coordinates: x right, y down, as in
                                                (m4r); anf4r converts to ANF's y-up jitterOffset). Expected in [-0.5,0.5]. */
    float motion_scale_x, motion_scale_y;    /* converts motion texture values to pixels (previous - current) */
    float pre_exposure;                      /* unused (the ANF tonemap assumes 1) */
    int reset, depth_reversed;
} anf4r_upscale_desc;

typedef struct anf4r_interpolate_desc {
    anf4r_image color, prev_color, depth, motion, output;  /* color/output at output size; depth/motion at input size */
    float motion_scale_x, motion_scale_y;                  /* to input pixels, previous - current */
    float delta_time, near_plane, far_plane;
    int depth_reversed, reset;
} anf4r_interpolate_desc;

typedef struct anf4r_framegen_desc {
    anf4r_image backbuffer, depth, motion;
    float motion_scale_x, motion_scale_y;   /* to render pixels, previous - current */
    float delta_time, near_plane, far_plane;
    int depth_reversed, reset;
} anf4r_framegen_desc;

ANF4R_API int anf4r_create(anf4r_context** out, PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance,
                           VkPhysicalDevice physical, VkDevice device, uint32_t queue_family, const char* data_dir);
ANF4R_API void anf4r_destroy(anf4r_context* ctx);
/* 0 on success. ANF SR is fixed 2.0x: out_w/out_h must equal 2*in. max_in_* is the largest content size that will be used. */
ANF4R_API int anf4r_upscale_init(anf4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t max_in_w, uint32_t max_in_h);
ANF4R_API int anf4r_upscale(anf4r_context* ctx, VkCommandBuffer cb, const anf4r_upscale_desc* desc);
ANF4R_API int anf4r_interpolate_init(anf4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t in_w, uint32_t in_h);
ANF4R_API int anf4r_interpolate(anf4r_context* ctx, VkCommandBuffer cb, const anf4r_interpolate_desc* desc);
/* Returns 1 when the generated frame was written to desc->backbuffer, 0 = nothing generated (first frame / reset), negative = error. */
ANF4R_API int anf4r_framegen_capture(anf4r_context* ctx, VkCommandBuffer cb, const anf4r_framegen_desc* desc);
ANF4R_API int anf4r_framegen_restore(anf4r_context* ctx, VkCommandBuffer cb, const anf4r_image* backbuffer);
ANF4R_API const char* anf4r_last_error(anf4r_context* ctx);

#ifdef __cplusplus
}
#endif
