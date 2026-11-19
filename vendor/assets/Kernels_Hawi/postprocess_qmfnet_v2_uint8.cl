#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void postprocess_qmfnet_v2(
    __read_only  image2d_t   input_color_lr,
    __read_only  image2d_t   previous_frame_hr,
    __global     uchar*      graph_output,
    __write_only image2d_t   output_color_hr,
    __write_only image2d_t   output_tonemaped_hr,
    __write_only image2d_t   output_features_hr,
    float2                   quantized_jitter_offsets,
    int                      output_width,
    int                      output_height,
    int                      low_res_tensor_width
#ifdef ENABLE_DEBUG
    , __global     half4*      debug_candidate       // RGBA FP16 — matches cl_half4 allocation in C++
    , __global     float2*     debug_sampling_grid   // 2D float vector (RG32F) — matches cl_float2 allocation in C++
    , __global     half4*      debug_output_color    // RGBA FP16 — matches cl_half4 allocation in C++
    , __global     float2*     debug_anchor_grid     // 2D float vector (RG32F) — matches cl_float2 allocation in C++
    , __global     half4*      debug_past_frame      // RGBA FP16 — matches cl_half4 allocation in C++
#endif
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= output_width || y >= output_height) {
        return;
    }

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // Unpack offsets and alpha from the graph_output buffer
    // Standard: The graph output is in LR resolution (e.g. 540x960) with 16 channels (4x4 spatial block packed into depth)
    // Strided: The graph output is in 0.5x LR resolution (e.g. 270x480) with 64 channels (8x8 spatial block packed into depth)
    // We need to perform "PixelShuffle" / "DepthToSpace" here to map HR coordinates (x,y) to the correct channel in the buffer

    const float scaling_factor = 2.0f;

    int lr_width = low_res_tensor_width;
    int channel_total = 64;

    // prod-3: replaced back-to-back PixelShuffle(2)->PixelShuffle(2) with a single DepthToSpaceDCR(4).
    // Matches QMFNetV2StridedFP32/INT8: self.depth_to_space = DepthToSpaceDCR(upscale_factor=s2d_factor_output**2)
    // in qmfnet/models/v2.py.
    int sub_x = x % 4;
    int sub_y = y % 4;

    int spatial_idx = sub_y * 4 + sub_x;

    int lr_x = x / 4;
    int lr_y = y / 4;
    
    // Index in the flattened LR buffer (NHWC layout)
    // Actually, graph_output is uchar*, so we can address bytes directly
    int base_idx = (lr_y * lr_width + lr_x) * channel_total;

    // Read bytes: DCR layout (matches DepthToSpaceDCR in qmfnet/models/v2.py)
    // channel = spatial_idx * num_base_channels + base_channel  (not CRD: base_channel * stride + spatial_idx)
    uchar raw_offset_x = graph_output[base_idx + spatial_idx * 4 + 0];
    uchar raw_offset_y = graph_output[base_idx + spatial_idx * 4 + 1];
    uchar raw_alpha    = graph_output[base_idx + spatial_idx * 4 + 2];
    uchar raw_feature  = graph_output[base_idx + spatial_idx * 4 + 3];

    // Convert to float and normalized [0, 1]
    // The model outputs values roughly in [-128, 127] packed as uint8 [0, 255] by adding 128.
    // So raw value 0 -> real value -128. raw value 128 -> real value 0.
    // To unquantize to [0,1], the python code effectively does: (real_val + 128) / 255.
    // Which simplifies to: raw_val / 255.0f.
    
    float offset_x = (float)raw_offset_x / 255.0f;
    float offset_y = (float)raw_offset_y / 255.0f;
    float alpha    = (float)raw_alpha    / 255.0f;
    float feature  = (float)raw_feature  / 255.0f;

    // 1. Generate Anchor Grid (reusing logic from the other kernel)
    // This is the coordinate in the low-res input space
    float2 anchor_grid;
    anchor_grid.x = (float)x - 0.5f * (scaling_factor - 1.0f);
    anchor_grid.y = (float)y - 0.5f * (scaling_factor - 1.0f);
    anchor_grid /= scaling_factor;

    anchor_grid.x -= quantized_jitter_offsets.x;
    anchor_grid.y += quantized_jitter_offsets.y;

    anchor_grid = floor(anchor_grid);

    anchor_grid.x = fmax(anchor_grid.x, 0.0f);
    anchor_grid.y = fmax(anchor_grid.y, 0.0f);

#ifdef ENABLE_DEBUG
    if (debug_anchor_grid) {
        debug_anchor_grid[y * output_width + x] = anchor_grid;
    }
#endif

    // 2. Calculate final sampling coordinates
    float2 sampling_grid = anchor_grid + (float2)(offset_x, offset_y);

    // 3. Sample the candidate color
    half4 candidate = read_imageh(input_color_lr, sampler, sampling_grid + 0.5f);

#ifdef ENABLE_DEBUG
    if (debug_candidate) {
        debug_candidate[y * output_width + x] = candidate;
    }
    if (debug_sampling_grid) {
        debug_sampling_grid[y * output_width + x] = sampling_grid;
    }
#endif

    // 4. Sample the previous frame
    // Coordinates for previous_frame_hr are just the output pixel coordinates
    int2 prev_coord = (int2)(x, y);
    half4 past_frame = read_imageh(previous_frame_hr, sampler, convert_float2(prev_coord) + 0.5f);

#ifdef ENABLE_DEBUG
    if (debug_past_frame) {
        debug_past_frame[y * output_width + x] = past_frame;
    }
#endif

    // 5. Alpha blend
    half alpha_h = (half)alpha;
    half4 output_color = alpha_h * past_frame + (1.0h - alpha_h) * candidate;

#ifdef ENABLE_DEBUG
    if (debug_output_color) {
        debug_output_color[y * output_width + x] = output_color;
    }
#endif

    // 6. Write to output
    int2 out_coord = (int2)(x, y);

    // output_tonemaped_hr should get the blended result (which is in tonemapped space)
    write_imageh(output_tonemaped_hr, out_coord, output_color);
    write_imagef(output_features_hr, out_coord, (float4)(feature, 0.0f, 0.0f, 0.0f));

    // Linear blending model: input/output are already in linear color space.
    // Write the blended result directly — no inverse tonemapping needed.
    write_imageh(output_color_hr, out_coord, output_color);
}
