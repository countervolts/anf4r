// This is the baseline MxM kernel for Pakala
// -----------------------------------------------------
// Tile size is M:64 N:64 K:{8(FP32),16(FP16),32(INT8)}
// A and C are rowMajor
// B is colMahor
// -----------------------------------------------------
// Loads A via TP into GPR
// Loads B via SP into GPR
// Stores B into LM for Broadasting to ALUs
// Uses divergent A0 to lock ALU section
// Has 512 ALU cycles per iteration


// example 
//.\tcu.exe  -u:13 -FP32 -IH:18 -IW:18 -IC:16 -FH:3 -FW:3 -OC:128 -unevenCCHEbanksMapping -genqasm -kernels_dir:../Kernels_Lanai


#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_extra_vector_types : enable
#pragma OPENCL EXTENSION cl_qcom_filter_bicubic : enable

#define xstr(s) str(s)
#define str(s) #s

#define BYTES_PER_WORD       4

#define MATRIX_B_ON_CHIP_IN_BYTES (2)
#define SPML_MODE_FLAG
#define LM_FIRST_256B  0
#define LM_SECOND_256B 512
#define LM_THIRD_256B  1024
#define LM_FOURTH_256B 1536
#define LM_FIRST_512B  0
#define LM_SECOND_512B 1024

// #define MODEL_INPUT_STRIDE_BYTES 28


// #define BICUBIC
#define BILINEAR
// #define POINT

// use 16 bit float motion vectors
#define MV_FP16

// static objects have motion
#define FINAL_MV_INPUT

#define MV_SCALE_X -1
#define MV_SCALE_Y +1

#define MM_SETUP()      \
    if (get_global_id(0) == 123456789) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regDepth_d24s8.xyzw = read_imagef(color_lr,   sampler_point, convert_float2(uv));\
       regDepth_d24s8.xyzw += convert_float4(read_imageh(depth_lr, get_global_id(0))); \
       regDepth_d24s8.xyzw += convert_float4(read_imagef(mv_lr, get_global_id(0))); \
       regDepth_d24s8.xyzw += convert_float4(read_imagei(prev_color_hr, sampler, convert_float2(uv)));\
       regDepth_d24s8.xyzw += convert_float4(read_imagei(prev_feature_hr, sampler, convert_float2(uv)));\
       write_imagef(model_input, 0, regDepth_d24s8.xyzw);\
       write_imagef(color_lr_rgba_fp16, 0, regDepth_d24s8.xyzw);\
       write_imagef(previous_color_warped_hr_rgba_fp16, 0, regDepth_d24s8.xyzw);\
    }\
    __asm__ __volatile__( \
                        "movs.u32u32 sr50.x, 0;" \
                        "movs.u32u32 sr50.y, 11;" \
                        "movs.u32u32 sr50.z, 22;" \
                        "movs.u32u32 sr50.w, 0x3F;" \
                        "movs.u32u32 sr51.x, 0x3F;" \
                        "movs.u32u32 sr51.y, 0x1F;" \
                        "movs.u32u32 sr51.z, 6;" \
                        "movs.u32u32 sr51.w, 17;" \
                        "movs.u32u32 sr52.x, 27;" \
                        "movs.u32u32 sr52.y, 16;" \
                        "movs.u32u32 sr52.z, 0;" \
                        "movs.u32u32 sr52.w, 8;" \
                        "movs.u32u32 sr53.x, 16;" \
                        "movs.u32u32 sr53.y, 24;" \
                        "movs.u32u32 sr53.z, 0xFF;" \
                        "movs.u32u32 sr53.w, 0xFF00;" \
                        "movs.u32u32 sr54.x, 0xFF0000;" \
                        "movs.u32u32 sr54.y, 0xFF000000;" \
                        : \
                        : "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : "sr50.x", "sr50.y", "sr50.z", "sr50.w", "sr51.x", "sr51.y", "sr51.z", "sr51.w" \
                       );


inline void mm_load_depth_d24s8_mv_g16r16(float16* depth_ptr, float8* mv_0_ptr, float8* mv_1_ptr, float8* mv_2_ptr, int x_index, int y_index, int image_width_lr)
{

   // size_t x_index = get_global_id(0);
   // size_t y_index = get_global_id(1);

   uint depth_coordInWords0 = (y_index - 1) * image_width_lr + (x_index - 1);
   uint depth_coordInWords1 = (y_index + 0) * image_width_lr + (x_index - 1);
   uint depth_coordInWords2 = (y_index + 1) * image_width_lr + (x_index - 1);

   if (x_index == 0 && y_index == 0) {      
      depth_coordInWords1 ++; // This is to fix out-of-bound for the first index in a vload3
   }
   if (x_index == 0 && y_index == 1) {
      depth_coordInWords0 ++; // This is to fix out-of-bound for the first index in a vload3
   }

    __asm__ __volatile__(
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 0>,   %[depth_coordInWords0], 66;" 
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 3>,   %[depth_coordInWords1], 66;" 
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 6>,   %[depth_coordInWords2], 66;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV0].s!< 0>, %[depth_coordInWords0], 82;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV1].s!< 0>, %[depth_coordInWords1], 82;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV2].s!< 0>, %[depth_coordInWords2], 82;"
#else
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 0>,   %[depth_coordInWords0], 65;" 
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 3>,   %[depth_coordInWords1], 65;" 
                        "isam.v.1d.s2en.mode6.base0 (u32)(xyzo)%[rD].s!< 6>,   %[depth_coordInWords2], 65;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV0].s!< 0>, %[depth_coordInWords0], 81;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV1].s!< 0>, %[depth_coordInWords1], 81;"
                        "isam.v.1d.s2en.mode6.base0 (f32)(xyzo)%[rMV2].s!< 0>, %[depth_coordInWords2], 81;"
#endif
                        "(ss)nop;"
                        // "(sy)nop;"
                        :  [rD] "=r" (*depth_ptr),
                           [rMV0] "=r" (*mv_0_ptr),
                           [rMV1] "=r" (*mv_1_ptr),
                           [rMV2] "=r" (*mv_2_ptr)
                        :  [depth_coordInWords0] "r" (depth_coordInWords0),
                           [depth_coordInWords1] "r" (depth_coordInWords1),
                           [depth_coordInWords2] "r" (depth_coordInWords2)
                        :  
                       );

      return;
}



inline float mm_load_color_lr_r11g11b10(uint2 color_lr_uv)
{
    __private float   regColor;

    __asm__ __volatile__(
                        // "isam.s2en.mode6.base0 (f32)(xooo)%[rC], %[color_lr_uv], 32;"                        
#ifdef E17
                        "isam.s2en.mode6.base0 (f32)(xooo)%[rC], %[color_lr_uv], 49;"                        
#else
                        "isam.s2en.mode6.base0 (f32)(xooo)%[rC], %[color_lr_uv], 48;"                        
#endif
                        "(ss)nop;"
                        // "(sy)nop;"
                        :  [rC] "=r" (regColor)
                        :  [color_lr_uv] "r" (color_lr_uv)
                        :  
                       );

      return regColor;
}

inline half4 mm_unpack_r11g11b10(float color_lr_r11g11b10, half pre_exposure)
{
    __private half4   regUnpacked;
    __private float4  exponents;
    __private float4  mantisa;
    __private half    max_color_val = 0.0f;    
    short ten = 10;

    __asm__ __volatile__(
                        "(rpt2)shrm.b32 %[mantisa].x,   (r)sr50.x, %[r11g11b10], (r)sr50.w;"                        
                        "(rpt2)shrm.b32 %[exponents].x, (r)sr51.z, %[r11g11b10],    sr51.y;" 
                        "nop;"   
                        "(rpt1)shlg.b32 %[mantisa].x, %[four], (r)%[mantisa].x, %[zero];"                    
                        "      shlg.b32 %[mantisa].z, %[five],    %[mantisa].z, %[zero];"  
                        "(rpt3)nop;"
                        "shlg.b16 %[rUnpacked].x, %[ten], %[exponents].x.lo, %[mantisa].x.lo;"                  
                        "shlg.b16 %[rUnpacked].y, %[ten], %[exponents].y.lo, %[mantisa].y.lo;"                  
                        "(nop3)shlg.b16 %[rUnpacked].z, %[ten], %[exponents].z.lo, %[mantisa].z.lo;"
                        "(rpt1)nop;"  
                        "sqrt %[rUnpacked].x, %[rUnpacked].x;" 
                        "sqrt %[rUnpacked].y, %[rUnpacked].y;"
                        "sqrt %[rUnpacked].z, %[rUnpacked].z;"
                        "(ss)nop;"
                        "(nop3) cmps.u.gt hr0.y, %[rUnpacked].x, %[rUnpacked].y;"   
                        "(nop3)sel.b16 %[max_color], %[rUnpacked].x, hr0.y, %[rUnpacked].y;"
                        "(nop3) cmps.u.gt hr0.y, %[max_color], %[rUnpacked].z;"
                        "(nop3)sel.b16 %[max_color], %[max_color].x, hr0.y, %[rUnpacked].z;"
                        "(nop3)add.f %[max_color], %[max_color], %[pre_exposure];"
                        "(rpt2)nop;"
                        "rcp     %[max_color], %[max_color];"
                        "mov.f16f16   %[rUnpacked].w, 0;"    // Set Alpha channel to 0
                        "(ss)(rpt2)mul.f   %[rUnpacked].x, (r)%[rUnpacked].x, %[max_color];"                        
                                        
                        // Deal with the 0 and subnorm later

                        :  [rUnpacked] "=r" (regUnpacked),
                           [exponents] "+r" (exponents),
                           [mantisa] "+r" (mantisa),
                           [max_color] "+r" (max_color_val)
                        :  [r11g11b10] "r" (color_lr_r11g11b10),
                           [pre_exposure]    "r" (pre_exposure),
                           [five]    "r" (0x5),
                           [four]    "r" (0x4),
                           [zero]    "r" (0x0),
                           [ten]    "r" (ten)                           
                        :  "hr0.y"
                       );

      return regUnpacked;
}

inline void mm_unpack_d24s8(float16 *depth_d24s8_ptr, const float d24s8_multiplier)
{    
   uint   mask = 0x00FFFFFF;
    __asm__ __volatile__(
#ifdef DEPTH_FP32
                        "nop;"
#else         
                        "(rpt3)shrm.b32 %[rD].s!< 0>,   0, (r)%[rD].s!< 0>, %[mask];"                        
                        "(rpt3)shrm.b32 %[rD].s!< 4>,   0, (r)%[rD].s!< 4>, %[mask];"
                        "      shrm.b32 %[rD].s!< 8>,   0,    %[rD].s!< 8>, %[mask];"
                        "(rpt3)cov.u32f32   %[rD].s!< 0>,      (r)%[rD].s!< 0>;"
                        "(rpt3)cov.u32f32   %[rD].s!< 4>,      (r)%[rD].s!< 4>;"
                        "      cov.u32f32   %[rD].s!< 8>,      (r)%[rD].s!< 8>;"
                        "(rpt3)mul.f    %[rD].s!< 0>,      (r)%[rD].s!< 0>, %[d24s8_multiplier];"
                        "(rpt3)mul.f    %[rD].s!< 4>,      (r)%[rD].s!< 4>, %[d24s8_multiplier];"
                        "      mul.f    %[rD].s!< 8>,         %[rD].s!< 8>, %[d24s8_multiplier];"
#endif                        
                        // Deal with the 0 and subnorm later

                        :  [rD] "+r" (*depth_d24s8_ptr)
                        :  [d24s8_multiplier] "r" (d24s8_multiplier),
                           [mask] "r" (mask)
                        :  
                       );

      return;
}

inline void mm_unpack_g16r16(float8 *mv_0_ptr, float8 *mv_1_ptr, float8 *mv_2_ptr, const float g16r16_multiplier)
{    
   uint   mask = 0x0000FFFF;
    __asm__ __volatile__(
#ifndef MV_FP16
                        "(rpt1)cov.u16f32 %[rMV_0].s!< 4>,  (r)%[rMV_0].s!< 2>.lo;"
                        "(rpt1)cov.u16f32 %[rMV_1].s!< 4>,  (r)%[rMV_1].s!< 2>.lo;"
                        "(rpt1)cov.u16f32 %[rMV_2].s!< 4>,  (r)%[rMV_2].s!< 2>.lo;"
                        "(rpt1)cov.u16f32 %[rMV_0].s!< 2>,  (r)%[rMV_0].s!< 1>.lo;"
                        "(rpt1)cov.u16f32 %[rMV_1].s!< 2>,  (r)%[rMV_1].s!< 1>.lo;"
                        "(rpt1)cov.u16f32 %[rMV_2].s!< 2>,  (r)%[rMV_2].s!< 1>.lo;"
                        "      cov.u16f32 %[rMV_0].s!< 1>,     %[rMV_0].s!< 0>.hi;"
                        "      cov.u16f32 %[rMV_1].s!< 1>,     %[rMV_1].s!< 0>.hi;"
                        "      cov.u16f32 %[rMV_2].s!< 1>,     %[rMV_2].s!< 0>.hi;"
                        "      cov.u16f32 %[rMV_0].s!< 0>,     %[rMV_0].s!< 0>.lo;"
                        "      cov.u16f32 %[rMV_1].s!< 0>,     %[rMV_1].s!< 0>.lo;"
                        "      cov.u16f32 %[rMV_2].s!< 0>,     %[rMV_2].s!< 0>.lo;"
                        "nop;"
                         "(rpt3)mul.f     %[rMV_0].s!< 0>,  (r)%[rMV_0].s!< 0>, %[g16r16_multiplier];"
                         "(rpt1)mul.f     %[rMV_0].s!< 4>,  (r)%[rMV_0].s!< 4>, %[g16r16_multiplier];"
                         "(rpt3)mul.f     %[rMV_1].s!< 0>,  (r)%[rMV_1].s!< 0>, %[g16r16_multiplier];"
                         "(rpt1)mul.f     %[rMV_1].s!< 4>,  (r)%[rMV_1].s!< 4>, %[g16r16_multiplier];"
                         "(rpt3)mul.f     %[rMV_2].s!< 0>,  (r)%[rMV_2].s!< 0>, %[g16r16_multiplier];"
                         "(rpt1)mul.f     %[rMV_2].s!< 4>,  (r)%[rMV_2].s!< 4>, %[g16r16_multiplier];"
#else
                        "(rpt1)cov.f16f32 %[rMV_0].s!< 4>,  (r)%[rMV_0].s!< 2>.lo;"
                        "(rpt1)cov.f16f32 %[rMV_1].s!< 4>,  (r)%[rMV_1].s!< 2>.lo;"
                        "(rpt1)cov.f16f32 %[rMV_2].s!< 4>,  (r)%[rMV_2].s!< 2>.lo;"
                        "(rpt1)cov.f16f32 %[rMV_0].s!< 2>,  (r)%[rMV_0].s!< 1>.lo;"
                        "(rpt1)cov.f16f32 %[rMV_1].s!< 2>,  (r)%[rMV_1].s!< 1>.lo;"
                        "(rpt1)cov.f16f32 %[rMV_2].s!< 2>,  (r)%[rMV_2].s!< 1>.lo;"
                        "      cov.f16f32 %[rMV_0].s!< 1>,     %[rMV_0].s!< 0>.hi;"
                        "      cov.f16f32 %[rMV_1].s!< 1>,     %[rMV_1].s!< 0>.hi;"
                        "      cov.f16f32 %[rMV_2].s!< 1>,     %[rMV_2].s!< 0>.hi;"
                        "      cov.f16f32 %[rMV_0].s!< 0>,     %[rMV_0].s!< 0>.lo;"
                        "      cov.f16f32 %[rMV_1].s!< 0>,     %[rMV_1].s!< 0>.lo;"
                        "      cov.f16f32 %[rMV_2].s!< 0>,     %[rMV_2].s!< 0>.lo;"
#endif       
                         // Now check if mv is 0 do motion calculation, else velocity calculation
                         
                        
                        // Deal with the 0 and subnorm later

                        :  [rMV_0] "+r" (*mv_0_ptr),
                           [rMV_1] "+r" (*mv_1_ptr),
                           [rMV_2] "+r" (*mv_2_ptr)
                        :  [g16r16_multiplier] "r" (g16r16_multiplier),
                           [mask] "r" (mask)
                        :  
                       );

      return;
}

inline void mm_calculate_mv( float2 *staticMV_ptr, float2 velocity, float depth, const float16 clipConst, 
                                    float x_index, float y_index, const float4 jitter_offsets,
                                    const float velocity_multiplier, const float velocity_bias, int image_width_lr, int image_height_lr)
{    
   
   
   float2 uv = (float2)((((float)x_index   + 0.5)/image_width_lr)*2 - 1, 1 - 2*(((float)y_index   + 0.5)/image_height_lr));   
   float2 pos = (float2)((float)(depth), 1.0);
   float4 prev_clip_pos = (float4)(0.0);
   float2 jitter_compensation = (float2)jitter_offsets.zw - (float2)jitter_offsets.xy;
   jitter_compensation.x /= image_width_lr;
   jitter_compensation.y /= image_height_lr;
   jitter_compensation *= 2;

   
   
    __asm__ __volatile__(
#ifndef FINAL_MV_INPUT      
                           "cmps.u.ne p0.x, %[rVelocity].x, 0;"
                           "(rpt5)nop;"
                           "br p0.x, #12;"
                           "(rpt3)mad.f32 %[rC].x, (r)%[clipConst].s!< 0>, %[uv].x,            %[zero];"
                           "(rpt3)mad.f32 %[rC].x, (r)%[clipConst].s!< 4>, %[uv].y, (r)(last)%[rC].x;"
                           "(rpt3)mad.f32 %[rC].x, (r)%[clipConst].s!< 8>, %[pos].x,  (r)(last)%[rC].x;"
                           "(rpt3)mad.f32 %[rC].x, (r)%[clipConst].s!<12>, %[pos].y,  (r)(last)%[rC].x;"
                           "(rpt5)nop;"
                           "rcp     %[rC].w, %[rC].w;"                           
                           "(ss)(rpt1)mul.f %[rC].x, (r)%[rC].x, %[rC].w;"
                           "(rpt3)nop;"                           
                           "(rpt1)add.f %[rStaticMV].x, (r)%[uv].x, (r)(neg)%[rC].x;"  
                           "(rpt1)nop;"                         
                           "(rpt1)add.f %[rStaticMV].x, (r)%[rStaticMV].x, (r)(neg)%[jitter_compensation].x;"
                           "(jp)br !p0.x, #4;"
                   
                           "(rpt1)add.f %[rStaticMV].x, (r)%[rVelocity].x, (neg)%[rVel_bias];"
                           "(rpt1)nop;"
                           "(rpt1)mul.f %[rStaticMV].x, (r)%[rStaticMV].x, %[rVel_mul];"
                           "(jp)nop;"
#else
                           "(rpt1)mov.f32f32 %[rStaticMV].x, (r)%[rVelocity].x;"
#endif                            


                        // Deal with the 0 and subnorm later

                        :  [rStaticMV] "=r" (*staticMV_ptr),
                           [rC] "+r" (prev_clip_pos)
                        :  [clipConst] "c" (clipConst), 
                           [rVelocity] "c" (velocity),  
                           [rVel_mul]  "c" (velocity_multiplier),                         
                           [rVel_bias]  "c" (velocity_bias),
                           [rD] "r" (depth),
                           [uv] "r" (uv),
                           [pos] "r" (pos),
                           [jitter_compensation] "r" (jitter_compensation),
                           [zero]                    "c" (0x00000000)
                        :  "p0.x"
                       );

      return;
}


inline void mm_find_max_depth_index(float2* max_uv_ptr, float* max_depth_ptr, float2* max_mv_ptr, float16 regDepth, float8 regMV_0, float8 regMV_1, float8 regMV_2, uint x_index, uint y_index, int image_width_lr, int image_height_lr)
{

   // boundry check, make sure OOB depths will not get selected in the max.
   if (x_index == 0 && y_index == 0) { 
      regDepth.s!<5> = regDepth.s!<4>;
      regDepth.s!<4> = regDepth.s!<3>;
      regMV_1.s!<5> = regMV_1.s!<3>;
      regMV_1.s!<4> = regMV_1.s!<2>;
      regMV_1.s!<3> = regMV_1.s!<1>;
      regMV_1.s!<2> = regMV_1.s!<0>;
   }
   else if (x_index == 0 && y_index == 1) {
      regDepth.s!<2> = regDepth.s!<1>;
      regDepth.s!<1> = regDepth.s!<0>;
      regMV_0.s!<5> = regMV_0.s!<3>;
      regMV_0.s!<4> = regMV_0.s!<2>;
      regMV_0.s!<3> = regMV_0.s!<1>;
      regMV_0.s!<2> = regMV_0.s!<0>;      
   }
   if (x_index == 0) {
      regDepth.s!<0> = -1.0;
      regDepth.s!<3> = -1.0;   
      regDepth.s!<6> = -1.0;   
   }
   if (x_index == image_width_lr-1) {
      regDepth.s!<2> = -1.0;
      regDepth.s!<5> = -1.0;   
      regDepth.s!<8> = -1.0;  
   }
   if (y_index == 0) {
      regDepth.s!<0> = -1.0;
      regDepth.s!<1> = -1.0;
      regDepth.s!<2> = -1.0;
   }
   if (y_index == image_height_lr-1) {
      regDepth.s!<6> = -1.0;
      regDepth.s!<7> = -1.0;
      regDepth.s!<8> = -1.0;
   }

   (*max_uv_ptr).x = x_index - 1;
   (*max_uv_ptr).y = y_index - 1;
   (*max_depth_ptr) = regDepth.s!<0>;
   (*max_mv_ptr).x = regMV_0.s!<0>;
   (*max_mv_ptr).y = regMV_0.s!<1>;

   if (regDepth.s!<1> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index;
      (*max_uv_ptr).y = y_index - 1;
      (*max_depth_ptr) = regDepth.s!<1>;
      (*max_mv_ptr).x = regMV_0.s!<2>;
      (*max_mv_ptr).y = regMV_0.s!<3>;
   }

   if (regDepth.s!<2> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index + 1;
      (*max_uv_ptr).y = y_index - 1;
      (*max_depth_ptr) = regDepth.s!<2>;
      (*max_mv_ptr).x = regMV_0.s!<4>;
      (*max_mv_ptr).y = regMV_0.s!<5>;
   }

   if (regDepth.s!<3> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index - 1;
      (*max_uv_ptr).y = y_index;
      (*max_depth_ptr) = regDepth.s!<3>;
      (*max_mv_ptr).x = regMV_1.s!<0>;
      (*max_mv_ptr).y = regMV_1.s!<1>;
   }

   if (regDepth.s!<4> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index;
      (*max_uv_ptr).y = y_index;
      (*max_depth_ptr) = regDepth.s!<4>;
      (*max_mv_ptr).x = regMV_1.s!<2>;
      (*max_mv_ptr).y = regMV_1.s!<3>;
   }

   if (regDepth.s!<5> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index + 1;
      (*max_uv_ptr).y = y_index;
      (*max_depth_ptr) = regDepth.s!<5>;
      (*max_mv_ptr).x = regMV_1.s!<4>;
      (*max_mv_ptr).y = regMV_1.s!<5>;
   }

   if (regDepth.s!<6> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index - 1;
      (*max_uv_ptr).y = y_index + 1;
      (*max_depth_ptr) = regDepth.s!<6>;
      (*max_mv_ptr).x = regMV_2.s!<0>;
      (*max_mv_ptr).y = regMV_2.s!<1>;
   }

   if (regDepth.s!<7> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index;
      (*max_uv_ptr).y = y_index + 1;
      (*max_depth_ptr) = regDepth.s!<7>;
      (*max_mv_ptr).x = regMV_2.s!<2>;
      (*max_mv_ptr).y = regMV_2.s!<3>;
   }

   if (regDepth.s!<8> > (*max_depth_ptr)) {
      (*max_uv_ptr).x = x_index + 1;
      (*max_uv_ptr).y = y_index + 1;
      (*max_depth_ptr) = regDepth.s!<8>;
      (*max_mv_ptr).x = regMV_2.s!<4>;
      (*max_mv_ptr).y = regMV_2.s!<5>;
   }

   
   return;                   

}




inline void mm_load_clip_to_prev_clip(__global const float* restrict clip_to_prev_clip, const float16 clipConst)
{
    __asm__ __volatile__(
                        "ldg.k c[&%[clipConst]], g[%[clip_to_prev_clip]], 4;"
                        "(ss)nop;"
                        :
                        :  [clip_to_prev_clip] "u" (clip_to_prev_clip),
                           [clipConst]  "c" (clipConst)
                        :
                       );
}


inline void mm_store(half4 color_hdr, float depth, float4 sample_0, float4 sample_1, float4 sample_2, float4 sample_3, float4 feature_0, int tile_id, int apron, int image_width_lr)
{
   int x_index = get_global_id(0);
   int y_index = get_global_id(1);   
   uint coordInWords = ((y_index)    * image_width_lr + x_index) * MODEL_INPUT_STRIDE_BYTES / 4;

   if (tile_id == 0) {
      y_index -= apron;
   }
   uint2 coord = (uint2)((uint)x_index,   (uint)y_index);
   float8 model_input = (float8)0.0;
   // float8 warped_color_hdr_fp16 = (float8)0.0;    
   half16 warped_color_hdr_fp16 = (half16)0.0;

   // uint coord_hr_0 = ((y_index * 2)     * INPUT_WIDTH*2 + x_index*2) * 2;
   // uint coord_hr_2 = ((y_index * 2 + 1) * INPUT_WIDTH*2 + x_index*2) * 2;

   uint2 coord_hr_0 = (uint2)((uint)x_index*2,       (uint)y_index*2);
   uint2 coord_hr_1 = (uint2)((uint)x_index*2 + 1,   (uint)y_index*2);
   uint2 coord_hr_2 = (uint2)((uint)x_index*2,       (uint)y_index*2 + 1);
   uint2 coord_hr_3 = (uint2)((uint)x_index*2 + 1,   (uint)y_index*2 + 1);

   // uint4 temp;
   // size_t sub_block_id_m = get_local_id(0);
   // short fiber_odd = (sub_block_id_m%2 == 1);

   // int x_index_even = x_index - (sub_block_id_m%2);
   // uint coordInWords = (((y_index)    * INPUT_WIDTH + x_index_even) * MODEL_INPUT_STRIDE_BYTES / 4) + (sub_block_id_m%2)*4;
   

   __asm__ __volatile__(
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rC_HDR].x, %[coord].x, 8;" 
                        // "cmps.u.ne p0.y, %[finber_odd], 0;"

                        // "(rpt2)cov.f32f16 %[rWarped_color].s!< 0>.lo, (r)%[rS0].s!<0>;"
                        // "(rpt2)cov.f32f16 %[rWarped_color].s!< 2>.lo, (r)%[rS1].s!<0>;"
                        // "(rpt2)cov.f32f16 %[rWarped_color].s!< 4>.lo, (r)%[rS2].s!<0>;"
                        // "(rpt2)cov.f32f16 %[rWarped_color].s!< 6>.lo, (r)%[rS3].s!<0>;"
                        "(rpt2)cov.f32f16 %[rWarped_color].s!< 0>, (r)%[rS0].s!<0>;"
                        "(rpt2)cov.f32f16 %[rWarped_color].s!< 4>, (r)%[rS1].s!<0>;"
                        "(rpt2)cov.f32f16 %[rWarped_color].s!< 8>, (r)%[rS2].s!<0>;"
                        "(rpt2)cov.f32f16 %[rWarped_color].s!<12>, (r)%[rS3].s!<0>;"

                        "(rpt2)cov.f16f32 %[model_input].s!<0>, (r)%[rC_HDR].s!<0>;" 
                        "      mov.f32f32 %[model_input].s!<3>,    %[rD].s!<0>;" 
                        "(rpt1)mov.f32f32 %[model_input].s!<4>, (r)%[rS0].s!<1>;" 
                        "(rpt1)mov.f32f32 %[model_input].s!<6>, (r)%[rS1].s!<2>;" 
                        "      mov.f32f32 %[rS0].s!<1>,            %[rS1].s!<0>;"
                        "      mov.f32f32 %[rS0].s!<2>,            %[rS2].s!<0>;"
                        "      mov.f32f32 %[rS0].s!<3>,            %[rS3].s!<0>;"
                        "      mov.f32f32 %[rS1].s!<0>,            %[model_input].s!<4>;"
                        "      mov.f32f32 %[rS1].s!<2>,            %[rS2].s!<1>;"
                        "      mov.f32f32 %[rS1].s!<3>,            %[rS3].s!<1>;"
                        "      mov.f32f32 %[rS2].s!<0>,            %[model_input].s!<5>;"
                        "      mov.f32f32 %[rS2].s!<1>,            %[model_input].s!<6>;"
                        "      mov.f32f32 %[rS2].s!<3>,            %[rS3].s!<2>;"                        
                        "(ss)nop;"      

                        // "stib.untyped.u32.1d.4.unpacked.mode4.base0 %[rWarped_color].s!< 0>, %[coord_hr_0].x, 9;" 
                        // "stib.untyped.u32.1d.4.unpacked.mode4.base0 %[rWarped_color].s!< 4>, %[coord_hr_2].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 0>, %[coord_hr_0].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 4>, %[coord_hr_1].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 8>, %[coord_hr_2].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!<12>, %[coord_hr_3].x, 9;"  

                        "(rpt3)mul.f  %[model_input].s!<0>,  (r)%[model_input].s!<0>,  %[f255];"
                        "(rpt3)mul.f  %[rF].s!<0>,  (r)%[rF].s!<0>,  %[f255];"
                        "(rpt3)mul.f  %[rS0].s!<0>,  (r)%[rS0].s!<0>,  %[f255];"
                        "(rpt3)mul.f  %[rS1].s!<0>,  (r)%[rS1].s!<0>,  %[f255];"
                        "(rpt3)mul.f  %[rS2].s!<0>,  (r)%[rS2].s!<0>,  %[f255];"
                        // "(rpt2)mul.f  %[rS3].s!<0>,  (r)%[rS3].s!<0>,  %[f255];"
                        "(rpt3)rndaz.f %[model_input].s!< 0>,  (r)%[model_input].s!< 0>;"
                        "(rpt3)rndaz.f %[rF].s!< 0>,  (r)%[rF].s!< 0>;"
                        "(rpt3)rndaz.f %[rS0].s!< 0>,  (r)%[rS0].s!< 0>;"
                        "(rpt3)rndaz.f %[rS1].s!< 0>,  (r)%[rS1].s!< 0>;"
                        "(rpt3)rndaz.f %[rS2].s!< 0>,  (r)%[rS2].s!< 0>;"
                        // "(rpt2)rndaz.f %[rS3].s!< 0>,  (r)%[rS3].s!< 0>;"
                        "(rpt3)max.f  %[model_input].s!<0>,  (r)%[model_input].s!<0>,  %[zero];"
                        "(rpt3)max.f  %[rF].s!<0>,  (r)%[rF].s!<0>,  %[zero];"
                        "(rpt3)max.f  %[rS0].s!<0>,  (r)%[rS0].s!<0>,  %[zero];"
                        "(rpt3)max.f  %[rS1].s!<0>,  (r)%[rS1].s!<0>,  %[zero];"
                        "(rpt3)max.f  %[rS2].s!<0>,  (r)%[rS2].s!<0>,  %[zero];"
                        // "(rpt2)max.f  %[rS3].s!<0>,  (r)%[rS3].s!<0>,  %[zero];"
                        "(rpt3)min.f  %[model_input].s!<0>,  (r)%[model_input].s!<0>,  %[f255];"
                        "(rpt3)min.f  %[rF].s!<0>,  (r)%[rF].s!<0>,  %[f255];"
                        "(rpt3)min.f  %[rS0].s!<0>,  (r)%[rS0].s!<0>,  %[f255];"
                        "(rpt3)min.f  %[rS1].s!<0>,  (r)%[rS1].s!<0>,  %[f255];"
                        "(rpt3)min.f  %[rS2].s!<0>,  (r)%[rS2].s!<0>,  %[f255];"
                        // "(rpt2)min.f  %[rS3].s!<0>,  (r)%[rS3].s!<0>,  %[f255];"
                        "(rpt3)cov.f32u32 %[model_input].s!<0>,        (r)%[model_input].s!<0>;"
                        "(rpt3)cov.f32u32 %[rF].s!<0>,        (r)%[rF].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS0].s!<0>,        (r)%[rS0].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS1].s!<0>,        (r)%[rS1].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS2].s!<0>,        (r)%[rS2].s!<0>;"
                        // "(rpt2)cov.f32u32 %[rS3].s!<0>,        (r)%[rS3].s!<0>;"
#ifndef UINT8_PACKING
                        "(rpt3)sub.s      %[model_input].s!<0>,        (r)%[model_input].s!<0>,         128;"
                        "(rpt3)sub.s      %[rF].s!<0>,        (r)%[rF].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS0].s!<0>,        (r)%[rS0].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS1].s!<0>,        (r)%[rS1].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS2].s!<0>,        (r)%[rS2].s!<0>,         128;"
                        // "(rpt2)sub.s      %[rS3].s!<0>,        (r)%[rS3].s!<0>,         128;"
#endif
                        "(rpt3)shlm.b32 %[model_input].s!<0>,  (r)sr52.z, (r)%[model_input].s!<0>,  (r)sr53.z;"
                        "(rpt3)shlm.b32 %[rF].s!<0>,   (r)sr52.z, (r)%[rF].s!<0>,   (r)sr53.z;"
                        "(rpt3)shlm.b32 %[rS0].s!<0>,  (r)sr52.z, (r)%[rS0].s!<0>,  (r)sr53.z;"
                        "(rpt3)shlm.b32 %[rS1].s!<0>,  (r)sr52.z, (r)%[rS1].s!<0>,  (r)sr53.z;"
                        "(rpt3)shlm.b32 %[rS2].s!<0>,  (r)sr52.z, (r)%[rS2].s!<0>,  (r)sr53.z;"
                        // "(rpt3)shlm.b32 %[rS3].s!<0>,  (r)sr52.z, (r)%[rS3].s!<0>,  (r)sr53.z;"                        
                        "       or.b  %[model_input].s!<0>,     %[model_input].s!<1>,  %[model_input].s!<0>;"
                        "       or.b  %[rF].s!<0>,     %[rF].s!<1>,  %[rF].s!<0>;"
                        "       or.b  %[rS0].s!<0>,     %[rS0].s!<1>,  %[rS0].s!<0>;"
                        "       or.b  %[rS1].s!<0>,     %[rS1].s!<1>,  %[rS1].s!<0>;"
                        "       or.b  %[rS2].s!<0>,     %[rS2].s!<1>,  %[rS2].s!<0>;"
                        // "       or.b  %[rS3].s!<0>,     %[rS3].s!<1>,  %[rS3].s!<0>;"
                        "       or.b  %[model_input].s!<0>,     %[model_input].s!<2>,  %[model_input].s!<0>;"
                        "       or.b  %[rF].s!<0>,     %[rF].s!<2>,  %[rF].s!<0>;"
                        "       or.b  %[rS0].s!<0>,     %[rS0].s!<2>,  %[rS0].s!<0>;"
                        "       or.b  %[rS1].s!<0>,     %[rS1].s!<2>,  %[rS1].s!<0>;"
                        "       or.b  %[rS2].s!<0>,     %[rS2].s!<2>,  %[rS2].s!<0>;"
                        // "       or.b  %[rS3].s!<0>,     %[rS3].s!<2>,  %[rS3].s!<0>;"
                        "       or.b  %[model_input].s!<0>,     %[model_input].s!<3>,  %[model_input].s!<0>;"
                        "       or.b  %[rF].s!<0>,     %[rF].s!<3>,  %[rF].s!<0>;"
                        "       or.b  %[rS0].s!<0>,     %[rS0].s!<3>,  %[rS0].s!<0>;"
                        "       or.b  %[rS1].s!<0>,     %[rS1].s!<3>,  %[rS1].s!<0>;"
                        "       or.b  %[rS2].s!<0>,     %[rS2].s!<3>,  %[rS2].s!<0>;"
                        // "       or.b  %[rS3].s!<0>,     %[rS3].s!<3>,  %[rS3].s!<0>;"
                        "mov.u32u32 %[model_input].s!<1>, %[rF].s!<0>;"
                        // "mov.u32u32 %[model_input].s!<1>, %[rS0].s!<0>;"
                        "mov.u32u32 %[model_input].s!<2>, %[rS0].s!<0>;"
                        "mov.u32u32 %[model_input].s!<3>, %[rS1].s!<0>;"
                        "mov.u32u32 %[model_input].s!<4>, %[rS2].s!<0>;"
#ifndef UINT8_PACKING
                        "(rpt2)mov.u32u32 %[model_input].s!<5>, %[neg128];"
#else
                        "(rpt2)mov.u32u32 %[model_input].s!<5>, %[zero];"
#endif
       "(rpt2)nop;"
                        // "br !p0.y, #3;"
                        // "(rpt3)mov.u32u32  %[rT].x, (r)%[model_input].s!<4>;"
                        // "(rpt3)mov.u32u32  %[model_input].s!<4>, (r)%[model_input].s!<0>;"
                        // "(jp)shfl.xor.w4.u32 %[model_input].s!<4>,  %[model_input].s!<4>, 1;"
                        // "shfl.xor.w4.u32 %[model_input].s!<5>,  %[model_input].s!<5>, 1;"
                        // "shfl.xor.w4.u32 %[model_input].s!<6>,  %[model_input].s!<6>, 1;"
                        // "shfl.xor.w4.u32 %[model_input].s!<7>,  %[model_input].s!<7>, 1;"
                        // "(ss)br !p0.y, #3;"
                        // "(rpt3)mov.u32u32  %[model_input].s!<0>, (r)%[model_input].s!<4>;"
                        // "(rpt3)mov.u32u32  %[model_input].s!<4>, (r)%[rT].x;"
                        // "(jp)nop;"
                        // "(rpt3)nop;"
                       
                        // "stib.untyped.u32.1d.4.unpacked.mode4.base0          %[model_input].s!<0>,    %[coordInWords], 10;"
                        // "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset8  %[model_input].s!<4>,    %[coordInWords], 10;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0          %[model_input].s!<0>,    %[coordInWords], 10;"
#if MODEL_INPUT_STRIDE_BYTES == 20
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0.offset4  %[model_input].s!<4>,    %[coordInWords], 10;"
#else
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4  %[model_input].s!<4>,    %[coordInWords], 10;"
#endif
                        "(ss)nop;"
                        :  [model_input] "+r" (model_input)
                        :  [rC_HDR] "r" (color_hdr),
                           [rWarped_color] "r" (warped_color_hdr_fp16),
                           [rD]   "r" (depth),
                           [rS0]   "r" (sample_0),
                           [rS1]   "r" (sample_1),
                           [rS2]   "r" (sample_2),
                           [rS3]   "r" (sample_3),
                           [rF]   "r" (feature_0),
                           [coord] "r" (coord),
                           [coord_hr_0] "r" (coord_hr_0),
                           [coord_hr_1] "r" (coord_hr_1),
                           [coord_hr_2] "r" (coord_hr_2),
                           [coord_hr_3] "r" (coord_hr_3),
                           [coordInWords] "r" (coordInWords),
                           [f255]  "c" (0x437F0000),
                           [zero]      "c" (0x00000000), 
                           [neg128]      "c" (0x80808080)
                        :
                       );

}

inline void mm_store_zero(int input_width)
{
   size_t x_index = get_global_id(0);
   size_t y_index = get_global_id(1);
   uint coordInWords = ((y_index)    * input_width + x_index) * MODEL_INPUT_STRIDE_BYTES / 4;

   uint2 coord = (uint2)((uint)x_index,   (uint)y_index);
   float8 model_input = (float8)0.0;
   half16 warped_color_hdr_fp16 = (half16)0.0;
   half4 color_hdr = (half4)0.0;

    

   uint2 coord_hr_0 = (uint2)((uint)x_index*2,       (uint)y_index*2);
   uint2 coord_hr_1 = (uint2)((uint)x_index*2 + 1,   (uint)y_index*2);
   uint2 coord_hr_2 = (uint2)((uint)x_index*2,       (uint)y_index*2 + 1);
   uint2 coord_hr_3 = (uint2)((uint)x_index*2 + 1,   (uint)y_index*2 + 1);

   __asm__ __volatile__(
                        // "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rC_HDR].x, %[coord].x, 7;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rC_HDR].x, %[coord].x, 8;"                         
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 0>, %[coord_hr_0].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 4>, %[coord_hr_1].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!< 8>, %[coord_hr_2].x, 9;" 
                        "stib.typed.f16.2d.4.unpacked.mode4.base0 %[rWarped_color].s!<12>, %[coord_hr_3].x, 9;"                                                                                                                
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0          %[model_input].s!<0>,    %[coordInWords], 10;"
#if MODEL_INPUT_STRIDE_BYTES == 20
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0.offset4  %[model_input].s!<4>,    %[coordInWords], 10;"
#else
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4  %[model_input].s!<4>,    %[coordInWords], 10;"
#endif
                        "(ss)nop;"
                        :  [model_input] "+r" (model_input)
                        :  [rC_HDR] "r" (color_hdr),
                           [rWarped_color] "r" (warped_color_hdr_fp16),                                                      
                           [coord] "r" (coord),
                           [coord_hr_0] "r" (coord_hr_0),
                           [coord_hr_1] "r" (coord_hr_1),
                           [coord_hr_2] "r" (coord_hr_2),
                           [coord_hr_3] "r" (coord_hr_3),
                           [coordInWords] "r" (coordInWords)
                        :  
                       );

}

inline void mm_sample_prev_feature(float4 *sample_0_ptr, float4 *sample_1_ptr, float4 *sample_2_ptr, float4 *sample_3_ptr, float4 *feature_0_ptr, float2 feature_hr_uv0, float2 feature_hr_uv1, float2 feature_hr_uv2, float2 feature_hr_uv3)
{

    __asm__ __volatile__(
                        // "isam.s2en.mode6.base0 (f32)(xooo)%[rC], %[color_lr_uv], 32;" 
#ifdef E17
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_0].x, %[feature_hr_uv0],  96;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_1].x, %[feature_hr_uv1],  96;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_2].x, %[feature_hr_uv2],  96;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_3].x, %[feature_hr_uv3],  96;"                       
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].x,   %[feature_hr_uv0], 112;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].y,   %[feature_hr_uv1], 112;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].z,   %[feature_hr_uv2], 112;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].w,   %[feature_hr_uv3], 112;"
#else
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_0].x, %[feature_hr_uv0],  98;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_1].x, %[feature_hr_uv1],  98;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_2].x, %[feature_hr_uv2],  98;"                       
                        "sam.s2en.mode6.base0 (f32)(xyzw)%[rS_3].x, %[feature_hr_uv3],  98;"                       
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].x,   %[feature_hr_uv0], 114;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].y,   %[feature_hr_uv1], 114;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].z,   %[feature_hr_uv2], 114;"
                        "sam.s2en.mode6.base0 (f32)(xOOO)%[rF].w,   %[feature_hr_uv3], 114;"
#endif
                        "(ss)nop;"
                        "(sy)nop;"
                        :  [rS_0] "=r" (*sample_0_ptr),
                           [rS_1] "=r" (*sample_1_ptr),
                           [rS_2] "=r" (*sample_2_ptr),
                           [rS_3] "=r" (*sample_3_ptr),
                           [rF] "=r" (*feature_0_ptr)
                        :  [feature_hr_uv0] "r" (feature_hr_uv0),
                           [feature_hr_uv1] "r" (feature_hr_uv1),
                           [feature_hr_uv2] "r" (feature_hr_uv2),
                           [feature_hr_uv3] "r" (feature_hr_uv3)
                        :  
                       );

      return;
}





__kernel void preprocess(  
                           __read_only image2d_t         color_lr,                                                                  
                           __read_only image1d_buffer_t  depth_lr,
                           __read_only image1d_buffer_t  mv_lr,
                           __global const float* restrict clip_to_prev_clip,
                           const float4         jitter_offsets,
                           __read_only image2d_t        prev_color_hr,
                           __read_only image2d_t        prev_feature_hr,                           
                           float                         scale_factor,   
                           half                          pre_exposure,                                                 
                           __write_only image2d_t        color_lr_rgba_fp16, // This is unpacked hdr color for blending in the last layer,
                           __write_only image2d_t        previous_color_warped_hr_rgba_fp16,
                           __write_only image1d_buffer_t model_input, // This is the packed input model,                            
                           int                          apron,
                           int                          tile_id,
                           int                          num_tiles,
                           int                          image_width_lr,
                           int                          image_height_lr,
                           int                          cropped_width_lr,
                           int                          cropped_height_lr)
{
   

#ifdef BICUBIC   
   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | QCOM_CLK_FILTER_BICUBIC;
#else
#ifdef BILINEAR 
   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
#else
   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;   
#endif
#endif
   sampler_t sampler_point = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;   


   int x_index = get_global_id(0);
   int y_index = get_global_id(1) + (image_height_lr / num_tiles) * tile_id - apron;
   // size_t y_index = get_global_id(1);
   float16 regDepth_d24s8;
   float8 velocity_g16r16_0;
   float8 velocity_g16r16_1;
   float8 velocity_g16r16_2;

   if ((y_index < 0) || (y_index >= image_height_lr)) {
      return;
   }
   // else {
      // if (y_index < 0) {
         
      //    y_index = 0;
      // }
      // if (y_index >= INPUT_HEIGHT) {
      //    y_index = INPUT_HEIGHT-1;
      // }
       
      // uint   lr_coord = y_index * INPUT_WIDTH + x_index;

      float2 uv = (float2)((float)x_index,   (float)y_index);
      uint2  uv_uint = (uint2)((uint)x_index,   (uint)y_index);
      // float2 uv_float = (float2)((float)x_index + 0.5,   (float)y_index + 0.5);
      
      // if (x_index >= 8 | y_index >= 8 ) return;
      // if (y_index < 304 | y_index >= 312 ) return;
      // if (x_index < 352 | x_index >= 360 | y_index < 312 | y_index >= 320 ) return;

      float   regColor;
      const float d24s8_multiplier = (float)(1/16777215.0);
      const float g16r16_multiplier = (float)(1/65535.0);
      const float velocity_multiplier = (float)(1/(0.499*0.5));
      const float velocity_bias = (float)(32767.0 / 65535.0);
      const float16 clipConst = (float16)(0);

      MM_SETUP()
      float sample_r11g11b10 = mm_load_color_lr_r11g11b10(uv_uint);
      
      
      
      mm_load_depth_d24s8_mv_g16r16(&regDepth_d24s8, &velocity_g16r16_0, &velocity_g16r16_1, &velocity_g16r16_2, x_index, y_index, image_width_lr);                            

      mm_load_clip_to_prev_clip(clip_to_prev_clip, clipConst);   

      __asm__ __volatile__("(sy)nop;");   
      half4 sample_r11g11b10_unpacked = mm_unpack_r11g11b10(sample_r11g11b10, pre_exposure);
      mm_unpack_d24s8(&regDepth_d24s8, d24s8_multiplier);

      float2 max_uv = (float2)(0.0);
      float max_depth = -1.0;
      float2 max_mv = (float2)(0.0);

      mm_unpack_g16r16(&velocity_g16r16_0, &velocity_g16r16_1, &velocity_g16r16_2, g16r16_multiplier);

      mm_find_max_depth_index(&max_uv, &max_depth, &max_mv, regDepth_d24s8, velocity_g16r16_0, velocity_g16r16_1, velocity_g16r16_2, x_index, y_index, image_width_lr, image_height_lr);

      float2 final_motion = (float2)(0.0);
      mm_calculate_mv( &final_motion, max_mv, max_depth, clipConst, 
                        max_uv.x, max_uv.y, jitter_offsets,
                        velocity_multiplier, velocity_bias, image_width_lr, image_height_lr);

      // Input motion is in uv space at the cropped resolution
      final_motion.x *= cropped_width_lr  * MV_SCALE_X;
      final_motion.y *= cropped_height_lr * MV_SCALE_Y;

      float2 sample_uv_0 = (float2)(((2*uv.x)     + (2 * final_motion.x) + 0.5), ((2*uv.y)     - (2 * final_motion.y) + 0.5));  // regMV_min.x x 2?????
      float2 sample_uv_1 = (float2)(((2*uv.x)     + (2 * final_motion.x) + 1.5), ((2*uv.y)     - (2 * final_motion.y) + 0.5));
      float2 sample_uv_2 = (float2)(((2*uv.x)     + (2 * final_motion.x) + 0.5), ((2*uv.y)     - (2 * final_motion.y) + 1.5));
      float2 sample_uv_3 = (float2)(((2*uv.x)     + (2 * final_motion.x) + 1.5), ((2*uv.y)     - (2 * final_motion.y) + 1.5));            
   
      
      float4 sample_0;
      float4 sample_1;
      float4 sample_2;
      float4 sample_3;   
      float4 feature_0;

      mm_sample_prev_feature(&sample_0, &sample_1, &sample_2, &sample_3, &feature_0, sample_uv_0, sample_uv_1, sample_uv_2, sample_uv_3);

      mm_store(sample_r11g11b10_unpacked, regDepth_d24s8.s4, sample_0, sample_1, sample_2, sample_3, feature_0, tile_id, apron, image_width_lr);
   // }


}
