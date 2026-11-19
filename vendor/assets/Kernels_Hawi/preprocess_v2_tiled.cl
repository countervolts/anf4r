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


#define BICUBIC
// #define BILINEAR
// #define POINT


#define MM_SETUP()      \
    if (get_global_id(0) == 123456789) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regDepth_0.xyzw  = convert_float4(read_imageh(color_lr, get_global_id(0))); \
       regDepth_0.xyzw += convert_float4(read_imageh(depth_lr, get_global_id(0))); \
       regDepth_0.xyzw += convert_float4(read_imagef(mv_lr, get_global_id(0))); \
       regDepth_0.xyzw += convert_float4(read_imagei(prev_color_hr, sampler, uv));\
       regDepth_0.xyzw += convert_float4(read_imagei(prev_feature_hr, sampler, uv));\
       write_imagef(model_input, 0, regDepth_0.xyzw);\
    }\
    __asm__ __volatile__( \
                        "movs.u32u32 sr48.x, 0;"\
                        "movs.u32u32 sr48.y, 8;"\
                        "movs.u32u32 sr48.z, 16;"\
                        "movs.u32u32 sr48.w, 24;"\
                        : \
                        : "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : "sr48.x", "sr48.y", "sr48.z", "sr48.w"\
                       );


inline float4 mm_load_depth(uint depth_coordInWords)
{
    __private float4   regDepth;

    __asm__ __volatile__(
                        "isam.v.1d.s2en.mode6.base0        (f32)(xyzo)%[rD].s!< 0>, %[depth_coordInWords], 48;"                        
                        "(ss)nop;"
                        :  [rD] "=r" (regDepth)
                        :  [depth_coordInWords] "r" (depth_coordInWords)
                        :  
                       );

      return regDepth;
}

inline float8 mm_load_mv(uint mv_coordInWords)
{
    __private float8   regMV;

    __asm__ __volatile__(
                        "isam.v.1d.s2en.mode6.base0         (f32)(xyzo)%[rMV].s!< 0>, %[mv_coordInWords], 64;"                        
                        // "isam.v.1d.s2en.mode6.base0.offset4 (f32)(xyoo)%[rMV].s!< 4>, %[mv_coordInWords], 64;"                        
                        "(ss)nop;"
                        :  [rMV] "=r" (regMV)
                        :  [mv_coordInWords] "r" (mv_coordInWords)
                        :  
                       );

      return regMV;
}


inline float mm_load_color(uint color_coordInWords)
{
    __private float   regColor;


    __asm__ __volatile__(                        
                        // "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rC].s!< 0>, %[color_coordInWords], 2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xooo)%[rC].s0,  %[color_coordInWords], 32;"
                        "(ss)nop;"
                       
                        :  [rC] "=r" (regColor)
                        :  [color_coordInWords] "r" (color_coordInWords)
                        :  
                       );

      return regColor;
}

inline float2 mm_load_mvs_xy(uint mvs_coordInWords, short shouldLoad)
{
    __private float2   regMvs;
   

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"
                        "(rpt5)nop;"
                        "br !p0.x, #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyoo)%[rM].s0,  %[mvs_coordInWords], 64;"
                        "(jp)nop;"                                                
                        :  [rM] "=r" (regMvs)
                        :  [mvs_coordInWords] "r" (mvs_coordInWords),
                           [shouldLoad]   "r" (shouldLoad)
                        :  "p0.x"
                       );

      return regMvs;
}

inline float2 mm_find_min_mv(float4 regDepth_0, float4 regDepth_1, float4 regDepth_2, float8 regMV_0, float8 regMV_1, float8 regMV_2, uint x_index, uint y_index)
{

   // if (x_index == 0 && y_index == 0) {
   //    regDepth_1.z = regDepth_1.y;
   //    regDepth_1.y = regDepth_1.x;
   // }
   // if (x_index == 0 && y_index == 1) {
   //    regDepth_0.z = regDepth_0.y;
   //    regDepth_0.y = regDepth_0.x;
   // }

   float2 mv_min;
   short  linearIndex = get_local_id(0);
    __asm__ __volatile__(
                        // "mova  a0, %[linearIndex];"  
                        "(nop3) cmps.u.eq p0.y, %[x_index], 0;"
                        "       cmps.u.eq p0.z, %[y_index], 0;"
                        "(nop3) cmps.u.eq p0.w, %[y_index], 1;"
                        "cov.f16f32 %[rMV0].s!<5>, %[rMV0].s!<2>.hi;"
                        "cov.f16f32 %[rMV1].s!<5>, %[rMV1].s!<2>.hi;"
                        "cov.f16f32 %[rMV2].s!<5>, %[rMV2].s!<2>.hi;"
                        "cov.f16f32 %[rMV0].s!<4>, %[rMV0].s!<2>.lo;"
                        "cov.f16f32 %[rMV1].s!<4>, %[rMV1].s!<2>.lo;"
                        "cov.f16f32 %[rMV2].s!<4>, %[rMV2].s!<2>.lo;"
                        "cov.f16f32 %[rMV0].s!<3>, %[rMV0].s!<1>.hi;"
                        "cov.f16f32 %[rMV1].s!<3>, %[rMV1].s!<1>.hi;"
                        "cov.f16f32 %[rMV2].s!<3>, %[rMV2].s!<1>.hi;"
                        "cov.f16f32 %[rMV0].s!<2>, %[rMV0].s!<1>.lo;"
                        "cov.f16f32 %[rMV1].s!<2>, %[rMV1].s!<1>.lo;"
                        "cov.f16f32 %[rMV2].s!<2>, %[rMV2].s!<1>.lo;"
                        "cov.f16f32 %[rMV0].s!<1>, %[rMV0].s!<0>.hi;"
                        "cov.f16f32 %[rMV1].s!<1>, %[rMV1].s!<0>.hi;"
                        "cov.f16f32 %[rMV2].s!<1>, %[rMV2].s!<0>.hi;"
                        "cov.f16f32 %[rMV0].s!<0>, %[rMV0].s!<0>.lo;"
                        "cov.f16f32 %[rMV1].s!<0>, %[rMV1].s!<0>.lo;"
                        "cov.f16f32 %[rMV2].s!<0>, %[rMV2].s!<0>.lo;"
                        "(rpt3)nop;"
                        "br !p0.y, #11;"                        
                        "br !p0.z, #5;"
                        "mov.f32f32 %[rD1].z, %[rD1].y;"
                        "mov.f32f32 %[rD1].y, %[rD1].x;"
                        "mov.f32f32 %[rMV1].z, %[rMV1].y;"
                        "mov.f32f32 %[rMV1].y, %[rMV1].x;"
                        "(jp)br !p0.w, #5;"
                        "mov.f32f32 %[rD0].z, %[rD0].y;"
                        "mov.f32f32 %[rD0].y, %[rD0].x;"
                        "mov.f32f32 %[rMV0].z, %[rMV0].y;"
                        "mov.f32f32 %[rMV0].y, %[rMV0].x;"
                        "(jp)nop;"
                        "(nop3) cmps.u.ne r0.y, %[y_index], 0;"    
                        "(rpt2)sel.b32 %[rD0].s!< 0>, (r)%[rD0].s!< 0>, r0.y, %[max_val];"
                        "(nop3) cmps.u.lt r0.y, %[y_index], %[height];"   
                        "(rpt2)sel.b32 %[rD2].s!< 0>, (r)%[rD2].s!< 0>, r0.y, %[max_val];"
                        "(nop3) cmps.u.ne r0.y, %[x_index], 0;"   
                        "sel.b32 %[rD0].s!< 0>, (r)%[rD0].s!< 0>, r0.y, %[max_val];"
                        "sel.b32 %[rD1].s!< 0>, (r)%[rD1].s!< 0>, r0.y, %[max_val];"
                        "sel.b32 %[rD2].s!< 0>, (r)%[rD2].s!< 0>, r0.y, %[max_val];"
                        "(nop3) cmps.u.lt r0.y, %[x_index], %[width];"   
                        "sel.b32 %[rD0].s!< 2>, (r)%[rD0].s!< 2>, r0.y, %[max_val];"
                        "sel.b32 %[rD1].s!< 2>, (r)%[rD1].s!< 2>, r0.y, %[max_val];"
                        "sel.b32 %[rD2].s!< 2>, (r)%[rD2].s!< 2>, r0.y, %[max_val];"
                        "(rpt1)mov.f32f32 %[mv_min], (r)%[rMV0].s!< 0>;"
                        // "(rpt1)nop;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD0].s!< 1>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD0].s!< 1>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV0].s!< 2>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD0].s!< 2>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD0].s!< 2>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV0].s!< 4>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD1].s!< 0>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD1].s!< 0>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV1].s!< 0>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD1].s!< 1>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD1].s!< 1>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV1].s!< 2>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD1].s!< 2>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD1].s!< 2>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV1].s!< 4>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD2].s!< 0>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD2].s!< 0>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV2].s!< 0>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD2].s!< 1>;"
                        "(nop1) sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD2].s!< 1>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV2].s!< 2>;"
                        "(nop3) cmps.u.le r0.y, %[rD0].s!< 0>, %[rD2].s!< 2>;"
                        "sel.b32 %[rD0].s!< 0>, %[rD0].s!< 0>, r0.y, %[rD2].s!< 2>;"
                        "(rpt1)sel.b32 %[mv_min].s!< 0>, (r)%[mv_min].s!< 0>, r0.y, (r)%[rMV2].s!< 4>;"
                        :  [mv_min] "=r" (mv_min) 
                        :  [rD0] "r" (regDepth_0),
                           [rD1] "r" (regDepth_1),
                           [rD2] "r" (regDepth_2),
                           [rMV0] "r" (regMV_0),
                           [rMV1] "r" (regMV_1),
                           [rMV2] "r" (regMV_2),
                           [x_index]  "r" (x_index),
                           [y_index]  "r" (y_index),
                           [width]   "c" (INPUT_WIDTH -1),
                           [height]  "c" (INPUT_HEIGHT-1),
                           [linearIndex]             "r" (linearIndex),
                           [max_val]  "c" (0x7f7fffff)                           
                        :  "r0.y", "r0.z", "p0.y", "p0.z", "p0.w", "sr48.x"
                       );

   return  mv_min;                   

}

inline float4 mm_quantize_depth(float regDepth)
{
   float4 quantized_depth;
   float4 quantized_depth_frac;

    __asm__ __volatile__(                        
                        "mov.f32f32 %[rD_q].s!<0>, %[rD].s!<0>;" 
                        "mul.f  %[rD_q].s!<1>, %[rD].s!<0>, %[f255];"
                        "mul.f  %[rD_q].s!<2>, %[rD].s!<0>, %[f255e2];"
                        "(nop1)mul.f  %[rD_q].s!<3>, %[rD].s!<0>, %[f255e3];"                        
                        "(rpt2)floor.f %[rD_q_f].s!<1>, (r)%[rD_q].s!<1>;"
                        "mov.f32f32 %[rD_q_f].s!<0>, %[rD].s!<0>;" 
                        "(rpt2)add.f   %[rD_q_f].s!<1>, (r)(neg)%[rD_q_f].s!<1>, (r)%[rD_q].s!<1>;"
                        "(rpt3)nop;"
                        "(rpt2)mad.f32 %[rD_q_f].s!<0>, %[f255_rcp], (r)(neg)%[rD_q_f].s!<1>, (r)%[rD_q_f].s!<0>;"
                        "mad.f32 %[rD_q_f].s!<3>, %[f255_rcp], (neg)%[rD_q_f].s!<3>, (r)%[rD_q_f].s!<3>;"
                        "(rpt3)mul.f  %[rD_q_f].s!<0>, (r)%[rD_q_f].s!<0>,  %[f255];"
                        "(rpt3)rndaz.f %[rD_q_f].s!<0>,  (r)%[rD_q_f].s!<0>;"
                        "(rpt3)cov.f32u32 %[rD_q_f].s!<0>, (r)%[rD_q_f].s!<0>;"
                        "(rpt3)sub.s %[rD_q_f].s!<0>, (r)%[rD_q_f].s!<0>, 128;"
                        "(rpt3)and.b  %[rD_q].s!<0>, (r)%[rD_q_f].s!<0>,  255;" // s0-s3                             
                        :  [rD_q]         "+r" (quantized_depth),
                           [rD_q_f]         "+r" (quantized_depth_frac)
                        :  [rD]           "r" (regDepth),                           
                           [f255]         "c" (0x437F0000),                           
                           [f255e2]       "c" (0x477E0100),
                           [f255e3]       "c" (0x4B7D02FF),
                           [f255_rcp]     "c" (0x3B808081)
                        :  "p0.x"
                       );

   return quantized_depth;
}

inline void mm_find_mvs(uint local_id, short shouldLoad, float2 mvs_value)
{

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"                        
                        "cmps.u.eq p0.y, %[local_id], sr48.x;"
                        "(rpt4)nop;"
                        "br !p0.x, #3;"
                        "(jp)br !p0.y, #2;"
                        "(rpt1)mov.f32f32 sr48.y, (r)%[mvs_value].x;"                        
                        "(jp)(ss)nop;"
                        :  
                        :  [local_id] "r" (local_id),
                           [shouldLoad] "r" (shouldLoad),
                           [mvs_value] "r" (mvs_value)                           
                        :  "p0.x", "p0.y", "sr48.x", "sr48.y", "sr48.z"
                       );

}

// Add comment about what the CL code will look like


inline float2 mm_get_final_mvs(float scale_factor)
{
   float2 mvs_final;
    __asm__ __volatile__(
                        // "(rpt1)mul.f    %[rMVS].s0, %[scale_factor], (r)sr48.y;"
                        "mad.f32    %[rMVS].s0, %[scale_factor], sr48.z, 0.5;"
                        "(ul)mad.f32    %[rMVS].s1, %[scale_factor], sr48.y, (neg)0.5;"
                        // "(rpt1)mov.f32f32 %[rMVS], (r)sr48.y;" 
                        "(ss)nop;"
                        :  [rMVS] "=r" (mvs_final)
                        :  [scale_factor] "r" (scale_factor)                           
                        :  "sr48.y", "sr48.z"
                       );

   return mvs_final;
}


inline void mm_store(float4 sample_0, float4 sample_1, float4 sample_2, float4 sample_3, float4 feature, float regColor, float4 regDepth, int tile_id)
{
   size_t x_index = get_global_id(0);
   size_t y_index = get_global_id(1);

   // if (tile_id == 0) y_index -= 4;

   uint modelInputStrideInWords = MODEL_INPUT_STRIDE_BYTES / 4;

   uint coordInWords = ((y_index)    * INPUT_WIDTH + x_index) * modelInputStrideInWords;   
   float8 model_input;
   // uint coordInWords_color    = (y_index_lr)    * INPUT_WIDTH * modelInputStrideInWords + x_index_lr * modelInputStrideInWords;
   // // uint coordInWords_depth    = (y_index_lr)    * INPUT_WIDTH * 7 + x_index_lr * 7 + 1;
   // // uint coordInWords_features = (y_index_lr)    * INPUT_WIDTH * 7 + x_index_lr * 7 + 2;// + (get_global_id(1) % 2)*2 + (get_global_id(0)%2);
   // uint coordInWords_sample   = (y_index_lr)    * INPUT_WIDTH * modelInputStrideInWords + x_index_lr * modelInputStrideInWords + 3 + (get_global_id(1) % 2)*2 + (get_global_id(0)%2);

   // short  linearIndex = get_local_id(0);
   // uint fiber_01 = ((get_local_id(1))   * 8 + get_local_id(0) + 1);
   // uint fiber_10 = ((get_local_id(1)+1) * 8 + get_local_id(0));
   // uint fiber_11 = ((get_local_id(1)+1) * 8 + get_local_id(0) + 1);

   __asm__ __volatile__(
                        "      mov.f32f32 %[rS00].s!<3>,           %[rF].s!<0>;"
                        "      mov.f32f32 %[rS01].s!<3>,           %[rF].s!<1>;"
                        "      mov.f32f32 %[rS10].s!<3>,           %[rF].s!<2>;"
                        "      mov.f32f32 %[rS11].s!<3>,           %[rF].s!<3>;"
                        "(rpt3)shrm.b32   %[model_input].s!<0>, (r)sr48.x,  %[rC],        255;"
                        "(rpt3)sub.s      %[model_input].s!<0>, (r)%[model_input].s!<0>,  128;"
                        "(rpt3)and.b      %[model_input].s!<0>, (r)%[model_input].s!<0>,  255;"
                        "      mov.f32f32 %[model_input].s!<5>,    %[rD].s!<0>;"
                        "(rpt2)shl.b      %[rD].s!<1>,          (r)%[rD].s!<1>,     (r)sr48.y;"
                        "(rpt3)mul.f      %[rS00].s!<0>,        (r)%[rS00].s!<0>,     %[f255];"
                        "(rpt3)mul.f      %[rS01].s!<0>,        (r)%[rS01].s!<0>,     %[f255];"
                        "(rpt3)mul.f      %[rS10].s!<0>,        (r)%[rS10].s!<0>,     %[f255];"
                        "(rpt3)mul.f      %[rS11].s!<0>,        (r)%[rS11].s!<0>,     %[f255];"
                        // "(rpt3)rndaz.f    %[rS00].s!<0>,        (r)%[rS00].s!<0>;"
                        // "(rpt3)rndaz.f    %[rS01].s!<0>,        (r)%[rS01].s!<0>;"
                        // "(rpt3)rndaz.f    %[rS10].s!<0>,        (r)%[rS10].s!<0>;"
                        // "(rpt3)rndaz.f    %[rS11].s!<0>,        (r)%[rS11].s!<0>;"
                        "(rpt3)max.f      %[rS00].s!<0>,        (r)%[rS00].s!<0>,     %[zero];"
                        "(rpt3)max.f      %[rS01].s!<0>,        (r)%[rS01].s!<0>,     %[zero];"
                        "(rpt3)max.f      %[rS10].s!<0>,        (r)%[rS10].s!<0>,     %[zero];"
                        "(rpt3)max.f      %[rS11].s!<0>,        (r)%[rS11].s!<0>,     %[zero];"
                        "(rpt3)min.f      %[rS00].s!<0>,        (r)%[rS00].s!<0>,     %[f255];"
                        "(rpt3)min.f      %[rS01].s!<0>,        (r)%[rS01].s!<0>,     %[f255];"
                        "(rpt3)min.f      %[rS10].s!<0>,        (r)%[rS10].s!<0>,     %[f255];"
                        "(rpt3)min.f      %[rS11].s!<0>,        (r)%[rS11].s!<0>,     %[f255];"
                        "(rpt2)shl.b      %[model_input].s!<1>, (r)%[model_input].s!<1>,  (r)sr48.y;"                          
                        "(rpt3)cov.f32u32 %[rS00].s!<0>,        (r)%[rS00].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS10].s!<0>,        (r)%[rS10].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS01].s!<0>,        (r)%[rS01].s!<0>;"
                        "(rpt3)cov.f32u32 %[rS11].s!<0>,        (r)%[rS11].s!<0>;"
                        "      or.b       %[model_input].s!<0>,    %[model_input].s!<1>,  %[model_input].s!<0>;"
                        "      or.b       %[model_input].s!<5>,    %[model_input].s!<5>,  %[rD].s!<1>;"  
                        "(rpt3)sub.s      %[rS00].s!<0>,        (r)%[rS00].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS01].s!<0>,        (r)%[rS01].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS10].s!<0>,        (r)%[rS10].s!<0>,         128;"
                        "(rpt3)sub.s      %[rS11].s!<0>,        (r)%[rS11].s!<0>,         128;"
                        "      or.b       %[model_input].s!<0>,    %[model_input].s!<2>,  %[model_input].s!<0>;"  
                        "      or.b       %[model_input].s!<5>,    %[model_input].s!<5>,  %[rD].s!<2>;"  
                        "(rpt3)and.b      %[rS00].s!<0>,        (r)%[rS00].s!<0>,         255;" 
                        "(rpt3)and.b      %[rS01].s!<0>,        (r)%[rS01].s!<0>,         255;" 
                        "(rpt3)and.b      %[rS10].s!<0>,        (r)%[rS10].s!<0>,         255;" 
                        "(rpt3)and.b      %[rS11].s!<0>,        (r)%[rS11].s!<0>,         255;" 
                        "      or.b       %[model_input].s!<0>,    %[model_input].s!<3>,  %[model_input].s!<0>;" 
                        "      or.b       %[model_input].s!<5>,    %[model_input].s!<5>,  %[rD].s!<3>;"                                      
                        "(rpt3)shl.b      %[rS01].s!<0>,        (r)%[rS01].s!<0>,      sr48.y;"
                        "(rpt3)shl.b      %[rS10].s!<0>,        (r)%[rS10].s!<0>,      sr48.z;"
                        "(rpt3)shl.b      %[rS11].s!<0>,        (r)%[rS11].s!<0>,      sr48.w;"                        
                        "(rpt3)mov.f32f32 %[model_input].s!<1>, (r)%[rS00].s!<0>;"
                        "(rpt3)or.b       %[model_input].s!<1>, (r)%[model_input].s!<1>,  (r)%[rS01].s!<0>;"
                        "(rpt3)or.b       %[model_input].s!<1>, (r)%[model_input].s!<1>,  (r)%[rS10].s!<0>;"
                        "(rpt3)or.b       %[model_input].s!<1>, (r)%[model_input].s!<1>,  (r)%[rS11].s!<0>;"


                        // "(rpt3)mov.f32f32 %[model_input].s!< 0>, %[zero];"
                        // "(rpt3)mov.f32f32 %[model_input].s!< 4>, %[zero];"
                        // "mov.f32f32 %[model_input].s!< 0>, %[rC];"
                        // "and.b          %[model_input].s!<0>,             %[rC], 255;" // s0-s3"
                        // "(rpt1)shrm.b32 %[model_input].s!<1>, (r)sr48.x,  %[rC], 255;"                        
                        // "shl.b %[rD].s!<0>,  %[rD].s!<0>,  sr48.z;"                   
                        // // "(rpt2)sub.s    %[model_input].s!<0>, (r)%[model_input].s!<0>,  128;"                         
                        // "(rpt1)shl.b %[rD].s!<2>,  (r)%[rD].s!<2>,  (r)sr48.x;" // s1<<8, s2<<16, 
                        // // "(rpt2)and.b    %[model_input].s!<0>, (r)%[model_input].s!<0>,  255;" // s0-s3                         
                        
                        //                             
                        
                        // "or.b     %[model_input].s!<0>,  %[model_input].s!<2>,  %[model_input].s!<0>;"     
                        
                        
                        // "(rpt3)shl.b %[rS00].s!<0>,  (r)%[rS00].s!<0>,  sr48.z;"    
                        // "(rpt3)shl.b %[rS10].s!<0>,  (r)%[rS10].s!<0>,  sr48.x;"
                        // "(rpt3)shl.b %[rS11].s!<0>,  (r)%[rS11].s!<0>,  sr48.y;"
                        // "or.b  %[model_input].s!<0>,  %[model_input].s!<0>,  %[rD].s!<0>;"
                        // "or.b  %[model_input].s!<1>,  %[model_input].s!<1>,  %[rD].s!<1>;"
                        // "or.b  %[model_input].s!<2>,  %[model_input].s!<2>,  %[rS01].s!<3>;"
                        // "or.b  %[model_input].s!<3>,  %[model_input].s!<3>,  %[rS01].s!<0>;"
                        // "or.b  %[model_input].s!<4>,  %[model_input].s!<4>,  %[rS01].s!<1>;"
                        // "or.b  %[model_input].s!<5>,  %[model_input].s!<5>,  %[rS01].s!<2>;"
                        // "or.b  %[model_input].s!<1>,  %[model_input].s!<1>,  %[rD].s!<2>;"
                        // "or.b  %[model_input].s!<2>,  %[model_input].s!<2>,  %[rS10].s!<3>;"
                        // "or.b  %[model_input].s!<3>,  %[model_input].s!<3>,  %[rS10].s!<0>;"
                        // "or.b  %[model_input].s!<4>,  %[model_input].s!<4>,  %[rS10].s!<1>;"
                        // "or.b  %[model_input].s!<5>,  %[model_input].s!<5>,  %[rS10].s!<2>;"
                        // "or.b  %[model_input].s!<1>,  %[model_input].s!<1>,  %[rD].s!<3>;"
                        // "or.b  %[model_input].s!<2>,  %[model_input].s!<2>,  %[rS11].s!<3>;"
                        // "or.b  %[model_input].s!<3>,  %[model_input].s!<3>,  %[rS11].s!<0>;"
                        // "or.b  %[model_input].s!<4>,  %[model_input].s!<4>,  %[rS11].s!<1>;"
                        // "or.b  %[model_input].s!<5>,  %[model_input].s!<5>,  %[rS11].s!<2>;"
                        // "or.b  %[model_input].s!<1>,  %[model_input].s!<1>,  %[rS00].s!<3>;"
                        // "or.b  %[model_input].s!<2>,  %[model_input].s!<2>,  %[rS00].s!<0>;"
                        // "or.b  %[model_input].s!<3>,  %[model_input].s!<3>,  %[rS00].s!<1>;"
                        // "or.b  %[model_input].s!<4>,  %[model_input].s!<4>,  %[rS00].s!<2>;"
                        // "nop;"
                        // "(nop3)or.b  %[rD_q].s!<0>,  %[rD_q].s!<1>,  %[rD_q].s!<0>;"        
                        // "(nop3)or.b  %[rD_q].s!<0>,  %[rD_q].s!<2>,  %[rD_q].s!<0>;"   
                        // "or.b  %[rD_q].s!<0>,  %[rD_q].s!<3>,  %[rD_q].s!<0>;"   
                        // "(rpt3)nop;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0             %[model_input].s!<0>,    %[coordInWords], 7;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4     %[model_input].s!<4>,    %[coordInWords], 7;"
                        "(ss)nop;"                        
                        :  [model_input] "+r" (model_input),
                           [rS00] "+r" (sample_0),
                           [rS01] "+r" (sample_1),
                           [rS10] "+r" (sample_2),
                           [rS11] "+r" (sample_3),
                           [rF] "+r" (feature),
                           [rC] "+r" (regColor),
                           [rD] "+r" (regDepth)
                        :  [coordInWords]          "r" (coordInWords),
                           [f255]                     "c" (0x437F0000),
                           [zero]      "c" (0x00000000)
                        :  "sr48.x", "sr48.y", "sr48.z", "sr48.w"
                       );

}



__kernel void preprocess(  __read_only image1d_buffer_t color_lr,
                           __read_only image1d_buffer_t depth_lr,
                           __read_only image1d_buffer_t mv_lr,          
                           __read_only image2d_t        prev_color_hr,
                           __read_only image2d_t        prev_feature_hr,
                           float                         scale_factor,                                                    
                           __write_only image1d_buffer_t model_input, // This is the packed input model, 
                           int                          apron,
                           int                          tile_id

)
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

   size_t x_index = get_global_id(0);
   size_t y_index = get_global_id(1) + (INPUT_HEIGHT / NUM_TILES) * tile_id - apron;

   if ((y_index < 0) || (y_index >= INPUT_HEIGHT)) return;

   uint   lr_coord = y_index * INPUT_WIDTH + x_index;

   float2 uv = (float2)((float)x_index,   (float)y_index);
   
   // if (x_index >= 8 | y_index >= 8 ) return;

   float4  regDepth_0;
   float4  regDepth_1;
   float4  regDepth_2;
   float4  regDepth_quant;
   uint    min_depth_index;
   float8  regMV_0;
   float8  regMV_1;
   float8  regMV_2;   
   float2  regMV_min;
   float   regColor;
   

   MM_SETUP()

  
   uint depth_coordInWords0 = (y_index - 1) * INPUT_WIDTH + (x_index - 1);
   uint depth_coordInWords1 = (y_index + 0) * INPUT_WIDTH + (x_index - 1);
   uint depth_coordInWords2 = (y_index + 1) * INPUT_WIDTH + (x_index - 1);

   if (x_index == 0 && y_index == 0) {      
      depth_coordInWords1 ++; // This is to fix out-of-bound for the first index in a vload3
   }
   if (x_index == 0 && y_index == 1) {
      depth_coordInWords0 ++; // This is to fix out-of-bound for the first index in a vload3
   }
   

   regDepth_0 = mm_load_depth(depth_coordInWords0);
   regDepth_1 = mm_load_depth(depth_coordInWords1);
   regDepth_2 = mm_load_depth(depth_coordInWords2);
   regColor   = mm_load_color(lr_coord);
   regMV_0    = mm_load_mv(depth_coordInWords0);
   regMV_1    = mm_load_mv(depth_coordInWords1);
   regMV_2    = mm_load_mv(depth_coordInWords2);   
   

   __asm__ __volatile__("(sy)nop;");
   

   regMV_min = mm_find_min_mv(regDepth_0, regDepth_1, regDepth_2, regMV_0, regMV_1, regMV_2, x_index, y_index);
   regDepth_quant = mm_quantize_depth(regDepth_1.s1);


      
   float2 sample_uv_0 = (float2)((2*uv.x     + 2 * regMV_min.y + 0.5), (2*uv.y     - 2 * regMV_min.x) + 0.5);  // regMV_min.x x 2?????
   float2 sample_uv_1 = (float2)((2*uv.x + 1 + 2 * regMV_min.y + 0.5), (2*uv.y     - 2 * regMV_min.x) + 0.5);
   float2 sample_uv_2 = (float2)((2*uv.x     + 2 * regMV_min.y + 0.5), (2*uv.y + 1 - 2 * regMV_min.x) + 0.5);
   float2 sample_uv_3 = (float2)((2*uv.x + 1 + 2 * regMV_min.y + 0.5), (2*uv.y + 1 - 2 * regMV_min.x) + 0.5);         
   float2 sample_uv_f = (float2)((uv.x     + regMV_min.y + 0.5), (uv.y     - regMV_min.x) + 0.5);    

   // float2 sample_uv_0 = (float2)((2*uv.x     + 2 * regMV_min.y), (2*uv.y     - 2 * regMV_min.x));  // regMV_min.x x 2?????
   // float2 sample_uv_1 = (float2)((2*uv.x + 1 + 2 * regMV_min.y), (2*uv.y     - 2 * regMV_min.x));
   // float2 sample_uv_2 = (float2)((2*uv.x     + 2 * regMV_min.y), (2*uv.y + 1 - 2 * regMV_min.x));
   // float2 sample_uv_3 = (float2)((2*uv.x + 1 + 2 * regMV_min.y), (2*uv.y + 1 - 2 * regMV_min.x));         
   // float2 sample_uv_f = (float2)((uv.x     + regMV_min.y), (uv.y     - regMV_min.x));    
  
   
   float4 sample_0 = read_imagef(prev_color_hr,   sampler, sample_uv_0);
   float4 sample_1 = read_imagef(prev_color_hr,   sampler, sample_uv_1);
   float4 sample_2 = read_imagef(prev_color_hr,   sampler, sample_uv_2);
   float4 sample_3 = read_imagef(prev_color_hr,   sampler, sample_uv_3);
   float4 feature_0 = read_imagef(prev_feature_hr, sampler, sample_uv_0);
   float4 feature_1 = read_imagef(prev_feature_hr, sampler, sample_uv_1);
   float4 feature_2 = read_imagef(prev_feature_hr, sampler, sample_uv_2);
   float4 feature_3 = read_imagef(prev_feature_hr, sampler, sample_uv_3);
   feature_0.y = feature_1.x;
   feature_0.z = feature_2.x;
   feature_0.w = feature_3.x;


   mm_store(sample_0, sample_1, sample_2, sample_3, feature_0, regColor, regDepth_quant, tile_id);


}
