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


#define MATRIX_B_ON_CHIP_IN_BYTES (1024*2)
#define SPML_MODE_FLAG
#define LM_FIRST_256B  0
#define LM_SECOND_256B 512
#define LM_THIRD_256B  1024
#define LM_FOURTH_256B 1536
#define LM_FIRST_512B  0
#define LM_SECOND_512B 1024


#define BICUBIC
// #define BILINEAR
// #define POINT


#define MM_SETUP()      \
    if (get_global_id(0) == 123456789) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regDepth.xyzw  = convert_float4(read_imageh(depth, get_global_id(0))); \
       regDepth.xyzw += convert_float4(read_imagef(mv, get_global_id(0))); \
       regDepth.xyzw += convert_float4(read_imagei(input_image, sampler, uv));\
       regDepth.xyzw += convert_float4(read_imagei(input_features, sampler, uv));\
       write_imagef(quantized_depth, 0, regDepth.xyzw);\
    }\
    __asm__ __volatile__( \
                        "nop;"\
                        : \
                        : "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : \
                       );


inline float16 mm_load_depth(uint depth_coordInWords0, uint depth_coordInWords1, uint depth_coordInWords2, uint depth_coordInWords3, short shouldLoad)
{
    __private float16   regDepth;


    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"
                        "getone #5;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0 %[rD].s!< 0>, %[depth_coordInWords0], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0 %[rD].s!< 4>, %[depth_coordInWords1], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0 %[rD].s!< 8>, %[depth_coordInWords2], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0 %[rD].s!<12>, %[depth_coordInWords3], 2;"
                        "(jp)br !p0.x, #2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rD].s!< 0>, %[depth_coordInWords0], 2;"
                        "(jp)(ss)nop;"
                        :  [rD] "=r" (regDepth)
                        :  [depth_coordInWords0] "r" (depth_coordInWords0),
                           [depth_coordInWords1] "r" (depth_coordInWords1),
                           [depth_coordInWords2] "r" (depth_coordInWords2),
                           [depth_coordInWords3] "r" (depth_coordInWords3),
                           [shouldLoad]   "r" (shouldLoad)
                        :  "p0.x"
                       );

      return regDepth;
}


inline float4 mm_load_mvs(uint mvs_coordInWords, float scale_factor)
{
    __private float4   regMvs;


    __asm__ __volatile__(
                        "nop;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyoo)%[rM].s0,  %[mvs_coordInWords], 48;"
                        "(sy)nop;"                        
                        "(rpt1)mov.f32f32 %[rM].s2, (r)%[rM].s0;"
                        "(rpt3)mul.f    %[rM].s0, %[scale_factor], (r)%[rM].s0;"
                        :  [rM] "=r" (regMvs)
                        :  [mvs_coordInWords] "r" (mvs_coordInWords),
                           [scale_factor] "r" (scale_factor)
                        :  
                       );

      return regMvs;
}

inline float2 mm_load_mvs_xy(uint mvs_coordInWords, short shouldLoad)
{
    __private float2   regMvs;
   

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"
                        "(rpt5)nop;"
                        "br !p0.x, #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyoo)%[rM].s0,  %[mvs_coordInWords], 48;"
                        "(jp)nop;"                                                
                        :  [rM] "=r" (regMvs)
                        :  [mvs_coordInWords] "r" (mvs_coordInWords),
                           [shouldLoad]   "r" (shouldLoad)
                        :  "p0.x"
                       );

      return regMvs;
}

inline void mm_find_min_depth(float16 regDepth)
{

    __asm__ __volatile__(
                        "getone #48;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 1>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 1>;"
                        "(nop2)sel.b32 r0.z, 0, r0.y, 1;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 2>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 2>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 2;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 3>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 3>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 3;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 4>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 4>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 4;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 5>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 5>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 5;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 6>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 6>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 6;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 7>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 7>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 7;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 8>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 8>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 8;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!< 9>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!< 9>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 9;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<10>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<10>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 10;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<11>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<11>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 11;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<12>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<12>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 12;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<13>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<13>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 13;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<14>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<14>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 14;"
                        "(nop3) cmps.u.le r0.y, %[rD].s!< 0>, %[rD].s!<15>;"
                        "sel.b32 %[rD].s!< 0>, %[rD].s!< 0>, r0.y, %[rD].s!<15>;"
                        "(nop2)sel.b32 r0.z, r0.z, r0.y, 15;"
                        "(rpt1)nop;"
                        "mov.f32f32 sr48.x, r0.z;"                        
                        "(jp)(ss)nop;"
                        :  
                        :  [rD] "r" (regDepth)                           
                        :  "r0.y", "r0.z", "sr48.x"
                       );

}

inline float4 mm_quantize_depth(float16 regDepth, short shouldLoad)
{
   float4 quantized_depth;
   float4 quantized_depth_frac;

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"                        
                        "(rpt5)nop;"
                        "br !p0.x, #14;"
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
                        "(rpt3)cov.f32u32 %[rD_q_f].s!<0>, (r)%[rD_q_f].s!<0>;"
                        "(rpt3)and.b  %[rD_q].s!<0>, (r)%[rD_q_f].s!<0>,  255;" // s0-s3                        
                        "(jp)nop;"
                        :  [rD_q]         "+r" (quantized_depth),
                           [rD_q_f]         "+r" (quantized_depth_frac)
                        :  [rD]           "r" (regDepth),                           
                           [shouldLoad]   "r" (shouldLoad),
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
                        "mad.f32    %[rMVS].s1, %[scale_factor], sr48.y, (neg)0.5;"
                        // "(rpt1)mov.f32f32 %[rMVS], (r)sr48.y;" 
                        "(ss)nop;"
                        :  [rMVS] "=r" (mvs_final)
                        :  [scale_factor] "r" (scale_factor)                           
                        :  "sr48.y", "sr48.z"
                       );

   return mvs_final;
}


inline void mm_store(float4 sample, float4 feature, float4 quantized_depth, short shouldLoad_depth)
{
   size_t x_index = get_global_id(0);
   size_t y_index = get_global_id(1);
   uint   x_index_lr = x_index / 2;
   uint   y_index_lr = y_index / 2;

   uint coordInWords = (y_index)    * 2*INPUT_WIDTH + x_index;
   uint coordInWords_depth = (y_index_lr)    * INPUT_WIDTH + x_index_lr;

   __asm__ __volatile__(
                        // "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */     
                        "cmps.u.ne p0.x, %[shouldLoad], 0;"
                        "movs.u32u32 sr48.x, 8;"
                        "movs.u32u32 sr48.y, 16;"
                        "movs.u32u32 sr48.z, 24;"
                        "(rpt3)mul.f %[rS00].s!<0>, (r)%[rS00].s!<0>, %[f255];"
                        "mul.f       %[rF00].s!<0>,    %[rF00].s!<0>, %[f255];"
                        "(rpt3)cov.f32u32 %[rS00].s!<0>, (r)%[rS00].s!<0>;"
                        "      cov.f32u32 %[rF00].s!<0>,    %[rF00].s!<0>;"
                        "(rpt3)and.b %[rS00].s!<0>,  (r)%[rS00].s!<0>,  255;" // s0-s3                        
                        "(ss) nop;"
                        "(rpt2)shl.b %[rS00].s!<1>,  (r)%[rS00].s!<1>,  (r)sr48.x;" // s1<<8, s2<<16, s3<<24 
                        "(rpt2)shl.b %[rD00].s!<1>,  (r)%[rD00].s!<1>,  (r)sr48.x;" // s1<<8, s2<<16, s3<<24 
                        // "(rpt3)nop;"                       
                        "(nop3)or.b  %[rS00].s!<0>,  %[rS00].s!<1>,  %[rS00].s!<0>;"     
                        "(nop3)or.b  %[rD00].s!<0>,  %[rD00].s!<1>,  %[rD00].s!<0>;"                                           
                        "(nop3)or.b  %[rS00].s!<0>,  %[rS00].s!<2>,  %[rS00].s!<0>;"  // s0 readable, no need for nop    
                        "(nop3)or.b  %[rD00].s!<0>,  %[rD00].s!<2>,  %[rD00].s!<0>;"                    
                        "or.b  %[rS00].s!<0>,  %[rS00].s!<3>,  %[rS00].s!<0>;"    
                        "or.b  %[rD00].s!<0>,  %[rD00].s!<3>,  %[rD00].s!<0>;"                                            
                        "(rpt5)nop;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0 %[rS00].s!<0>, %[coordInWords], 7;"
                        "stib.untyped.u8.1d.1.unpacked.mode4.base0  %[rF00].s!<0>.lo, %[coordInWords], 8;"
                        "br !p0.x, #2;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0 %[rD00].s!<0>, %[coordInWords_depth], 6;"
                        "(jp)nop;"
                        :  [rS00] "+r" (sample),
                           [rF00] "+r" (feature),
                           [rD00] "+r" (quantized_depth)
                        :  [coordInWords]          "r" (coordInWords),
                           [coordInWords_depth]          "r" (coordInWords_depth),
                           [shouldLoad]             "r" (shouldLoad_depth),
                           [f255]                     "c" (0x437F0000)
                        :  "p0.x"
                       );

}



__kernel void preprocess(  __read_only image1d_buffer_t depth,
                           __read_only image1d_buffer_t mv,          
                           __read_only image2d_t        input_image,
                           __read_only image2d_t        input_features,
                           float                         scale_factor,                         
                           __write_only image1d_buffer_t quantized_depth,
                           __write_only image1d_buffer_t output_image,
                           __write_only image1d_buffer_t output_features

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
   size_t y_index = get_global_id(1);
   float2 uv = (float2)((float)x_index,   (float)y_index);
   
   // if (x_index >= 8 | y_index >= 8 ) return;


   float16 regDepth;
   float2  mvs;
   

   MM_SETUP()

  
   uint depth_coordInWords0 = (y_index / 2    ) * INPUT_WIDTH + (x_index / 2);
   uint depth_coordInWords1 = (y_index / 2 + 1) * INPUT_WIDTH + (x_index / 2);
   uint depth_coordInWords2 = (y_index / 2 + 2) * INPUT_WIDTH + (x_index / 2);
   uint depth_coordInWords3 = (y_index / 2 + 3) * INPUT_WIDTH + (x_index / 2);
   uint depth_coordInWords  = (y_index / 2    ) * INPUT_WIDTH + (x_index / 2);

   uint mvs_coordInWords = 2 * ((y_index / 2    ) * INPUT_WIDTH + (x_index / 2)); 

   uint local_id = (get_local_id(1) / 2 ) * 4 + (get_local_id(0) / 2); 
   short shouldLoad = (get_local_id(0) % 2 == 0)&(get_local_id(1) % 2 == 0);
   short shouldLoad_depth = (get_local_id(0) % 2 == 1)&(get_local_id(1) % 2 == 1);

   regDepth = mm_load_depth(depth_coordInWords0, depth_coordInWords1, depth_coordInWords2, depth_coordInWords3, shouldLoad_depth);
   mvs      = mm_load_mvs_xy(mvs_coordInWords, shouldLoad);
   
   __asm__ __volatile__("(sy)nop;");

   mm_find_min_depth(regDepth);
   float4 q_depth = mm_quantize_depth(regDepth, shouldLoad_depth);

   mm_find_mvs(local_id, shouldLoad, mvs);

   float2  mvs_final = mm_get_final_mvs(scale_factor);
      
   float2 sample_uv = (float2)(uv.x + mvs_final.x, uv.y - mvs_final.y);         
   
   float4 sample = read_imagef(input_image, sampler, sample_uv);// * (float4)(255, 255, 255, 255);
   float4 feature = read_imagef(input_features, sampler, sample_uv);// * (float4)(255, 255, 255, 255);

   mm_store(sample, feature, q_depth, shouldLoad_depth);


}
