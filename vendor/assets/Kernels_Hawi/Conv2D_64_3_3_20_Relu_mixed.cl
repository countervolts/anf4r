// This is the baseline Conv+Relu INT8 kernel for Kaanapali

#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_extra_vector_types : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable

#define xstr(s) str(s)
#define str(s) #s

#define BYTES_PER_WORD       4
#define BYTES_PER_HALF_WORD  2

#ifdef THREE_NSP
#ifdef USE_MALU
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*3)
#else
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*32)
#endif
#define SPML_MODE_FLAG
#define LM_FIRST_256B   /* 0   *3 */ 0
#define LM_SECOND_256B  /* 256 *3 */ 768
#define LM_THIRD_256B   /* 512 *3 */ 1536
#define LM_FOURTH_256B  /* 768 *3 */ 2304
#define LM_FIRST_128B   /* 0   *3 */ 0
#define LM_SECOND_128B  /* 128 *3 */ 384
#define LM_THIRD_128B   /* 256 *3 */ 768
#define LM_FOURTH_128B  /* 384 *3 */ 1152
#define LM_FIFTH_128B   /* 512 *3 */ 1536
#define LM_SIXTH_128B   /* 640 *3 */ 1920
#define LM_SEVENTH_128B /* 768 *3 */ 2304
#define LM_EIGHTH_128B  /* 896 *3 */ 2688
#define LM_FIRST_512B   /* 0   *3 */ 0
#define LM_SECOND_512B  /* 512 *3 */ 1536
#else
#ifdef USE_MALU
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*2)
#else
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*32)
#endif
#define SPML_MODE_FLAG
#define LM_FIRST_256B   /* 0   *2 */ 0
#define LM_SECOND_256B  /* 256 *2 */ 512
#define LM_THIRD_256B   /* 512 *2 */ 1024
#define LM_FOURTH_256B  /* 768 *2 */ 1536
#define LM_FIRST_128B   /* 0   *2 */ 0
#define LM_SECOND_128B  /* 128 *2 */ 256
#define LM_THIRD_128B   /* 256 *2 */ 512
#define LM_FOURTH_128B  /* 384 *2 */ 768
#define LM_FIFTH_128B   /* 512 *2 */ 1024
#define LM_SIXTH_128B   /* 640 *2 */ 1280
#define LM_SEVENTH_128B /* 768 *2 */ 1536
#define LM_EIGHTH_128B  /* 896 *2 */ 1792
#define LM_FIRST_512B   /* 0   *2 */ 0
#define LM_SECOND_512B  /* 512 *2 */ 1024
#endif

#define BYTES_PER_OPERAND 2
#define BYTES_PER_RESULT  2
#define FULL_K_MASK       32

#define USE_PR
#ifdef USE_PR
#define PRIORITYRESET "priorityreset;"
#else
#define PRIORITYRESET
#endif

#ifdef USE_MALU
#define BARSET  __asm__ __volatile__("bar.set;");
#define BARSYNC __asm__ __volatile__("bar.sync;");
#else
#define BARSET
#define BARSYNC
#endif

#ifdef USE_TILE
#define TILE_INPUT_OFFSET input_offset
#define TILE_OUTPUT_OFFSET output_offset
#else
#define TILE_INPUT_OFFSET 0 
#define TILE_OUTPUT_OFFSET 0 
#endif


#define MATRIX_C_ON_MALUMEM_IN_KBYTES (8)
#define REG_C_DATATYPE float32

//----------------------------------------------------------------------------//
// ALU section
//----------------------------------------------------------------------------//

#define ALU_32_CYCLES(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, ss, ul) \
                     #ss"(rpt3)pkmad.f16 "#rC", (r)l"#b0", "#rA0", (r)      "#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b1", "#rA1", (r)(last)"#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b2", "#rA2", (r)(last)"#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b3", "#rA3", (r)(last)"#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b4", "#rA4", (r)(last)"#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b5", "#rA5", (r)(last)"#rC";"\
                        "(rpt3)pkmad.f16 "#rC", (r)l"#b6", "#rA6", (r)(last)"#rC";"\
                     #ul"(rpt3)pkmad.f16 "#rC", (r)l"#b7", "#rA7", (r)(last)"#rC";"

//----------------------------------------------------------------------------//
// mm_setup_info 
//----------------------------------------------------------------------------//
// NOTE: matB is layout needs to be Kfirst
typedef struct
{
    uint b_localOffsetInBytes;
    uint b_globalOffsetInWords00;
    uint b_globalOffsetInWords16;
#ifdef USE_MALU
    uint b_globalOffsetInWords_malu;
#endif
    uint b_localOffsetInBytes_scale_bias;

} mm_setup_info;


//----------------------------------------------------------------------------//
// MM SETUP 
//----------------------------------------------------------------------------//
#define MM_SETUP(matrix_B)      \
    if (get_global_id(0) == 12345) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regC.xyzw  = convert_float4(read_imageh(matrix_A, get_global_id(0))); \
       regC.xyzw += convert_float4(read_imagef(matrix_B, get_global_id(0))); \
       write_imagef(matrix_C, 0, regC.xyzw);\
       write_imagef(matrix_C_mask, 0, regC.xyzw);\
    }\
    __asm__ __volatile__( \
                        "movs.u32u32 sr50.x, 0;" \
                        "movs.u32u32 sr50.y, 8;" \
                        "movs.u32u32 sr50.z, 16;" \
                        "movs.u32u32 sr50.w, 24;" \
                        "movs.u32u32 sr51.x, 0xFF;" \
                        "movs.u32u32 sr51.y, 0xFF00;" \
                        "movs.u32u32 sr51.z, 0xFF0000;" \
                        "movs.u32u32 sr51.w, 0xFF000000;" \
                        : \
                        : "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : "sr50.x", "sr50.y", "sr50.z", "sr50.w", "sr51.x", "sr51.y", "sr51.z", "sr51.w" \
                       );

//----------------------------------------------------------------------------//
// Update mm_setup_info 
//----------------------------------------------------------------------------//
inline mm_setup_info mm_setup()
{

    mm_setup_info setup_info;
    size_t sub_block_id_m = get_local_id(0); 

    uint   b_localOffsetInHalfWords = (sub_block_id_m/32)*32 + ((((sub_block_id_m)>>0)&1)<<3) + ((((sub_block_id_m)>>1)&1)<<4) + ((((sub_block_id_m)>>2)&1)<<0) + ((((sub_block_id_m)>>3)&1)<<1) + ((((sub_block_id_m)>>4)&1)<<2);
    setup_info.b_localOffsetInBytes = b_localOffsetInHalfWords * 2;

   //  uint   b_localOffsetInElements     = (sub_block_id_m/16)*16 + ((((sub_block_id_m)>>0)&1)<<2) + ((((sub_block_id_m)>>1)&1)<<3) + ((((sub_block_id_m)>>2)&1)<<0) + ((((sub_block_id_m)>>3)&1)<<1);
   //  setup_info.b_localOffsetInBytes    = b_localOffsetInElements * BYTES_PER_RESULT;

    // For Kailua/Lanai/Hamoa 
    // setup_info.b_localOffsetInBytes += MATRIX_B_ON_CHIP_IN_BYTES - (2*64) - ((setup_info.b_localOffsetInBytes>>6) * (3*64)); // instead of jumping forward by 64, jump backwards by 64 (so subtract 64 three times)
    

    setup_info.b_localOffsetInBytes_scale_bias = sub_block_id_m * BYTES_PER_RESULT;
    // setup_info.b_localOffsetInBytes_scale_bias += MATRIX_B_ON_CHIP_IN_BYTES - (2*64) - ((setup_info.b_localOffsetInBytes_scale_bias>>6) * (3*64));

    /* in xy below, the x means "first dimension" and y means "second dimension" - either can be K or N depending on layout */ 
    uint2  b_globalOffsetInWords_xy = {((sub_block_id_m%4)*(TILESIZE_K/BYTES_PER_WORD/4)), (sub_block_id_m>>2)}; 

    setup_info.b_globalOffsetInWords00   = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD;
#if (TILESIZE_N == 32 | TILESIZE_N == 64)    
    setup_info.b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords00 + (16 * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
#endif
#ifdef USE_MALU
    setup_info.b_globalOffsetInWords_malu   = (sub_block_id_m&0x7) + ((sub_block_id_m>>3)*(LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD));
#endif

#ifndef USE_SPML_MODE
   // 64 bytes = 512bits = one side of LM. So Bit5...0 are "address in one side", Bit6 is "side select"
   short addressInOneSide = ((setup_info.b_localOffsetInBytes>>0)&0x3F);
   #ifndef THREE_NSP
      short addressMSBs = (setup_info.b_localOffsetInBytes>>6);
      short newSideSelect = qcom_get_physical_sub_group_id()%2; // one bit mask for nSP ID
      setup_info.b_localOffsetInBytes = ((addressMSBs*2+newSideSelect)<<6) | (addressInOneSide); // times 2 for 2 nSP
   #else
      short addressMSBs = (setup_info.b_localOffsetInBytes>>6);
      short newSideSelect = qcom_get_physical_sub_group_id()%3; // wave id % 3 = nSP id for Honu
      setup_info.b_localOffsetInBytes = ((addressMSBs*3+newSideSelect)<<6) | (addressInOneSide); // times 3 for 3 nSP
   #endif
#endif
    
    return setup_info;
}


//----------------------------------------------------------------------------//
// load a 
//----------------------------------------------------------------------------//

#if defined(USE_MALU) && (IC==28)
#define compmask "xyzO"
#define rptNum "rpt2"
#elif defined(USE_MALU) && (IC==24)
#define compmask "xyOO"
#define rptNum "rpt1"
#else
#define compmask "xyzw"
#define rptNum "rpt3"
#endif


inline void mm_load_a_im2col_M64K64_INT8_1x1_zfill(uint8 *regA0, uint8 *regA1, 
                                                   uint8 *regA2, uint8 *regA3, 
                                                   uint matAcoord, uint zfill)
{
    __asm__ __volatile__(
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0].s0,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)      %[rA0].s4,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset8  (f32)(xyzw)      %[rA1].s0,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset12 (f32)(xyzw)      %[rA1].s4,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset16 (f32)(xyzw)      %[rA2].s0,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset20 (f32)(xyzw)      %[rA2].s4,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset24 (f32)(xyzw)      %[rA3].s0,  %[texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset28 (f32)(xyzw)      %[rA3].s4,  %[texCoord], 16;" 
                        "unlock;"
                        "(ss)nop;"
                        "(sy)nop;"                        
                        "cmps.u.gt p0, %[zfill], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)mov.u32u32 %[rA0].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rA0].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rA1].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rA1].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rA2].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rA2].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rA3].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rA3].s4, %[zero];"
                        "prede;"                   
                        :  [rA0] "=r" (*regA0), [rA1] "=r" (*regA1),
                           [rA2] "=r" (*regA2), [rA3] "=r" (*regA3)  
                        :  [texCoord] "r" (matAcoord / (4)),
                           [zfill] "r" (zfill),
                           // [zero]      "c" (0x80808080)
                           [zero]      "c" (0x00000000)
                        :  "p0.x"
                       );
}


#ifndef USE_MALU

inline void mm_load_b_N32K64_INT8_1x1(uint4 *regB0, uint4 *regB1,
                                      uint4 *regB2, uint4 *regB3,
                                      uint subMatrixWStartInElements,
                                      mm_setup_info setup_info)
{

   size_t sub_block_id_m = get_local_id(0);
   uint b_coordInWords_00 = (subMatrixWStartInElements/(4/BYTES_PER_OPERAND)) + (setup_info.b_globalOffsetInWords00);
   uint b_coordInWords_16 = (subMatrixWStartInElements/(4/BYTES_PER_OPERAND)) + (setup_info.b_globalOffsetInWords16);
   int active_fiber = (sub_block_id_m < 16); // This is hard-coded for now. The second 16 step in N dimension, only has 4 rows left (16 fibers)

   __asm__ __volatile__(
                        "cmps.u.eq p0.y, %[active_fiber], 0;"
                        "(rpt3)mov.u32u32 %[rB0].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB1].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB2].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB3].s0, %[zero];"                                                                    
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0          %[rB0].s0, %[b_coordInWords_00], 2;"                        
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset8  %[rB1].s0, %[b_coordInWords_00], 2;"                                                
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset16 %[rB2].s0, %[b_coordInWords_00], 2;"                        
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset24 %[rB3].s0, %[b_coordInWords_00], 2;"
                        "br p0.y, #5;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0          %[rB0].s2, %[b_coordInWords_16], 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset8  %[rB1].s2, %[b_coordInWords_16], 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset16 %[rB2].s2, %[b_coordInWords_16], 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset24 %[rB3].s2, %[b_coordInWords_16], 2;"                        
                        "(jp)(ss)nop;"
                        :  [rB0] "=r" (*regB0), [rB1] "=r" (*regB1),
                           [rB2] "=r" (*regB2), [rB3] "=r" (*regB3)
                        :   
                           [active_fiber]        "r" (active_fiber),  
                           [b_coordInWords_00] "r" (b_coordInWords_00),
                           [b_coordInWords_16] "r" (b_coordInWords_16),
                           [zero]                "c" (0x00000000)
                        :  "p0.y"
                       );


}

#else

inline void mm_load_b_N32K64_INT8_1x1(uint8 *regB0, uint8 *regB1,
                                      uint8 *regB2, uint8 *regB3,
                                      uint subMatrixWStartInElements,
                                      mm_setup_info setup_info)
{


   size_t sub_block_id_m = get_local_id(0);
   uint b_coordInWords = (subMatrixWStartInElements/(4/BYTES_PER_OPERAND)) + (sub_block_id_m * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
   int active_fiber = (sub_block_id_m < OC);

   // use just stride instead of all these registers....
    
    __asm__ __volatile__(
                        "cmps.u.eq p0.y, %[active_fiber], 0;"
                        "(rpt3)mov.u32u32 %[rB0].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB0].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rB1].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB1].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rB2].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB2].s4, %[zero];"
                        "(rpt3)mov.u32u32 %[rB3].s0, %[zero];"
                        "(rpt3)mov.u32u32 %[rB3].s4, %[zero];"
                        "(rpt5)nop;"
                        "br p0.y, #9;"                        
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0          %[rB0].s0, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4  %[rB0].s4, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset8  %[rB1].s0, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset12 %[rB1].s4, %[b_coordInWords], 2;"                        
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset16 %[rB2].s0, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset20 %[rB2].s4, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset24 %[rB3].s0, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset28 %[rB3].s4, %[b_coordInWords], 2;"
                        "(jp)(ss)nop;" /* FIXME - compiler issue */
                        :  [rB0] "=r" (*regB0), [rB1] "=r" (*regB1),
                           [rB2] "=r" (*regB2), [rB3] "=r" (*regB3)
                        :  [b_coordInWords]      "r" (b_coordInWords),
                           [active_fiber]        "r" (active_fiber),
                           [zero]                "c" (0x00000000)
                        :  "p0.y"
                       );


}

#endif

inline void mm_load_bias_INT32(__global const float* restrict matrixBias, const int32 biasConst)
{
    __asm__ __volatile__(
                        "ldg.k c[&%[biasConst]], g[%[matrixBias]], 4;"
                        "(ss)nop;"
                        :
                        :  [matrixBias] "u" (matrixBias),
                           [biasConst]  "c" (biasConst)
                        :
                       );
}

inline void mm_load_scale_FP32(__global const float* restrict matrixScale, const float32 scaleConst)
{
    __asm__ __volatile__(
                        "ldg.k c[&%[scaleConst]], g[%[matrixScale]], 6;"
                        "(ss)nop;"
                        :
                        :  [matrixScale] "u" (matrixScale),
                           [scaleConst]  "c" (scaleConst)
                        :
                       );
}





#ifndef USE_MALU

inline void mm_mad_N32K64_INT8_1x1(float16 *regC_ptr,
                                   uint8 *regA0, uint8 *regA1, uint8 *regA2, uint8 *regA3,
                                   uint4 *regB0, uint4 *regB1, uint4 *regB2, uint4 *regB3,
                                   mm_setup_info setup_info)
{
    

   uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
   short  linearIndex = get_local_id(0);

    __asm__ __volatile__(
                        "nop;"
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_128B)  "], %[rB0].s0.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_128B) "], %[rB0].s0.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_128B)  "], %[rB0].s1.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_128B) "], %[rB0].s1.hi, 1;"
                       PRIORITYRESET
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0].s0.lo, %[rA0].s0.hi, %[rA0].s2.lo, %[rA0].s2.hi, %[rA0].s4.lo, %[rA0].s4.hi, %[rA0].s6.lo, %[rA0].s6.hi, 00.x, 08.x, 01.x, 09.x, 02.x, 10.x, 03.x, 11.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIFTH_128B)  "], %[rB0].s2.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SIXTH_128B)  "], %[rB0].s2.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SEVENTH_128B)"], %[rB0].s3.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_EIGHTH_128B) "], %[rB0].s3.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0].s1.lo, %[rA0].s1.hi, %[rA0].s3.lo, %[rA0].s3.hi, %[rA0].s5.lo, %[rA0].s5.hi, %[rA0].s7.lo, %[rA0].s7.hi, 16.x, 24.x, 17.x, 25.x, 18.x, 26.x, 19.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0].s0.lo, %[rA0].s0.hi, %[rA0].s2.lo, %[rA0].s2.hi, %[rA0].s4.lo, %[rA0].s4.hi, %[rA0].s6.lo, %[rA0].s6.hi, 04.x, 12.x, 05.x, 13.x, 06.x, 14.x, 07.x, 15.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0].s1.lo, %[rA0].s1.hi, %[rA0].s3.lo, %[rA0].s3.hi, %[rA0].s5.lo, %[rA0].s5.hi, %[rA0].s7.lo, %[rA0].s7.hi, 20.x, 28.x, 21.x, 29.x, 22.x, 30.x, 23.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0].s0.lo, %[rA0].s0.hi, %[rA0].s2.lo, %[rA0].s2.hi, %[rA0].s4.lo, %[rA0].s4.hi, %[rA0].s6.lo, %[rA0].s6.hi, 32.x, 40.x, 33.x, 41.x, 34.x, 42.x, 35.x, 43.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_128B)  "], %[rB1].s0.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_128B) "], %[rB1].s0.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_128B)  "], %[rB1].s1.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_128B) "], %[rB1].s1.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0].s1.lo, %[rA0].s1.hi, %[rA0].s3.lo, %[rA0].s3.hi, %[rA0].s5.lo, %[rA0].s5.hi, %[rA0].s7.lo, %[rA0].s7.hi, 48.x, 56.x, 49.x, 57.x, 50.x, 58.x, 51.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0].s0.lo, %[rA0].s0.hi, %[rA0].s2.lo, %[rA0].s2.hi, %[rA0].s4.lo, %[rA0].s4.hi, %[rA0].s6.lo, %[rA0].s6.hi, 36.x, 44.x, 37.x, 45.x, 38.x, 46.x, 39.x, 47.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0].s1.lo, %[rA0].s1.hi, %[rA0].s3.lo, %[rA0].s3.hi, %[rA0].s5.lo, %[rA0].s5.hi, %[rA0].s7.lo, %[rA0].s7.hi, 52.x, 60.x, 53.x, 61.x, 54.x, 62.x, 55.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1].s0.lo, %[rA1].s0.hi, %[rA1].s2.lo, %[rA1].s2.hi, %[rA1].s4.lo, %[rA1].s4.hi, %[rA1].s6.lo, %[rA1].s6.hi, 00.x, 08.x, 01.x, 09.x, 02.x, 10.x, 03.x, 11.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIFTH_128B)  "], %[rB1].s2.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SIXTH_128B)  "], %[rB1].s2.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SEVENTH_128B)"], %[rB1].s3.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_EIGHTH_128B) "], %[rB1].s3.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1].s1.lo, %[rA1].s1.hi, %[rA1].s3.lo, %[rA1].s3.hi, %[rA1].s5.lo, %[rA1].s5.hi, %[rA1].s7.lo, %[rA1].s7.hi, 16.x, 24.x, 17.x, 25.x, 18.x, 26.x, 19.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1].s0.lo, %[rA1].s0.hi, %[rA1].s2.lo, %[rA1].s2.hi, %[rA1].s4.lo, %[rA1].s4.hi, %[rA1].s6.lo, %[rA1].s6.hi, 04.x, 12.x, 05.x, 13.x, 06.x, 14.x, 07.x, 15.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1].s1.lo, %[rA1].s1.hi, %[rA1].s3.lo, %[rA1].s3.hi, %[rA1].s5.lo, %[rA1].s5.hi, %[rA1].s7.lo, %[rA1].s7.hi, 20.x, 28.x, 21.x, 29.x, 22.x, 30.x, 23.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1].s0.lo, %[rA1].s0.hi, %[rA1].s2.lo, %[rA1].s2.hi, %[rA1].s4.lo, %[rA1].s4.hi, %[rA1].s6.lo, %[rA1].s6.hi, 32.x, 40.x, 33.x, 41.x, 34.x, 42.x, 35.x, 43.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_128B)  "], %[rB2].s0.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_128B) "], %[rB2].s0.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_128B)  "], %[rB2].s1.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_128B) "], %[rB2].s1.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1].s1.lo, %[rA1].s1.hi, %[rA1].s3.lo, %[rA1].s3.hi, %[rA1].s5.lo, %[rA1].s5.hi, %[rA1].s7.lo, %[rA1].s7.hi, 48.x, 56.x, 49.x, 57.x, 50.x, 58.x, 51.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1].s0.lo, %[rA1].s0.hi, %[rA1].s2.lo, %[rA1].s2.hi, %[rA1].s4.lo, %[rA1].s4.hi, %[rA1].s6.lo, %[rA1].s6.hi, 36.x, 44.x, 37.x, 45.x, 38.x, 46.x, 39.x, 47.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1].s1.lo, %[rA1].s1.hi, %[rA1].s3.lo, %[rA1].s3.hi, %[rA1].s5.lo, %[rA1].s5.hi, %[rA1].s7.lo, %[rA1].s7.hi, 52.x, 60.x, 53.x, 61.x, 54.x, 62.x, 55.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2].s0.lo, %[rA2].s0.hi, %[rA2].s2.lo, %[rA2].s2.hi, %[rA2].s4.lo, %[rA2].s4.hi, %[rA2].s6.lo, %[rA2].s6.hi, 00.x, 08.x, 01.x, 09.x, 02.x, 10.x, 03.x, 11.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIFTH_128B)  "], %[rB2].s2.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SIXTH_128B)  "], %[rB2].s2.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SEVENTH_128B)"], %[rB2].s3.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_EIGHTH_128B) "], %[rB2].s3.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2].s1.lo, %[rA2].s1.hi, %[rA2].s3.lo, %[rA2].s3.hi, %[rA2].s5.lo, %[rA2].s5.hi, %[rA2].s7.lo, %[rA2].s7.hi, 16.x, 24.x, 17.x, 25.x, 18.x, 26.x, 19.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2].s0.lo, %[rA2].s0.hi, %[rA2].s2.lo, %[rA2].s2.hi, %[rA2].s4.lo, %[rA2].s4.hi, %[rA2].s6.lo, %[rA2].s6.hi, 04.x, 12.x, 05.x, 13.x, 06.x, 14.x, 07.x, 15.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2].s1.lo, %[rA2].s1.hi, %[rA2].s3.lo, %[rA2].s3.hi, %[rA2].s5.lo, %[rA2].s5.hi, %[rA2].s7.lo, %[rA2].s7.hi, 20.x, 28.x, 21.x, 29.x, 22.x, 30.x, 23.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2].s0.lo, %[rA2].s0.hi, %[rA2].s2.lo, %[rA2].s2.hi, %[rA2].s4.lo, %[rA2].s4.hi, %[rA2].s6.lo, %[rA2].s6.hi, 32.x, 40.x, 33.x, 41.x, 34.x, 42.x, 35.x, 43.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_128B)  "], %[rB3].s0.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_128B) "], %[rB3].s0.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_128B)  "], %[rB3].s1.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_128B) "], %[rB3].s1.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2].s1.lo, %[rA2].s1.hi, %[rA2].s3.lo, %[rA2].s3.hi, %[rA2].s5.lo, %[rA2].s5.hi, %[rA2].s7.lo, %[rA2].s7.hi, 48.x, 56.x, 49.x, 57.x, 50.x, 58.x, 51.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2].s0.lo, %[rA2].s0.hi, %[rA2].s2.lo, %[rA2].s2.hi, %[rA2].s4.lo, %[rA2].s4.hi, %[rA2].s6.lo, %[rA2].s6.hi, 36.x, 44.x, 37.x, 45.x, 38.x, 46.x, 39.x, 47.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2].s1.lo, %[rA2].s1.hi, %[rA2].s3.lo, %[rA2].s3.hi, %[rA2].s5.lo, %[rA2].s5.hi, %[rA2].s7.lo, %[rA2].s7.hi, 52.x, 60.x, 53.x, 61.x, 54.x, 62.x, 55.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA3].s0.lo, %[rA3].s0.hi, %[rA3].s2.lo, %[rA3].s2.hi, %[rA3].s4.lo, %[rA3].s4.hi, %[rA3].s6.lo, %[rA3].s6.hi, 00.x, 08.x, 01.x, 09.x, 02.x, 10.x, 03.x, 11.x, (ss),    )
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_FIFTH_128B)  "], %[rB3].s2.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SIXTH_128B)  "], %[rB3].s2.hi, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_SEVENTH_128B)"], %[rB3].s3.lo, 1;"
                       "stl.u16 l[%[b_localOffsetInBytes]+  "xstr(LM_EIGHTH_128B) "], %[rB3].s3.hi, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA3].s1.lo, %[rA3].s1.hi, %[rA3].s3.lo, %[rA3].s3.hi, %[rA3].s5.lo, %[rA3].s5.hi, %[rA3].s7.lo, %[rA3].s7.hi, 16.x, 24.x, 17.x, 25.x, 18.x, 26.x, 19.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA3].s0.lo, %[rA3].s0.hi, %[rA3].s2.lo, %[rA3].s2.hi, %[rA3].s4.lo, %[rA3].s4.hi, %[rA3].s6.lo, %[rA3].s6.hi, 04.x, 12.x, 05.x, 13.x, 06.x, 14.x, 07.x, 15.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA3].s1.lo, %[rA3].s1.hi, %[rA3].s3.lo, %[rA3].s3.hi, %[rA3].s5.lo, %[rA3].s5.hi, %[rA3].s7.lo, %[rA3].s7.hi, 20.x, 28.x, 21.x, 29.x, 22.x, 30.x, 23.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA3].s0.lo, %[rA3].s0.hi, %[rA3].s2.lo, %[rA3].s2.hi, %[rA3].s4.lo, %[rA3].s4.hi, %[rA3].s6.lo, %[rA3].s6.hi, 32.x, 40.x, 33.x, 41.x, 34.x, 42.x, 35.x, 43.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA3].s1.lo, %[rA3].s1.hi, %[rA3].s3.lo, %[rA3].s3.hi, %[rA3].s5.lo, %[rA3].s5.hi, %[rA3].s7.lo, %[rA3].s7.hi, 48.x, 56.x, 49.x, 57.x, 50.x, 58.x, 51.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA3].s0.lo, %[rA3].s0.hi, %[rA3].s2.lo, %[rA3].s2.hi, %[rA3].s4.lo, %[rA3].s4.hi, %[rA3].s6.lo, %[rA3].s6.hi, 36.x, 44.x, 37.x, 45.x, 38.x, 46.x, 39.x, 47.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA3].s1.lo, %[rA3].s1.hi, %[rA3].s3.lo, %[rA3].s3.hi, %[rA3].s5.lo, %[rA3].s5.hi, %[rA3].s7.lo, %[rA3].s7.hi, 52.x, 60.x, 53.x, 61.x, 54.x, 62.x, 55.x, 63.x,     ,(ul))

                        :  [rC] "+r" (*regC_ptr)
                        :  [rA0] "r" (*regA0), [rA1] "r" (*regA1), [rA2] "r" (*regA2), [rA3] "r" (*regA3),
                           [rB0] "r" (*regB0), [rB1] "r" (*regB1), [rB2] "r" (*regB2), [rB3] "r" (*regB3),
                           [b_localOffsetInBytes]  "r" (b_localOffsetInBytes), 
                            [linearIndex]  "r" (linearIndex)
                        :  "r36.x"
                       );



}

#else

inline void mm_mad_N32K64_INT8_1x1(float16 *regC_ptr,
                                   uint8 *regA0, uint8 *regA1, uint8 *regA2, uint8 *regA3,
                                   uint8 *regB0, uint8 *regB1, uint8 *regB2, uint8 *regB3,
                                   mm_setup_info setup_info)
{
    
    short4 temp1;
    short4 temp2;


    __asm__ __volatile__(
                        
                        
                                           
                     //   "ALLOC_MALUM;"
                     //   "stm.m64n32.l.u16 m[0], (rmajor)%[rC].s!<0>;"
                     //   "(ss)nop;"
                       "malu_wmma.accu.m64n32.f16f16 m[0], (linear)(rmajor)%[rA0].s0, (linear)(cmajor)%[rB0].s0, m[0];"
                       "malu_wmma.accu.m64n32.f16f16 m[0], (linear)(rmajor)%[rA1].s0, (linear)(cmajor)%[rB1].s0, m[0];"
                       "malu_wmma.accu.m64n32.f16f16 m[0], (linear)(rmajor)%[rA2].s0, (linear)(cmajor)%[rB2].s0, m[0];"
                       "malu_wmma.accu.m64n32.f16f16 m[0], (linear)(rmajor)%[rA3].s0, (linear)(cmajor)%[rB3].s0, m[0];"                  
                     //   "ldm.m64n32.l.u16 (rmajor)%[rC].s!<0>, m[0];"
                     //   "DEALLOC_MALUM;"
                       "(ss)nop;"
                        :  [rC] "+r" (*regC_ptr),
                           [rA0] "+r" (*regA0), [rA1] "+r" (*regA1), [rA2] "+r" (*regA2), [rA3] "+r" (*regA3),
                           [rB0] "+r" (*regB0), [rB1] "+r" (*regB1), [rB2] "+r" (*regB2), [rB3] "+r" (*regB3),
                           [rT1]  "=r" (temp1),
                           [rT2]  "=r" (temp2)
                        :  [mask]  "r" ((short)0x00FF), 
                            [mask_sign]  "r" ((short)0x0001), 
                            [neg256]                    "r" ((short)0xFF00)
                        : 
                       );

}

#endif


//----------------------------------------------------------------------------//
// requantization + Relu
//----------------------------------------------------------------------------//
inline void requant_N20(float16 *regC_ptr, const float32 scaleConst, int32 biasConst, const float max, const float min)
{


   __asm__ __volatile__(
                        "nop;"
#ifdef USE_MALU                        
                        "ldm.m64n32.l.u16 (rmajor)%[rC].s!<0>, m[0];"
                        "(ss)nop;"
                        "DEALLOC_MALUM;"
#endif                       
                        "(rpt3)add.f             %[rC].s!< 0>.lo,  (r)%[rC].s!< 0>.lo,     (r)%[biasConst].s!< 0>.lo;"
                        "(rpt3)add.f             %[rC].s!< 2>.lo,  (r)%[rC].s!< 2>.lo,     (r)%[biasConst].s!< 2>.lo;"
                        "(rpt3)add.f             %[rC].s!< 4>.lo,  (r)%[rC].s!< 4>.lo,     (r)%[biasConst].s!< 4>.lo;"
                        "(rpt3)add.f             %[rC].s!< 6>.lo,  (r)%[rC].s!< 6>.lo,     (r)%[biasConst].s!< 6>.lo;"
                        "(rpt3)add.f             %[rC].s!< 8>.lo,  (r)%[rC].s!< 8>.lo,     (r)%[biasConst].s!< 8>.lo;"                                                
                        :  [rC] "+r" (*regC_ptr)
                        :  [scaleConst]  "c" (scaleConst),
                           [biasConst]   "c" (biasConst),
                           [min]         "c" (min),
                           [max]         "c" (max),
                           [point_five]  "c" (0x3F000000)
                        :
                       );



}

//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N20(float16 *regC_ptr)
{
   uint   mask0 = get_local_id(0) & 0xFC;
   uint   mask1 = mask0 + 1;
   uint   mask2 = mask0 + 2;
   uint   mask3 = mask0 + 3;
   uint   tmp0, tmp1, tmp2;
   uint   id_mod_4 = (get_local_id(0) % 4);

   __asm__ __volatile__(      
                        "nop;"   
                        "mov.u16u16 %[rC].s!< 9>.lo,  %[rC].s!< 0>.hi;"                                      
                        "mov.u16u16 %[rC].s!< 9>.hi,  %[rC].s!< 1>.lo;"
                        "mov.u16u16 %[rC].s!<10>.lo,  %[rC].s!< 1>.hi;"
                        "mov.u16u16 %[rC].s!<10>.hi,  %[rC].s!< 2>.lo;"
                        "mov.u16u16 %[rC].s!<11>.lo,  %[rC].s!< 3>.lo;"
                        "mov.u16u16 %[rC].s!<11>.hi,  %[rC].s!< 3>.hi;"
                        "mov.u16u16 %[rC].s!<12>.lo,  %[rC].s!< 4>.lo;"
                        "mov.u16u16 %[rC].s!<12>.hi,  %[rC].s!< 4>.hi;"
                        "mov.u16u16 %[rC].s!<13>.lo,  %[rC].s!< 5>.hi;"
                        "mov.u16u16 %[rC].s!<13>.hi,  %[rC].s!< 6>.lo;"
                        "mov.u16u16 %[rC].s!<14>.lo,  %[rC].s!< 6>.hi;"
                        "mov.u16u16 %[rC].s!<14>.hi,  %[rC].s!< 7>.lo;"                        
                        "mov.u16u16 %[rC].s!< 0>.hi,  %[rC].s!<10>.hi;"
                        "mov.u16u16 %[rC].s!< 1>.lo,  %[rC].s!<12>.lo;"
                        "mov.u16u16 %[rC].s!< 1>.hi,  %[rC].s!<13>.hi;"
                        "mov.u16u16 %[rC].s!< 2>.lo,  %[rC].s!< 9>.lo;"
                        "mov.u16u16 %[rC].s!< 3>.lo,  %[rC].s!<12>.hi;"
                        "mov.u16u16 %[rC].s!< 3>.hi,  %[rC].s!<14>.lo;"
                        "mov.u16u16 %[rC].s!< 4>.lo,  %[rC].s!< 9>.hi;"
                        "mov.u16u16 %[rC].s!< 4>.hi,  %[rC].s!<11>.lo;"
                        "mov.u16u16 %[rC].s!< 5>.hi,  %[rC].s!<14>.hi;"
                        "mov.u16u16 %[rC].s!< 6>.lo,  %[rC].s!<10>.lo;"
                        "mov.u16u16 %[rC].s!< 6>.hi,  %[rC].s!<11>.hi;"
                        "mov.u16u16 %[rC].s!< 7>.lo,  %[rC].s!<13>.lo;"
                        
                        // "(rpt3)shlm.b32 %[rC].s!<0>,  sr50.x, (r)%[rC].s!<0>,  sr51.x;" // s0<<0, s1<<8, s2<<16, s3<<24
                        // "(rpt3)shlm.b32 %[rC].s!<4>,  sr50.y, (r)%[rC].s!<4>,  sr51.y;"
                        // "(rpt3)shlm.b32 %[rC].s!<8>,  sr50.z, (r)%[rC].s!<8>,  sr51.z;"
                        // "(rpt3)shlm.b32 %[rC].s!<12>, sr50.w, (r)%[rC].s!<12>, sr51.w;"
                        // "or.b  %[rC].s!< 0>,  %[rC].s!< 4>,  %[rC].s!< 0>;"
                        // "or.b  %[rC].s!< 1>,  %[rC].s!< 5>,  %[rC].s!< 1>;"
                        // "or.b  %[rC].s!< 2>,  %[rC].s!< 6>,  %[rC].s!< 2>;"
                        // "or.b  %[rC].s!< 3>,  %[rC].s!< 7>,  %[rC].s!< 3>;"
                        // "or.b  %[rC].s!< 0>,  %[rC].s!< 8>,  %[rC].s!< 0>;"  // s0 readable, no need for nop
                        // "or.b  %[rC].s!< 1>,  %[rC].s!< 9>,  %[rC].s!< 1>;"
                        // "or.b  %[rC].s!< 2>,  %[rC].s!<10>,  %[rC].s!< 2>;"
                        // "or.b  %[rC].s!< 3>,  %[rC].s!<11>,  %[rC].s!< 3>;"
                        // "or.b  %[rC].s!< 0>,  %[rC].s!<12>,  %[rC].s!< 0>;"
                        // "or.b  %[rC].s!< 1>,  %[rC].s!<13>,  %[rC].s!< 1>;"
                        // "or.b  %[rC].s!< 2>,  %[rC].s!<14>,  %[rC].s!< 2>;"
                        // "or.b  %[rC].s!< 3>,  %[rC].s!<15>,  %[rC].s!< 3>;"
                        // // "(ss)nop;"
                        // // "predt #14;"
                        // "shfl.gen.w4.u32 %[tmp0].x,  %[rC].s!<16>, %[mask1];"
                        // "shfl.gen.w4.u32 %[tmp1].x,  %[rC].s!<16>, %[mask2];"
                        // "shfl.gen.w4.u32 %[tmp2].x,  %[rC].s!<16>, %[mask3];"
                        // "(ss)shlm.b32 %[rC].s!<16>,  sr50.x, %[rC].s!<16>,  sr51.x;"
                        // "    shlm.b32 %[tmp0].x,     sr50.y, %[tmp0].x,     sr51.y;"
                        // "    shlm.b32 %[tmp1].x,     sr50.z, %[tmp1].x,     sr51.z;"
                        // "    shlm.b32 %[tmp2].x,     sr50.w, %[tmp2].x,     sr51.w; nop;"                        
                        // "or.b  %[rC].s!<16>,  %[tmp0].x,  %[rC].s!<16>; (rpt3)nop;"
                        // "or.b  %[rC].s!<16>,  %[tmp1].x,  %[rC].s!<16>; (rpt3)nop;"
                        // "or.b  %[rC].s!<16>,  %[tmp2].x,  %[rC].s!<16>; (rpt3)nop;"
                        // // "prede;"
                        :  [rC] "+r" (*regC_ptr),
                           [tmp0] "=r" (tmp0),
                           [tmp1] "=r" (tmp1),
                           [tmp2] "=r" (tmp2)
                        :  [mask1]        "r" (mask1),
                           [mask2]        "r" (mask2),
                           [mask3]        "r" (mask3),
                           [id_mod4]      "r" (id_mod_4) 
                        :  "p0.x"
                       );

}

inline void mm_store_c_N20(float32 *regC_ptr, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"                        
                        "(ss)nop;"
                        :  [rC]              "+r" (*regC_ptr)
                        :  [c_coordInWords]  "r" (c_coordInWords)
                        :
                       );
}

inline void mm_store_c_N16_D2S(float16 *regC_ptr, const int scale_factor, uint global_id_m, uint subMatrixCStartInElements, int output_width)
{
    uint sub_block_id_h = global_id_m / output_width;
    uint sub_block_id_w = global_id_m % output_width;
    uint d2s_col = get_global_id(2) * TILESIZE_N / (scale_factor*scale_factor);

    uint   id_mod_4 = (get_local_id(0) % 4);

    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInHalfWords = (subMatrixCStartInElements + sub_block_id_m);

    uint c_coordInWords00 =  2*((scale_factor * sub_block_id_h)     * (output_width*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords01 =  2*((scale_factor * sub_block_id_h)     * (output_width*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords10 =  2*((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords11 =  2*((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords00 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor) + (scale_factor * sub_block_id_w)) * 2   / (scale_factor * scale_factor) + d2s_col;
   //  uint f_coordInHalfWords01 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords10 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor) + (scale_factor * sub_block_id_w)) * 2    / (scale_factor * scale_factor) + d2s_col;
   //  uint f_coordInHalfWords11 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
   //  uint c_coordInWordsD  =  ((sub_block_id_h)     * (OUT_WIDTH) + (sub_block_id_w));
    
   __asm__ __volatile__(
                        "nop;"
                        // "cmps.u.gt p0, %[id_mod4], 0; nop;" 
                        "stib.untyped.u32.1d.2.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords00], 3;"
                        "stib.untyped.u32.1d.2.unpacked.mode4.base0           %[rC].s!<2>,  %[c_coordInWords01], 3;"
                        "stib.untyped.u32.1d.2.unpacked.mode4.base0           %[rC].s!<4>,  %[c_coordInWords10], 3;"
                        "stib.untyped.u32.1d.2.unpacked.mode4.base0           %[rC].s!<6>,  %[c_coordInWords11], 3;"
                        // "(ss)nop;"
                        // "predf #2;"
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<8>.lo,  %[c_coordInHalfWords], 4;"
                        // // "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<26>.lo, %[f_coordInHalfWords10],  5;"
                        // "prede;"  
                        "(ss)nop;"                                            
                        :  [rC]                 "+r" (*regC_ptr)
                        :  [c_coordInWords00]   "r" (c_coordInWords00),
                           [c_coordInWords01]   "r" (c_coordInWords01),
                           [c_coordInWords10]   "r" (c_coordInWords10),
                           [c_coordInWords11]   "r" (c_coordInWords11),
                           [c_coordInHalfWords] "r" (c_coordInHalfWords)
                        :  
                       );
}

// Reload part of matrix C (calculated by previous tile), instead of calculating
inline void mm_reload_c_N32(float32 *regC_ptr, uint subMatrixCStartInElements, uint subMatrixCStartInElements_load_from)
{
    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInWords           = (subMatrixCStartInElements           + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
    uint c_coordInWords_load_from = (subMatrixCStartInElements_load_from + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords_load_from], 3;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords_load_from], 3;"
                        "bar.set;"
                        PRIORITYRESET
                        "(ss)(sy)nop;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"                        
                        "(ss)nop;"
                        :  [rC]                       "+r" (*regC_ptr)
                        :  [c_coordInWords]            "r" (c_coordInWords),
                           [c_coordInWords_load_from]  "r" (c_coordInWords_load_from)
                        :
                       );
}

inline void mm_init_memc(float16 regC)
{
    __asm__ __volatile__(
#ifdef USE_MALU      
      "ALLOC_MALUM;"
      "stm.m64n32.l.u16 m[0], (rmajor)%[rC].s!< 0>;"
#endif      
      "(ss)nop;"                        
      :
      :  [rC] "r" (regC),
         "malu_matrix_c_size"(MATRIX_C_ON_MALUMEM_IN_KBYTES)
      :
      ); 
}

__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
__kernel void Conv_MxM_Baseline(__read_only    image1d_buffer_t matrix_A, // [IHxIW, IC]
                                __read_only    image1d_buffer_t matrix_W, // [OC, FHxFWxIC]
                                __global const float* restrict  matrix_B, // Bias
                                __global const float* restrict  matrix_S, // Scale
                                __read_write   image1d_buffer_t matrix_C, // [OHxOW, OC]
                                __read_write   image1d_buffer_t matrix_C_mask, // [OHxOW, OC]
                                __global       int*             sync_control, // Sync buffer for persistent waves coordination
                                         const float            min,   
                                         const float            max,                                
                                         const int              total_blocks, // Total workload in blocks.
                                         const int              input_width,
                                         const int              input_height,
                                         const int              input_offset,  // used for tiling
                                         const int              output_offset, // used for tiling
                                         const int              sync_buffer_start_index,         
                                         const int              next_layer_sync_buffer_size, // Need to clear the sync buffer for the next layer
                                         const int              apron_top // used for tiling
)
{

   // Previously for IC:32 kernel we unrolled the entire 3x3 filter. We loaded everything and ran 9 back-to-back wwma calls.
   // In IC:96, we load each row of the 3x3 filter, execute 9 (3 columns and 3 blocks of K32 each) back-to-back wmma calls, and load the next row, ...
   // This kernel has IC:64, we still load each row, but execute 6 (3 columns and 2 blocks of 32 each) back-to-back wmma calls. Let's see if that needs to change later

   //  __private uint4 regBlft[2], regBmid[2], regBrgt[2];
   //  __private uint8 regAlft[2], regAmid[2], regArgt[2];
   //  __private float32 regC;

#ifndef USE_MALU
    __private uint4 regB[4];
#else
    __private uint8 regB[4];
#endif    
    __private uint8 regA[4];   

    __private float16 regC = (float16)0.0; 
    

    size_t block_id_n = get_global_id(2);
    const uint  col = block_id_n * TILESIZE_N;

    size_t total_block_n = get_global_size(2);

    MM_SETUP(matrix_W)
    mm_setup_info setup_info = mm_setup();
    
    const uint strideWinElements = LINE_STRIDE_MATRIX_W_IN_BYTES;
    
    
    
    
    const float32 scaleConst = (float32)(0);
    const int32   biasConst  = (int32)(0);
    mm_load_scale_FP32(matrix_S + col, scaleConst);
    mm_load_bias_INT32(matrix_B + col, biasConst);

    // Reset sync_contro buffer for the next leyaer
    if ((get_global_linear_id() == 0)) {
      for (int i = 0; i < next_layer_sync_buffer_size; i++)
         atomic_and(sync_control+i, 0);
    }

     
   uint  block_id_m;
   const int my_layer_index = sync_buffer_start_index + block_id_n;
   if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
   block_id_m = sub_group_shuffle(block_id_m, 0);

   int output_width = input_width / STRIDE;   

   //------------------------------------------------------------------------//
   // MxM 
   //------------------------------------------------------------------------//
   #pragma unroll 1
   while (block_id_m < total_blocks)
   {      

      const uint row = block_id_m * TILESIZE_M + TILE_OUTPUT_OFFSET;
      uint gidx_m = get_local_id(0) + TILESIZE_M * block_id_m;
      uint out_col_id = gidx_m % output_width;
      int out_row_id = gidx_m / output_width;
      int input_col_id1 = STRIDE * out_col_id;
      int input_col_id0 = input_col_id1 - DILATION;
      int input_col_id2 = input_col_id1 + DILATION;

         
      int input_row_id = STRIDE * out_row_id + TILE_INPUT_OFFSET + ((apron_top > 0) ? 1 : 0);
      uint matAcoord[9];
      matAcoord[3] = (input_row_id * input_width + input_col_id0) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[4] = (input_row_id * input_width + input_col_id1) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[5] = (input_row_id * input_width + input_col_id2) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[0] = matAcoord[3] - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[1] = matAcoord[4] - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[2] = matAcoord[5] - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);       
      matAcoord[6] = matAcoord[3] + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[7] = matAcoord[4] + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      matAcoord[8] = matAcoord[5] + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES); 
      
      uint zfill[9];
      zfill[3] = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      zfill[4] = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      zfill[5] = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      zfill[0] = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      zfill[1] = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      zfill[2] = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      zfill[6] = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      zfill[7] = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      zfill[8] = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;

      regC = (float16)0.0; 


      mm_init_memc(regC);
      
      // First row
      // BARSYNC
      #pragma unroll 1
      for (int i = 0; i < 9; i++) {
         mm_load_b_N32K64_INT8_1x1( &regB[0], &regB[1],
                                 &regB[2], &regB[3],
                                 ((col * LINE_STRIDE_MATRIX_W_IN_BYTES + i*IC)),
                                 setup_info);

         mm_load_a_im2col_M64K64_INT8_1x1_zfill(&regA[0], &regA[1], 
                                                &regA[2], &regA[3], 
                                                matAcoord[i], zfill[i]);

         // BARSET
         // __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

         mm_mad_N32K64_INT8_1x1( &regC, &regA[0], &regA[1], &regA[2], &regA[3],
                                 &regB[0], &regB[1], &regB[2], &regB[3],
                                 setup_info);
      }
      

      if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
            

      requant_N20(&regC, scaleConst, biasConst, max, min);
      mm_prepare_store_c_N20(&regC);
      
      const uint subMatrixCStartInElements = (row + col);
      mm_store_c_N16_D2S(&regC, 2, gidx_m + TILE_OUTPUT_OFFSET, subMatrixCStartInElements, output_width);
      block_id_m = sub_group_shuffle(block_id_m, 0);
      // BARSYNC

   }
}
