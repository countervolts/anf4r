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

#define xstr(s) str(s)
#define str(s) #s

#define BYTES_PER_WORD       4
#define BYTES_PER_HALF_WORD  2

#ifndef ENABLE_A_TRANS
#define A_LAYOUT_K_FIRST 1
#else
#define A_LAYOUT_K_FIRST 0
#endif

#ifndef ENABLE_B_TRANS
#define B_LAYOUT_K_FIRST 1
#else
#define B_LAYOUT_K_FIRST 0
#endif

#ifndef ENABLE_C_TRANS
#define C_LAYOUT_N_FIRST 1
#else
#define C_LAYOUT_N_FIRST 0
#endif


#define MATRIX_B_ON_CHIP_IN_BYTES (1024*2)
#define SPML_MODE_FLAG
#define LM_FIRST_256B  0
#define LM_SECOND_256B 512
#define LM_THIRD_256B  1024
#define LM_FOURTH_256B 1536
#define LM_FIRST_512B  0
#define LM_SECOND_512B 1024


#define INT8  0
#define FP16  1
#define FP32  2

#if (TILESIZE_N == 32)
#define MATRIX_C_ON_MALUMEM_IN_KBYTES (16)
#define REG_C_DATATYPE float32
#else
FAIL - need to add code if you want to use other tiles
#endif

#define BYTES_PER_OPERAND 1
#define BYTES_PER_RESULT  4
#define FULL_K_MASK       32

#define USE_PR
#ifdef USE_PR
#define PRIORITYRESET "priorityreset;"
#else
#define PRIORITYRESET
#endif

#if (USE_MALU == 1)
#define RESERVE_REGISTER "r28.x"
#else
#define RESERVE_REGISTER "r44.x"
#endif



//----------------------------------------------------------------------------//
// ALU section
//----------------------------------------------------------------------------//
#if (IC == 24)
#define ALU_32_CYCLES(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b5", (signed)"#rA5", (r)(last)"#rC";"

#elif (IC == 28)
#define ALU_32_CYCLES(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b5", (signed)"#rA5", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b6", (signed)"#rA6", (r)(last)"#rC";"                     
#else
#define ALU_32_CYCLES(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b5", (signed)"#rA5", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b6", (signed)"#rA6", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b7", (signed)"#rA7", (r)(last)"#rC";"
#endif

#if (IC == 4)
#define ALU_16_CYCLES(rC, rA0, rA1, rA2, rA3, b0, b1, b2, b3, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"
#else
#define ALU_16_CYCLES(rC, rA0, rA1, rA2, rA3, b0, b1, b2, b3, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"                     
#endif

//----------------------------------------------------------------------------//
// mm_setup_info 
//----------------------------------------------------------------------------//
// NOTE: matB is layout needs to be Kfirst
typedef struct
{
    uint b_localOffsetInBytes;
    uint b_globalOffsetInWords00;
    uint b_globalOffsetInWords16;
    uint b_globalOffsetInWords32;
    uint b_globalOffsetInWords48;
    uint b_localOffsetInBytes_scale_bias;

} mm_setup_info;


//----------------------------------------------------------------------------//
// MM SETUP 
//----------------------------------------------------------------------------//

#define MM_SETUP(matrix_B, block_id_m, block_id_n)      \
    if (get_global_id(0) == 12345) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regC.xyzw  = convert_float4(read_imageh(matrix_A, get_global_id(0))); \
       regC.xyzw += convert_float4(read_imagef(matrix_B, get_global_id(0))); \
       regC.xyzw += convert_float4(read_imagef(features_history_hr, get_global_id(0))); \
       write_imagef(hidden_features, 0, regC.xyzw);\
       write_imagef(candidate_up, 0, regC.xyzw);\
    }\
    __asm__ __volatile__( \
                        "nop;"\
                        : \
                        : SPML_MODE_FLAG "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : \
                       );





#define LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES 28
//----------------------------------------------------------------------------//
// Update mm_setup_info 
//----------------------------------------------------------------------------//
inline mm_setup_info mm_setup()
{

    mm_setup_info setup_info;
    size_t sub_block_id_m = get_local_id(0); 

    uint   b_localOffsetInElements     = (sub_block_id_m/16)*16 + ((((sub_block_id_m)>>0)&1)<<2) + ((((sub_block_id_m)>>1)&1)<<3) + ((((sub_block_id_m)>>2)&1)<<0) + ((((sub_block_id_m)>>3)&1)<<1);
    setup_info.b_localOffsetInBytes    = b_localOffsetInElements * BYTES_PER_RESULT;

    // For Kailua/Lanai/Hamoa 
    // setup_info.b_localOffsetInBytes += MATRIX_B_ON_CHIP_IN_BYTES - (2*64) - ((setup_info.b_localOffsetInBytes>>6) * (3*64)); // instead of jumping forward by 64, jump backwards by 64 (so subtract 64 three times)
    

    setup_info.b_localOffsetInBytes_scale_bias = sub_block_id_m * BYTES_PER_RESULT;
    // setup_info.b_localOffsetInBytes_scale_bias += MATRIX_B_ON_CHIP_IN_BYTES - (2*64) - ((setup_info.b_localOffsetInBytes_scale_bias>>6) * (3*64));

    /* in xy below, the x means "first dimension" and y means "second dimension" - either can be K or N depending on layout */ 
    uint2  b_globalOffsetInWords_xy = {((sub_block_id_m%4)*(TILESIZE_K*BYTES_PER_OPERAND/BYTES_PER_WORD/4)), (sub_block_id_m>>2)}; 

    setup_info.b_globalOffsetInWords00   = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD;
#if (TILESIZE_N == 32 | TILESIZE_N == 64)    
    setup_info.b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords00 + (16 * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
#endif
#if (TILESIZE_N == 64)
    setup_info.b_globalOffsetInWords32   = setup_info.b_globalOffsetInWords00 + (32 * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
    setup_info.b_globalOffsetInWords48   = setup_info.b_globalOffsetInWords00 + (48 * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
#endif

    short addressInOneSide = ((setup_info.b_localOffsetInBytes>>0)&0x3F);
    short origSideSelect = ((setup_info.b_localOffsetInBytes>>6)&1);
    short addressMSBs = (setup_info.b_localOffsetInBytes>>7);
    short newSideSelect = qcom_get_physical_sub_group_id()&1;
    setup_info.b_localOffsetInBytes = (addressMSBs<<8) | (origSideSelect<<7) | (newSideSelect<<6) | (addressInOneSide);

    addressInOneSide = ((setup_info.b_localOffsetInBytes_scale_bias>>0)&0x3F);
    origSideSelect = ((setup_info.b_localOffsetInBytes_scale_bias>>6)&1);
    addressMSBs = (setup_info.b_localOffsetInBytes_scale_bias>>7);
    newSideSelect = qcom_get_physical_sub_group_id()&1;
    setup_info.b_localOffsetInBytes_scale_bias = (addressMSBs<<8) | (origSideSelect<<7) | (newSideSelect<<6) | (addressInOneSide);
    
    // uint physical_wave_id = qcom_get_physical_sub_group_id();
    // if (physical_wave_id&1) // odd wave, use the odd LM section
    //   setup_info.b_localOffsetInBytes += 64;
    return setup_info;
}


//----------------------------------------------------------------------------//
// load a 
//----------------------------------------------------------------------------//


inline uint4 mm_load_a_im2col_M64K16_INT8(uint matA_readRow_plus_step, uint zfill)
{
    __private uint4 regA;
    size_t sub_block_id_m = get_local_id(0);

    uint a_texCoord = matA_readRow_plus_step / (4/BYTES_PER_OPERAND);
    uint cmpVal = 0;

    __asm__ __volatile__(
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA].s0,  %[a_texCoord], 16;"
                        "(sy)nop;"
                        "cmps.u.gt %[cmpVal], %[zfill],  3; (rpt2)nop;  sel.b32 %[rA].s3, %[zero], %[cmpVal], %[rA].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill],  7; (rpt2)nop;  sel.b32 %[rA].s2, %[zero], %[cmpVal], %[rA].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill], 11; (rpt2)nop;  sel.b32 %[rA].s1, %[zero], %[cmpVal], %[rA].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill], 15; (rpt2)nop;  sel.b32 %[rA].s0, %[zero], %[cmpVal], %[rA].s0;"
                        "(ss)nop;"
                        :  [rA] "=r" (regA),
                           [cmpVal] "+r" (cmpVal)
                        :  [a_texCoord] "r" (a_texCoord),
                           [zfill]      "r" (zfill),
                           [zero]       "c" (0x00000000)
                        : 
                       );

    return regA;
}


inline void mm_load_a_im2col_M64K16_INT8_3x(uint4 *regA0_ptr,
                                             uint4 *regA1_ptr,
                                             uint4 *regA2_ptr, 
                                             uint matA_readRow_plus_step0,
                                             uint matA_readRow_plus_step1,
                                             uint matA_readRow_plus_step2,
                                             uint zfill0,
                                             uint zfill1,
                                             uint zfill2)
{
    size_t sub_block_id_m = get_local_id(0);

    uint a_texCoord0 = matA_readRow_plus_step0 / (4/BYTES_PER_OPERAND);
    uint a_texCoord1 = matA_readRow_plus_step1 / (4/BYTES_PER_OPERAND);
    uint a_texCoord2 = matA_readRow_plus_step2 / (4/BYTES_PER_OPERAND);
    uint cmpVal = 0;

    __asm__ __volatile__(
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA0].s0,  %[a_texCoord0], 16;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA1].s0,  %[a_texCoord1], 16;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA2].s0,  %[a_texCoord2], 16;"
                        "unlock;"
                        "(sy)nop;"
                        "cmps.u.gt %[cmpVal], %[zfill0],  3; (rpt2)nop;  sel.b32 %[rA0].s3, %[zero], %[cmpVal], %[rA0].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill0],  7; (rpt2)nop;  sel.b32 %[rA0].s2, %[zero], %[cmpVal], %[rA0].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 11; (rpt2)nop;  sel.b32 %[rA0].s1, %[zero], %[cmpVal], %[rA0].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 15; (rpt2)nop;  sel.b32 %[rA0].s0, %[zero], %[cmpVal], %[rA0].s0;"
                        "cmps.u.gt %[cmpVal], %[zfill1],  3; (rpt2)nop;  sel.b32 %[rA1].s3, %[zero], %[cmpVal], %[rA1].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill1],  7; (rpt2)nop;  sel.b32 %[rA1].s2, %[zero], %[cmpVal], %[rA1].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 11; (rpt2)nop;  sel.b32 %[rA1].s1, %[zero], %[cmpVal], %[rA1].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 15; (rpt2)nop;  sel.b32 %[rA1].s0, %[zero], %[cmpVal], %[rA1].s0;"
                        "cmps.u.gt %[cmpVal], %[zfill2],  3; (rpt2)nop;  sel.b32 %[rA2].s3, %[zero], %[cmpVal], %[rA2].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill2],  7; (rpt2)nop;  sel.b32 %[rA2].s2, %[zero], %[cmpVal], %[rA2].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 11; (rpt2)nop;  sel.b32 %[rA2].s1, %[zero], %[cmpVal], %[rA2].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 15; (rpt2)nop;  sel.b32 %[rA2].s0, %[zero], %[cmpVal], %[rA2].s0;"
                        "(ss)nop;"
                        :  [rA0] "=r" (*regA0_ptr),
                           [rA1] "=r" (*regA1_ptr),
                           [rA2] "=r" (*regA2_ptr),
                           [cmpVal] "+r" (cmpVal)
                        :  [a_texCoord0] "r" (a_texCoord0),
                           [a_texCoord1] "r" (a_texCoord1),
                           [a_texCoord2] "r" (a_texCoord2),
                           [zfill0]      "r" (zfill0),
                           [zfill1]      "r" (zfill1),
                           [zfill2]      "r" (zfill2),
                           [zero]       "c" (0x80808080)
                        : 
                       );

}


inline uint8 mm_load_a_im2col_M64K32_INT8(uint matA_readRow_plus_step, uint zfill)
{
    __private uint8 regA;
    size_t sub_block_id_m = get_local_id(0);
    uint a_texCoord = matA_readRow_plus_step / (4/BYTES_PER_OPERAND);
    uint cmpVal = 0;

    __asm__ __volatile__(
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA].s0,  %[a_texCoord], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA].s4,  %[a_texCoord], 16;" 
                        "unlock;" 
                        "(sy)nop;"
                        "cmps.u.gt %[cmpVal], %[zfill], 3;  (rpt2)nop;  sel.b32 %[rA].s7, %[zero], %[cmpVal], %[rA].s7;"
                        "cmps.u.gt %[cmpVal], %[zfill], 7;  (rpt2)nop;  sel.b32 %[rA].s6, %[zero], %[cmpVal], %[rA].s6;"
                        "cmps.u.gt %[cmpVal], %[zfill], 11; (rpt2)nop;  sel.b32 %[rA].s5, %[zero], %[cmpVal], %[rA].s5;"
                        "cmps.u.gt %[cmpVal], %[zfill], 15; (rpt2)nop;  sel.b32 %[rA].s4, %[zero], %[cmpVal], %[rA].s4;"
                        "cmps.u.gt %[cmpVal], %[zfill], 19; (rpt2)nop;  sel.b32 %[rA].s3, %[zero], %[cmpVal], %[rA].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill], 23; (rpt2)nop;  sel.b32 %[rA].s2, %[zero], %[cmpVal], %[rA].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill], 27; (rpt2)nop;  sel.b32 %[rA].s1, %[zero], %[cmpVal], %[rA].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill], 31; (rpt2)nop;  sel.b32 %[rA].s0, %[zero], %[cmpVal], %[rA].s0;"
                        "(ss)nop;"
                        :  [rA] "=r" (regA),
                           [cmpVal] "+r" (cmpVal)
                        :  [a_texCoord] "r" (a_texCoord),
                           [zfill]      "r" (zfill),
                           [zero]       "c" (0x00000000)
                        : 
                       );

    return regA;
}


#if(IC==32 || IC==28)
inline void mm_load_a_im2col_M64K32_INT8_3x(uint8 *regA0_ptr,
                                            uint8 *regA1_ptr,
                                            uint8 *regA2_ptr, 
                                            uint matA_readRow_plus_step0,
                                            uint matA_readRow_plus_step1,
                                            uint matA_readRow_plus_step2,
                                            uint zfill0,
                                            uint zfill1,
                                            uint zfill2)
{
    
    size_t sub_block_id_m = get_local_id(0);

    uint a_texCoord0 = matA_readRow_plus_step0 / (4/BYTES_PER_OPERAND);
    uint a_texCoord1 = matA_readRow_plus_step1 / (4/BYTES_PER_OPERAND);
    uint a_texCoord2 = matA_readRow_plus_step2 / (4/BYTES_PER_OPERAND);
    uint cmpVal0;
    uint cmpVal1;
    uint cmpVal2;


    __asm__ __volatile__(
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA0].s0,  %[a_texCoord0], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA0].s4,  %[a_texCoord0], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA1].s0,  %[a_texCoord1], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA1].s4,  %[a_texCoord1], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA2].s0,  %[a_texCoord2], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA2].s4,  %[a_texCoord2], 16;" 
                        "unlock;" 
                        "cmps.u.gt %[cmpVal0], %[zfill0], 7;"
                        "cmps.u.gt %[cmpVal1], %[zfill1], 7;"
                        "cmps.u.gt %[cmpVal2], %[zfill2], 7;"
                        "(sy)nop;"         
#if (USE_MALU != 1)                        
                        PRIORITYRESET  
#endif               
                        "(rpt3)sel.b32 %[rA0].s4, %[zero], %[cmpVal0], (r)%[rA0].s4;"
                        "(rpt3)sel.b32 %[rA0].s0, %[zero], %[cmpVal0], (r)%[rA0].s0;"
                        "(rpt3)sel.b32 %[rA1].s4, %[zero], %[cmpVal1], (r)%[rA1].s4;"
                        "(rpt3)sel.b32 %[rA1].s0, %[zero], %[cmpVal1], (r)%[rA1].s0;"
                        "(rpt3)sel.b32 %[rA2].s4, %[zero], %[cmpVal2], (r)%[rA2].s4;"
                        "(rpt3)sel.b32 %[rA2].s0, %[zero], %[cmpVal2], (r)%[rA2].s0;"
#if ((IC==28) && (USE_MALU == 1))
                        "mov.u32u32 %[rA0].s7, 0;"
                        "mov.u32u32 %[rA1].s7, 0;"
                        "mov.u32u32 %[rA2].s7, 0;"
#endif
                        "(ss)nop;"
                        :  [rA0] "=r" (*regA0_ptr),
                           [rA1] "=r" (*regA1_ptr),
                           [rA2] "=r" (*regA2_ptr),
                           [cmpVal0] "=r" (cmpVal0),
                           [cmpVal1] "=r" (cmpVal1),
                           [cmpVal2] "=r" (cmpVal2)
                        :  [a_texCoord0] "r" (a_texCoord0),
                           [a_texCoord1] "r" (a_texCoord1),
                           [a_texCoord2] "r" (a_texCoord2),
                           [zfill0]      "r" (zfill0),
                           [zfill1]      "r" (zfill1),
                           [zfill2]      "r" (zfill2),
                           [zero]       "c" (0x80808080)
                        :  "p0.x", "p0.y", "p0.z"
                       );

}
#else
inline void mm_load_a_im2col_M64K32_INT8_3x(uint8 *regA0_ptr,
                                            uint8 *regA1_ptr,
                                            uint8 *regA2_ptr, 
                                            uint matA_readRow_plus_step0,
                                            uint matA_readRow_plus_step1,
                                            uint matA_readRow_plus_step2,
                                            uint zfill0,
                                            uint zfill1,
                                            uint zfill2)
{
    
    size_t sub_block_id_m = get_local_id(0);

    uint a_texCoord0 = matA_readRow_plus_step0 / (4/BYTES_PER_OPERAND);
    uint a_texCoord1 = matA_readRow_plus_step1 / (4/BYTES_PER_OPERAND);
    uint a_texCoord2 = matA_readRow_plus_step2 / (4/BYTES_PER_OPERAND);
    uint cmpVal = 0;


    __asm__ __volatile__(
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA0].s0,  %[a_texCoord0], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA0].s4,  %[a_texCoord0], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA1].s0,  %[a_texCoord1], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA1].s4,  %[a_texCoord1], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)%[rA2].s0,  %[a_texCoord2], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)(xyzw)%[rA2].s4,  %[a_texCoord2], 16;" 
                        "unlock;" 
                        "(sy)nop;"
                        PRIORITYRESET
                        "cmps.u.gt %[cmpVal], %[zfill0], 3;  (rpt2)nop;  sel.b32 %[rA0].s7, %[zero], %[cmpVal], %[rA0].s7;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 7;  (rpt2)nop;  sel.b32 %[rA0].s6, %[zero], %[cmpVal], %[rA0].s6;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 11; (rpt2)nop;  sel.b32 %[rA0].s5, %[zero], %[cmpVal], %[rA0].s5;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 15; (rpt2)nop;  sel.b32 %[rA0].s4, %[zero], %[cmpVal], %[rA0].s4;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 19; (rpt2)nop;  sel.b32 %[rA0].s3, %[zero], %[cmpVal], %[rA0].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 23; (rpt2)nop;  sel.b32 %[rA0].s2, %[zero], %[cmpVal], %[rA0].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 27; (rpt2)nop;  sel.b32 %[rA0].s1, %[zero], %[cmpVal], %[rA0].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill0], 31; (rpt2)nop;  sel.b32 %[rA0].s0, %[zero], %[cmpVal], %[rA0].s0;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 3;  (rpt2)nop;  sel.b32 %[rA1].s7, %[zero], %[cmpVal], %[rA1].s7;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 7;  (rpt2)nop;  sel.b32 %[rA1].s6, %[zero], %[cmpVal], %[rA1].s6;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 11; (rpt2)nop;  sel.b32 %[rA1].s5, %[zero], %[cmpVal], %[rA1].s5;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 15; (rpt2)nop;  sel.b32 %[rA1].s4, %[zero], %[cmpVal], %[rA1].s4;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 19; (rpt2)nop;  sel.b32 %[rA1].s3, %[zero], %[cmpVal], %[rA1].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 23; (rpt2)nop;  sel.b32 %[rA1].s2, %[zero], %[cmpVal], %[rA1].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 27; (rpt2)nop;  sel.b32 %[rA1].s1, %[zero], %[cmpVal], %[rA1].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill1], 31; (rpt2)nop;  sel.b32 %[rA1].s0, %[zero], %[cmpVal], %[rA1].s0;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 3;  (rpt2)nop;  sel.b32 %[rA2].s7, %[zero], %[cmpVal], %[rA2].s7;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 7;  (rpt2)nop;  sel.b32 %[rA2].s6, %[zero], %[cmpVal], %[rA2].s6;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 11; (rpt2)nop;  sel.b32 %[rA2].s5, %[zero], %[cmpVal], %[rA2].s5;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 15; (rpt2)nop;  sel.b32 %[rA2].s4, %[zero], %[cmpVal], %[rA2].s4;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 19; (rpt2)nop;  sel.b32 %[rA2].s3, %[zero], %[cmpVal], %[rA2].s3;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 23; (rpt2)nop;  sel.b32 %[rA2].s2, %[zero], %[cmpVal], %[rA2].s2;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 27; (rpt2)nop;  sel.b32 %[rA2].s1, %[zero], %[cmpVal], %[rA2].s1;"
                        "cmps.u.gt %[cmpVal], %[zfill2], 31; (rpt2)nop;  sel.b32 %[rA2].s0, %[zero], %[cmpVal], %[rA2].s0;"
                        "(ss)nop;"
                        :  [rA0] "=r" (*regA0_ptr),
                           [rA1] "=r" (*regA1_ptr),
                           [rA2] "=r" (*regA2_ptr),
                           [cmpVal] "+r" (cmpVal)
                        :  [a_texCoord0] "r" (a_texCoord0),
                           [a_texCoord1] "r" (a_texCoord1),
                           [a_texCoord2] "r" (a_texCoord2),
                           [zfill0]      "r" (zfill0),
                           [zfill1]      "r" (zfill1),
                           [zfill2]      "r" (zfill2),
                           [zero]       "c" (0x80808080)
                        : 
                       );

}
#endif


//----------------------------------------------------------------------------//
// load b 
//----------------------------------------------------------------------------//


inline uint4 mm_load_b_N16K16_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint4   regB;

    size_t sub_block_id_m = get_local_id(0);
    short  isOddHalfWave = (get_local_id(1))&1;
    uint   b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00 = setup_info.b_globalOffsetInWords00;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "(rpt4)nop;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline uint4 mm_load_b_N32K16_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint4   regB;

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;
    uint   b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords16;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB].s1, %[b_coordInWords].x, %[b_globalOffsetInWords16];"
                        "(rpt4)nop;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s1, %[rB].s1, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (b_globalOffsetInWords16),                     
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline uint4 mm_load_b_N64K16_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint4   regB;

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;
    uint   b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords16;
    uint   b_globalOffsetInWords32   = setup_info.b_globalOffsetInWords32;
    uint   b_globalOffsetInWords48   = setup_info.b_globalOffsetInWords48;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB].s1, %[b_coordInWords].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB].s2, %[b_coordInWords].x, %[b_globalOffsetInWords32];"
                        "add.u %[rB].s3, %[b_coordInWords].x, %[b_globalOffsetInWords48];"
                        "(rpt2)nop;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"                     
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s1, %[rB].s1, 2;"                  
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s2, %[rB].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB].s3, %[rB].s3, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (b_globalOffsetInWords16),
                           [b_globalOffsetInWords32] "r" (b_globalOffsetInWords32),
                           [b_globalOffsetInWords48] "r" (b_globalOffsetInWords48),
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline uint8 mm_load_b_N16K32_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint8   regB;

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "(rpt5)nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),                         
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline uint8 mm_load_b_N32K32_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint8   regB;

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;
    uint   b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords16;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB].s2, %[b_coordInWords].x, %[b_globalOffsetInWords16];"
                        "(rpt4)nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s2, %[rB].s2, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (b_globalOffsetInWords16),                        
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline void mm_load_b_N32K32_INT8_3x(   uint8 *regB0_ptr,
                                        uint8 *regB1_ptr,
                                        uint8 *regB2_ptr, 
                                        uint subMatrixBStartInElements0, 
                                        uint subMatrixBStartInElements1, 
                                        uint subMatrixBStartInElements2, 
                                        mm_setup_info setup_info)
{

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords0 = (subMatrixBStartInElements0/(4/BYTES_PER_OPERAND));
    uint b_coordInWords1 = (subMatrixBStartInElements1/(4/BYTES_PER_OPERAND));
    uint b_coordInWords2 = (subMatrixBStartInElements2/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;
    uint   b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords16;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB0].s0, %[b_coordInWords0].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0].s2, %[b_coordInWords0].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB1].s0, %[b_coordInWords1].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1].s2, %[b_coordInWords1].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2].s0, %[b_coordInWords2].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2].s2, %[b_coordInWords2].x, %[b_globalOffsetInWords16];"
                        "nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0].s0, %[rB0].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0].s2, %[rB0].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1].s0, %[rB1].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1].s2, %[rB1].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2].s0, %[rB2].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2].s2, %[rB2].s2, 2;"
                        :  [rB0] "=r" (*regB0_ptr),
                           [rB1] "=r" (*regB1_ptr),
                           [rB2] "=r" (*regB2_ptr)
                        :  [b_coordInWords0] "r" (b_coordInWords0),
                           [b_coordInWords1] "r" (b_coordInWords1),
                           [b_coordInWords2] "r" (b_coordInWords2),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (b_globalOffsetInWords16),                        
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

}

inline uint8 mm_load_b_MALU_N32K32_INT8(uint subMatrixBStartInElements)
{
    __private uint8   regB;
    size_t sub_block_id_m = get_local_id(0);
    short  isFirst32Fibers = (get_local_id(0) < 32);
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND)) + (sub_block_id_m * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
    __asm__ __volatile__(
                        "cmps.u.eq p0.x, %[isFirst32Fibers], 1;"
                        "(rpt4)nop;"
                        "(ss)nop;"
                        "predt #3;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0         %[rB].s0, %[b_coordInWords], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4 %[rB].s4, %[b_coordInWords], 2;"
                        "(jp)prede;"
                        "(ss)nop;" /* FIXME - compiler issue */
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords]  "r" (b_coordInWords),
                           [isFirst32Fibers] "r" (isFirst32Fibers)
                        :  "p0.x"
                       );
      return regB;
}

inline void mm_load_b_MALU_N32K32_INT8_3x(  uint8 *regB0_ptr,
                                            uint8 *regB1_ptr,
                                            uint8 *regB2_ptr,
                                            uint subMatrixBStartInElements0,
                                            uint subMatrixBStartInElements1,
                                            uint subMatrixBStartInElements2)
{
    size_t sub_block_id_m = get_local_id(0);
    short  isFirst32Fibers = (get_local_id(0) < 32);
    uint b_coordInWords0 = (subMatrixBStartInElements0/(4/BYTES_PER_OPERAND)) + (sub_block_id_m * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
    uint b_coordInWords1 = (subMatrixBStartInElements1/(4/BYTES_PER_OPERAND)) + (sub_block_id_m * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
    uint b_coordInWords2 = (subMatrixBStartInElements2/(4/BYTES_PER_OPERAND)) + (sub_block_id_m * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
    __asm__ __volatile__(
                        "cmps.u.eq p0.x, %[isFirst32Fibers], 1;"
                        "(rpt4)nop;"
                        "(ss)nop;"
                        "predt #7;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0         %[rB0].s0, %[b_coordInWords0], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4 %[rB0].s4, %[b_coordInWords0], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0         %[rB1].s0, %[b_coordInWords1], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4 %[rB1].s4, %[b_coordInWords1], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0         %[rB2].s0, %[b_coordInWords2], 2;"
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0.offset4 %[rB2].s4, %[b_coordInWords2], 2;"
                        "(jp)prede;"
                        "(ss)nop;" /* FIXME - compiler issue */
                        :  [rB0] "=r" (*regB0_ptr),
                           [rB1] "=r" (*regB1_ptr),
                           [rB2] "=r" (*regB2_ptr)
                        :  [b_coordInWords0]  "r" (b_coordInWords0),
                           [b_coordInWords1]  "r" (b_coordInWords1),
                           [b_coordInWords2]  "r" (b_coordInWords2),
                           [isFirst32Fibers] "r" (isFirst32Fibers)
                        :  "p0.x"
                       );
}

inline uint8 mm_load_b_N64K32_INT8(uint subMatrixBStartInElements, mm_setup_info setup_info)
{
    __private uint8   regB;

    size_t sub_block_id_m = get_local_id(0);
    short isOddHalfWave = (get_local_id(1))&1;
    uint b_coordInWords = (subMatrixBStartInElements/(4/BYTES_PER_OPERAND));
    uint   b_globalOffsetInWords00   = setup_info.b_globalOffsetInWords00;
    uint   b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords16;
    uint   b_globalOffsetInWords32   = setup_info.b_globalOffsetInWords32;
    uint   b_globalOffsetInWords48   = setup_info.b_globalOffsetInWords48;

    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB].s0, %[b_coordInWords].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB].s2, %[b_coordInWords].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB].s4, %[b_coordInWords].x, %[b_globalOffsetInWords32];"
                        "add.u %[rB].s6, %[b_coordInWords].x, %[b_globalOffsetInWords48];"
                        "(rpt2)nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s0, %[rB].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s2, %[rB].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s4, %[rB].s4, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB].s6, %[rB].s6, 2;"
                        :  [rB] "=r" (regB)
                        :  [b_coordInWords] "r" (b_coordInWords),
                           [b_globalOffsetInWords00] "r" (b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (b_globalOffsetInWords16),
                           [b_globalOffsetInWords32] "r" (b_globalOffsetInWords32),
                           [b_globalOffsetInWords48] "r" (b_globalOffsetInWords48),
                           [isOddHalfWave]           "r" (isOddHalfWave)
                        :  "p0.x"
                       );

      return regB;
}

inline float16 mm_load_prev_frame(uint subMatrixPrevFrameStartInElements)
{
    __private float16   regPrevFrame;

    size_t sub_block_id_m = get_local_id(0);
    short  linearIndex = get_local_id(0);

    uint c_coordInWords = (subMatrixPrevFrameStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES/BYTES_PER_WORD));

    __asm__ __volatile__(                        
                        "ldib.untyped.u32.1d.4.unpacked.mode4.base0           %[rPF].s!<0>,  %[c_coordInWords], 3;" // 3???                        
                        "(ss)nop;"
                        :  [rPF] "=r" (regPrevFrame)
                        :  [c_coordInWords]          "r" (c_coordInWords)
                        :  
                       );

      return regPrevFrame;
}

inline uint mm_load_bias_INT32(__global int* restrict matrix_bias)
{
    __private uint   regBias;

    uint b_coordInWords = get_local_id(0);
    short isActive = b_coordInWords < TILESIZE_N;  
    uint b_coordInBytes =   b_coordInWords * 4;

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[isActive], 0;"
                        "(rpt5)nop;"
                        "predt #2;"
                        "ldg.a.f32  %[rB], g[%[matBias]+%[b_coordInBytes]+0], 1;"
                        "(jp)prede;"
                        // "(sy)nop;"             // remove           
                        "(ss)nop;"             // remove           
                        :  [rB]                "=r" (regBias)
                        :  [matBias]            "c"  (matrix_bias),
                           [b_coordInBytes]     "r" (b_coordInBytes),
                           [isActive]           "r" (isActive)
                        :  "p0.x"
                       );

      return regBias;
}

inline float mm_load_scale_FP32(__global float* restrict matrix_scale)
{
    __private float   regScale;

    uint b_coordInWords = get_local_id(0);
    short isActive = b_coordInWords < TILESIZE_N;  
    uint b_coordInBytes =   b_coordInWords * 4;

    __asm__ __volatile__(
                        "cmps.u.ne p0.x, %[isActive], 0;"
                        "(rpt5)nop;"
                        "predt #2;"
                        "ldg.a.f32  %[rB], g[%[matScale]+%[b_coordInBytes]+0], 1;"
                        "(jp)prede;"
                        "(ss)nop;"
                        // "(sy)nop;"             // remove           
                        :  [rB]                "=r" (regScale)
                        :  [matScale]           "c"  (matrix_scale),
                           [b_coordInBytes]     "r" (b_coordInBytes),
                           [isActive]           "r" (isActive)
                        :  "p0.x"
                       );

      return regScale;
}


//----------------------------------------------------------------------------//
// mad
//----------------------------------------------------------------------------//


inline void mm_mad_N64K32_INT8(float64* regC_ptr, uint8 regA, uint8 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB].s1, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s4, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB].s5, 1;"
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<32>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s6, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB].s7, 1;"
                       ALU_32_CYCLES(%[rC].s!<36>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<40>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<44>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<48>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<52>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<56>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<60>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}

inline void mm_mad_N64K16_INT8(float64* regC_ptr, uint4 regA, uint4 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s1, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s2, 1;"
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<32>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s3, 1;"
                       ALU_16_CYCLES(%[rC].s!<36>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<40>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<44>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<48>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       ALU_16_CYCLES(%[rC].s!<52>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<56>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<60>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 44.x, 45.x, 46.x, 47.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}

inline void mm_mad_N32K32_INT8(float32* regC_ptr, uint8 regA, uint8 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB].s1, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )                       
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}
///// check #.x in LM
inline void mm_mad_N32K16_INT8(float32* regC_ptr, uint4 regA, uint4 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB].s1, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )                       
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 44.x, 45.x, 46.x, 47.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}

inline void mm_mad_N16K32_INT8(float16* regC_ptr, uint8 regA, uint8 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB].s1, 1;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, %[rA].s4, %[rA].s5, %[rA].s6, %[rA].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}

inline void mm_mad_N16K16_INT8(float16* regC_ptr, uint4 regA, uint4 regB, mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA].s0, %[rA].s1, %[rA].s2, %[rA].s3, 12.x, 13.x, 14.x, 15.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB] "r" (regB),
                           [rA] "r" (regA),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );

}

#if (USE_MALU == 1)
inline void mm_mad_MALU_N32K32_INT8_3x_no_accum(
                                  uint8 *regA0_ptr,
                                  uint8 *regA1_ptr,
                                  uint8 *regA2_ptr,
                                  uint8 *regB0_ptr,
                                  uint8 *regB1_ptr,
                                  uint8 *regB2_ptr)
{
    __asm__ __volatile__(
                        PRIORITYRESET  
                        "malu_wmma.m64n32.s8s32  m[0], (linear)%[rA0].s0, (linear)%[rB0].s0;"
                        "malu_wmma.accu.m64n32.s8s32  m[0], (linear)%[rA1].s0, (linear)%[rB1].s0, m[0];"
                        "malu_wmma.accu.m64n32.s8s32  m[0], (linear)%[rA2].s0, (linear)%[rB2].s0, m[0];"
                        "(ss)nop;"
                        :
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr)
                        :
                       );
}
inline void mm_mad_MALU_N32K32_INT8_3x(
                                  uint8 *regA0_ptr,
                                  uint8 *regA1_ptr,
                                  uint8 *regA2_ptr,
                                  uint8 *regB0_ptr,
                                  uint8 *regB1_ptr,
                                  uint8 *regB2_ptr)
{
    __asm__ __volatile__(
                        PRIORITYRESET  
                        "malu_wmma.accu.m64n32.s8s32  m[0], (linear)%[rA0].s0, (linear)%[rB0].s0, m[0];"
                        "malu_wmma.accu.m64n32.s8s32  m[0], (linear)%[rA1].s0, (linear)%[rB1].s0, m[0];"
                        "malu_wmma.accu.m64n32.s8s32  m[0], (linear)%[rA2].s0, (linear)%[rB2].s0, m[0];"
                        "(ss)nop;"
                        :
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr)
                        :
                       );
}
#else
inline void mm_mad_N32K32_INT8_3x(float32 *regC_ptr,
                                  uint8 *regA0_ptr,
                                  uint8 *regA1_ptr, 
                                  uint8 *regA2_ptr, 
                                  uint8 *regB0_ptr,
                                  uint8 *regB1_ptr,
                                  uint8 *regB2_ptr,
                                  mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0].s1, 1;"          
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )                       
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )                       
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )                       
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );
}
#endif

inline void mm_mad_N16K32_INT8_3x(float16* regC_ptr,
                                  uint8 *regA0_ptr,
                                  uint8 *regA1_ptr, 
                                  uint8 *regA2_ptr, 
                                  uint8 *regB0_ptr,
                                  uint8 *regB1_ptr,
                                  uint8 *regB2_ptr,
                                  mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "(sy)mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0].s1, 1;"                 
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1].s1, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, %[rA0].s4, %[rA0].s5, %[rA0].s6, %[rA0].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, %[rA1].s4, %[rA1].s5, %[rA1].s6, %[rA1].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, %[rA2].s4, %[rA2].s5, %[rA2].s6, %[rA2].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );
}

inline void mm_mad_N32K16_INT8_3x(float32 *regC_ptr,
                                  uint4 *regA0_ptr,
                                  uint4 *regA1_ptr, 
                                  uint4 *regA2_ptr, 
                                  uint4 *regB0_ptr,
                                  uint4 *regB1_ptr,
                                  uint4 *regB2_ptr,
                                  mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "(sy)mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0].s1, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )                       
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1].s1, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )                       
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2].s1, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )                       
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 44.x, 45.x, 46.x, 47.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );
}

inline void mm_mad_N16K16_INT8_3x(float16* regC_ptr,
                                  uint4 *regA0_ptr,
                                  uint4 *regA1_ptr, 
                                  uint4 *regA2_ptr, 
                                  uint4 *regB0_ptr,
                                  uint4 *regB1_ptr,
                                  uint4 *regB2_ptr,
                                  mm_setup_info setup_info)
{
    uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes;
    short  linearIndex = get_local_id(0);
  
    __asm__ __volatile__(
                       "(sy)mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA0].s0, %[rA0].s1, %[rA0].s2, %[rA0].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2].s0, 1;"
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1].s0, %[rA1].s1, %[rA1].s2, %[rA1].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2].s0, %[rA2].s1, %[rA2].s2, %[rA2].s3, 12.x, 13.x, 14.x, 15.x,     ,(ul))
                        :  [rC] "+r" (*regC_ptr)
                        :  [rB0] "r" (*regB0_ptr),
                           [rA0] "r" (*regA0_ptr),
                           [rB1] "r" (*regB1_ptr),
                           [rA1] "r" (*regA1_ptr),
                           [rB2] "r" (*regB2_ptr),
                           [rA2] "r" (*regA2_ptr),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex)
                        :  
                       );
}




//----------------------------------------------------------------------------//
// activation
//----------------------------------------------------------------------------//
inline void run_activation_N16(float16  *regC_ptr) // fp16
{
#ifdef ReLU
   __asm__ __volatile__(
                        "(rpt3)sel.s32 %[rC].s!< 0>, (r)%[rC].s!< 0>, (r)%[rC].s!< 0>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!< 4>, (r)%[rC].s!< 4>, (r)%[rC].s!< 4>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!< 8>, (r)%[rC].s!< 8>, (r)%[rC].s!< 8>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<12>, (r)%[rC].s!<12>, (r)%[rC].s!<12>, %[zero];"                        
                        :  [rC] "+r" (*regC_ptr)
                        :  [zero] "c" (0x00000000)
                        :  
                       );
#endif
}

//----------------------------------------------------------------------------//
// activation
//----------------------------------------------------------------------------//
inline void run_activation_N32(float32  *regC_ptr) // fp16
{
#ifdef ReLU
   __asm__ __volatile__(
                        "(rpt3)sel.s32 %[rC].s!< 0>, (r)%[rC].s!< 0>, (r)%[rC].s!< 0>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!< 4>, (r)%[rC].s!< 4>, (r)%[rC].s!< 4>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!< 8>, (r)%[rC].s!< 8>, (r)%[rC].s!< 8>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<12>, (r)%[rC].s!<12>, (r)%[rC].s!<12>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<16>, (r)%[rC].s!<16>, (r)%[rC].s!<16>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<20>, (r)%[rC].s!<20>, (r)%[rC].s!<20>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<24>, (r)%[rC].s!<24>, (r)%[rC].s!<24>, %[zero];"
                        "(rpt3)sel.s32 %[rC].s!<28>, (r)%[rC].s!<28>, (r)%[rC].s!<28>, %[zero];"                        
                        :  [rC] "+r" (*regC_ptr)
                        :  [zero] "c" (0x00000000)
                        :  
                       );
#endif
}


//----------------------------------------------------------------------------//
// requantization 
//----------------------------------------------------------------------------//

inline void requant_N32(float32 *regC_ptr, float w_scale, float io_scale, int bias, float max, float min, int o_offset, mm_setup_info setup_info) // fp16
{
   w_scale = w_scale * io_scale;
   uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes_scale_bias;
   __asm__ __volatile__(
                        "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B)  "], %[bias],    1;"
                        "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B) "], %[w_scale], 1;"
                        "(ss)nop;"
                        "(rpt3)add.s             %[rC].s!< 0>,  (r)%[rC].s!< 0>, (r)l0.x;"
                        "(rpt3)add.s             %[rC].s!< 4>,  (r)%[rC].s!< 4>, (r)l1.x;"
                        "(rpt3)add.s             %[rC].s!< 8>,  (r)%[rC].s!< 8>, (r)l2.x;"
                        "(rpt3)add.s             %[rC].s!<12>,  (r)%[rC].s!<12>, (r)l3.x;"
                        "(rpt3)add.s             %[rC].s!<16>,  (r)%[rC].s!<16>, (r)l4.x;"
                        "(rpt3)add.s             %[rC].s!<20>,  (r)%[rC].s!<20>, (r)l5.x;"
                        "(rpt3)add.s             %[rC].s!<24>,  (r)%[rC].s!<24>, (r)l6.x;"
                        "(rpt3)add.s             %[rC].s!<28>,  (r)%[rC].s!<28>, (r)l7.x;"
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.s32f32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.s32f32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)cov.s32f32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.s32f32        %[rC].s!<28>,  (r)%[rC].s!<28>;"
                        "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,  (r)l16.x;"
                        "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,  (r)l17.x;"
                        "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,  (r)l18.x;"
                        "(rpt3)mul.f             %[rC].s!<12>,  (r)%[rC].s!<12>,  (r)l19.x;"
                        "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,  (r)l20.x;"
                        "(rpt3)mul.f             %[rC].s!<20>,  (r)%[rC].s!<20>,  (r)l21.x;"
                        "(rpt3)mul.f             %[rC].s!<24>,  (r)%[rC].s!<24>,  (r)l22.x;"
                        "(rpt3)mul.f             %[rC].s!<28>,  (r)%[rC].s!<28>,  (r)l23.x;"
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     %[min];"
                        "(rpt3)min.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     %[max];"
                        "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32s32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.f32s32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)cov.f32s32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.f32s32        %[rC].s!<28>,  (r)%[rC].s!<28>;"
                        "(rpt3)add.s             %[rC].s!< 0>,  (r)%[rC].s!< 0>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!< 4>,  (r)%[rC].s!< 4>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!< 8>,  (r)%[rC].s!< 8>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!<12>,  (r)%[rC].s!<12>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!<16>,  (r)%[rC].s!<16>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!<20>,  (r)%[rC].s!<20>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!<24>,  (r)%[rC].s!<24>, %[o_offset];"
                        "(rpt3)add.s             %[rC].s!<28>,  (r)%[rC].s!<28>, %[o_offset];"
                        :  [rC] "+r" (*regC_ptr)
                        :  [w_scale] "r" (w_scale),
                           [io_scale] "r" (io_scale),
                           [bias] "r" (bias),
                           [min] "r" (min),
                           [max] "r" (max),
                           [o_offset] "r" (o_offset),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes)
                        :  
                       );

}

//----------------------------------------------------------------------------//
// requantization 
//----------------------------------------------------------------------------//

inline void requant_blend_N24(float32 *regC_ptr, float16 prev_frame, float w_scale, int bias, float max, float min, float o_scale, float o_scale_rcp, mm_setup_info setup_info) // fp16
{
   short  linearIndex = get_local_id(0);
   uint   b_localOffsetInBytes = setup_info.b_localOffsetInBytes_scale_bias;
   short isActive = linearIndex < TILESIZE_N;  
   w_scale = w_scale;

   __asm__ __volatile__(
                        PRIORITYRESET
                        "mova  a0, %[linearIndex];"
                        "cmps.u.ne p0.x, %[isActive], 0;"
                        "(rpt5)nop;"
                        "predt #3;"
                        "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B)  "], %[bias],    1;"
                        "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B) "], %[w_scale], 1;"   
                        "(jp)prede;"    
                        "shrm.b32 %[prev_frame].s!< 4>,  0, %[prev_frame].s!< 1>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 5>,  8, %[prev_frame].s!< 1>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 6>, 16, %[prev_frame].s!< 1>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 7>, 24, %[prev_frame].s!< 1>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 8>,  0, %[prev_frame].s!< 2>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 9>,  8, %[prev_frame].s!< 2>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<10>, 16, %[prev_frame].s!< 2>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<11>, 24, %[prev_frame].s!< 2>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<12>,  0, %[prev_frame].s!< 3>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<13>,  8, %[prev_frame].s!< 3>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<14>, 16, %[prev_frame].s!< 3>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!<15>, 24, %[prev_frame].s!< 3>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 3>, 24, %[prev_frame].s!< 0>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 2>, 16, %[prev_frame].s!< 0>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 1>,  8, %[prev_frame].s!< 0>,  %[mask];"
                        "shrm.b32 %[prev_frame].s!< 0>,  0, %[prev_frame].s!< 0>,  %[mask];"                                                  
                        "(rpt3)add.s    %[prev_frame].s!< 4>,  (r)%[prev_frame].s!< 4>,  128;"
                        "(rpt3)add.s    %[prev_frame].s!< 0>,  (r)%[prev_frame].s!< 0>,  128;"
                        "(rpt3)add.s    %[prev_frame].s!< 8>,  (r)%[prev_frame].s!< 8>,  128;"
                        "(rpt3)add.s    %[prev_frame].s!<12>,  (r)%[prev_frame].s!<12>,  128;"
                        "(rpt3)and.b %[prev_frame].s!< 0>,  (r)%[prev_frame].s!< 0>,  255;"
                        "(rpt3)and.b %[prev_frame].s!< 4>,  (r)%[prev_frame].s!< 4>,  255;"
                        "(rpt3)and.b %[prev_frame].s!< 8>,  (r)%[prev_frame].s!< 8>,  255;"                        
                        "(rpt3)and.b %[prev_frame].s!<12>,  (r)%[prev_frame].s!<12>,  255;"                        
                        "(rpt3)cov.s32f32        %[prev_frame].s!< 0>,  (r)%[prev_frame].s!< 0>;"
                        "(rpt3)cov.s32f32        %[prev_frame].s!< 4>,  (r)%[prev_frame].s!< 4>;"
                        "(rpt3)cov.s32f32        %[prev_frame].s!< 8>,  (r)%[prev_frame].s!< 8>;"
                        "(rpt3)cov.s32f32        %[prev_frame].s!<12>,  (r)%[prev_frame].s!<12>;"
                        "(rpt3)mul.f             %[prev_frame].s!< 0>,  (r)%[prev_frame].s!< 0>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame].s!< 4>,  (r)%[prev_frame].s!< 4>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame].s!< 8>,  (r)%[prev_frame].s!< 8>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame].s!<12>,  (r)%[prev_frame].s!<12>,   %[scale];"
                        "(ss)nop;"
                        "(rpt3)add.s             %[rC].s!< 0>,  (r)%[rC].s!< 0>, (r)l0.x;"
                        "(rpt3)add.s             %[rC].s!< 4>,  (r)%[rC].s!< 4>, (r)l1.x;"
                        "(rpt3)add.s             %[rC].s!< 8>,  (r)%[rC].s!< 8>, (r)l2.x;"
                        "(rpt3)add.s             %[rC].s!<12>,  (r)%[rC].s!<12>, (r)l3.x;"
                        "(rpt3)add.s             %[rC].s!<16>,  (r)%[rC].s!<16>, (r)l4.x;"
                        "(rpt3)add.s             %[rC].s!<20>,  (r)%[rC].s!<20>, (r)l5.x;"
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.s32f32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.s32f32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,  (r)l16.x;"
                        "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,  (r)l17.x;"
                        "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,  (r)l18.x;"
                        "(rpt3)mul.f             %[rC].s!<12>,  (r)%[rC].s!<12>,  (r)l19.x;"
                        "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,  (r)l20.x;"
                        "(rpt3)mul.f             %[rC].s!<20>,  (r)%[rC].s!<20>,  (r)l21.x;"
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[min];"
                        "(rpt3)min.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[max];" // round
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)rndaz.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"                        
                        "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[unsigned_offset];"
                        "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,   %[scale];" // Can it be combined by w_scale?
                        "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,   %[scale];"
                        "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,   %[scale];"
                        "(rpt3)mul.f             %[rC].s!<12>,  (r)%[rC].s!<12>,   %[scale];"
                        "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,   %[scale];"
                        "(rpt3)mul.f             %[rC].s!<20>,  (r)%[rC].s!<20>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame].s!< 0>,  (r)%[prev_frame].s!< 0>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame].s!< 4>,  (r)%[prev_frame].s!< 4>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame].s!< 8>,  (r)%[prev_frame].s!< 8>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame].s!<12>,  (r)%[prev_frame].s!<12>,  (r)%[rC].s!<20>;"
                        "(rpt3)add.f             %[rC].s!<20>,  %[one], (r)(neg)%[rC].s!<20>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[rC].s!< 0>,  (r)%[rC].s!<20>, (r)%[prev_frame].s!< 0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[rC].s!< 4>,  (r)%[rC].s!<20>, (r)%[prev_frame].s!< 4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[rC].s!< 8>,  (r)%[rC].s!<20>, (r)%[prev_frame].s!< 8>;"
                        "(rpt3)mad.f32           %[rC].s!<12>,  (r)%[rC].s!<12>,  (r)%[rC].s!<20>, (r)%[prev_frame].s!<12>;"
                        "(rpt3)mul.f             %[rC].s!< 0>,  %[scale_rcp],  (r)%[rC].s!< 0>;"
                        "(rpt3)mul.f             %[rC].s!< 4>,  %[scale_rcp],  (r)%[rC].s!< 4>;"
                        "(rpt3)mul.f             %[rC].s!< 8>,  %[scale_rcp],  (r)%[rC].s!< 8>;"
                        "(rpt3)mul.f             %[rC].s!<12>,  %[scale_rcp],  (r)%[rC].s!<12>;"      
                        "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        :  [rC] "+r" (*regC_ptr)
                        :  [prev_frame] "+r" (prev_frame),
                           [w_scale] "r" (w_scale),
                           [scale] "r" (o_scale),
                           [scale_rcp] "r" (o_scale_rcp),
                           [bias] "r" (bias),
                           [min] "r" (min),
                           [max] "r" (max),
                           [b_localOffsetInBytes]    "r" (b_localOffsetInBytes),
                           [mask]                    "r" ((uint)0x000000FF), 
                           [linearIndex]             "r" (linearIndex),
                           [isActive] "r" (isActive),
                           [unsigned_offset]       "c" (0x43000000),
                           [one]       "c" (0x3F800000)
                        :  
                       );

}




//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_store_c_N64(float64 regC, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    short  linearIndex = get_local_id(0);

    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
                        "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */     
                    #ifdef ENABLE_INT8ACC
                        "(rpt3)and.b %[rC].s!<0>,  (r)%[rC].s!<0>,  255;" // s0-s3
                        "(rpt3)and.b %[rC].s!<4>,  (r)%[rC].s!<4>,  255;" // s4-s7
                        "(rpt3)and.b %[rC].s!<8>,  (r)%[rC].s!<8>,  255;" // s8-s11
                        "(rpt3)and.b %[rC].s!<12>, (r)%[rC].s!<12>, 255;" // s12-s15
                        "(rpt3)and.b %[rC].s!<16>, (r)%[rC].s!<16>, 255;" // s16-s19
                        "(rpt3)and.b %[rC].s!<20>, (r)%[rC].s!<20>, 255;" // s20-s23
                        "(rpt3)and.b %[rC].s!<24>, (r)%[rC].s!<24>, 255;" // s24-s27
                        "(rpt3)and.b %[rC].s!<28>, (r)%[rC].s!<28>, 255;" // s28-s31
                        "(rpt3)and.b %[rC].s!<32>, (r)%[rC].s!<32>, 255;" // s0-s3
                        "(rpt3)and.b %[rC].s!<36>, (r)%[rC].s!<36>, 255;" // s4-s7
                        "(rpt3)and.b %[rC].s!<40>, (r)%[rC].s!<40>, 255;" // s8-s11
                        "(rpt3)and.b %[rC].s!<44>, (r)%[rC].s!<44>, 255;" // s12-s15
                        "(rpt3)and.b %[rC].s!<48>, (r)%[rC].s!<48>, 255;" // s16-s19
                        "(rpt3)and.b %[rC].s!<52>, (r)%[rC].s!<52>, 255;" // s20-s23
                        "(rpt3)and.b %[rC].s!<56>, (r)%[rC].s!<56>, 255;" // s24-s27
                        "(rpt3)and.b %[rC].s!<60>, (r)%[rC].s!<60>, 255;" // s28-s31
                        "movs.u32u32 sr48.x, 8;"
                        "movs.u32u32 sr48.y, 16;"
                        "movs.u32u32 sr48.z, 24;"
                        "(ss) nop;"
                        "(rpt2)shl.b %[rC].s!<1>,  (r)%[rC].s!<1>,  (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<5>,  (r)%[rC].s!<5>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<9>,  (r)%[rC].s!<9>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<13>, (r)%[rC].s!<13>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<17>, (r)%[rC].s!<17>, (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<21>, (r)%[rC].s!<21>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<25>, (r)%[rC].s!<25>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<29>, (r)%[rC].s!<29>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<33>, (r)%[rC].s!<33>, (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<37>, (r)%[rC].s!<37>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<41>, (r)%[rC].s!<41>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<45>, (r)%[rC].s!<45>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<49>, (r)%[rC].s!<49>, (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<53>, (r)%[rC].s!<53>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<57>, (r)%[rC].s!<57>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<61>, (r)%[rC].s!<61>, (r)sr48.x;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<1>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<5>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<9>,  %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<13>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<17>, %[rC].s!<16>;"
                        "or.b  %[rC].s!<20>, %[rC].s!<21>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<25>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<29>, %[rC].s!<28>;"
                        "or.b  %[rC].s!<32>, %[rC].s!<33>, %[rC].s!<32>;"
                        "or.b  %[rC].s!<36>, %[rC].s!<37>, %[rC].s!<36>;"
                        "or.b  %[rC].s!<40>, %[rC].s!<41>, %[rC].s!<40>;"
                        "or.b  %[rC].s!<44>, %[rC].s!<45>, %[rC].s!<44>;"
                        "or.b  %[rC].s!<48>, %[rC].s!<49>, %[rC].s!<48>;"
                        "or.b  %[rC].s!<52>, %[rC].s!<53>, %[rC].s!<52>;"
                        "or.b  %[rC].s!<56>, %[rC].s!<57>, %[rC].s!<56>;"
                        "or.b  %[rC].s!<60>, %[rC].s!<61>, %[rC].s!<60>;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<2>,  %[rC].s!<0>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<4>,  %[rC].s!<6>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<10>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<14>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<18>, %[rC].s!<16>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<20>, %[rC].s!<22>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<26>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<30>, %[rC].s!<28>;"
                        "or.b  %[rC].s!<32>, %[rC].s!<34>, %[rC].s!<32>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<36>, %[rC].s!<38>, %[rC].s!<36>;"
                        "or.b  %[rC].s!<40>, %[rC].s!<42>, %[rC].s!<40>;"
                        "or.b  %[rC].s!<44>, %[rC].s!<46>, %[rC].s!<44>;"
                        "or.b  %[rC].s!<48>, %[rC].s!<50>, %[rC].s!<48>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<52>, %[rC].s!<54>, %[rC].s!<52>;"
                        "or.b  %[rC].s!<56>, %[rC].s!<58>, %[rC].s!<56>;"
                        "or.b  %[rC].s!<60>, %[rC].s!<62>, %[rC].s!<60>;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<3>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<7>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<11>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<15>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<19>, %[rC].s!<16>;"
                        "or.b  %[rC].s!<20>, %[rC].s!<23>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<27>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<31>, %[rC].s!<28>;"
                        "or.b  %[rC].s!<32>, %[rC].s!<35>, %[rC].s!<32>;"
                        "or.b  %[rC].s!<36>, %[rC].s!<39>, %[rC].s!<36>;"
                        "or.b  %[rC].s!<40>, %[rC].s!<43>, %[rC].s!<40>;"
                        "or.b  %[rC].s!<44>, %[rC].s!<47>, %[rC].s!<44>;"
                        "or.b  %[rC].s!<48>, %[rC].s!<51>, %[rC].s!<48>;"
                        "or.b  %[rC].s!<52>, %[rC].s!<55>, %[rC].s!<52>;"
                        "or.b  %[rC].s!<56>, %[rC].s!<59>, %[rC].s!<56>;"
                        "or.b  %[rC].s!<60>, %[rC].s!<63>, %[rC].s!<60>;"
                        "mov.s32s32 %[rC].s!<1>, %[rC].s!<4>;"
                        "mov.s32s32 %[rC].s!<2>, %[rC].s!<8>;"
                        "mov.s32s32 %[rC].s!<3>, %[rC].s!<12>;"
                        "(rpt4)nop;"
                        "mov.s32s32 %[rC].s!<4>, %[rC].s!<16>;"
                        "mov.s32s32 %[rC].s!<5>, %[rC].s!<20>;"
                        "mov.s32s32 %[rC].s!<6>, %[rC].s!<24>;"
                        "mov.s32s32 %[rC].s!<7>, %[rC].s!<28>;"
                        "(rpt4)nop;"
                        "mov.s32s32 %[rC].s!<8>, %[rC].s!<32>;"
                        "mov.s32s32 %[rC].s!<9>, %[rC].s!<36>;"
                        "mov.s32s32 %[rC].s!<10>, %[rC].s!<40>;"
                        "mov.s32s32 %[rC].s!<11>, %[rC].s!<44>;"
                        "(rpt4)nop;"
                        "mov.s32s32 %[rC].s!<12>, %[rC].s!<48>;"
                        "mov.s32s32 %[rC].s!<13>, %[rC].s!<52>;"
                        "mov.s32s32 %[rC].s!<14>, %[rC].s!<56>;"
                        "mov.s32s32 %[rC].s!<15>, %[rC].s!<60>;"
                        "(rpt4)nop;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset8   %[rC].s!<8>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset12  %[rC].s!<12>, %[c_coordInWords], 3;"
                    #else               
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset8   %[rC].s!<8>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset12  %[rC].s!<12>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset16  %[rC].s!<16>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset20  %[rC].s!<20>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset24  %[rC].s!<24>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset28  %[rC].s!<28>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset32  %[rC].s!<32>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset36  %[rC].s!<36>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset40  %[rC].s!<40>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset44  %[rC].s!<44>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset48  %[rC].s!<48>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset52  %[rC].s!<52>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset56  %[rC].s!<56>, %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset60  %[rC].s!<60>, %[c_coordInWords], 3;"
                    #endif
                        :  [rC] "+r" (regC)
                        :  [linearIndex]             "r" (linearIndex),
                           [c_coordInWords]          "r" (c_coordInWords)
                        :  
                       );

}

inline void mm_store_c_N32(float32 regC, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    short  linearIndex = get_local_id(0);

    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
                        "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */  
                        "(rpt3)and.b %[rC].s!<0>,  (r)%[rC].s!<0>,  255;" // s0-s3
                        "(rpt3)and.b %[rC].s!<4>,  (r)%[rC].s!<4>,  255;" // s4-s7
                        "(rpt3)and.b %[rC].s!<8>,  (r)%[rC].s!<8>,  255;" // s8-s11
                        "(rpt3)and.b %[rC].s!<12>, (r)%[rC].s!<12>, 255;" // s12-s15
                        "(rpt3)and.b %[rC].s!<16>, (r)%[rC].s!<16>, 255;" // s16-s19
                        "(rpt3)and.b %[rC].s!<20>, (r)%[rC].s!<20>, 255;" // s20-s23
                        "(rpt3)and.b %[rC].s!<24>, (r)%[rC].s!<24>, 255;" // s24-s27
                        "(rpt3)and.b %[rC].s!<28>, (r)%[rC].s!<28>, 255;" // s28-s31
                        "movs.u32u32 sr48.x, 8;"
                        "movs.u32u32 sr48.y, 16;"
                        "movs.u32u32 sr48.z, 24;"
                        "(ss) nop;"
                        "(rpt2)shl.b %[rC].s!<1>,  (r)%[rC].s!<1>,  (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<5>,  (r)%[rC].s!<5>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<9>,  (r)%[rC].s!<9>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<13>, (r)%[rC].s!<13>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<17>, (r)%[rC].s!<17>, (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<21>, (r)%[rC].s!<21>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<25>, (r)%[rC].s!<25>, (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<29>, (r)%[rC].s!<29>, (r)sr48.x;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<1>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<5>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<9>,  %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<13>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<17>, %[rC].s!<16>;"
                        "or.b  %[rC].s!<20>, %[rC].s!<21>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<25>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<29>, %[rC].s!<28>;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<2>,  %[rC].s!<0>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<4>,  %[rC].s!<6>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<10>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<14>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<18>, %[rC].s!<16>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<20>, %[rC].s!<22>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<26>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<30>, %[rC].s!<28>;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<3>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<7>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<11>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<15>, %[rC].s!<12>;"
                        "or.b  %[rC].s!<16>, %[rC].s!<19>, %[rC].s!<16>;"
                        "or.b  %[rC].s!<20>, %[rC].s!<23>, %[rC].s!<20>;"
                        "or.b  %[rC].s!<24>, %[rC].s!<27>, %[rC].s!<24>;"
                        "or.b  %[rC].s!<28>, %[rC].s!<31>, %[rC].s!<28>;"
                        "mov.s32s32 %[rC].s!<1>, %[rC].s!<4>;"
                        "mov.s32s32 %[rC].s!<2>, %[rC].s!<8>;"
                        "mov.s32s32 %[rC].s!<3>, %[rC].s!<12>;"
                        "(rpt4)nop;"
                        "mov.s32s32 %[rC].s!<4>, %[rC].s!<16>;"
                        "mov.s32s32 %[rC].s!<5>, %[rC].s!<20>;"
                        "mov.s32s32 %[rC].s!<6>, %[rC].s!<24>;"
                        "mov.s32s32 %[rC].s!<7>, %[rC].s!<28>;"
                        "(rpt4)nop;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"                        
                        :  [rC] "+r" (regC)
                        :  [linearIndex]             "r" (linearIndex),
                           [c_coordInWords]          "r" (c_coordInWords)
                        :  
                       );

}



inline void mm_store_c_N16(float16 regC, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    short  linearIndex = get_local_id(0);

    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));

    
   __asm__ __volatile__(
                        "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */  
                        "(rpt3)and.b %[rC].s!<0>,  (r)%[rC].s!<0>,  255;" // s0-s3
                        "(rpt3)and.b %[rC].s!<4>,  (r)%[rC].s!<4>,  255;" // s4-s7
                        "(rpt3)and.b %[rC].s!<8>,  (r)%[rC].s!<8>,  255;" // s8-s11
                        "(rpt3)and.b %[rC].s!<12>, (r)%[rC].s!<12>, 255;" // s12-s15
                        "movs.u32u32 sr48.x, 8;"
                        "movs.u32u32 sr48.y, 16;"
                        "movs.u32u32 sr48.z, 24;"
                        "(ss) nop;"
                        "(rpt2)shl.b %[rC].s!<1>,  (r)%[rC].s!<1>,  (r)sr48.x;" // s1<<8, s2<<16, s3<<24
                        "(rpt2)shl.b %[rC].s!<5>,  (r)%[rC].s!<5>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<9>,  (r)%[rC].s!<9>,  (r)sr48.x;"
                        "(rpt2)shl.b %[rC].s!<13>, (r)%[rC].s!<13>, (r)sr48.x;"
                        "or.b  %[rC].s!<0>,  %[rC].s!<1>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<5>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<9>,  %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<13>, %[rC].s!<12>;"                        
                        "or.b  %[rC].s!<0>,  %[rC].s!<2>,  %[rC].s!<0>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!<4>,  %[rC].s!<6>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<10>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<14>, %[rC].s!<12>;"                        
                        "or.b  %[rC].s!<0>,  %[rC].s!<3>,  %[rC].s!<0>;"
                        "or.b  %[rC].s!<4>,  %[rC].s!<7>,  %[rC].s!<4>;"
                        "or.b  %[rC].s!<8>,  %[rC].s!<11>, %[rC].s!<8>;"
                        "or.b  %[rC].s!<12>, %[rC].s!<15>, %[rC].s!<12>;" 
                        "nop;"                       
                        "mov.s32s32 %[rC].s!<1>, %[rC].s!<4>;"
                        "mov.s32s32 %[rC].s!<2>, %[rC].s!<8>;"
                        "mov.s32s32 %[rC].s!<3>, %[rC].s!<12>;"
                        "(rpt5)nop;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        :  [rC] "+r" (regC)
                        :  [linearIndex]             "r" (linearIndex),
                           [c_coordInWords]          "r" (c_coordInWords)
                        :  
                       );

}

inline void mm_store_c_D2S_N16(float32 regC, int scale_factor)
{
   //  size_t sub_block_id_m = get_local_id(0);
   //  short  linearIndex = get_local_id(0);

    uint global_id_m = get_global_id(1) * TILESIZE_M + get_global_id(0);
    uint sub_block_id_h = global_id_m / OUT_WIDTH;
    uint sub_block_id_w = global_id_m % OUT_WIDTH;
    uint d2s_col = get_global_id(2) * TILESIZE_N / (scale_factor*scale_factor);

    uint c_coordInWords00 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords01 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords10 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords11 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;

    
   __asm__ __volatile__(
                        // "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */    
                        "(rpt3)and.b %[rC].s!<0>,  (r)%[rC].s!<0>,  255;" // s0-s3
                        "(rpt3)and.b %[rC].s!<4>,  (r)%[rC].s!<4>,  255;" // s4-s7
                        "(rpt3)and.b %[rC].s!<8>,  (r)%[rC].s!<8>,  255;" // s8-s11
                        "(rpt3)and.b %[rC].s!<12>, (r)%[rC].s!<12>, 255;" // s12-s15                                     
                        "movs.u32u32 sr48.x, 8;"
                        "movs.u32u32 sr48.y, 16;"
                        "movs.u32u32 sr48.z, 24;"
                        "(ss)nop;"
                        "(rpt3)shl.b %[rC].s!< 4>,  (r)%[rC].s!< 4>,     sr48.x;"
                        "(rpt3)shl.b %[rC].s!< 8>,  (r)%[rC].s!< 8>,     sr48.y;"
                        "(rpt3)shl.b %[rC].s!<12>,  (r)%[rC].s!<12>,     sr48.z;"                        
                        "or.b  %[rC].s!< 0>,  %[rC].s!< 4>,  %[rC].s!< 0>;"
                        "or.b  %[rC].s!< 1>,  %[rC].s!< 5>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!< 6>,  %[rC].s!< 2>;"
                        "or.b  %[rC].s!< 3>,  %[rC].s!< 7>,  %[rC].s!< 3>;"
                        "or.b  %[rC].s!< 0>,  %[rC].s!< 8>,  %[rC].s!< 0>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!< 1>,  %[rC].s!< 9>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!<10>,  %[rC].s!< 2>;"
                        "or.b  %[rC].s!< 3>,  %[rC].s!<11>,  %[rC].s!< 3>;"
                        "or.b  %[rC].s!< 0>,  %[rC].s!<12>,  %[rC].s!< 0>;"
                        "or.b  %[rC].s!< 1>,  %[rC].s!<13>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!<14>,  %[rC].s!< 2>;"
                    "(ul)or.b  %[rC].s!< 3>,  %[rC].s!<15>,  %[rC].s!< 3>;"                        
                        "(rpt4)nop;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords00], 5;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<1>,  %[c_coordInWords01], 5;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<2>,  %[c_coordInWords10], 5;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<3>,  %[c_coordInWords11], 5;"                    
                        :  [rC] "+r" (regC)
                        :  [c_coordInWords00]          "r" (c_coordInWords00), 
                           [c_coordInWords01]          "r" (c_coordInWords01),                           
                           [c_coordInWords10]          "r" (c_coordInWords10),
                           [c_coordInWords11]          "r" (c_coordInWords11)
                        :  RESERVE_REGISTER
                       );

}

inline void mm_init_memc(REG_C_DATATYPE regC)
{
    __asm__ __volatile__(
                        "stm.m64n32.l.u32 m[0], %[rC].s!< 0>;"
                        "(ss)nop;"
                        :
                        :  [rC] "r" (regC)
                        :
                       );
}

inline void mm_alloc_memc()
{
    __asm__ __volatile__(
                        "ALLOC_MALUM;"
                        "(ss)nop;"
                        :
                        :  "malu_matrix_c_size"(MATRIX_C_ON_MALUMEM_IN_KBYTES)
                        :
                       );


}



__kernel void Conv_MxM_Baseline(__read_only image1d_buffer_t matrix_A,             // [IHxIW, IC]
                                __read_only image1d_buffer_t matrix_W,             // [OC, FHxFWxIC]
                                __read_only image1d_buffer_t features_history_hr,  // Previous frame
                                   __global int*  restrict   matrix_B, // Bias
                                   __global float* restrict  matrix_S, // Scale
                                                     float        min,   
                                                     float        max,
                                                     float    o_scale,   
                                                     float    o_scale_rcp,   
                                                     int   d2s_factor,
                                __write_only image1d_buffer_t hidden_features,
                                __write_only image1d_buffer_t candidate_up

)
{
    size_t sub_block_id_m = get_local_id(0);
    size_t block_id_m = get_global_id(1);
    size_t block_id_n = get_global_id(2);

   // if (block_id_m > 0 | block_id_n > 0) return;


// Specify regC size based on TILESIZE_N and TYPE

#if   (TILESIZE_N == 64)
    __private float64 regC = (float64)(0);
#elif (TILESIZE_N == 32)
    __private float32 regC = (float32)(0);
    __private float16 prev_frame = (float16)(0);
#elif (TILESIZE_N == 16)
    __private float16 regC = (float16)(0);
#endif


// Specify regA and regB size based on TILESIZE_K and TYPE
// regB is actually depends on TILESIZE_N too, but ignoring that for simplicity
// Supported sizes for now: FP32: 8  Fp16: 8  INT8: 16, 32

#if (TILESIZE_K == 16)
#ifndef WIDE_LOAD
    __private uint4 regA;
    __private uint4 regB;   
#else
    __private uint4 regA[3];
    __private uint4 regB[3];
#endif
#elif (TILESIZE_K == 32)
#ifndef WIDE_LOAD
    __private uint8 regA;
    __private uint8 regB;
#else
    __private uint8 regA[3];
    __private uint8 regB[3];
#endif
#endif


    float scale = mm_load_scale_FP32(matrix_S);
    int   bias  = mm_load_bias_INT32(matrix_B);

    const uint row = block_id_m * TILESIZE_M;
    const uint col = block_id_n * TILESIZE_N; 

    const uint stridePrevFrameinElements = LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES/BYTES_PER_RESULT;

    uint subMatrixPrevFrameStartInElements = row * stridePrevFrameinElements + col + 3; // Hardcoded for now. starting slice index 12 / 4 = 3

    prev_frame = mm_load_prev_frame(subMatrixPrevFrameStartInElements);

    const uint strideWinElements = LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_OPERAND;
    const uint strideCinElements = LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_RESULT;

    //------------------------------------------------------------------------//
    // setup 
    //------------------------------------------------------------------------//
    MM_SETUP(matrix_W, block_id_m, block_id_n)

    mm_setup_info setup_info = mm_setup();
    

    uint gidx_m = get_local_id(0) + TILESIZE_M * get_global_id(1); // fibers along M
    uint out_col_id = gidx_m % OUT_WIDTH; // output image col index ,  output is [OHxOW, OC]
    uint out_row_id = gidx_m / OUT_WIDTH; // output image row index

    __private uint matA_readRow0;
    __private uint matA_readRow1;
    __private uint matA_readRow2;

   
    __private uint matA_zfill0=0;
    __private uint matA_zfill1=0;
    __private uint matA_zfill2=0;
    __private uint zfill_partialK = 0;


   // for odd  sized filter, offset will be -filter_w/2
   // for even sized filter, offset will be 0
    int filter_offset_h = (FILTER_HEIGHT % 2 == 0)? 0 : FILTER_HEIGHT/2;
    int filter_offset_w = (FILTER_WIDTH  % 2 == 0)? 0 : FILTER_WIDTH/2;

    //------------------------------------------------------------------------//
    // MxM 
    //------------------------------------------------------------------------//
    #pragma unroll 1
    for (uint step = 0; step < TOTAL_K; step += TILESIZE_K)
    {
        uint subMatrixWStartInElements = col * strideWinElements + step; // B is Kfirst
         
        zfill_partialK = (TOTAL_K - TILESIZE_K < (int)step) ? TILESIZE_K - (TOTAL_K - step) /* last iteration zero mask */ : 0 /* no zero masking */;
         #pragma unroll 1
        for (int filter_row = 0; filter_row < FILTER_HEIGHT; filter_row++)
        {
#ifndef WIDE_LOAD            
            for (int filter_col = 0; filter_col < FILTER_WIDTH; filter_col++)
            {
#ifdef IMPLICIT_PAD

                int input_row_id = STRIDE * out_row_id + DILATION * (filter_row - filter_offset_h); // height
                int input_col_id = STRIDE * out_col_id + DILATION * (filter_col - filter_offset_w); // width

                matA_zfill0 = zfill_partialK;
                if ((input_row_id < 0) || (input_row_id >= INPUT_HEIGHT) || (input_col_id < 0) || (input_col_id >= INPUT_WIDTH) )   // out of boundary cases
                    matA_zfill0 += FULL_K_MASK; // mask all K values (8 for FP32, 16 for FP16, 32 for INT8)
#else
                int input_row_id = STRIDE * out_row_id + DILATION * filter_row;
                int input_col_id = STRIDE * out_col_id + DILATION * filter_col;
                matA_zfill0 = zfill_partialK;
#endif
                matA_readRow0 = (input_row_id * INPUT_WIDTH + input_col_id) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
                

#if (TILESIZE_K == 16)                
#if (TILESIZE_N == 16)
                regB = mm_load_b_N16K16_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K16_INT8(matA_readRow0 + step, matA_zfill0);
                mm_mad_N16K16_INT8(&regC, regA, regB, setup_info);
#elif (TILESIZE_N == 32)
                regB = mm_load_b_N32K16_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K16_INT8(matA_readRow0 + step, matA_zfill0);
                mm_mad_N32K16_INT8(&regC, regA, regB, setup_info);
#elif (TILESIZE_N == 64)
                regB = mm_load_b_N64K16_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K16_INT8(matA_readRow0 + step, matA_zfill0);
                mm_mad_N64K16_INT8(&regC, regA, regB, setup_info);
#endif
#elif (TILESIZE_K == 32)                
#if (TILESIZE_N == 16)
                regB =  mm_load_b_N16K32_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K32_INT8(matA_readRow0 + step, matA_zfill0);
                        mm_mad_N16K32_INT8(&regC, regA, regB, setup_info);
#elif (TILESIZE_N == 32)
                regB =  mm_load_b_N32K32_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K32_INT8(matA_readRow0 + step, matA_zfill0);
                        mm_mad_N32K32_INT8(&regC, regA, regB, setup_info);
#elif (TILESIZE_N == 64)
                regB =  mm_load_b_N64K32_INT8(subMatrixWStartInElements, setup_info);
                regA =  mm_load_a_im2col_M64K32_INT8(matA_readRow0 + step, matA_zfill0);
                        mm_mad_N64K32_INT8(&regC, regA, regB, setup_info);
#endif                
#endif

                subMatrixWStartInElements += IC;
            }
#else  // 3wide load

                matA_zfill0 = zfill_partialK;
                matA_zfill1 = zfill_partialK;
                matA_zfill2 = zfill_partialK;

#ifdef IMPLICIT_PAD
                int input_row_id = STRIDE * out_row_id + DILATION * (filter_row - 1); // NOTE: change to FILTER_HEIGHT/2 for general cases
                // int input_col_id = STRIDE * out_col_id + DILATION * (filter_col - 1);
                int input_col_id1 = STRIDE * out_col_id;
                int input_col_id0 = input_col_id1 - DILATION;
                int input_col_id2 = input_col_id1 + DILATION;

                if ((input_row_id < 0) || (input_row_id >= INPUT_HEIGHT) || (input_col_id0 < 0) || (input_col_id0 >= INPUT_WIDTH) )  matA_zfill0 += FULL_K_MASK;
                if ((input_row_id < 0) || (input_row_id >= INPUT_HEIGHT) || (input_col_id1 < 0) || (input_col_id1 >= INPUT_WIDTH) )  matA_zfill1 += FULL_K_MASK;
                if ((input_row_id < 0) || (input_row_id >= INPUT_HEIGHT) || (input_col_id2 < 0) || (input_col_id2 >= INPUT_WIDTH) )  matA_zfill2 += FULL_K_MASK;
#else
                int input_row_id   = STRIDE * out_row_id + DILATION * filter_row;
                // int input_col_id   = STRIDE * out_col_id + DILATION * filter_col;
                int input_col_id0 = STRIDE * out_col_id;
                int input_col_id1 = input_col_id0 + DILATION;
                int input_col_id2 = input_col_id1 + DILATION;
#endif
                int input_row_start = input_row_id * INPUT_WIDTH; 
                matA_readRow0 = (input_row_start + input_col_id0) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
                matA_readRow1 = (input_row_start + input_col_id1) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
                matA_readRow2 = (input_row_start + input_col_id2) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);

#if (USE_MALU == 1)
                if (filter_row == 0) {
                    mm_alloc_memc();
                }
#endif

#if (TILESIZE_K == 16)                
#if (TILESIZE_N == 16)
                regB[0] =  mm_load_b_N16K16_INT8(subMatrixWStartInElements,      setup_info);
                regB[1] =  mm_load_b_N16K16_INT8(subMatrixWStartInElements+IC,   setup_info);
                regB[2] =  mm_load_b_N16K16_INT8(subMatrixWStartInElements+2*IC, setup_info);
                           mm_load_a_im2col_M64K16_INT8_3x(&regA[0], &regA[1], &regA[2], matA_readRow0 + step, matA_readRow1 + step, matA_readRow2 + step, matA_zfill0, matA_zfill1, matA_zfill2);
                           mm_mad_N16K16_INT8_3x(&regC, &regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2], setup_info);
#elif (TILESIZE_N == 32)
                regB[0] =  mm_load_b_N32K16_INT8(subMatrixWStartInElements,      setup_info);
                regB[1] =  mm_load_b_N32K16_INT8(subMatrixWStartInElements+IC,   setup_info);
                regB[2] =  mm_load_b_N32K16_INT8(subMatrixWStartInElements+2*IC, setup_info);
                           mm_load_a_im2col_M64K16_INT8_3x(&regA[0], &regA[1], &regA[2], matA_readRow0 + step, matA_readRow1 + step, matA_readRow2 + step, matA_zfill0, matA_zfill1, matA_zfill2);
                           mm_mad_N32K16_INT8_3x(&regC, &regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2], setup_info);
#endif
#elif (TILESIZE_K == 32)                
#if (TILESIZE_N == 16)
                regB[0] =  mm_load_b_N16K32_INT8(subMatrixWStartInElements,      setup_info);
                regB[1] =  mm_load_b_N16K32_INT8(subMatrixWStartInElements+IC,   setup_info);
                regB[2] =  mm_load_b_N16K32_INT8(subMatrixWStartInElements+2*IC, setup_info);
                           mm_load_a_im2col_M64K32_INT8_3x(&regA[0], &regA[1], &regA[2], matA_readRow0 + step, matA_readRow1 + step, matA_readRow2 + step, matA_zfill0, matA_zfill1, matA_zfill2);
                           mm_mad_N16K32_INT8_3x(&regC, &regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2], setup_info);
#elif (TILESIZE_N == 32)
#if (USE_MALU == 1)
                           mm_load_b_MALU_N32K32_INT8_3x(&regB[0], &regB[1], &regB[2], subMatrixWStartInElements, subMatrixWStartInElements+IC, subMatrixWStartInElements+2*IC);                
                           mm_load_a_im2col_M64K32_INT8_3x(&regA[0], &regA[1], &regA[2], matA_readRow0 + step, matA_readRow1 + step, matA_readRow2 + step, matA_zfill0, matA_zfill1, matA_zfill2);
                           if (filter_row == 0)                          
                                mm_mad_MALU_N32K32_INT8_3x_no_accum(&regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2]);
                           else 
                                mm_mad_MALU_N32K32_INT8_3x(&regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2]);
#else
                           mm_load_b_N32K32_INT8_3x(&regB[0], &regB[1], &regB[2], subMatrixWStartInElements, subMatrixWStartInElements+IC, subMatrixWStartInElements+2*IC,      setup_info);
                           mm_load_a_im2col_M64K32_INT8_3x(&regA[0], &regA[1], &regA[2], matA_readRow0 + step, matA_readRow1 + step, matA_readRow2 + step, matA_zfill0, matA_zfill1, matA_zfill2);
                           mm_mad_N32K32_INT8_3x(&regC, &regA[0], &regA[1], &regA[2], &regB[0], &regB[1], &regB[2], setup_info);
#endif
#endif                
#endif


                subMatrixWStartInElements += 3 * IC;  // NOTE: work on next 1x3
#endif                
                
        }
    }


#if (USE_MALU == 1)
    __asm__ __volatile__(
                        "ldm.m64n32.l.u32 %[rC].s!< 0>, m[0];"
                        "(ss)nop;"
                        "DEALLOC_MALUM;"
                           :  [rC]                       "=r" (regC)
                           :
                           :
                       );
#endif

//------------------------------------------------------------------------//
// Requantization 
//------------------------------------------------------------------------//

#if   (TILESIZE_N == 64)
     requant_N64(&regC, matrix_S[0], o_offset);
#elif (TILESIZE_N == 32)
     requant_blend_N24(&regC, prev_frame, scale, bias, max, min, o_scale, o_scale_rcp, setup_info); // fp16
   //   requant_N32(&regC, scale, i_o_scale, bias, max, min, o_offset, setup_info);
     
// #elif (TILESIZE_N == 16)
//      requant_N16(&regC, scale, i_o_scale, bias, max, min, o_offset, setup_info);
#endif     




//------------------------------------------------------------------------//
// Store 
//------------------------------------------------------------------------//

mm_store_c_D2S_N16(regC, d2s_factor);

// #ifndef D2S
//    uint subMatrixCStartInElements = C_LAYOUT_N_FIRST ? row * strideCinElements + col : col * strideCinElements + row;
// #if   (TILESIZE_N == 64)
//     mm_store_c_N64(regC, subMatrixCStartInElements);
// #elif (TILESIZE_N == 32)
//     mm_store_c_N32(regC, subMatrixCStartInElements);
// #elif (TILESIZE_N == 16)
//     mm_store_c_N16(regC, subMatrixCStartInElements);
// #endif     
// #else
// #if   (TILESIZE_N == 64)
//     mm_store_c_D2S_N64(regC, d2s_factor);
// #elif (TILESIZE_N == 32)
//     mm_store_c_D2S_N32(regC, d2s_factor);
// #elif (TILESIZE_N == 16)
//     mm_store_c_D2S_N16(regC, d2s_factor);
// #endif   
// #endif
    
}
