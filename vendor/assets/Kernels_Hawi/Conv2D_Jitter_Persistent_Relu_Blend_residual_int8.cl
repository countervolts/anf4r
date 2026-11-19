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

#define MATRIX_B_ON_CHIP_IN_BYTES (1024*2)
#define LM_FIRST_256B  0
#define LM_SECOND_256B 512
#define LM_THIRD_256B  1024
#define LM_FOURTH_256B 1536
#define LM_FIRST_512B  0
#define LM_SECOND_512B 1024

#define BYTES_PER_OPERAND 1
#define BYTES_PER_RESULT  4
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
#define TILE_PREV_FRAME_OFFSET prevframe_offset
#define TILE_OUT_OFFSET output_offset
#define TILE_INPUT_OFFSET input_offset
#else
#define TILE_PREV_FRAME_OFFSET 0 
#define TILE_OUT_OFFSET 0 
#define TILE_INPUT_OFFSET 0 
#endif

#define MATRIX_C_ON_MALUMEM_IN_KBYTES (16)
#define REG_C_DATATYPE float32

// #define LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES 32

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
#define ALU_32_CYCLES_NO_ACC(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, zero, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", "#zero";"\
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
#define ALU_32_CYCLES_NO_ACC(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, zero, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", "#zero";"\
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
#define ALU_32_CYCLES_NO_ACC(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, zero, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", "#zero";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b5", (signed)"#rA5", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b6", (signed)"#rA6", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b7", (signed)"#rA7", (r)(last)"#rC";"
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
#define MM_SETUP(matrix_B)      \
    if (get_global_id(0) == 12345) /* needed to make sure compiler doesn't optimize away this texture (never hit)*/ \
    {\
       regC.xyzw  = convert_float4(read_imageh(matrix_A, get_global_id(0))); \
       regC.xyzw += convert_float4(read_imagef(matrix_B, get_global_id(0))); \
       regC.xyzw += convert_float4(read_imagef(features_history_hr, get_global_id(0))); \
       write_imagef(candidate_up, 0, regC.xyzw);\
       write_imageh(features, 0, convert_half4(regC.xyzw));\
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
    
    return setup_info;
}


//----------------------------------------------------------------------------//
// load a 
//----------------------------------------------------------------------------//

#if defined(USE_MALU) && (IC==28)
#define compmask "xyzO"
#define rptNum "rpt2"
// #define compmask "xyzw"
// #define rptNum "rpt3"
#else
#define compmask "xyzw"
#define rptNum "rpt3"
#endif

inline void mm_load_a_im2col_M64K32_INT8_3x3_zfill(uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint matAcoord0top, uint matAcoord1top, uint matAcoord2top, uint zfill0top, uint zfill1top, uint zfill2top,
                                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint matAcoord0mid, uint matAcoord1mid, uint matAcoord2mid, uint zfill0mid, uint zfill1mid, uint zfill2mid,
                                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint matAcoord0bot, uint matAcoord1bot, uint matAcoord2bot, uint zfill0bot, uint zfill1bot, uint zfill2bot)
{
    __asm__ __volatile__(
#if defined(USE_MALU) && (IC==28)
                        "mov.u32u32 %[rA0top].s7, 0;"
                        "mov.u32u32 %[rA1top].s7, 0;"
                        "mov.u32u32 %[rA2top].s7, 0;"
                        "mov.u32u32 %[rA0mid].s7, 0;"
                        "mov.u32u32 %[rA1mid].s7, 0;"
                        "mov.u32u32 %[rA2mid].s7, 0;"
                        "mov.u32u32 %[rA0bot].s7, 0;"
                        "mov.u32u32 %[rA1bot].s7, 0;"
                        "mov.u32u32 %[rA2bot].s7, 0;"
#endif
                        "cmps.u.gt p0, %[zfill0top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0top].s0,  %[texCoord0top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0top].s4,  %[texCoord0top], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1top].s0,  %[texCoord1top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1top].s4,  %[texCoord1top], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2top].s0,  %[texCoord2top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2top].s4,  %[texCoord2top], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA2top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0mid].s4,  %[texCoord0mid], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1mid].s0,  %[texCoord1mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1mid].s4,  %[texCoord1mid], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2mid].s0,  %[texCoord2mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2mid].s4,  %[texCoord2mid], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA2mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0bot].s0,  %[texCoord0bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0bot].s4,  %[texCoord0bot], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0bot].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1bot].s0,  %[texCoord1bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1bot].s4,  %[texCoord1bot], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1bot].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2bot].s0,  %[texCoord2bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2bot].s4,  %[texCoord2bot], 16;" 
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA2bot].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2bot].s4, %[zero];"
                        "prede;"
                        "(ss)nop;"
                        :  [rA0top] "=r" (*regA0top), [rA1top] "=r" (*regA1top), [rA2top] "=r" (*regA2top),
                           [rA0mid] "=r" (*regA0mid), [rA1mid] "=r" (*regA1mid), [rA2mid] "=r" (*regA2mid),
                           [rA0bot] "=r" (*regA0bot), [rA1bot] "=r" (*regA1bot), [rA2bot] "=r" (*regA2bot)  
                        :  [texCoord0top] "r" (matAcoord0top / (4/BYTES_PER_OPERAND)), [texCoord1top] "r" (matAcoord1top / (4/BYTES_PER_OPERAND)), [texCoord2top] "r" (matAcoord2top / (4/BYTES_PER_OPERAND)),
                           [texCoord0mid] "r" (matAcoord0mid / (4/BYTES_PER_OPERAND)), [texCoord1mid] "r" (matAcoord1mid / (4/BYTES_PER_OPERAND)), [texCoord2mid] "r" (matAcoord2mid / (4/BYTES_PER_OPERAND)),
                           [texCoord0bot] "r" (matAcoord0bot / (4/BYTES_PER_OPERAND)), [texCoord1bot] "r" (matAcoord1bot / (4/BYTES_PER_OPERAND)), [texCoord2bot] "r" (matAcoord2bot / (4/BYTES_PER_OPERAND)),
                           [zfill0top] "r" (zfill0top), [zfill1top] "r" (zfill1top), [zfill2top] "r" (zfill2top),
                           [zfill0mid] "r" (zfill0mid), [zfill1mid] "r" (zfill1mid), [zfill2mid] "r" (zfill2mid),
                           [zfill0bot] "r" (zfill0bot), [zfill1bot] "r" (zfill1bot), [zfill2bot] "r" (zfill2bot),
                           [zero]      "c" (0x80808080)
                        :  "p0.x"
                       );
}

//----------------------------------------------------------------------------//
// load b 
//----------------------------------------------------------------------------//
inline void mm_load_b_N32K32_INT8_3x3(uint4 *regB0top, uint4 *regB1top, uint4 *regB2top, uint matrixBcoord0top, uint matrixBcoord1top, uint matrixBcoord2top,
                                      uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid, uint matrixBcoord0mid, uint matrixBcoord1mid, uint matrixBcoord2mid,
                                      uint4 *regB0bot, uint4 *regB1bot, uint4 *regB2bot, uint matrixBcoord0bot, uint matrixBcoord1bot, uint matrixBcoord2bot,
                                      mm_setup_info setup_info)
{
#ifndef USE_MALU
    size_t sub_block_id_m = get_local_id(0);
    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB0top].s0, %[matrixBcoord0top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0top].s2, %[matrixBcoord0top].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB1top].s0, %[matrixBcoord1top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1top].s2, %[matrixBcoord1top].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2top].s0, %[matrixBcoord2top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2top].s2, %[matrixBcoord2top].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB0mid].s0, %[matrixBcoord0mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0mid].s2, %[matrixBcoord0mid].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB1mid].s0, %[matrixBcoord1mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1mid].s2, %[matrixBcoord1mid].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2mid].s0, %[matrixBcoord2mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2mid].s2, %[matrixBcoord2mid].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB0bot].s0, %[matrixBcoord0bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0bot].s2, %[matrixBcoord0bot].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1bot].s2, %[matrixBcoord1bot].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s2, %[matrixBcoord2bot].x, %[b_globalOffsetInWords16];"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0top].s0, %[rB0top].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0top].s2, %[rB0top].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1top].s0, %[rB1top].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1top].s2, %[rB1top].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2top].s0, %[rB2top].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2top].s2, %[rB2top].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0mid].s0, %[rB0mid].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0mid].s2, %[rB0mid].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1mid].s0, %[rB1mid].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1mid].s2, %[rB1mid].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2mid].s0, %[rB2mid].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2mid].s2, %[rB2mid].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0bot].s0, %[rB0bot].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0bot].s2, %[rB0bot].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 2;"
                        :  [rB0top] "=r" (*regB0top), [rB1top] "=r" (*regB1top), [rB2top] "=r" (*regB2top),
                           [rB0mid] "=r" (*regB0mid), [rB1mid] "=r" (*regB1mid), [rB2mid] "=r" (*regB2mid),
                           [rB0bot] "=r" (*regB0bot), [rB1bot] "=r" (*regB1bot), [rB2bot] "=r" (*regB2bot)
                        :  [matrixBcoord0top] "r" ((matrixBcoord0top/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1top] "r" ((matrixBcoord1top/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2top] "r" ((matrixBcoord2top/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord0mid] "r" ((matrixBcoord0mid/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1mid] "r" ((matrixBcoord1mid/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2mid] "r" ((matrixBcoord2mid/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord0bot] "r" ((matrixBcoord0bot/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1bot] "r" ((matrixBcoord1bot/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2bot] "r" ((matrixBcoord2bot/(4/BYTES_PER_OPERAND))),
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (setup_info.b_globalOffsetInWords16)
                        :
                       );
#else
    size_t sub_block_id_m = get_local_id(0);
    short  isFirst32Fibers = (get_local_id(0) < 32);
    __asm__ __volatile__(
                        "add.u %[rB0top].s0, %[matrixBcoord0top].x, %[fiberOffset];"
                        "add.u %[rB1top].s0, %[matrixBcoord1top].x, %[fiberOffset];"
                        "add.u %[rB2top].s0, %[matrixBcoord2top].x, %[fiberOffset];"
                        "add.u %[rB0mid].s0, %[matrixBcoord0mid].x, %[fiberOffset];"
                        "add.u %[rB1mid].s0, %[matrixBcoord1mid].x, %[fiberOffset];"
                        "add.u %[rB2mid].s0, %[matrixBcoord2mid].x, %[fiberOffset];"
                        "add.u %[rB0bot].s0, %[matrixBcoord0bot].x, %[fiberOffset];"
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[fiberOffset];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[fiberOffset];"
                        "add.u %[rB0top].s1, %[rB0top].s0,          %[strideTimesEight];"
                        "add.u %[rB1top].s1, %[rB1top].s0,          %[strideTimesEight];"
                        "add.u %[rB2top].s1, %[rB2top].s0,          %[strideTimesEight];"
                        "add.u %[rB0mid].s1, %[rB0mid].s0,          %[strideTimesEight];"
                        "add.u %[rB1mid].s1, %[rB1mid].s0,          %[strideTimesEight];"
                        "add.u %[rB2mid].s1, %[rB2mid].s0,          %[strideTimesEight];"
                        "add.u %[rB0bot].s1, %[rB0bot].s0,          %[strideTimesEight];"
                        "add.u %[rB1bot].s1, %[rB1bot].s0,          %[strideTimesEight];"
                        "add.u %[rB2bot].s1, %[rB2bot].s0,          %[strideTimesEight];"
                        "add.u %[rB0top].s2, %[rB0top].s1,          %[strideTimesEight];"
                        "add.u %[rB1top].s2, %[rB1top].s1,          %[strideTimesEight];"
                        "add.u %[rB2top].s2, %[rB2top].s1,          %[strideTimesEight];"
                        "add.u %[rB0mid].s2, %[rB0mid].s1,          %[strideTimesEight];"
                        "add.u %[rB1mid].s2, %[rB1mid].s1,          %[strideTimesEight];"
                        "add.u %[rB2mid].s2, %[rB2mid].s1,          %[strideTimesEight];"
                        "add.u %[rB0bot].s2, %[rB0bot].s1,          %[strideTimesEight];"
                        "add.u %[rB1bot].s2, %[rB1bot].s1,          %[strideTimesEight];"
                        "add.u %[rB2bot].s2, %[rB2bot].s1,          %[strideTimesEight];"
                        "add.u %[rB0top].s3, %[rB0top].s2,          %[strideTimesEight];"
                        "add.u %[rB1top].s3, %[rB1top].s2,          %[strideTimesEight];"
                        "add.u %[rB2top].s3, %[rB2top].s2,          %[strideTimesEight];"
                        "add.u %[rB0mid].s3, %[rB0mid].s2,          %[strideTimesEight];"
                        "add.u %[rB1mid].s3, %[rB1mid].s2,          %[strideTimesEight];"
                        "add.u %[rB2mid].s3, %[rB2mid].s2,          %[strideTimesEight];"
                        "add.u %[rB0bot].s3, %[rB0bot].s2,          %[strideTimesEight];"
                        "add.u %[rB1bot].s3, %[rB1bot].s2,          %[strideTimesEight];"
                        "add.u %[rB2bot].s3, %[rB2bot].s2,          %[strideTimesEight];"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s0, %[rB0top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s1, %[rB0top].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s2, %[rB0top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s3, %[rB0top].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s0, %[rB1top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s1, %[rB1top].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s2, %[rB1top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s3, %[rB1top].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s0, %[rB2top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s1, %[rB2top].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s2, %[rB2top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s3, %[rB2top].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s0, %[rB0mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s1, %[rB0mid].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s2, %[rB0mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s3, %[rB0mid].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s0, %[rB1mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s1, %[rB1mid].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s2, %[rB1mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s3, %[rB1mid].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s0, %[rB2mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s1, %[rB2mid].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s2, %[rB2mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s3, %[rB2mid].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s0, %[rB0bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s1, %[rB0bot].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s2, %[rB0bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s3, %[rB0bot].s3, 2;"
#ifndef USE_ALU
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s0, %[rB1bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s1, %[rB1bot].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s2, %[rB1bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s3, %[rB1bot].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s0, %[rB2bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s1, %[rB2bot].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s2, %[rB2bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s3, %[rB2bot].s3, 2;"
#else
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1bot].s2, %[matrixBcoord1bot].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s2, %[matrixBcoord2bot].x, %[b_globalOffsetInWords16];"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 2;"
#endif
                        "(ss)nop;" /* FIXME - compiler issue */
                        :  [rB0top] "=r" (*regB0top), [rB1top] "=r" (*regB1top), [rB2top] "=r" (*regB2top),
                           [rB0mid] "=r" (*regB0mid), [rB1mid] "=r" (*regB1mid), [rB2mid] "=r" (*regB2mid),
                           [rB0bot] "=r" (*regB0bot), [rB1bot] "=r" (*regB1bot), [rB2bot] "=r" (*regB2bot)
                        :  [matrixBcoord0top] "r" ((matrixBcoord0top/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1top] "r" ((matrixBcoord1top/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2top] "r" ((matrixBcoord2top/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord0mid] "r" ((matrixBcoord0mid/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1mid] "r" ((matrixBcoord1mid/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2mid] "r" ((matrixBcoord2mid/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord0bot] "r" ((matrixBcoord0bot/(4/BYTES_PER_OPERAND))), 
                           [matrixBcoord1bot] "r" ((matrixBcoord1bot/(4/BYTES_PER_OPERAND))),
                           [matrixBcoord2bot] "r" ((matrixBcoord2bot/(4/BYTES_PER_OPERAND))),
                           [strideTimesEight] "c" (8*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD),
                           [fiberOffset]      "r" ((sub_block_id_m&0x7) + ((sub_block_id_m>>3)*(LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD))),
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (setup_info.b_globalOffsetInWords16)
                        :
                       );
#endif
}

inline void mm_load_bias_INT32(__global const float* restrict matrixBias, const int32 biasConst)
{
    __asm__ __volatile__(
                        "ldg.k c[&%[biasConst]], g[%[matrixBias]], 8;"
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
                        "ldg.k c[&%[scaleConst]], g[%[matrixScale]], 8;"
                        "(ss)nop;"
                        :
                        :  [matrixScale] "u" (matrixScale),
                           [scaleConst]  "c" (scaleConst)
                        :
                       );
}

inline float4 mm_load_prev_frame(uint subMatrixPrevFrameStartInElements)
{
    __private float4   regPrevFrame;

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


//----------------------------------------------------------------------------//
// mad
//----------------------------------------------------------------------------//
inline void mm_mad_N24K32_INT8_3x3(float32 *regC_ptr,
                                   uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint4 *regB0top, uint4 *regB1top, uint4 *regB2top,
                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid,
                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint4 *regB0bot, uint4 *regB1bot, uint4 *regB2bot,
                                   mm_setup_info setup_info)
{
    short  linearIndex = get_local_id(0);
#ifdef USE_MALU
    __asm__ __volatile__(
                       "ALLOC_MALUM;"
                       // This Conv Kernel has N24, but MALU still have to do N32 
                       "malu_wmma.m64n32.s8s32       m[0], (linear)(rmajor)%[rA0top].s0, (folded)(cmajor)%[rB0top].s0;"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1top].s0, (folded)(cmajor)%[rB1top].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2top].s0, (folded)(cmajor)%[rB2top].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0mid].s0, (folded)(cmajor)%[rB0mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1mid].s0, (folded)(cmajor)%[rB1mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2mid].s0, (folded)(cmajor)%[rB2mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0bot].s0, (folded)(cmajor)%[rB0bot].s0, m[0];"
#ifndef USE_ALU
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1bot].s0, (folded)(cmajor)%[rB1bot].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2bot].s0, (folded)(cmajor)%[rB2bot].s0, m[0];"
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "DEALLOC_MALUM;"
                       "(ss)nop;"
#else
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1bot].s1, 1;" 
                       "DEALLOC_MALUM;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2bot].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,(ul))
#endif
                        :  [rC] "=r" (*regC_ptr)
                        :  [rB0top] "r" (*regB0top), [rA0top] "r" (*regA0top), [rB1top] "r" (*regB1top), [rA1top] "r" (*regA1top), [rB2top] "r" (*regB2top), [rA2top] "r" (*regA2top),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0bot] "r" (*regB0bot), [rA0bot] "r" (*regA0bot), [rB1bot] "r" (*regB1bot), [rA1bot] "r" (*regA1bot), [rB2bot] "r" (*regB2bot), [rA2bot] "r" (*regA2bot),
                           [b_localOffsetInBytes]    "r" (setup_info.b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex),
                           [zero]                    "c" (0x00000000),
                           "malu_matrix_c_size"(MATRIX_C_ON_MALUMEM_IN_KBYTES)
                        : 
                       );
#else
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0top].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0top].s1, 1;"          
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 0>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0top].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0top].s3, 1;"
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 4>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 8>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<12>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<16>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1top].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1top].s1, 1;" 
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<20>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x, %[zero],     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1top].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1top].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2top].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2top].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2top].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2top].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0mid].s1, 1;"          
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0mid].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0mid].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1mid].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1mid].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1mid].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2mid].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2mid].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2mid].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0bot].s1, 1;"          
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1bot].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2bot].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,(ul))
                        :  [rC] "=r" (*regC_ptr)
                        :  [rB0top] "r" (*regB0top), [rA0top] "r" (*regA0top), [rB1top] "r" (*regB1top), [rA1top] "r" (*regA1top), [rB2top] "r" (*regB2top), [rA2top] "r" (*regA2top),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0bot] "r" (*regB0bot), [rA0bot] "r" (*regA0bot), [rB1bot] "r" (*regB1bot), [rA1bot] "r" (*regA1bot), [rB2bot] "r" (*regB2bot), [rA2bot] "r" (*regA2bot),
                           [b_localOffsetInBytes]    "r" (setup_info.b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex),
                           [zero]                    "c" (0x00000000)
                        : 
                       );
#endif
}


//----------------------------------------------------------------------------//
// requantization + Blending + Relu
//----------------------------------------------------------------------------//

inline void requant_blend_N24(float32 *regC_ptr, float4 prev_frame, const float32 scaleConst, int32 biasConst, const float max, const float min, const float o_scale, const float o_scale_rcp) // fp16
{

   float16 prev_frame_fp32;
   
   __asm__ __volatile__(
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 0>,  (r)sr50.x, %[prev_frame].s!< 0>, 255;"                        
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 4>,  (r)sr50.x, %[prev_frame].s!< 1>, 255;"  
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 8>,  (r)sr50.x, %[prev_frame].s!< 2>, 255;"  
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!<12>,  (r)sr50.x, %[prev_frame].s!< 3>, 255;"                          
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  128;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,  255;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  255;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  255;"                        
                        "(rpt3)and.b             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  255;"                        
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,   %[scale];"
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.s32f32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.s32f32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[scaleConst].s!<0>,  (r)%[rC].s!< 0>,    (r)%[biasConst].s!<0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[scaleConst].s!<4>,  (r)%[rC].s!< 4>,    (r)%[biasConst].s!<4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[scaleConst].s!<8>,  (r)%[rC].s!< 8>,    (r)%[biasConst].s!<8>;"
                        "(rpt3)mad.f32           %[rC].s!<12>,  (r)%[scaleConst].s!<12>, (r)%[rC].s!<12>,    (r)%[biasConst].s!<12>;"
                        "(rpt3)mad.f32           %[rC].s!<16>,  (r)%[scaleConst].s!<16>, (r)%[rC].s!<16>,    (r)%[biasConst].s!<16>;"
                        "(rpt3)mad.f32           %[rC].s!<20>,  (r)%[scaleConst].s!<20>, (r)%[rC].s!<20>,    (r)%[biasConst].s!<20>;"
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)rndaz.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
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
                        "(rpt3)min.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[max];"
                        "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[unsigned_offset];"
                        "(rpt3)add.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[unsigned_offset];"
                        "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[scale];" // Can it be combined by w_scale?
                        "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[scale];"
                        "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[scale];"
                        "(rpt3)mul.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[scale];"
                        // "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[scale];"
                        "(rpt3)mul.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  (r)%[rC].s!<20>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  (r)%[rC].s!<20>;"
                        "(rpt3)add.f             %[rC].s!<20>,  %[one], (r)(neg)%[rC].s!<20>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[rC].s!< 0>,  (r)%[rC].s!<20>, (r)%[prev_frame_fp32].s!< 0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[rC].s!< 4>,  (r)%[rC].s!<20>, (r)%[prev_frame_fp32].s!< 4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[rC].s!< 8>,  (r)%[rC].s!<20>, (r)%[prev_frame_fp32].s!< 8>;"
                        "(rpt3)mad.f32           %[rC].s!<12>,  (r)%[rC].s!<12>,  (r)%[rC].s!<20>, (r)%[prev_frame_fp32].s!<12>;"
                        "(rpt3)mul.f             %[rC].s!< 0>,  %[scale_rcp],  (r)%[rC].s!< 0>;"
                        "(rpt3)mul.f             %[rC].s!< 4>,  %[scale_rcp],  (r)%[rC].s!< 4>;"
                        "(rpt3)mul.f             %[rC].s!< 8>,  %[scale_rcp],  (r)%[rC].s!< 8>;"
                        "(rpt3)mul.f             %[rC].s!<12>,  %[scale_rcp],  (r)%[rC].s!<12>;"      
                        "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32s32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        :  [rC] "+r" (*regC_ptr),
                           [prev_frame]     "+r" (prev_frame)
                        :  [prev_frame_fp32] "+r" (prev_frame_fp32),
                           [scaleConst]      "c" (scaleConst),
                           [biasConst]       "c" (biasConst),
                           [min]             "c" (min),
                           [max]             "c" (max),                           
                           [scale]           "c" (o_scale),
                           [scale_rcp]       "c" (o_scale_rcp),
                           [mask]            "c" ((uint)0x000000FF),
                           [unsigned_offset] "c" (0x43000000),
                           [one]             "c" (0x3F800000)
                        :  
                       );

}

//----------------------------------------------------------------------------//
// requantization + Blending + Relu
//----------------------------------------------------------------------------//

inline void requant_residual_blend_N20(float32 *regC_ptr, float4 prev_frame, const float32 scaleConst, int32 biasConst, const float max, const float min, const float o_scale, const float o_scale_rcp) // fp16
{

   float16 prev_frame_fp32;
   
   __asm__ __volatile__(
                        "(rpt3)mov.f32f32        %[rC].s!<24>,               (r)%[rC].s!<12>;"                                  // Moving learned features values to 24, making room for upsampled alpha channel 
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 0>,  (r)sr50.x, %[prev_frame].s!< 0>, 255;"                        
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 4>,  (r)sr50.x, %[prev_frame].s!< 1>, 255;"  
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!< 8>,  (r)sr50.x, %[prev_frame].s!< 2>, 255;"  
                        "(rpt3)shrm.b32          %[prev_frame_fp32].s!<12>,  (r)sr50.x, %[prev_frame].s!< 3>, 255;"                          
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  128;"
                        "(rpt3)add.s             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  128;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,  255;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  255;"
                        "(rpt3)and.b             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  255;"                        
                        "(rpt3)and.b             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  255;"                        
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>;"
                        "(rpt3)cov.s32f32        %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 0>,  (r)%[prev_frame_fp32].s!< 0>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,   %[scale];"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,   %[scale];"
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.s32f32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[scaleConst].s!<0>,  (r)%[rC].s!< 0>,    (r)%[biasConst].s!<0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[scaleConst].s!<4>,  (r)%[rC].s!< 4>,    (r)%[biasConst].s!<4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[scaleConst].s!<8>,  (r)%[rC].s!< 8>,    (r)%[biasConst].s!<8>;"
                        "(rpt3)mad.f32           %[rC].s!<24>,  (r)%[scaleConst].s!<12>, (r)%[rC].s!<24>,    (r)%[biasConst].s!<12>;"
                        "(rpt3)mad.f32           %[rC].s!<16>,  (r)%[scaleConst].s!<16>, (r)%[rC].s!<16>,    (r)%[biasConst].s!<16>;"                        
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[min];"
                        "(rpt3)min.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[max];"
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        // We used to do +128 / 255 to make it into 0-1 range
                        // "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[unsigned_offset];"
                        // "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[unsigned_offset];"
                        // "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[unsigned_offset];"
                        // "(rpt3)add.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[unsigned_offset];"
                        // "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[unsigned_offset];"
                        // "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[scale];"
                        // "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[scale];"
                        // "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[scale];"
                        // "(rpt3)mul.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[scale];"
                        // "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[scale];"
                        // Now we do + 0.5 / 127.5 to make it inot -1 1 range
                        "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[point5];"
                        "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[point5];"
                        "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[point5];"
                        "(rpt3)add.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[point5];"
                        "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[point5];"
                        "(rpt3)mul.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[onetwoseven_rcp];"
                        "(rpt3)mul.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[onetwoseven_rcp];"
                        "(rpt3)mul.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[onetwoseven_rcp];"
                        "(rpt3)mul.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[onetwoseven_rcp];"
                        "(rpt3)mul.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[onetwoseven_rcp];"
                        "(rpt3)max.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     0;"
                        "(rpt3)max.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     0;"
                        "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[prev_frame_fp32].s!< 0>;"  // Add residual for RRRR
                        "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[prev_frame_fp32].s!< 1>;"  // Add residual for GGGG
                        "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[prev_frame_fp32].s!< 2>;"  // Add residual for BBBB
                        "(rpt3)mov.f32f32        %[rC].s!<12>,                       %[prev_frame_fp32].s!< 3>;"  // Upsampled A channel to AAAA
                        // Blending
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 4>,  (r)%[prev_frame_fp32].s!< 4>,  (r)%[rC].s!<16>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!< 8>,  (r)%[prev_frame_fp32].s!< 8>,  (r)%[rC].s!<16>;"
                        "(rpt3)mul.f             %[prev_frame_fp32].s!<12>,  (r)%[prev_frame_fp32].s!<12>,  (r)%[rC].s!<16>;"
                        "(rpt3)add.f             %[rC].s!<16>,  %[one], (r)(neg)%[rC].s!<16>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[rC].s!< 0>,  (r)%[rC].s!<16>, (r)%[prev_frame_fp32].s!< 4>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[rC].s!< 4>,  (r)%[rC].s!<16>, (r)%[prev_frame_fp32].s!< 8>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[rC].s!< 8>,  (r)%[rC].s!<16>, (r)%[prev_frame_fp32].s!<12>;"
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,      0;"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,      0;"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,      0;"
                        "(rpt3)min.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>, %[one];"
                        "(rpt3)min.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>, %[one];"
                        "(rpt3)min.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>, %[one];"
                        "(rpt3)mul.f             %[rC].s!< 0>,  %[scale_rcp],  (r)%[rC].s!< 0>;"
                        "(rpt3)mul.f             %[rC].s!< 4>,  %[scale_rcp],  (r)%[rC].s!< 4>;"
                        "(rpt3)mul.f             %[rC].s!< 8>,  %[scale_rcp],  (r)%[rC].s!< 8>;"
                        "(rpt3)mul.f             %[rC].s!<12>,  %[scale_rcp],  (r)%[rC].s!<12>;"      
                        "(rpt3)mul.f             %[rC].s!<24>,  %[scale_rcp],  (r)%[rC].s!<24>;"  
                        // "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        // "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        // "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        // "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        // "(rpt3)rndaz.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"                        
                        "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32s32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        :  [rC] "+r" (*regC_ptr),
                           [prev_frame]     "+r" (prev_frame)
                        :  [prev_frame_fp32] "+r" (prev_frame_fp32),
                           [scaleConst]      "c" (scaleConst),
                           [biasConst]       "c" (biasConst),
                           [min]             "c" (min),
                           [max]             "c" (max),                           
                           [scale]           "c" (o_scale),
                           [scale_rcp]       "c" (o_scale_rcp),
                           [mask]            "c" ((uint)0x000000FF),
                           [unsigned_offset] "c" (0x43000000),
                           [point5]          "c" (0x3F000000),
                           [onetwoseven_rcp] "c" (0x3C008081),
                           [one]             "c" (0x3F800000)
                        :  
                       );

}


//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N16(float32 *regC_ptr)
{
   __asm__ __volatile__(
                        "(rpt3)shlm.b32 %[rC].s!<0>,  sr50.x,    (r)%[rC].s!<0>,  sr51.x;" // s0<<0, s1<<8, s2<<16, s3<<24
                        "(rpt3)shlm.b32 %[rC].s!<4>,  sr50.y,    (r)%[rC].s!<4>,  sr51.y;"
                        "(rpt3)shlm.b32 %[rC].s!<8>,  sr50.z,    (r)%[rC].s!<8>,  sr51.z;"
                        "(rpt3)shlm.b32 %[rC].s!<12>, sr50.w,    (r)%[rC].s!<12>, sr51.w;"
                        // "(rpt3)shlm.b32 %[rC].s!<24>, (r)sr50.x, (r)%[rC].s!<24>, (r)sr51.x;"
                        // Now writing Features in space mode
                        "(rpt1)shlm.b32 %[rC].s!<24>, (r)sr50.x, (r)%[rC].s!<24>, (r)sr51.x;"
                        "(rpt1)shlm.b32 %[rC].s!<26>, (r)sr50.x, (r)%[rC].s!<26>, (r)sr51.x;"
                        "or.b  %[rC].s!< 0>,  %[rC].s!< 4>,  %[rC].s!< 0>;"
                        "or.b  %[rC].s!< 1>,  %[rC].s!< 5>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!< 6>,  %[rC].s!< 2>;"
                        "or.b  %[rC].s!< 3>,  %[rC].s!< 7>,  %[rC].s!< 3>;"
                        // "or.b  %[rC].s!<24>,  %[rC].s!<25>,  %[rC].s!<24>;"
                        "or.b  %[rC].s!<24>,  %[rC].s!<25>,  %[rC].s!<24>;"
                        "or.b  %[rC].s!< 0>,  %[rC].s!< 8>,  %[rC].s!< 0>;"  // s0 readable, no need for nop
                        "or.b  %[rC].s!< 1>,  %[rC].s!< 9>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!<10>,  %[rC].s!< 2>;"
                        "or.b  %[rC].s!< 3>,  %[rC].s!<11>,  %[rC].s!< 3>;"
                        // "or.b  %[rC].s!<24>,  %[rC].s!<26>,  %[rC].s!<24>;"
                        "or.b  %[rC].s!<26>,  %[rC].s!<27>,  %[rC].s!<26>;"
                        "or.b  %[rC].s!< 0>,  %[rC].s!<12>,  %[rC].s!< 0>;"
                        "or.b  %[rC].s!< 1>,  %[rC].s!<13>,  %[rC].s!< 1>;"
                        "or.b  %[rC].s!< 2>,  %[rC].s!<14>,  %[rC].s!< 2>;"
                        "or.b  %[rC].s!< 3>,  %[rC].s!<15>,  %[rC].s!< 3>;"
                        // "or.b  %[rC].s!<24>,  %[rC].s!<27>,  %[rC].s!<24>;"
                        // this is a test for debugging
                        // "(rpt3)mov.f32f32        %[rC].s!<24>,               (r)%[rC].s!<0>;" 
                        :  [rC] "+r" (*regC_ptr)
                        :
                        :  
                       );

}


inline void mm_store_c_N16_D2S(float32 *regC_ptr, const int scale_factor, uint global_id_m)
{
    uint sub_block_id_h = global_id_m / OUT_WIDTH;
    uint sub_block_id_w = global_id_m % OUT_WIDTH;
    uint d2s_col = get_global_id(2) * TILESIZE_N / (scale_factor*scale_factor);

    uint c_coordInWords00 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords01 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords10 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords11 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords00 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w)) * 2   / (scale_factor * scale_factor) + d2s_col;
   //  uint f_coordInHalfWords01 =  ((scale_factor * sub_block_id_h)     * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords10 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w)) * 2    / (scale_factor * scale_factor) + d2s_col;
   //  uint f_coordInHalfWords11 =  ((scale_factor * sub_block_id_h + 1) * (OUT_WIDTH*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
   //  uint c_coordInWordsD  =  ((sub_block_id_h)     * (OUT_WIDTH) + (sub_block_id_w));
    
   __asm__ __volatile__(
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords00], 4;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<1>,  %[c_coordInWords01], 4;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<2>,  %[c_coordInWords10], 4;"
                        "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<3>,  %[c_coordInWords11], 4;"
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<24>.lo, %[f_coordInHalfWords00],  5;"  
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<26>.lo, %[f_coordInHalfWords10],  5;"  
                        "(ss)nop;"                                            
                        :  [rC]               "+r" (*regC_ptr)
                        :  [c_coordInWords00]  "r" (c_coordInWords00),
                           [c_coordInWords01]  "r" (c_coordInWords01),
                           [c_coordInWords10]  "r" (c_coordInWords10),
                           [c_coordInWords11]  "r" (c_coordInWords11),
                           [f_coordInHalfWords00]   "r" (f_coordInHalfWords00),
                           [f_coordInHalfWords10]   "r" (f_coordInHalfWords10)
                        :
                       );
}

__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
__kernel void Conv_MxM_Baseline(__read_only    image1d_buffer_t matrix_A, // [IHxIW, IC]
                                __read_only    image1d_buffer_t matrix_W, // [OC, FHxFWxIC]
                                __read_only    image1d_buffer_t features_history_hr,  // Previous frame
                                __global const float* restrict  matrix_B, // Bias
                                __global const float* restrict  matrix_S, // Scale
                                         const float            min,   
                                         const float            max,   
                                         const float            o_scale,   
                                         const float            o_scale_rcp,  
                                         const int              d2s_factor,
                                __write_only   image1d_buffer_t candidate_up, // [OHxOW, OC]
                                __write_only   image1d_buffer_t features,     // [OHxOW, OC]
                                         const int              jitter_index,
                                         const int              total_blocks,
                                         const int              input_height,
                                         const int              prevframe_offset, // used for tiling 
                                         const int              input_offset,     // used for tiling 
                                         const int              output_offset,    // used for tiling 
                                         const int              layer_index,                                         
                                __global       int*             sync_control,
                                         const int              apron_top
)
{
// #define TOTAL_BLOCKS (OUT_WIDTH*OUT_HEIGHT/TILESIZE_M)

    __private uint4 regBtop[3], regBmid[3], regBbot[3];
    __private uint8 regAtop[3], regAmid[3], regAbot[3];
    __private float32 regC;
    __private float4 prev_frame;
    uint subMatrixCStartInElements;

    size_t block_id_n = get_global_id(2);
    uint  col = block_id_n * TILESIZE_N;

    MM_SETUP(matrix_W)
    mm_setup_info setup_info = mm_setup();
    
    const uint strideWinElements = LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_OPERAND;
    const uint strideCinElements = LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_RESULT;
    uint subMatrixWStartInElements = ((OC*IC*FILTER_HEIGHT*FILTER_WIDTH*jitter_index)) + col * strideWinElements; // B is Kfirst
    const uint stridePrevFrameinElements = LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES/BYTES_PER_RESULT;
    mm_load_b_N32K32_INT8_3x3(&regBtop[0], &regBtop[1], &regBtop[2], subMatrixWStartInElements+0*IC, subMatrixWStartInElements+1*IC, subMatrixWStartInElements+2*IC,
                              &regBmid[0], &regBmid[1], &regBmid[2], subMatrixWStartInElements+3*IC, subMatrixWStartInElements+4*IC, subMatrixWStartInElements+5*IC,
                              &regBbot[0], &regBbot[1], &regBbot[2], subMatrixWStartInElements+6*IC, subMatrixWStartInElements+7*IC, subMatrixWStartInElements+8*IC, setup_info);
    const float32 scaleConst = (float32)(0);
    const int32   biasConst  = (int32)(0);
    mm_load_scale_FP32(matrix_S + (OC*jitter_index), scaleConst);
    mm_load_bias_INT32(matrix_B + (OC*jitter_index), biasConst);    
    if ((get_global_linear_id() == 0)) atomic_and(sync_control, 0);
   //  if ((get_global_id(0) == 0) && (get_global_id(1) == 0) && (get_global_id(2) == 0)) atomic_and(sync_control, 0);
    uint  block_id_m;
    if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+layer_index);
    block_id_m = sub_group_shuffle(block_id_m, 0);    

    //------------------------------------------------------------------------//
    // MxM 
    //------------------------------------------------------------------------//
    #pragma unroll 1
    while (block_id_m < total_blocks)
    {
        uint row = block_id_m * TILESIZE_M + TILE_PREV_FRAME_OFFSET;
        uint gidx_m = get_local_id(0) + TILESIZE_M * block_id_m;
        uint out_col_id = gidx_m % OUT_WIDTH;
        uint out_row_id = gidx_m / OUT_WIDTH;
        int input_col_id1 = STRIDE * out_col_id;
        int input_col_id0 = input_col_id1 - DILATION;
        int input_col_id2 = input_col_id1 + DILATION;

         
        int input_row_id = STRIDE * out_row_id + TILE_INPUT_OFFSET + ((apron_top > 0) ? 1 : 0);
        uint matAcoord0mid = (input_row_id * INPUT_WIDTH + input_col_id0) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord1mid = (input_row_id * INPUT_WIDTH + input_col_id1) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord2mid = (input_row_id * INPUT_WIDTH + input_col_id2) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord0top = matAcoord0mid - DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord1top = matAcoord1mid - DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord2top = matAcoord2mid - DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND); 
        uint matAcoord0bot = matAcoord0mid + DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord1bot = matAcoord1mid + DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
        uint matAcoord2bot = matAcoord2mid + DILATION*INPUT_WIDTH*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND); 

        uint zfill0mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill1mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill2mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill0top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill1top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill2top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill0bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill1bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;
        uint zfill2bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= INPUT_WIDTH)) ? FULL_K_MASK : 0;     

        uint subMatrixPrevFrameStartInElements = row * stridePrevFrameinElements + col; // Hardcoded for now. starting slice index 12 / 4 = 3   

        BARSYNC
        prev_frame = mm_load_prev_frame(subMatrixPrevFrameStartInElements);
        mm_load_a_im2col_M64K32_INT8_3x3_zfill(&regAtop[0], &regAtop[1], &regAtop[2], matAcoord0top, matAcoord1top, matAcoord2top, zfill0top, zfill1top, zfill2top,
                                               &regAmid[0], &regAmid[1], &regAmid[2], matAcoord0mid, matAcoord1mid, matAcoord2mid, zfill0mid, zfill1mid, zfill2mid,
                                               &regAbot[0], &regAbot[1], &regAbot[2], matAcoord0bot, matAcoord1bot, matAcoord2bot, zfill0bot, zfill1bot, zfill2bot);

        if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+layer_index);
        BARSET
        __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

        mm_mad_N24K32_INT8_3x3(&regC, &regAtop[0], &regAtop[1], &regAtop[2], &regBtop[0], &regBtop[1], &regBtop[2],
                                      &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                      &regAbot[0], &regAbot[1], &regAbot[2], &regBbot[0], &regBbot[1], &regBbot[2], setup_info);

        
        requant_residual_blend_N20(&regC, prev_frame, scaleConst, biasConst, max, min, o_scale, o_scale_rcp);
        mm_prepare_store_c_N16(&regC);
        subMatrixCStartInElements = row * strideCinElements + col;
        mm_store_c_N16_D2S(&regC, d2s_factor, gidx_m + TILE_OUT_OFFSET);
        block_id_m = sub_group_shuffle(block_id_m, 0);
        BARSYNC
    }
}
