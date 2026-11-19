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
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*3)
#define SPML_MODE_FLAG
#define LM_FIRST_256B  /* 0   *3 */  0
#define LM_SECOND_256B /* 256 *3 */  768
#define LM_THIRD_256B  /* 512 *3 */  1536
#define LM_FOURTH_256B /* 768 *3 */  2304
#define LM_FIRST_512B  /* 0   *3 */  0
#define LM_SECOND_512B /* 512 *3 */  1536
#else
#define MATRIX_B_ON_CHIP_IN_BYTES (1024*2)
#define SPML_MODE_FLAG
#define LM_FIRST_256B  /* 0   *2 */  0
#define LM_SECOND_256B /* 256 *2 */  512
#define LM_THIRD_256B  /* 512 *2 */  1024
#define LM_FOURTH_256B /* 768 *2 */  1536
#define LM_FIRST_512B  /* 0   *2 */  0
#define LM_SECOND_512B /* 512 *2 */  1024
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


#define MATRIX_C_ON_MALUMEM_IN_KBYTES (16)
#define REG_C_DATATYPE float32

//----------------------------------------------------------------------------//
// ALU section
//----------------------------------------------------------------------------//
#if (IC == 20)
#define ALU_32_CYCLES(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"
#define ALU_32_CYCLES_NO_ACC(rC, rA0, rA1, rA2, rA3, rA4, rA5, rA6, rA7, b0, b1, b2, b3, b4, b5, b6, b7, zero, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", "#zero";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b4", (signed)"#rA4", (r)(last)"#rC";"
#elif (IC == 24)
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
       write_imagef(matrix_C, 0, regC.xyzw);\
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
                        "movs.u32u32 sr49.x, 0xFF00FF00;" \
                        "movs.u32u32 sr49.y, 0x00FF00FF;" \
                        : \
                        : "lm_size"(MATRIX_B_ON_CHIP_IN_BYTES) \
                        : "sr50.x", "sr50.y", "sr50.z", "sr50.w", "sr51.x", "sr51.y", "sr51.z", "sr51.w", "sr49.x", "sr49.y" \
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
#ifndef THREE_NSP
    short addressMSBs = (setup_info.b_localOffsetInBytes>>6);
    short newSideSelect = qcom_get_physical_sub_group_id()%2; // one bit mask for nSP ID    
    setup_info.b_localOffsetInBytes = ((addressMSBs*2+newSideSelect)<<6) | (addressInOneSide); // times 2 for 2 nSP
#else
    short addressMSBs = (setup_info.b_localOffsetInBytes>>6);
    short newSideSelect = qcom_get_physical_sub_group_id()%3; // wave id % 3 = nSP id for Honu
    setup_info.b_localOffsetInBytes = ((addressMSBs*3+newSideSelect)<<6) | (addressInOneSide); // times 3 for 3 nSP
#endif    

    addressInOneSide = ((setup_info.b_localOffsetInBytes_scale_bias>>0)&0x3F);
#ifndef THREE_NSP
    addressMSBs = (setup_info.b_localOffsetInBytes_scale_bias>>6);
    newSideSelect = qcom_get_physical_sub_group_id()%2; // one bit mask for nSP ID    
    setup_info.b_localOffsetInBytes_scale_bias = ((addressMSBs*2+newSideSelect)<<6) | (addressInOneSide); // times 2 for 2 nSP
#else
    addressMSBs = (setup_info.b_localOffsetInBytes_scale_bias>>6);
    newSideSelect = qcom_get_physical_sub_group_id()%3; // wave id % 3 = nSP id for Honu
    setup_info.b_localOffsetInBytes_scale_bias = ((addressMSBs*3+newSideSelect)<<6) | (addressInOneSide); // times 3 for 3 nSP
#endif   
    return setup_info;
}


//----------------------------------------------------------------------------//
// load a 
//----------------------------------------------------------------------------//

// #if defined(USE_MALU) && (IC==28)
// #define compmask "xyzO"
// #define rptNum "(rpt2)"
// #elif defined(USE_MALU) && (IC==24)
// #define compmask "xyOO"
// #define rptNum "(rpt1)"
// #elif defined(USE_MALU) && (IC==20)
// #define compmask "xOOO"
// #define rptNum ""
// #else
#define compmask "xyzw"
// #define rptNum "(rpt3)"
// #endif


inline void mm_load_a_im2col_M64K32_INT8_3x3_zfill(uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint matAcoord0top, uint matAcoord1top, uint matAcoord2top, uint zfill0top, uint zfill1top, uint zfill2top,
                                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint matAcoord0mid, uint matAcoord1mid, uint matAcoord2mid, uint zfill0mid, uint zfill1mid, uint zfill2mid,
                                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint matAcoord0bot, uint matAcoord1bot, uint matAcoord2bot, uint zfill0bot, uint zfill1bot, uint zfill2bot, uint full_wave)
{
    __asm__ __volatile__(
                        "cmps.u.gt p0.y, %[full_wave], 0;"
                        "lock;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0top].s0,  %[texCoord0top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0top].s4,  %[texCoord0top], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1top].s0,  %[texCoord1top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1top].s4,  %[texCoord1top], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2top].s0,  %[texCoord2top], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2top].s4,  %[texCoord2top], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0mid].s4,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1mid].s0,  %[texCoord1mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1mid].s4,  %[texCoord1mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2mid].s0,  %[texCoord2mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2mid].s4,  %[texCoord2mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0bot].s0,  %[texCoord0bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0bot].s4,  %[texCoord0bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1bot].s0,  %[texCoord1bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1bot].s4,  %[texCoord1bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2bot].s0,  %[texCoord2bot], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2bot].s4,  %[texCoord2bot], 16;" 
                        "unlock;"
                        "(ss)nop;"
                        "bar.set;"
                        "(sy)nop;"
                        "(rpt2)mov.u32u32 %[rA0top].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA1top].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA2top].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA0mid].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA1mid].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA2mid].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA0bot].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA1bot].s5, 0;"
                        "(rpt2)mov.u32u32 %[rA2bot].s5, 0;" 
                        "br !p0.y, #55;"
                        "cmps.u.gt p0, %[zfill0top], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA0top].s0, %[zero];"
                        "mov.u32u32 %[rA0top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1top], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA1top].s0, %[zero];"
                        "mov.u32u32 %[rA1top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2top], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA2top].s0, %[zero];"
                        "mov.u32u32 %[rA2top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0mid], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA0mid].s0, %[zero];"
                        "mov.u32u32 %[rA0mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1mid], 7; (ss)(rpt5)nop;" 
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA1mid].s0, %[zero];"
                        "mov.u32u32 %[rA1mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2mid], 7; (ss)(rpt5)nop;" 
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA2mid].s0, %[zero];"
                        "mov.u32u32 %[rA2mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0bot], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA0bot].s0, %[zero];"
                        "mov.u32u32 %[rA0bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1bot], 7; (ss)(rpt5)nop;"
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA1bot].s0, %[zero];"
                        "mov.u32u32 %[rA1bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2bot], 7; (ss)(rpt5)nop;" 
                        "predt;"
                        "(rpt3)    mov.u32u32 %[rA2bot].s0, %[zero];"
                        "mov.u32u32 %[rA2bot].s4, %[zero];"
                        "prede;"
                        "(jp)nop;"
                        :  [rA0top] "=r" (*regA0top), [rA1top] "=r" (*regA1top), [rA2top] "=r" (*regA2top),
                           [rA0mid] "=r" (*regA0mid), [rA1mid] "=r" (*regA1mid), [rA2mid] "=r" (*regA2mid),
                           [rA0bot] "=r" (*regA0bot), [rA1bot] "=r" (*regA1bot), [rA2bot] "=r" (*regA2bot)  
                        :  [texCoord0top] "r" (matAcoord0top / (4/BYTES_PER_OPERAND)), [texCoord1top] "r" (matAcoord1top / (4/BYTES_PER_OPERAND)), [texCoord2top] "r" (matAcoord2top / (4/BYTES_PER_OPERAND)),
                           [texCoord0mid] "r" (matAcoord0mid / (4/BYTES_PER_OPERAND)), [texCoord1mid] "r" (matAcoord1mid / (4/BYTES_PER_OPERAND)), [texCoord2mid] "r" (matAcoord2mid / (4/BYTES_PER_OPERAND)),
                           [texCoord0bot] "r" (matAcoord0bot / (4/BYTES_PER_OPERAND)), [texCoord1bot] "r" (matAcoord1bot / (4/BYTES_PER_OPERAND)), [texCoord2bot] "r" (matAcoord2bot / (4/BYTES_PER_OPERAND)),
                           [zfill0top] "r" (zfill0top), [zfill1top] "r" (zfill1top), [zfill2top] "r" (zfill2top),
                           [zfill0mid] "r" (zfill0mid), [zfill1mid] "r" (zfill1mid), [zfill2mid] "r" (zfill2mid),
                           [zfill0bot] "r" (zfill0bot), [zfill1bot] "r" (zfill1bot), [zfill2bot] "r" (zfill2bot),
                           [full_wave] "r" (full_wave),
#ifdef ZERO_PADDING                           
                           [zero]      "c" (0x00000000)
#else
                           [zero]      "c" (0x80808080)
#endif                                                      
                        :  "p0.x", "p0.y"
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
                        "sr50.x", "sr50.y", "sr50.z", "sr50.w", "sr51.x", "sr51.y", "sr51.z", "sr51.w", "sr49.x", "sr49.y"
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
                        "sr50.x", "sr50.y", "sr50.z", "sr50.w", "sr51.x", "sr51.y", "sr51.z", "sr51.w", "sr49.x", "sr49.y"
                       );
}


//----------------------------------------------------------------------------//
// mad
//----------------------------------------------------------------------------//
inline void mm_mad_N32K32_INT8_3x3(float32 *regC_ptr,
                                   uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint4 *regB0top, uint4 *regB1top, uint4 *regB2top,
                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid,
                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint4 *regB0bot, uint4 *regB1bot, uint4 *regB2bot,
                                   mm_setup_info setup_info)
{
    short  linearIndex = get_local_id(0);
#ifdef USE_MALU
    __asm__ __volatile__(
                       "ALLOC_MALUM;"
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
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
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<24>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<28>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x, %[zero],     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2bot].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
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
// requantization + Relu
//----------------------------------------------------------------------------//
inline void requant_N32(float32 *regC_ptr, const float32 scaleConst, int32 biasConst, const float max, const float min)
{

#ifndef INV_POW2      
   __asm__ __volatile__(
   
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.s32f32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.s32f32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)cov.s32f32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.s32f32        %[rC].s!<28>,  (r)%[rC].s!<28>;"
			               "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[scaleConst].s!<0>,  (r)%[rC].s!< 0>,  (r)%[biasConst].s!<0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[scaleConst].s!<4>,  (r)%[rC].s!< 4>,  (r)%[biasConst].s!<4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[scaleConst].s!<8>,  (r)%[rC].s!< 8>,  (r)%[biasConst].s!<8>;"
                        "(rpt3)mad.f32           %[rC].s!<12>,  (r)%[scaleConst].s!<12>, (r)%[rC].s!<12>,  (r)%[biasConst].s!<12>;"
                        "(rpt3)mad.f32           %[rC].s!<16>,  (r)%[scaleConst].s!<16>, (r)%[rC].s!<16>,  (r)%[biasConst].s!<16>;"
                        "(rpt3)mad.f32           %[rC].s!<20>,  (r)%[scaleConst].s!<20>, (r)%[rC].s!<20>,  (r)%[biasConst].s!<20>;"
                        "(rpt3)mad.f32           %[rC].s!<24>,  (r)%[scaleConst].s!<24>, (r)%[rC].s!<24>,  (r)%[biasConst].s!<24>;"
                        "(rpt3)mad.f32           %[rC].s!<28>,  (r)%[scaleConst].s!<28>, (r)%[rC].s!<28>,  (r)%[biasConst].s!<28>;"                                                
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;" ///< ANF SDK CHANGE - LINES 764-771 Uncommented due to quality reduction without these rounds
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)rndaz.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)rndaz.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)rndaz.f           %[rC].s!<28>,  (r)%[rC].s!<28>;"
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
                        :  [rC] "+r" (*regC_ptr)
                        :  [scaleConst]  "c" (scaleConst),
                           [biasConst]   "c" (biasConst),
                           [min]         "c" (min),
                           [max]         "c" (max),
                           [point_five]  "c" (0x3F000000)
                        :
                       );


#else
   short max_short = 0x007F;
   short min_short = 0xFF80;
   __asm__ __volatile__(                        
                        "(rpt3)mad.s24           %[rC].s!< 0>,  (r)%[scaleConst].s!< 0>, (r)%[rC].s!< 0>,  (r)%[biasConst].s!< 0>;"
                        "(rpt3)mad.s24           %[rC].s!< 4>,  (r)%[scaleConst].s!< 4>, (r)%[rC].s!< 4>,  (r)%[biasConst].s!< 4>;"
                        "(rpt3)mad.s24           %[rC].s!< 8>,  (r)%[scaleConst].s!< 8>, (r)%[rC].s!< 8>,  (r)%[biasConst].s!< 8>;"
                        "(rpt3)mad.s24           %[rC].s!<12>,  (r)%[scaleConst].s!<12>, (r)%[rC].s!<12>,  (r)%[biasConst].s!<12>;"
                        "(rpt3)mad.s24           %[rC].s!<16>,  (r)%[scaleConst].s!<16>, (r)%[rC].s!<16>,  (r)%[biasConst].s!<16>;" 
                        "(rpt3)mad.s24           %[rC].s!<20>,  (r)%[scaleConst].s!<20>, (r)%[rC].s!<20>,  (r)%[biasConst].s!<20>;"
                        "(rpt3)mad.s24           %[rC].s!<24>,  (r)%[scaleConst].s!<24>, (r)%[rC].s!<24>,  (r)%[biasConst].s!<24>;"
                        "(rpt3)mad.s24           %[rC].s!<28>,  (r)%[scaleConst].s!<28>, (r)%[rC].s!<28>,  (r)%[biasConst].s!<28>;"
                        "      max.s             %[rC].s!< 0>.lo,           %[rC].s!< 0>.hi,     %[min];"
                        "      max.s             %[rC].s!< 0>.hi,           %[rC].s!< 2>.hi,     %[min];"
                        "      max.s             %[rC].s!< 1>.lo,           %[rC].s!< 4>.hi,     %[min];"
                        "      max.s             %[rC].s!< 8>.lo,           %[rC].s!< 1>.hi,     %[min];"
                        "      max.s             %[rC].s!< 1>.hi,           %[rC].s!< 6>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!< 0>.lo,  (r)(last)%[rC].s!< 0>.lo,     %[max];"
                        "      min.s             %[rC].s!< 8>.lo,     (last)%[rC].s!< 8>.lo,     %[max];"
                        "      max.s             %[rC].s!< 2>.lo,           %[rC].s!< 8>.hi,     %[min];"
                        "      max.s             %[rC].s!< 2>.hi,           %[rC].s!<10>.hi,     %[min];"
                        "      max.s             %[rC].s!< 3>.lo,           %[rC].s!<12>.hi,     %[min];"
                        "      max.s             %[rC].s!< 8>.hi,           %[rC].s!< 3>.hi,     %[min];"
                        "      max.s             %[rC].s!< 3>.hi,           %[rC].s!<14>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!< 2>.lo,  (r)(last)%[rC].s!< 2>.lo,     %[max];"
                        "      min.s             %[rC].s!< 8>.hi,     (last)%[rC].s!< 8>.hi,     %[max];"
                        "      max.s             %[rC].s!< 4>.lo,           %[rC].s!<16>.hi,     %[min];"
                        "      max.s             %[rC].s!< 4>.hi,           %[rC].s!<18>.hi,     %[min];"
                        "      max.s             %[rC].s!< 5>.lo,           %[rC].s!<20>.hi,     %[min];"
                        "      max.s             %[rC].s!< 9>.lo,           %[rC].s!< 5>.hi,     %[min];"
                        "      max.s             %[rC].s!< 5>.hi,           %[rC].s!<22>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!< 4>.lo,  (r)(last)%[rC].s!< 4>.lo,     %[max];"
                        "      min.s             %[rC].s!< 9>.lo,     (last)%[rC].s!< 9>.lo,     %[max];"
                        "      max.s             %[rC].s!< 6>.lo,           %[rC].s!<24>.hi,     %[min];"
                        "      max.s             %[rC].s!< 6>.hi,           %[rC].s!<26>.hi,     %[min];"
                        "      max.s             %[rC].s!< 7>.lo,           %[rC].s!<28>.hi,     %[min];"
                        "      max.s             %[rC].s!<10>.lo,           %[rC].s!< 9>.hi,     %[min];"
                        "      max.s             %[rC].s!< 9>.hi,           %[rC].s!< 7>.hi,     %[min];"
                        "      max.s             %[rC].s!< 7>.hi,           %[rC].s!<30>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!< 6>.lo,  (r)(last)%[rC].s!< 6>.lo,     %[max];"
                        "      min.s             %[rC].s!< 9>.hi,     (last)%[rC].s!< 9>.hi,     %[max];"
                        "      max.s             %[rC].s!<10>.hi,           %[rC].s!<11>.hi,     %[min];"
                        "      max.s             %[rC].s!<11>.lo,           %[rC].s!<13>.hi,     %[min];"
                        "      max.s             %[rC].s!<11>.hi,           %[rC].s!<15>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!<10>.lo,  (r)(last)%[rC].s!<10>.lo,     %[max];"
                        "      max.s             %[rC].s!<12>.lo,           %[rC].s!<17>.hi,     %[min];"
                        "      max.s             %[rC].s!<12>.hi,           %[rC].s!<19>.hi,     %[min];"
                        "      max.s             %[rC].s!<13>.lo,           %[rC].s!<21>.hi,     %[min];"
                        "      max.s             %[rC].s!<13>.hi,           %[rC].s!<23>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!<12>.lo,  (r)(last)%[rC].s!<12>.lo,     %[max];"
                        "      max.s             %[rC].s!<14>.lo,           %[rC].s!<25>.hi,     %[min];"
                        "      max.s             %[rC].s!<14>.hi,           %[rC].s!<27>.hi,     %[min];"
                        "      max.s             %[rC].s!<15>.lo,           %[rC].s!<29>.hi,     %[min];"
                        "      max.s             %[rC].s!<15>.hi,           %[rC].s!<31>.hi,     %[min];"
                        "(rpt3)min.s             %[rC].s!<14>.lo,  (r)(last)%[rC].s!<14>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!< 0>.lo,        (r)%[rC].s!< 0>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!< 2>.lo,        (r)%[rC].s!< 2>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!< 4>.lo,        (r)%[rC].s!< 4>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!< 6>.lo,        (r)%[rC].s!< 6>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!< 8>.lo,        (r)%[rC].s!< 8>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!<10>.lo,        (r)%[rC].s!<10>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!<12>.lo,        (r)%[rC].s!<12>.lo,     %[max];"
                        "(rpt3)min.s             %[rC].s!<14>.lo,        (r)%[rC].s!<14>.lo,     %[max];"         
                        :  [rC] "+r" (*regC_ptr)
                        :  [scaleConst]  "c" (scaleConst),
                           [biasConst]   "c" (biasConst),
                           [min]         "c" (min_short),
                           [max]         "c" (max_short)
                        :
                       );

                        
#endif    
}

//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N32(float32 *regC_ptr)
{
   
   size_t sub_block_id_m = get_local_id(0);
   short fiber_odd = (sub_block_id_m%2 == 1);
#ifndef INV_POW2
   uint4 temp;
   __asm__ __volatile__(
                        "cmps.u.ne p0.y, %[finber_odd], 0;"
                        "(rpt3)shlm.b32 %[rC].s!<0>,  (r)sr50.x, (r)%[rC].s!<0>,  (r)sr51.x;" // s0<<0, s1<<8, s2<<16, s3<<24
                        "(rpt3)shlm.b32 %[rC].s!<4>,  (r)sr50.x, (r)%[rC].s!<4>,  (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<8>,  (r)sr50.x, (r)%[rC].s!<8>,  (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<12>, (r)sr50.x, (r)%[rC].s!<12>, (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<16>, (r)sr50.x, (r)%[rC].s!<16>, (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<20>, (r)sr50.x, (r)%[rC].s!<20>, (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<24>, (r)sr50.x, (r)%[rC].s!<24>, (r)sr51.x;"
                        "(rpt3)shlm.b32 %[rC].s!<28>, (r)sr50.x, (r)%[rC].s!<28>, (r)sr51.x;"
                  "(rpt1)or.b  %[rC].s!<0>,  (r)%[rC].s!<0>,  (r)%[rC].s!<2>;"
                  "(rpt1)or.b  %[rC].s!<4>,  (r)%[rC].s!<4>,  (r)%[rC].s!<6>;"
                  "(rpt1)or.b  %[rC].s!<8>,  (r)%[rC].s!<8>,  (r)%[rC].s!<10>;"
                  "(rpt1)or.b  %[rC].s!<12>, (r)%[rC].s!<12>, (r)%[rC].s!<14>;"
                  "(rpt1)or.b  %[rC].s!<16>, (r)%[rC].s!<16>, (r)%[rC].s!<18>;"
                  "(rpt1)or.b  %[rC].s!<20>, (r)%[rC].s!<20>, (r)%[rC].s!<22>;"
                  "(rpt1)or.b  %[rC].s!<24>, (r)%[rC].s!<24>, (r)%[rC].s!<26>;"
                  "(rpt1)or.b  %[rC].s!<28>, (r)%[rC].s!<28>, (r)%[rC].s!<30>;"
                        "or.b  %[rC].s!<0>,     %[rC].s!<0>,     %[rC].s!<1>;"
                        "or.b  %[rC].s!<1>,     %[rC].s!<4>,     %[rC].s!<5>;"
                        "or.b  %[rC].s!<2>,     %[rC].s!<8>,     %[rC].s!<9>;"
                        "or.b  %[rC].s!<3>,     %[rC].s!<12>,    %[rC].s!<13>;"
                        "or.b  %[rC].s!<4>,     %[rC].s!<16>,    %[rC].s!<17>;"
                        "or.b  %[rC].s!<5>,     %[rC].s!<20>,    %[rC].s!<21>;"
                        "or.b  %[rC].s!<6>,     %[rC].s!<24>,    %[rC].s!<25>;"
                        "or.b  %[rC].s!<7>,     %[rC].s!<28>,    %[rC].s!<29>;"
                        "br !p0.y, #3;"
                        "(rpt3)mov.u32u32  %[rT].x, (r)%[rC].s!<4>;"
                        "(rpt3)mov.u32u32  %[rC].s!<4>, (r)%[rC].s!<0>;"
                        "(jp)shfl.xor.w4.u32 %[rC].s!<4>,  %[rC].s!<4>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<5>,  %[rC].s!<5>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<6>,  %[rC].s!<6>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<7>,  %[rC].s!<7>, 1;"
                        "(ss)br !p0.y, #3;"
                        "(rpt3)mov.u32u32  %[rC].s!<0>, (r)%[rC].s!<4>;"
                        "(rpt3)mov.u32u32  %[rC].s!<4>, (r)%[rT].x;"
                        "(jp)nop;"
                        :  [rC] "+r" (*regC_ptr)
                        :  [rT]  "r" (temp),
                           [finber_odd]   "r" (fiber_odd)
                        :  "p0.y"
                       );
#else
   uint8 temp;
   __asm__ __volatile__(
                        "cmps.u.ne p0, %[finber_odd], 0;"
                        "(rpt3)shlm.b32 %[rC].s!< 8>,     sr50.y,     (r)%[rC].s!< 8>,           sr49.x;"
                        "(rpt3)andg.b32 %[rT].s!< 0>, (r)%[rC].s!< 0>,    sr49.y,      (r)(last)%[rC].s!< 8>;"
                        "(rpt3)shlm.b32 %[rC].s!<12>,     sr50.y,     (r)%[rC].s!<12>,           sr49.x;"
                        "(rpt3)andg.b32 %[rT].s!< 4>, (r)%[rC].s!< 4>,    sr49.y,      (r)(last)%[rC].s!<12>;"                        
                        "predt;"
                        "(rpt3)mov.u32u32  %[rC].s!<4>, (r)%[rT].s!<0>;"
                        "predf;"
                        "(rpt3)mov.u32u32  %[rC].s!<4>, (r)%[rT].s!<4>;"
                        "prede;"
                        "shfl.xor.w4.u32 %[rC].s!<4>,  %[rC].s!<4>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<5>,  %[rC].s!<5>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<6>,  %[rC].s!<6>, 1;"
                        "shfl.xor.w4.u32 %[rC].s!<7>,  %[rC].s!<7>, 1;"
                        "(ss)nop;"
                        "predt;"
                        "(rpt3)mov.u32u32  %[rC].s!<0>, (r)%[rC].s!<4>;"
                        "(rpt3)mov.u32u32  %[rC].s!<4>, (r)%[rT].s!<4>;"                        
                        "predf;"
                        "(rpt3)mov.u32u32  %[rC].s!<0>, (r)%[rT].s!<0>;"
                        "prede;"                        
                        :  [rC] "+r" (*regC_ptr)
                        :  [rT]  "r" (temp),
                           [finber_odd]   "r" (fiber_odd)
                        :  "p0.x"
                       );
#endif                       
}

inline void mm_store_c_N32(float32 *regC_ptr, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);   
   __asm__ __volatile__(
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"                        
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset8   %[rC].s!<4>,  %[c_coordInWords], 3;"                        
                        "(ss)nop;"
                        :  [rC]              "+r" (*regC_ptr)
                        :  [c_coordInWords]  "r" (subMatrixCStartInElements)
                        :
                       );
}


__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
__kernel void Conv_MxM_Baseline(__read_only    image1d_buffer_t matrix_A, // [IHxIW, IC]
                                __read_only    image1d_buffer_t matrix_W, // [OC, FHxFWxIC]
                                __global const float* restrict  matrix_B, // Bias
                                __global const float* restrict  matrix_S, // Scale
                                __read_write   image1d_buffer_t matrix_C, // [OHxOW, OC]
                                __global       int*             sync_control,
                                         const float            min,   
                                         const float            max,   
                                         const int              total_blocks,
                                         const int              input_width,
                                         const int              input_height,
                                         const int              input_offset, // used for tiling
                                         const int              output_offset,// used for tiling
                                         const int              sync_buffer_start_index,         
                                         const int              next_layer_sync_buffer_size, // Need to clear the sync buffer for the next layer
                                         const int              apron_top, // used for tiling                                               
                                         const int              jitter_index
)
{

    __private uint4 regBtop[3], regBmid[3], regBbot[3];
    __private uint8 regAtop[3], regAmid[3], regAbot[3];
    __private float32 regC;
    uint subMatrixCStartInElements;

#ifdef DETERMINISTIC
    uint num_CCHE = 3; 
    uint CCHE_id = get_group_id(1) / (get_num_groups(1) / num_CCHE); // CCHE 0, 1, 2    
#else
    uint num_CCHE = 1; 
    uint CCHE_id = 0; // CCHE 0, 1, 2    
#endif

    uint total_blocks_per_CCHE = (total_blocks+num_CCHE-1) / num_CCHE;
    uint block_offset = total_blocks_per_CCHE * CCHE_id;
    

    size_t block_id_n = get_global_id(2);
    uint  col = block_id_n * TILESIZE_N;

    size_t total_block_n = get_global_size(2);

    MM_SETUP(matrix_W)
    mm_setup_info setup_info = mm_setup();
    
    const uint strideWinElements = LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_OPERAND;
    const uint strideCinElements = LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_RESULT;
    uint subMatrixWStartInElements = ((OC*IC*FILTER_HEIGHT*FILTER_WIDTH*jitter_index)) + col * strideWinElements; // B is Kfirst
    mm_load_b_N32K32_INT8_3x3(&regBtop[0], &regBtop[1], &regBtop[2], subMatrixWStartInElements+0*IC, subMatrixWStartInElements+1*IC, subMatrixWStartInElements+2*IC,
                              &regBmid[0], &regBmid[1], &regBmid[2], subMatrixWStartInElements+3*IC, subMatrixWStartInElements+4*IC, subMatrixWStartInElements+5*IC,
                              &regBbot[0], &regBbot[1], &regBbot[2], subMatrixWStartInElements+6*IC, subMatrixWStartInElements+7*IC, subMatrixWStartInElements+8*IC, setup_info);
    const float32 scaleConst = (float32)(0);
    const int32   biasConst  = (int32)(0);
    mm_load_scale_FP32(matrix_S + (OC*jitter_index) + col, scaleConst);
    mm_load_bias_INT32(matrix_B + (OC*jitter_index) + col, biasConst);

    // Reset sync_contro buffer for the next leyaer
    if ((get_global_linear_id() == 0)) {
      for (int i = 0; i < next_layer_sync_buffer_size; i++)
         atomic_and(sync_control+sync_buffer_start_index+total_block_n*num_CCHE+i, 0);
    }

     
   // if ((get_global_id(0) == 0) && (get_global_id(1) == 0) && (get_global_id(2) == 0)) atomic_and(sync_control+layer_index+1, 0); // this is not working for some compiler issue. FILE JIRA

    uint  block_id_m;
   // const int my_layer_index = sync_buffer_start_index + block_id_n;
   const int my_layer_index = sync_buffer_start_index + block_id_n*num_CCHE + CCHE_id;
   if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
   block_id_m = sub_group_shuffle(block_id_m, 0);


   int output_width = input_width / STRIDE;
    //------------------------------------------------------------------------//
    // MxM 
    //------------------------------------------------------------------------//
    #pragma unroll 1
   //  while (block_id_m < total_blocks)
   while (block_id_m < total_blocks_per_CCHE)
   {
      // block_id_m += block_offset;
      // if(block_id_m >= total_blocks) return;
      // uint block_id_m_x = block_id_m % 5;
      // uint block_id_m_y = block_id_m / 5;
      // uint block_id_m_deterministic = block_id_m_y * 15 + block_id_m_x + CCHE_id*5;
      
      // uint row = block_id_m * TILESIZE_M + TILE_OUTPUT_OFFSET;
      uint gidx_m = get_local_id(0) + TILESIZE_M * block_id_m;
      uint out_col_id = gidx_m % (output_width/num_CCHE) + CCHE_id*(output_width/num_CCHE);
      uint out_row_id = gidx_m / (output_width/num_CCHE);

      uint row = (out_row_id*output_width + (out_col_id/2)*2 + TILE_OUTPUT_OFFSET) *LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD + (out_col_id%2)*4 + col;

      // ((sub_block_id_m/2) * 2*LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD + (sub_block_id_m%2)*4));
      

      int input_col_id1 = STRIDE * out_col_id;
      int input_col_id0 = input_col_id1 - DILATION;
      int input_col_id2 = input_col_id1 + DILATION;

         
      int input_row_id = STRIDE * out_row_id + TILE_INPUT_OFFSET + ((apron_top > 0) ? 1 : 0);
      uint matAcoord0mid = (input_row_id * input_width + input_col_id0) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord1mid = (input_row_id * input_width + input_col_id1) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord2mid = (input_row_id * input_width + input_col_id2) * (LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord0top = matAcoord0mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord1top = matAcoord1mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord2top = matAcoord2mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND); 
      uint matAcoord0bot = matAcoord0mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord1bot = matAcoord1mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND);
      uint matAcoord2bot = matAcoord2mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES/BYTES_PER_OPERAND); 

      uint full_wave = 0;

      uint zfill0mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;

      full_wave += zfill0mid;
      full_wave += zfill1mid;
      full_wave += zfill2mid;
      full_wave += zfill0top;
      full_wave += zfill1top;
      full_wave += zfill2top;
      full_wave += zfill0bot;
      full_wave += zfill1bot;
      full_wave += zfill2bot;

      BARSYNC
      mm_load_a_im2col_M64K32_INT8_3x3_zfill(&regAtop[0], &regAtop[1], &regAtop[2], matAcoord0top, matAcoord1top, matAcoord2top, zfill0top, zfill1top, zfill2top,
                                             &regAmid[0], &regAmid[1], &regAmid[2], matAcoord0mid, matAcoord1mid, matAcoord2mid, zfill0mid, zfill1mid, zfill2mid,
                                             &regAbot[0], &regAbot[1], &regAbot[2], matAcoord0bot, matAcoord1bot, matAcoord2bot, zfill0bot, zfill1bot, zfill2bot, full_wave);

      if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
      // BARSET
      // __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

      mm_mad_N32K32_INT8_3x3(&regC, &regAtop[0], &regAtop[1], &regAtop[2], &regBtop[0], &regBtop[1], &regBtop[2],
                                    &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regAbot[0], &regAbot[1], &regAbot[2], &regBbot[0], &regBbot[1], &regBbot[2], setup_info);

      requant_N32(&regC, scaleConst, biasConst, max, min);
      mm_prepare_store_c_N32(&regC);
      // subMatrixCStartInElements = row * strideCinElements + col;
      mm_store_c_N32(&regC, row);
      block_id_m = sub_group_shuffle(block_id_m, 0);
      BARSYNC
    }
}
