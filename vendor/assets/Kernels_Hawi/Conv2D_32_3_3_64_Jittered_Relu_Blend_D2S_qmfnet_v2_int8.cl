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
#define TILE_PREV_FRAME_OFFSET prevframe_offset
#define TILE_INPUT_OFFSET input_offset
#define TILE_OUTPUT_OFFSET output_offset
#else
#define TILE_PREV_FRAME_OFFSET 0 
#define TILE_INPUT_OFFSET 0 
#define TILE_OUTPUT_OFFSET 0 
#endif

#define MATRIX_C_ON_MALUMEM_IN_KBYTES (16)
#define REG_C_DATATYPE float32

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
       write_imagef(candidate_up, 0, regC.xyzw);\
       write_imagef(features, 0, regC.xyzw);\
       write_imagef(tonemaped_candidate_up, 0, regC.xyzw);\
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
#elif defined(USE_MALU) && (IC==24)          
                        "(rpt1)mov.u32u32 %[rA0top].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA1top].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA2top].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA0mid].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA1mid].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA2mid].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA0bot].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA1bot].s6, 0;"
                        "(rpt1)mov.u32u32 %[rA2bot].s6, 0;"             
#endif
                        "cmps.u.gt p0, %[zfill0top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0top].s0,  %[texCoord0top], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0top].s4,  %[texCoord0top], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0top].s0,  %[texCoord0top], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0top].s4,  %[texCoord0top], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1top].s0,  %[texCoord1top], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1top].s4,  %[texCoord1top], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1top].s0,  %[texCoord1top], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1top].s4,  %[texCoord1top], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2top], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2top].s0,  %[texCoord2top], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2top].s4,  %[texCoord2top], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2top].s0,  %[texCoord2top], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2top].s4,  %[texCoord2top], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA2top].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2top].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0mid].s4,  %[texCoord0mid], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0mid].s4,  %[texCoord0mid], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1mid].s0,  %[texCoord1mid], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1mid].s4,  %[texCoord1mid], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1mid].s0,  %[texCoord1mid], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1mid].s4,  %[texCoord1mid], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2mid], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2mid].s0,  %[texCoord2mid], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2mid].s4,  %[texCoord2mid], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2mid].s0,  %[texCoord2mid], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2mid].s4,  %[texCoord2mid], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA2mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0bot].s0,  %[texCoord0bot], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0bot].s4,  %[texCoord0bot], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0bot].s0,  %[texCoord0bot], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0bot].s4,  %[texCoord0bot], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA0bot].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1bot].s0,  %[texCoord1bot], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1bot].s4,  %[texCoord1bot], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1bot].s0,  %[texCoord1bot], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA1bot].s4,  %[texCoord1bot], 32;" 
#endif
                        "predt #3;"
                        "(rpt3)    mov.u32u32 %[rA1bot].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1bot].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2bot], 7; (ss)(rpt5)nop;"
                        "predf #3;"
#ifdef E17
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2bot].s0,  %[texCoord2bot], 33;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2bot].s4,  %[texCoord2bot], 33;" 
#else
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2bot].s0,  %[texCoord2bot], 32;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA2bot].s4,  %[texCoord2bot], 32;" 
#endif
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
inline void mm_load_b_N16K32_INT8_3x3(uint2 *regB0top, uint2 *regB1top, uint2 *regB2top, uint matrixBcoord0top, uint matrixBcoord1top, uint matrixBcoord2top,
                                      uint2 *regB0mid, uint2 *regB1mid, uint2 *regB2mid, uint matrixBcoord0mid, uint matrixBcoord1mid, uint matrixBcoord2mid,
                                      uint2 *regB0bot, uint2 *regB1bot, uint2 *regB2bot, uint matrixBcoord0bot, uint matrixBcoord1bot, uint matrixBcoord2bot,
                                      mm_setup_info setup_info)
{
#ifndef USE_MALU
    size_t sub_block_id_m = get_local_id(0);
    __asm__ __volatile__(
                        "nop;"
                        "add.u %[rB0top].s0, %[matrixBcoord0top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1top].s0, %[matrixBcoord1top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2top].s0, %[matrixBcoord2top].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0mid].s0, %[matrixBcoord0mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1mid].s0, %[matrixBcoord1mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2mid].s0, %[matrixBcoord2mid].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB0bot].s0, %[matrixBcoord0bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0top].s0, %[rB0top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1top].s0, %[rB1top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2top].s0, %[rB2top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0mid].s0, %[rB0mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1mid].s0, %[rB1mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2mid].s0, %[rB2mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0bot].s0, %[rB0bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 3;"
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
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00)
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
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s0, %[rB0top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s1, %[rB0top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s0, %[rB1top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s1, %[rB1top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s0, %[rB2top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s1, %[rB2top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s0, %[rB0mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s1, %[rB0mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s0, %[rB1mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s1, %[rB1mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s0, %[rB2mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s1, %[rB2mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s0, %[rB0bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s1, %[rB0bot].s1, 3;"
#ifndef USE_ALU
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s1, %[rB1bot].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s0, %[rB2bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s1, %[rB2bot].s1, 3;"
#else
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "(rpt2)nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 3;"
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
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00)
                        :
                       );
#endif
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
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0top].s0, %[rB0top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0top].s2, %[rB0top].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1top].s0, %[rB1top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1top].s2, %[rB1top].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2top].s0, %[rB2top].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2top].s2, %[rB2top].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0mid].s0, %[rB0mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0mid].s2, %[rB0mid].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1mid].s0, %[rB1mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1mid].s2, %[rB1mid].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2mid].s0, %[rB2mid].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2mid].s2, %[rB2mid].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0bot].s0, %[rB0bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB0bot].s2, %[rB0bot].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 3;"
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
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s0, %[rB0top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s1, %[rB0top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s2, %[rB0top].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0top].s3, %[rB0top].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s0, %[rB1top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s1, %[rB1top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s2, %[rB1top].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1top].s3, %[rB1top].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s0, %[rB2top].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s1, %[rB2top].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s2, %[rB2top].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2top].s3, %[rB2top].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s0, %[rB0mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s1, %[rB0mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s2, %[rB0mid].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0mid].s3, %[rB0mid].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s0, %[rB1mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s1, %[rB1mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s2, %[rB1mid].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1mid].s3, %[rB1mid].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s0, %[rB2mid].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s1, %[rB2mid].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s2, %[rB2mid].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2mid].s3, %[rB2mid].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s0, %[rB0bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s1, %[rB0bot].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s2, %[rB0bot].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0bot].s3, %[rB0bot].s3, 3;"
#ifndef USE_ALU
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s1, %[rB1bot].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s2, %[rB1bot].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB1bot].s3, %[rB1bot].s3, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s0, %[rB2bot].s0, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s1, %[rB2bot].s1, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s2, %[rB2bot].s2, 3;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB2bot].s3, %[rB2bot].s3, 3;"
#else
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1bot].s2, %[matrixBcoord1bot].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s2, %[matrixBcoord2bot].x, %[b_globalOffsetInWords16];"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 3;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 3;"
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
// mad
//----------------------------------------------------------------------------//
inline void mm_mad_N16K32_INT8_3x3(float16 *regC_ptr,
                                   uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint2 *regB0top, uint2 *regB1top, uint2 *regB2top,
                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint2 *regB0mid, uint2 *regB1mid, uint2 *regB2mid,
                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint2 *regB0bot, uint2 *regB1bot, uint2 *regB2bot,
                                   mm_setup_info setup_info)
{
    short  linearIndex = get_local_id(0);
#ifdef USE_MALU
    __asm__ __volatile__(
                       "ALLOC_MALUM;"
                       "malu_wmma.m64n16.s8s32       m[0], (linear)(rmajor)%[rA0top].s0, (folded)(cmajor)%[rB0top].s0;"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA1top].s0, (folded)(cmajor)%[rB1top].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA2top].s0, (folded)(cmajor)%[rB2top].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA0mid].s0, (folded)(cmajor)%[rB0mid].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA1mid].s0, (folded)(cmajor)%[rB1mid].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA2mid].s0, (folded)(cmajor)%[rB2mid].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA0bot].s0, (folded)(cmajor)%[rB0bot].s0, m[0];"
#ifndef USE_ALU
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA1bot].s0, (folded)(cmajor)%[rB1bot].s0, m[0];"
                       "malu_wmma.accu.m64n16.s8s32  m[0], (linear)(rmajor)%[rA2bot].s0, (folded)(cmajor)%[rB2bot].s0, m[0];"
                       "ldm.m64n16.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "DEALLOC_MALUM;"
                       "(ss)nop;"
#else
                       "ldm.m64n16.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1bot].s1, 1;" 
                       "DEALLOC_MALUM;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2bot].s1, 1;"    
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
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
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1top].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1top].s1, 1;"
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 4>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 8>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<12>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, %[rA0top].s4, %[rA0top].s5, %[rA0top].s6, %[rA0top].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x, %[zero],     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2top].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2top].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, %[rA1top].s4, %[rA1top].s5, %[rA1top].s6, %[rA1top].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )               
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0mid].s1, 1;"       
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, %[rA2top].s4, %[rA2top].s5, %[rA2top].s6, %[rA2top].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1mid].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, %[rA0mid].s4, %[rA0mid].s5, %[rA0mid].s6, %[rA0mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2mid].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, %[rA1mid].s4, %[rA1mid].s5, %[rA1mid].s6, %[rA1mid].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0bot].s1, 1;"          
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1bot].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, %[rA0bot].s4, %[rA0bot].s5, %[rA0bot].s6, %[rA0bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2bot].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2bot].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, %[rA1bot].s4, %[rA1bot].s5, %[rA1bot].s6, %[rA1bot].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )                       
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )                       
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, %[rA2bot].s4, %[rA2bot].s5, %[rA2bot].s6, %[rA2bot].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,(ul))                       
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
inline void requant_N16(float16 *regC_ptr, const float16 scaleConst, int16 biasConst, const float max, const float min)
{
   __asm__ __volatile__(
                        "(rpt3)cov.s32f32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.s32f32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.s32f32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
			               "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[scaleConst].s!<0>,  (r)%[rC].s!< 0>,  (r)%[biasConst].s!<0>;"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[scaleConst].s!<4>,  (r)%[rC].s!< 4>,  (r)%[biasConst].s!<4>;"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[scaleConst].s!<8>,  (r)%[rC].s!< 8>,  (r)%[biasConst].s!<8>;"
                        "(rpt3)mad.f32           %[rC].s!<12>,  (r)%[scaleConst].s!<12>, (r)%[rC].s!<12>,  (r)%[biasConst].s!<12>;"                      
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[min];"
                        "(rpt3)min.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[max];"
                        "(rpt3)min.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[max];"
                        "(rpt3)min.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[max];"
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        // "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        // "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[point_five];"
                        // "(rpt3)floor.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        // "(rpt3)floor.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        // "(rpt3)floor.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        // "(rpt3)floor.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[rC].s!< 0>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[rC].s!< 4>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[oneTwoEight];"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[rC].s!< 8>, %[reciprocal_255], %[oneTwoEightby255];"                        
                        "(rpt3)cov.f32f16        %[rC].s!< 8>.lo,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        :  [rC] "+r" (*regC_ptr)
                        :  [scaleConst]  "c" (scaleConst),
                           [biasConst]   "c" (biasConst),
                           [min]         "c" (min),
                           [max]         "c" (max),
                           [point_five]  "c" (0x3F000000),
                           [reciprocal_255] "r" (0x3B808080),
                           [oneTwoEightby255] "c" (0x3F008080),
                           [oneTwoEight] "c" (0x43000000)
                        :
                       );

}


//----------------------------------------------------------------------------//
// requantization + Relu
//----------------------------------------------------------------------------//
inline void requant_N32(float32 *regC_ptr, const float32 scaleConst, int32 biasConst, const float max, const float min)
{
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
                        "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)rndaz.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)rndaz.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)rndaz.f           %[rC].s!<28>,  (r)%[rC].s!<28>;"
                        // "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[point_five];"
                        // "(rpt3)add.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     %[point_five];"
                        // "(rpt3)floor.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        // "(rpt3)floor.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        // "(rpt3)floor.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        // "(rpt3)floor.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        // "(rpt3)floor.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        // "(rpt3)floor.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        // "(rpt3)floor.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        // "(rpt3)floor.f           %[rC].s!<28>,  (r)%[rC].s!<28>;"

                        "(rpt3)mad.f32           %[rC].s!< 0>,  (r)%[rC].s!< 0>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)mad.f32           %[rC].s!<16>,  (r)%[rC].s!<16>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)mad.f32           %[rC].s!< 4>,  (r)%[rC].s!< 4>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)mad.f32           %[rC].s!<20>,  (r)%[rC].s!<20>, %[reciprocal_255], %[oneTwoEightby255];"
                        "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[oneTwoEight];"
                        "(rpt3)add.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     %[oneTwoEight];"
                        "(rpt3)mad.f32           %[rC].s!< 8>,  (r)%[rC].s!< 8>, %[reciprocal_255], %[oneTwoEightby255];"                        
                        "(rpt3)mad.f32           %[rC].s!<24>,  (r)%[rC].s!<24>, %[reciprocal_255], %[oneTwoEightby255];"                        
                        "(rpt3)cov.f32f16        %[rC].s!< 8>.lo,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32f16        %[rC].s!<24>.lo,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32s32        %[rC].s!<28>,  (r)%[rC].s!<28>;"

                        // "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        // "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        // "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        // "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        // "(rpt3)cov.f32s32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        // "(rpt3)cov.f32s32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        // "(rpt3)cov.f32s32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        // "(rpt3)cov.f32s32        %[rC].s!<28>,  (r)%[rC].s!<28>;"
                        :  [rC] "+r" (*regC_ptr)
                        :  [scaleConst]  "c" (scaleConst),
                           [biasConst]   "c" (biasConst),
                           [min]         "c" (min),
                           [max]         "c" (max),
                           [point_five]  "c" (0x3F000000),
                           [reciprocal_255] "r" (0x3B808080),
                           [oneTwoEightby255] "c" (0x3F008080),
                           [oneTwoEight] "c" (0x43000000)
                        :
                       );

}


//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N32( float32 *regC_ptr, half16 *tonemaped_color_0, half16 *tonemaped_color_1, 
                                    half4 *sample_0, half4 *sample_1, half4 *sample_2, half4 *sample_3, half4 *sample_4, half4 *sample_5, half4 *sample_6, half4 *sample_7,
                                    half4 *pref_f_0, half4 *pref_f_1, half4 *pref_f_2, half4 *pref_f_3, half4 *pref_f_4, half4 *pref_f_5, half4 *pref_f_6, half4 *pref_f_7, 
                                    const half max, const half min) 
{
   half one = 1.0;
   __private half4    max_color_val_0; 
   __private half4    max_color_val_1; 
   __asm__ __volatile__(
                        "(rpt1)shlm.b32 %[rC].s!<12>, (r)sr50.x, (r)%[rC].s!<12>, (r)sr51.x;"
                        "(rpt1)shlm.b32 %[rC].s!<14>, (r)sr50.x, (r)%[rC].s!<14>, (r)sr51.x;"
                        "(rpt1)shlm.b32 %[rC].s!<28>, (r)sr50.x, (r)%[rC].s!<28>, (r)sr51.x;"
                        "(rpt1)shlm.b32 %[rC].s!<30>, (r)sr50.x, (r)%[rC].s!<30>, (r)sr51.x;"
                        "(rpt3)mul.f             %[rf00].s!< 0>,  (r)%[rf00].s!< 0>,  %[rC].s!< 8>.lo;"
                        "(rpt3)mul.f             %[rf01].s!< 0>,  (r)%[rf01].s!< 0>,  %[rC].s!< 8>.hi;"
                        "(rpt3)mul.f             %[rf10].s!< 0>,  (r)%[rf10].s!< 0>,  %[rC].s!< 9>.lo;"
                        "(rpt3)mul.f             %[rf11].s!< 0>,  (r)%[rf11].s!< 0>,  %[rC].s!< 9>.hi;"
                        "(rpt3)mul.f             %[rf02].s!< 0>,  (r)%[rf02].s!< 0>,  %[rC].s!<24>.lo;"
                        "(rpt3)mul.f             %[rf03].s!< 0>,  (r)%[rf03].s!< 0>,  %[rC].s!<24>.hi;"
                        "(rpt3)mul.f             %[rf12].s!< 0>,  (r)%[rf12].s!< 0>,  %[rC].s!<25>.lo;"
                        "(rpt3)mul.f             %[rf13].s!< 0>,  (r)%[rf13].s!< 0>,  %[rC].s!<25>.hi;"                                                
                        "(rpt3)add.f             %[rC].s!< 8>.lo,  %[one], (r)(neg)%[rC].s!< 8>.lo;"                        
                        "(rpt3)add.f             %[rC].s!<24>.lo,  %[one], (r)(neg)%[rC].s!<24>.lo;"                        
                        "or.b  %[rC].s!<12>,  %[rC].s!<13>,  %[rC].s!<12>;"
                        "or.b  %[rC].s!<14>,  %[rC].s!<15>,  %[rC].s!<14>;"
                        "or.b  %[rC].s!<28>,  %[rC].s!<29>,  %[rC].s!<28>;"
                        "or.b  %[rC].s!<30>,  %[rC].s!<31>,  %[rC].s!<30>;"
                        "(rpt3)mad.f16           %[tonemaped_color_0].s!< 0>,  (r)%[rS00].s!< 0>,  %[rC].s!< 8>.lo, (r)%[rf00].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_0].s!< 4>,  (r)%[rS01].s!< 0>,  %[rC].s!< 8>.hi, (r)%[rf01].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_0].s!< 8>,  (r)%[rS10].s!< 0>,  %[rC].s!< 9>.lo, (r)%[rf10].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_0].s!<12>,  (r)%[rS11].s!< 0>,  %[rC].s!< 9>.hi, (r)%[rf11].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_1].s!< 0>,  (r)%[rS02].s!< 0>,  %[rC].s!<24>.lo, (r)%[rf02].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_1].s!< 4>,  (r)%[rS03].s!< 0>,  %[rC].s!<24>.hi, (r)%[rf03].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_1].s!< 8>,  (r)%[rS12].s!< 0>,  %[rC].s!<25>.lo, (r)%[rf12].s!< 0>;"
                        "(rpt3)mad.f16           %[tonemaped_color_1].s!<12>,  (r)%[rS13].s!< 0>,  %[rC].s!<25>.hi, (r)%[rf13].s!< 0>;"
                        // inverse tonemap
                        "(rpt3)max.f             %[rS00].s!< 0>,  (r)%[tonemaped_color_0].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rS01].s!< 0>,  (r)%[tonemaped_color_0].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rS10].s!< 0>,  (r)%[tonemaped_color_0].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rS11].s!< 0>,  (r)%[tonemaped_color_0].s!<12>,     %[min];"
                        "(rpt3)max.f             %[rS02].s!< 0>,  (r)%[tonemaped_color_1].s!< 0>,     %[min];"
                        "(rpt3)max.f             %[rS03].s!< 0>,  (r)%[tonemaped_color_1].s!< 4>,     %[min];"
                        "(rpt3)max.f             %[rS12].s!< 0>,  (r)%[tonemaped_color_1].s!< 8>,     %[min];"
                        "(rpt3)max.f             %[rS13].s!< 0>,  (r)%[tonemaped_color_1].s!<12>,     %[min];"
                        "(rpt3)min.f             %[rS00].s!< 0>,  (r)%[rS00].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS01].s!< 0>,  (r)%[rS01].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS10].s!< 0>,  (r)%[rS10].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS11].s!< 0>,  (r)%[rS11].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS02].s!< 0>,  (r)%[rS02].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS03].s!< 0>,  (r)%[rS03].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS12].s!< 0>,  (r)%[rS12].s!< 0>,     %[max];"
                        "(rpt3)min.f             %[rS13].s!< 0>,  (r)%[rS13].s!< 0>,     %[max];"
                        
                        "cmps.u.gt hr1.x, %[rS00].x, %[rS00].y;"   
                        "cmps.u.gt hr1.y, %[rS01].x, %[rS01].y;"   
                        "cmps.u.gt hr1.z, %[rS10].x, %[rS10].y;"   
                        "cmps.u.gt hr1.w, %[rS11].x, %[rS11].y;"   
                        "cmps.u.gt hr2.x, %[rS02].x, %[rS02].y;"   
                        "cmps.u.gt hr2.y, %[rS03].x, %[rS03].y;"   
                        "cmps.u.gt hr2.z, %[rS12].x, %[rS12].y;"   
                        "cmps.u.gt hr2.w, %[rS13].x, %[rS13].y;"   
                        "sel.b16 %[max_color_0].x, %[rS00].x, hr1.x, %[rS00].y;"
                        "sel.b16 %[max_color_0].y, %[rS01].x, hr1.y, %[rS01].y;"
                        "sel.b16 %[max_color_0].z, %[rS10].x, hr1.z, %[rS10].y;"
                        "sel.b16 %[max_color_0].w, %[rS11].x, hr1.w, %[rS11].y;"
                        "sel.b16 %[max_color_1].x, %[rS02].x, hr2.x, %[rS02].y;"
                        "sel.b16 %[max_color_1].y, %[rS03].x, hr2.y, %[rS03].y;"
                        "sel.b16 %[max_color_1].z, %[rS12].x, hr2.z, %[rS12].y;"
                        "sel.b16 %[max_color_1].w, %[rS13].x, hr2.w, %[rS13].y;"
                        "cmps.u.gt hr1.x, %[max_color_0].x, %[rS00].z;"
                        "cmps.u.gt hr1.y, %[max_color_0].y, %[rS01].z;"
                        "cmps.u.gt hr1.z, %[max_color_0].z, %[rS10].z;"
                        "cmps.u.gt hr1.w, %[max_color_0].w, %[rS11].z;"
                        "cmps.u.gt hr2.x, %[max_color_1].x, %[rS02].z;"
                        "cmps.u.gt hr2.y, %[max_color_1].y, %[rS03].z;"
                        "cmps.u.gt hr2.z, %[max_color_1].z, %[rS12].z;"
                        "cmps.u.gt hr2.w, %[max_color_1].w, %[rS13].z;"
                        "sel.b16 %[max_color_0].x, %[max_color_0].x, hr1.x, %[rS00].z;"
                        "sel.b16 %[max_color_0].y, %[max_color_0].y, hr1.y, %[rS01].z;"
                        "sel.b16 %[max_color_0].z, %[max_color_0].z, hr1.z, %[rS10].z;"
                        "sel.b16 %[max_color_0].w, %[max_color_0].w, hr1.w, %[rS11].z;"
                        "sel.b16 %[max_color_1].x, %[max_color_1].x, hr2.x, %[rS02].z;"
                        "sel.b16 %[max_color_1].y, %[max_color_1].y, hr2.y, %[rS03].z;"
                        "sel.b16 %[max_color_1].z, %[max_color_1].z, hr2.z, %[rS12].z;"
                        "sel.b16 %[max_color_1].w, %[max_color_1].w, hr2.w, %[rS13].z;"
                        "cmps.u.gt hr1.x, %[max_color_0].x, %[rS00].w;"
                        "cmps.u.gt hr1.y, %[max_color_0].y, %[rS01].w;"
                        "cmps.u.gt hr1.z, %[max_color_0].z, %[rS10].w;"
                        "cmps.u.gt hr1.w, %[max_color_0].w, %[rS11].w;"
                        "cmps.u.gt hr2.x, %[max_color_1].x, %[rS02].w;"
                        "cmps.u.gt hr2.y, %[max_color_1].y, %[rS03].w;"
                        "cmps.u.gt hr2.z, %[max_color_1].z, %[rS12].w;"
                        "cmps.u.gt hr2.w, %[max_color_1].w, %[rS13].w;"
                        "sel.b16 %[max_color_0].x, %[max_color_0].x, hr1.x, %[rS00].w;"
                        "sel.b16 %[max_color_0].y, %[max_color_0].y, hr1.y, %[rS01].w;"
                        "sel.b16 %[max_color_0].z, %[max_color_0].z, hr1.z, %[rS10].w;"
                        "sel.b16 %[max_color_0].w, %[max_color_0].w, hr1.w, %[rS11].w;"
                        "sel.b16 %[max_color_1].x, %[max_color_1].x, hr2.x, %[rS02].w;"
                        "sel.b16 %[max_color_1].y, %[max_color_1].y, hr2.y, %[rS03].w;"
                        "sel.b16 %[max_color_1].z, %[max_color_1].z, hr2.z, %[rS12].w;"
                        "sel.b16 %[max_color_1].w, %[max_color_1].w, hr2.w, %[rS13].w;"
                        "add.f %[max_color_0].x, (neg)%[max_color_0].x, %[one];"
                        "add.f %[max_color_0].y, (neg)%[max_color_0].y, %[one];"
                        "add.f %[max_color_0].z, (neg)%[max_color_0].z, %[one];"
                        "add.f %[max_color_0].w, (neg)%[max_color_0].w, %[one];"
                        "add.f %[max_color_1].x, (neg)%[max_color_1].x, %[one];"
                        "add.f %[max_color_1].y, (neg)%[max_color_1].y, %[one];"
                        "add.f %[max_color_1].z, (neg)%[max_color_1].z, %[one];"
                        "add.f %[max_color_1].w, (neg)%[max_color_1].w, %[one];"
                        "(rpt2)nop;"
                        "rcp     %[max_color_0].x, %[max_color_0].x;"
                        "rcp     %[max_color_0].y, %[max_color_0].y;"
                        "rcp     %[max_color_0].z, %[max_color_0].z;"
                        "rcp     %[max_color_0].w, %[max_color_0].w;"
                        "rcp     %[max_color_1].x, %[max_color_1].x;"
                        "rcp     %[max_color_1].y, %[max_color_1].y;"
                        "rcp     %[max_color_1].z, %[max_color_1].z;"
                        "rcp     %[max_color_1].w, %[max_color_1].w;"                        
                        "(ss)(rpt3)mul.f   %[rS00].x, (r)%[rS00].x, %[max_color_0].x;" 
                        "    (rpt3)mul.f   %[rS01].x, (r)%[rS01].x, %[max_color_0].y;" 
                        "    (rpt3)mul.f   %[rS10].x, (r)%[rS10].x, %[max_color_0].z;" 
                        "    (rpt3)mul.f   %[rS11].x, (r)%[rS11].x, %[max_color_0].w;"
                        "    (rpt3)mul.f   %[rS02].x, (r)%[rS02].x, %[max_color_1].x;" 
                        "    (rpt3)mul.f   %[rS03].x, (r)%[rS03].x, %[max_color_1].y;" 
                        "    (rpt3)mul.f   %[rS12].x, (r)%[rS12].x, %[max_color_1].z;" 
                        "    (rpt3)mul.f   %[rS13].x, (r)%[rS13].x, %[max_color_1].w;"
                        // Gama correction
                        "    (rpt3)mul.f   %[rS00].x, (r)%[rS00].x, (r)%[rS00].x;" 
                        "    (rpt3)mul.f   %[rS01].x, (r)%[rS01].x, (r)%[rS01].x;" 
                        "    (rpt3)mul.f   %[rS10].x, (r)%[rS10].x, (r)%[rS10].x;" 
                        "    (rpt3)mul.f   %[rS11].x, (r)%[rS11].x, (r)%[rS11].x;"
                        "    (rpt3)mul.f   %[rS02].x, (r)%[rS02].x, (r)%[rS02].x;" 
                        "    (rpt3)mul.f   %[rS03].x, (r)%[rS03].x, (r)%[rS03].x;" 
                        "    (rpt3)mul.f   %[rS12].x, (r)%[rS12].x, (r)%[rS12].x;" 
                        "    (rpt3)mul.f   %[rS13].x, (r)%[rS13].x, (r)%[rS13].x;"
                        


                        :  [rC] "+r" (*regC_ptr),
                           [tonemaped_color_0] "+r" (*tonemaped_color_0),
                           [tonemaped_color_1] "+r" (*tonemaped_color_1),
                           [rS00] "+r" (*sample_0),
                           [rS01] "+r" (*sample_1),
                           [rS10] "+r" (*sample_2),
                           [rS11] "+r" (*sample_3),
                           [rS02] "+r" (*sample_4),
                           [rS03] "+r" (*sample_5),
                           [rS12] "+r" (*sample_6),
                           [rS13] "+r" (*sample_7),
                           [rf00] "+r" (*pref_f_0),
                           [rf01] "+r" (*pref_f_1),
                           [rf10] "+r" (*pref_f_2),
                           [rf11] "+r" (*pref_f_3),
                           [rf02] "+r" (*pref_f_4),
                           [rf03] "+r" (*pref_f_5),
                           [rf12] "+r" (*pref_f_6),
                           [rf13] "+r" (*pref_f_7),
                           [max_color_0] "+r" (max_color_val_0),
                           [max_color_1] "+r" (max_color_val_1)
                        :  [one]             "r" (one),
                           [min]         "c" (min),
                           [max]         "c" (max)
                        :  "hr1.x", "hr1.y", "hr1.z", "hr1.w",
                           "hr2.x", "hr2.y", "hr2.z", "hr2.w"
                       );

}

//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N20(float32 *regC_ptr)
{
   uint   mask0 = get_local_id(0) & 0xFC;
   uint   mask1 = mask0 + 1;
   uint   mask2 = mask0 + 2;
   uint   mask3 = mask0 + 3;
   uint2   tmp0, tmp1, tmp2;
   uint   id_mod_4 = (get_local_id(0) % 4);

   __asm__ __volatile__(                                               
                        "(rpt3)shlm.b32 %[rC].s!<0>,  sr50.x, (r)%[rC].s!<0>,  sr51.x;" // s0<<0, s1<<8, s2<<16, s3<<24
                        "(rpt3)shlm.b32 %[rC].s!<4>,  sr50.y, (r)%[rC].s!<4>,  sr51.y;"
                        "(rpt3)shlm.b32 %[rC].s!<8>,  sr50.z, (r)%[rC].s!<8>,  sr51.z;"
                        "(rpt3)shlm.b32 %[rC].s!<12>, sr50.w, (r)%[rC].s!<12>, sr51.w;"
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
                        "or.b  %[rC].s!< 3>,  %[rC].s!<15>,  %[rC].s!< 3>;"
                        // "(ss)nop;"
                        // "predt #14;"
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
                        "shfl.gen.w4.u32 %[tmp0].x,  %[rC].s!<16>, %[mask1];"
                        "shfl.gen.w4.u32 %[tmp0].y,  %[rC].s!<17>, %[mask1];"
                        "shfl.gen.w4.u32 %[tmp1].x,  %[rC].s!<16>, %[mask2];"
                        "shfl.gen.w4.u32 %[tmp1].y,  %[rC].s!<17>, %[mask2];"
                        "shfl.gen.w4.u32 %[tmp2].x,  %[rC].s!<16>, %[mask3];"
                        "shfl.gen.w4.u32 %[tmp2].y,  %[rC].s!<17>, %[mask3];"
                        "(ss)(rpt1)shlm.b32 %[rC].s!<16>,  sr50.x, (r)%[rC].s!<16>,  sr51.x;"
                        "    (rpt1)shlm.b32 %[tmp0].x,     sr50.y, (r)%[tmp0].x,     sr51.y;"
                        "    (rpt1)shlm.b32 %[tmp1].x,     sr50.z, (r)%[tmp1].x,     sr51.z;"
                        "    (rpt1)shlm.b32 %[tmp2].x,     sr50.w, (r)%[tmp2].x,     sr51.w; nop;"                        
                        "(rpt1)or.b  %[rC].s!<16>,  (r)%[tmp0].x,  (r)%[rC].s!<16>; (rpt3)nop;"
                        "(rpt1)or.b  %[rC].s!<16>,  (r)%[tmp1].x,  (r)%[rC].s!<16>; (rpt3)nop;"
                        "(rpt1)or.b  %[rC].s!<16>,  (r)%[tmp2].x,  (r)%[rC].s!<16>; (rpt3)nop;"
                        // "prede;"
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

inline void mm_store_c_N32(float32 *regC_ptr, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!<0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!<4>,  %[c_coordInWords], 3;"                        
                        "(ss)nop;"
                        :  [rC]              "+r" (*regC_ptr)
                        :  [c_coordInWords]  "r" (c_coordInWords)
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

inline void mm_store_c_N32_D2S(float32 *regC_ptr, const int scale_factor, uint global_id_m, uint subMatrixCStartInElements, 
                              half4 sample_0, half4 sample_1, half4 sample_2, half4 sample_3,
                              half4 sample_4, half4 sample_5, half4 sample_6, half4 sample_7,
                              half16 tonemaped_color_0, half16 tonemaped_color_1, int output_width)
{
    
    uint d2s_col = 0;//get_global_id(2) * TILESIZE_N / (scale_factor*scale_factor);
    uint sub_block_id_h = 2*(global_id_m / output_width) + get_global_id(2);
    uint sub_block_id_w = 2*(global_id_m % output_width);

    uint   id_mod_4 = (get_local_id(0) % 4);

    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m/BYTES_PER_WORD));

    uint c_coordInWords00 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords01 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords10 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w))     * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords11 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 1)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;

    uint c_coordInWords02 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 2)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords03 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 3)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords12 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 2)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;
    uint c_coordInWords13 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 3)) * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD/BYTES_PER_WORD / (scale_factor * scale_factor) + d2s_col;

    uint f_coordInHalfWords00 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w)) * 2  * 2  / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords10 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w)) * 2  * 2  / (scale_factor * scale_factor) + d2s_col;

    uint f_coordInHalfWords02 =  ((scale_factor * sub_block_id_h)     * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 2)) * 2  * 2  / (scale_factor * scale_factor) + d2s_col;
    uint f_coordInHalfWords12 =  ((scale_factor * sub_block_id_h + 1) * (output_width*scale_factor*scale_factor) + (scale_factor * sub_block_id_w + 2)) * 2  * 2  / (scale_factor * scale_factor) + d2s_col;
    
   __asm__ __volatile__(
                        // "cmps.u.gt p0, %[id_mod4], 0; nop;" 
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS00].s!<0>,  %[c_coordInWords00], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS01].s!<0>,  %[c_coordInWords01], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS10].s!<0>,  %[c_coordInWords10], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS11].s!<0>,  %[c_coordInWords11], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS02].s!<0>,  %[c_coordInWords02], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS03].s!<0>,  %[c_coordInWords03], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS12].s!<0>,  %[c_coordInWords12], 6;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[rS13].s!<0>,  %[c_coordInWords13], 6;"
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<12>.lo, %[f_coordInHalfWords00],  7;"  
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<14>.lo, %[f_coordInHalfWords10],  7;"  
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<28>.lo, %[f_coordInHalfWords02],  7;"  
                        "stib.untyped.u16.1d.1.unpacked.mode4.base0           %[rC].s!<30>.lo, %[f_coordInHalfWords12],  7;"  
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_0].s!<0>,   %[c_coordInWords00], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_0].s!<4>,   %[c_coordInWords01], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_0].s!<8>,   %[c_coordInWords10], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_0].s!<12>,  %[c_coordInWords11], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_1].s!<0>,   %[c_coordInWords02], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_1].s!<4>,   %[c_coordInWords03], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_1].s!<8>,   %[c_coordInWords12], 8;"
                        "stib.untyped.u16.1d.4.unpacked.mode4.base0           %[tonemaped_color_1].s!<12>,  %[c_coordInWords13], 8;"
                        "(ss)nop;"
                        // "predf #3;"
                        // "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<16>,  %[c_coordInWords], 4;"
                        // "stib.untyped.u32.1d.1.unpacked.mode4.base0           %[rC].s!<17>,  %[c_coordInWords], 5;"
                        // "prede;"  
                        // "(ss)nop;"                                            
                        :  [rC]               "+r" (*regC_ptr)
                        :  [c_coordInWords00]  "r" (c_coordInWords00),
                           [c_coordInWords01]  "r" (c_coordInWords01),
                           [c_coordInWords10]  "r" (c_coordInWords10),
                           [c_coordInWords11]  "r" (c_coordInWords11),
                           [c_coordInWords02]  "r" (c_coordInWords02),
                           [c_coordInWords03]  "r" (c_coordInWords03),
                           [c_coordInWords12]  "r" (c_coordInWords12),
                           [c_coordInWords13]  "r" (c_coordInWords13),
                           [c_coordInWords]    "r" (c_coordInWords),
                           [f_coordInHalfWords00]   "r" (f_coordInHalfWords00),
                           [f_coordInHalfWords10]   "r" (f_coordInHalfWords10),
                           [f_coordInHalfWords02]   "r" (f_coordInHalfWords02),
                           [f_coordInHalfWords12]   "r" (f_coordInHalfWords12),
                           [rS00]    "r" (sample_0),
                           [rS01]    "r" (sample_1),
                           [rS10]    "r" (sample_2),
                           [rS11]    "r" (sample_3),
                           [rS02]    "r" (sample_4),
                           [rS03]    "r" (sample_5),
                           [rS12]    "r" (sample_6),
                           [rS13]    "r" (sample_7),
                           [tonemaped_color_0] "r" (tonemaped_color_0),
                           [tonemaped_color_1] "r" (tonemaped_color_1),
                           [id_mod4]      "r" (id_mod_4) 
                        :  "p0.x"
                       );
}

// inline float4 mm_load_prev_frame(uint subMatrixPrevFrameStartInElements)
// {
//     __private float4   regPrevFrame;

//     size_t sub_block_id_m = get_local_id(0);
//     short  linearIndex = get_local_id(0);

//     uint c_coordInWords = (subMatrixPrevFrameStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_PREV_FRAME_IN_BYTES/BYTES_PER_WORD));

//     __asm__ __volatile__(                        
//                         "ldib.untyped.u32.1d.4.unpacked.mode4.base0           %[rPF].s!<0>,  %[c_coordInWords], 3;" // 3???                        
//                         "(ss)nop;"
//                         :  [rPF] "=r" (regPrevFrame)
//                         :  [c_coordInWords]          "r" (c_coordInWords)
//                         :  
//                        );

//       return regPrevFrame;
// }

__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
__kernel void Conv_MxM_Baseline(__read_only    image1d_buffer_t matrix_A, // [IHxIW, IC]
                                __read_only    image1d_buffer_t matrix_W, // [OC, FHxFWxIC]
                                __read_only    image2d_t input_color_lr,     // Current low res frame in HDR format (not really, it is actually RGBA16F)
                                __read_only    image2d_t previous_frame_hr,  // Previous frame in HDR format (not really, it is actually RGBA16F)
                                __global const float* restrict  matrix_B, // Bias
                                __global const float* restrict  matrix_S, // Scale                                
                                __read_write   image1d_buffer_t candidate_up, // [OHxOW, OC]
                                __read_write   image1d_buffer_t features, // [OHxOW, OC]
                                __read_write   image1d_buffer_t tonemaped_candidate_up, // [OHxOW, OC]
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
                                         const int              prevframe_offset,
                                         const int              jitter_index,
                                         const float2           quantized_jitter_offsets                                         
)
{
// #define TOTAL_BLOCKS (OUT_WIDTH*OUT_HEIGHT/TILESIZE_M)
   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    __private uint4 regBtop[3], regBmid[3], regBbot[3];
    __private uint8 regAtop[3], regAmid[3], regAbot[3];
    __private float32 regC;

    const half inverse_tonemap_min = 0.0;
    const half inverse_tonemap_max = 0.9999847;
          
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
         atomic_and(sync_control+i, 0);
    }

     
   // if ((get_global_id(0) == 0) && (get_global_id(1) == 0) && (get_global_id(2) == 0)) atomic_and(sync_control+layer_index+1, 0); // this is not working for some compiler issue. FILE JIRA

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
        uint row = block_id_m * TILESIZE_M + TILE_OUTPUT_OFFSET;
        uint gidx_m = get_local_id(0) + TILESIZE_M * block_id_m;
        uint out_col_id = gidx_m % output_width;
        int out_row_id = gidx_m / output_width;
        int input_col_id1 = STRIDE * out_col_id;
        int input_col_id0 = input_col_id1 - DILATION;
        int input_col_id2 = input_col_id1 + DILATION;

      //   if (out_row_id < 2*(apron_top-1)) {
      //       subMatrixCStartInElements = row * strideCinElements + col;
      //       uint subMatrixCStartInElements_load_from = (block_id_m * TILESIZE_M + OUT_WIDTH*OUT_HEIGHT) * strideCinElements + col;            
      //       BARSYNC
      //       if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+layer_index);
      //       mm_reload_c_N32(&regC, subMatrixCStartInElements, subMatrixCStartInElements_load_from);
            
      //       block_id_m = sub_group_shuffle(block_id_m, 0);
      //       BARSYNC
            
      //   }
      //   else{
         
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

            uint zfill0mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill1mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill2mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill0top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill1top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill2top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill0bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill1bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
            uint zfill2bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;

            BARSYNC
            mm_load_a_im2col_M64K32_INT8_3x3_zfill(&regAtop[0], &regAtop[1], &regAtop[2], matAcoord0top, matAcoord1top, matAcoord2top, zfill0top, zfill1top, zfill2top,
                                                   &regAmid[0], &regAmid[1], &regAmid[2], matAcoord0mid, matAcoord1mid, matAcoord2mid, zfill0mid, zfill1mid, zfill2mid,
                                                   &regAbot[0], &regAbot[1], &regAbot[2], matAcoord0bot, matAcoord1bot, matAcoord2bot, zfill0bot, zfill1bot, zfill2bot);

            if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
            BARSET
            __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

            mm_mad_N32K32_INT8_3x3(&regC, &regAtop[0], &regAtop[1], &regAtop[2], &regBtop[0], &regBtop[1], &regBtop[2],
                                          &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                          &regAbot[0], &regAbot[1], &regAbot[2], &regBbot[0], &regBbot[1], &regBbot[2], setup_info);

            requant_N32(&regC, scaleConst, biasConst, max, min);

            out_col_id = 2*out_col_id;
            out_row_id = 2*out_row_id + block_id_n;

            float2 uv_0 = (float2)((float)out_col_id - 0.25,     (float)out_row_id + TILE_PREV_FRAME_OFFSET - 0.25);
            float2 uv_1 = (float2)((float)out_col_id + 0.25,     (float)out_row_id + TILE_PREV_FRAME_OFFSET - 0.25);
            float2 uv_2 = (float2)((float)out_col_id - 0.25,     (float)out_row_id + TILE_PREV_FRAME_OFFSET + 0.25);
            float2 uv_3 = (float2)((float)out_col_id + 0.25,     (float)out_row_id + TILE_PREV_FRAME_OFFSET + 0.25);

            float2 uv_4 = (float2)((float)out_col_id - 0.25 + 1, (float)out_row_id + TILE_PREV_FRAME_OFFSET - 0.25);
            float2 uv_5 = (float2)((float)out_col_id + 0.25 + 1, (float)out_row_id + TILE_PREV_FRAME_OFFSET - 0.25);
            float2 uv_6 = (float2)((float)out_col_id - 0.25 + 1, (float)out_row_id + TILE_PREV_FRAME_OFFSET + 0.25);
            float2 uv_7 = (float2)((float)out_col_id + 0.25 + 1, (float)out_row_id + TILE_PREV_FRAME_OFFSET + 0.25);

            uv_0.x -= quantized_jitter_offsets.x;
            uv_1.x -= quantized_jitter_offsets.x;
            uv_2.x -= quantized_jitter_offsets.x;
            uv_3.x -= quantized_jitter_offsets.x;
            uv_4.x -= quantized_jitter_offsets.x;
            uv_5.x -= quantized_jitter_offsets.x;
            uv_6.x -= quantized_jitter_offsets.x;
            uv_7.x -= quantized_jitter_offsets.x;

            uv_0.y += quantized_jitter_offsets.y;
            uv_1.y += quantized_jitter_offsets.y;
            uv_2.y += quantized_jitter_offsets.y;
            uv_3.y += quantized_jitter_offsets.y;
            uv_4.y += quantized_jitter_offsets.y;
            uv_5.y += quantized_jitter_offsets.y;
            uv_6.y += quantized_jitter_offsets.y;
            uv_7.y += quantized_jitter_offsets.y;

            uv_0 = floor(uv_0);
            uv_1 = floor(uv_1);
            uv_2 = floor(uv_2);
            uv_3 = floor(uv_3);
            uv_4 = floor(uv_4);
            uv_5 = floor(uv_5);
            uv_6 = floor(uv_6);
            uv_7 = floor(uv_7);

            uv_0 = fmax(uv_0, 0);
            uv_1 = fmax(uv_1, 0);
            uv_2 = fmax(uv_2, 0);
            uv_3 = fmax(uv_3, 0);
            uv_4 = fmax(uv_4, 0);
            uv_5 = fmax(uv_5, 0);
            uv_6 = fmax(uv_6, 0);
            uv_7 = fmax(uv_7, 0);


            float2 sample_uv_0 = (float2)((uv_0.x +  regC.s!< 0> + 0.5), (uv_0.y +  regC.s!< 4>) + 0.5);  // regMV_min.x x 2?????
            float2 sample_uv_1 = (float2)((uv_1.x +  regC.s!< 1> + 0.5), (uv_1.y +  regC.s!< 5>) + 0.5);
            float2 sample_uv_2 = (float2)((uv_2.x +  regC.s!< 2> + 0.5), (uv_2.y +  regC.s!< 6>) + 0.5);
            float2 sample_uv_3 = (float2)((uv_3.x +  regC.s!< 3> + 0.5), (uv_3.y +  regC.s!< 7>) + 0.5);   
            float2 sample_uv_4 = (float2)((uv_4.x +  regC.s!<16> + 0.5), (uv_4.y +  regC.s!<20>) + 0.5);  // regMV_min.x x 2?????
            float2 sample_uv_5 = (float2)((uv_5.x +  regC.s!<17> + 0.5), (uv_5.y +  regC.s!<21>) + 0.5);
            float2 sample_uv_6 = (float2)((uv_6.x +  regC.s!<18> + 0.5), (uv_6.y +  regC.s!<22>) + 0.5);
            float2 sample_uv_7 = (float2)((uv_7.x +  regC.s!<19> + 0.5), (uv_7.y +  regC.s!<23>) + 0.5);   

            half4 sample_0 = read_imageh(input_color_lr,   sampler, sample_uv_0);
            half4 sample_1 = read_imageh(input_color_lr,   sampler, sample_uv_1);
            half4 sample_2 = read_imageh(input_color_lr,   sampler, sample_uv_2);
            half4 sample_3 = read_imageh(input_color_lr,   sampler, sample_uv_3);
            half4 sample_4 = read_imageh(input_color_lr,   sampler, sample_uv_4);
            half4 sample_5 = read_imageh(input_color_lr,   sampler, sample_uv_5);
            half4 sample_6 = read_imageh(input_color_lr,   sampler, sample_uv_6);
            half4 sample_7 = read_imageh(input_color_lr,   sampler, sample_uv_7);

            float2 uv = (float2)((float)out_col_id,   (float)out_row_id + TILE_PREV_FRAME_OFFSET);

            sample_uv_0 = (float2)((2*uv.x     + 0.5), (2*uv.y     ) + 0.5);  // regMV_min.x x 2?????
            sample_uv_1 = (float2)((2*uv.x + 1 + 0.5), (2*uv.y     ) + 0.5);
            sample_uv_2 = (float2)((2*uv.x     + 0.5), (2*uv.y + 1 ) + 0.5);
            sample_uv_3 = (float2)((2*uv.x + 1 + 0.5), (2*uv.y + 1 ) + 0.5);    

            sample_uv_4 = (float2)((2*uv.x + 2 + 0.5), (2*uv.y     ) + 0.5);  // regMV_min.x x 2?????
            sample_uv_5 = (float2)((2*uv.x + 3 + 0.5), (2*uv.y     ) + 0.5);
            sample_uv_6 = (float2)((2*uv.x + 2 + 0.5), (2*uv.y + 1 ) + 0.5);
            sample_uv_7 = (float2)((2*uv.x + 3 + 0.5), (2*uv.y + 1 ) + 0.5);    

            half4 prev_f_0 = read_imageh(previous_frame_hr, sampler, sample_uv_0);
            half4 prev_f_1 = read_imageh(previous_frame_hr, sampler, sample_uv_1);
            half4 prev_f_2 = read_imageh(previous_frame_hr, sampler, sample_uv_2);
            half4 prev_f_3 = read_imageh(previous_frame_hr, sampler, sample_uv_3); 

            half4 prev_f_4 = read_imageh(previous_frame_hr, sampler, sample_uv_4);
            half4 prev_f_5 = read_imageh(previous_frame_hr, sampler, sample_uv_5);
            half4 prev_f_6 = read_imageh(previous_frame_hr, sampler, sample_uv_6);
            half4 prev_f_7 = read_imageh(previous_frame_hr, sampler, sample_uv_7); 

            half16 tonemaped_color_0;
            half16 tonemaped_color_1;

            mm_prepare_store_c_N32( &regC, &tonemaped_color_0, &tonemaped_color_1, 
                                    &sample_0, &sample_1, &sample_2, &sample_3, &sample_4, &sample_5, &sample_6, &sample_7, 
                                    &prev_f_0, &prev_f_1, &prev_f_2, &prev_f_3, &prev_f_4, &prev_f_5, &prev_f_6, &prev_f_7,
                                    inverse_tonemap_max, inverse_tonemap_min);

            const uint subMatrixCStartInElements = (row + col)/BYTES_PER_RESULT;

            mm_store_c_N32_D2S(&regC, 2, gidx_m + TILE_OUTPUT_OFFSET, subMatrixCStartInElements, 
                              sample_0, sample_1, sample_2, sample_3,
                              sample_4, sample_5, sample_6, sample_7,
                              tonemaped_color_0, tonemaped_color_1, output_width);
            // mm_store_c_N16_D2S(&regC, 2, gidx_m + TILE_OUTPUT_OFFSET, subMatrixCStartInElements);
            block_id_m = sub_group_shuffle(block_id_m, 0);
            BARSYNC
      //   }
    }
}
