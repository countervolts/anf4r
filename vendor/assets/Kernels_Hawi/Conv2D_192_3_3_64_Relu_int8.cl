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
    uint2  b_globalOffsetInWords_xy = {((sub_block_id_m%4)*(TILESIZE_K/BYTES_PER_WORD/4)), (sub_block_id_m>>2)}; 

    setup_info.b_globalOffsetInWords00   = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD;
#if (TILESIZE_N == 32 | TILESIZE_N == 64)    
    setup_info.b_globalOffsetInWords16   = setup_info.b_globalOffsetInWords00 + (16 * LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD);
#endif
#ifdef USE_MALU
    setup_info.b_globalOffsetInWords_malu   = (sub_block_id_m&0x7) + ((sub_block_id_m>>3)*(LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD));
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
#elif defined(USE_MALU) && (IC==24)
#define compmask "xyOO"
#define rptNum "rpt1"
#else
#define compmask "xyzw"
#define rptNum "rpt3"
#endif

inline void mm_load_a_im2col_M64K96_INT8_1x3_zfill(uint8 *regA0lft, uint8 *regA1lft, uint8 *regA2lft, uint matAcoord0lft, uint zfilllft,
                                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint matAcoord0mid, uint zfillmid,
                                                   uint8 *regA0rgt, uint8 *regA1rgt, uint8 *regA2rgt, uint matAcoord0rgt, uint zfillrgt)
{
    __asm__ __volatile__(
                        "cmps.u.gt p0, %[zfilllft], 7; (ss)(rpt5)nop;"
                        "predf #7;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0lft].s0,  %[texCoord0lft], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0lft].s4,  %[texCoord0lft], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset8  (f32)(xyzw)      %[rA1lft].s0,  %[texCoord0lft], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset12 (f32)("compmask")%[rA1lft].s4,  %[texCoord0lft], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset16 (f32)(xyzw)      %[rA2lft].s0,  %[texCoord0lft], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset20 (f32)("compmask")%[rA2lft].s4,  %[texCoord0lft], 16;" 
                        "predt #7;"
                        "(rpt3)    mov.u32u32 %[rA0lft].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0lft].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA1lft].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1lft].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA2lft].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2lft].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfillmid], 7; (ss)(rpt5)nop;"
                        "predf #7;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0mid].s4,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset8  (f32)(xyzw)      %[rA1mid].s0,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset12 (f32)("compmask")%[rA1mid].s4,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset16 (f32)(xyzw)      %[rA2mid].s0,  %[texCoord0mid], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset20 (f32)("compmask")%[rA2mid].s4,  %[texCoord0mid], 16;" 
                        "predt #7;"
                        "(rpt3)    mov.u32u32 %[rA0mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0mid].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA1mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1mid].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA2mid].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2mid].s4, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfillrgt], 7; (ss)(rpt5)nop;"
                        "predf #7;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0rgt].s0,  %[texCoord0rgt], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset4  (f32)("compmask")%[rA0rgt].s4,  %[texCoord0rgt], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset8  (f32)(xyzw)      %[rA1rgt].s0,  %[texCoord0rgt], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset12 (f32)("compmask")%[rA1rgt].s4,  %[texCoord0rgt], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset16 (f32)(xyzw)      %[rA2rgt].s0,  %[texCoord0rgt], 16;" 
                        "isam.v.1d.s2en.mode6.base0.offset20 (f32)("compmask")%[rA2rgt].s4,  %[texCoord0rgt], 16;" 
                        "predt #7;"
                        "(rpt3)    mov.u32u32 %[rA0rgt].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA0rgt].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA1rgt].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA1rgt].s4, %[zero];"
                        "(rpt3)    mov.u32u32 %[rA2rgt].s0, %[zero];"
                        "("rptNum")mov.u32u32 %[rA2rgt].s4, %[zero];"
                        "prede;"
                        "(ss)nop;"
                        :  [rA0lft] "=r" (*regA0lft), [rA1lft] "=r" (*regA1lft), [rA2lft] "=r" (*regA2lft),
                           [rA0mid] "=r" (*regA0mid), [rA1mid] "=r" (*regA1mid), [rA2mid] "=r" (*regA2mid),
                           [rA0rgt] "=r" (*regA0rgt), [rA1rgt] "=r" (*regA1rgt), [rA2rgt] "=r" (*regA2rgt)  
                        :  [texCoord0lft] "r" (matAcoord0lft / (4)),
                           [texCoord0mid] "r" (matAcoord0mid / (4)),
                           [texCoord0rgt] "r" (matAcoord0rgt / (4)),
                           [zfilllft] "r" (zfilllft),
                           [zfillmid] "r" (zfillmid),
                           [zfillrgt] "r" (zfillrgt),
                           [zero]      "c" (0x80808080)
                        :  "p0.x"
                       );
}

//----------------------------------------------------------------------------//
// load b 
//----------------------------------------------------------------------------//
inline void mm_load_b_N32K96_INT8_1x3(uint4 *regB0lft, uint4 *regB1lft, uint4 *regB2lft,
                                      uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid,
                                      uint4 *regB0rgt, uint4 *regB1rgt, uint4 *regB2rgt, uint subMatrixWStartInElements,
                                      mm_setup_info setup_info)
{
   // use just stride instead of all these registers....
#ifndef USE_MALU
    size_t sub_block_id_m = get_local_id(0);
    __asm__ __volatile__(
                        "add.u %[rB2rgt].s0, %[subMatrixWStart].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2rgt].s2, %[subMatrixWStart].x, %[b_globalOffsetInWords16];"
                        "(rpt3)nop;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0          %[rB0lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0          %[rB0lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset8  %[rB1lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset8  %[rB1lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset16 %[rB2lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset16 %[rB2lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset48 %[rB0mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset48 %[rB0mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset56 %[rB1mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset56 %[rB1mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset64 %[rB2mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset64 %[rB2mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset96 %[rB0rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset96 %[rB0rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset104 %[rB1rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset104 %[rB1rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset112 %[rB2rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset112 %[rB2rgt].s2, %[rB2rgt].s2, 2;"
                        "(ss)nop;"
                        :  [rB0lft] "=r" (*regB0lft), [rB1lft] "=r" (*regB1lft), [rB2lft] "=r" (*regB2lft),
                           [rB0mid] "=r" (*regB0mid), [rB1mid] "=r" (*regB1mid), [rB2mid] "=r" (*regB2mid),
                           [rB0rgt] "=r" (*regB0rgt), [rB1rgt] "=r" (*regB1rgt), [rB2rgt] "=r" (*regB2rgt)                           
                        :   
                           [subMatrixWStart] "c" (subMatrixWStartInElements),     
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (setup_info.b_globalOffsetInWords16)
                        :  
                       );
#else
    size_t sub_block_id_m = get_local_id(0);
    __asm__ __volatile__(                        
                        "add.u %[rB2rgt].s0, %[matrixBcoord0lft].x, %[fiberOffset]; (rpt3)nop;"                        
                        "add.u %[rB2rgt].s1, %[rB2rgt].s0,          %[strideTimesEight]; (rpt3)nop;"                        
                        "add.u %[rB2rgt].s2, %[rB2rgt].s1,          %[strideTimesEight]; (rpt3)nop;"                        
                        "add.u %[rB2rgt].s3, %[rB2rgt].s2,          %[strideTimesEight]; (rpt3)nop;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0lft].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0         %[rB0lft].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset8 %[rB1lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset8 %[rB1lft].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset8 %[rB1lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset8 %[rB1lft].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset16 %[rB2lft].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset16 %[rB2lft].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset16 %[rB2lft].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset16 %[rB2lft].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset48 %[rB0mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset48 %[rB0mid].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset48 %[rB0mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset48 %[rB0mid].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset56 %[rB1mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset56 %[rB1mid].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset56 %[rB1mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset56 %[rB1mid].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset64 %[rB2mid].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset64 %[rB2mid].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset64 %[rB2mid].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset64 %[rB2mid].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset96 %[rB0rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset96 %[rB0rgt].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset96 %[rB0rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset96 %[rB0rgt].s3, %[rB2rgt].s3, 2;"
#ifndef USE_ALU
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset104 %[rB1rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset104 %[rB1rgt].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset104 %[rB1rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset104 %[rB1rgt].s3, %[rB2rgt].s3, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset112 %[rB2rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset112 %[rB2rgt].s1, %[rB2rgt].s1, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset112 %[rB2rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0.offset112 %[rB2rgt].s3, %[rB2rgt].s3, 2;"
#else
                        "(ss)nop;"
                        "(nop3)add.u %[rB2rgt].s0, %[matrixBcoord0lft].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2rgt].s2, %[rB2rgt].s0, %[strideTimesSixteen];"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset104 %[rB1rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset104 %[rB1rgt].s2, %[rB2rgt].s2, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset112 %[rB2rgt].s0, %[rB2rgt].s0, 2;"
                        "ldib.untyped.u32.1d.2.unpacked.mode4.base0.offset112 %[rB2rgt].s2, %[rB2rgt].s2, 2;"
#endif
                        "(ss)nop;" /* FIXME - compiler issue */
                        :  [rB0lft] "=r" (*regB0lft), [rB1lft] "=r" (*regB1lft), [rB2lft] "=r" (*regB2lft),
                           [rB0mid] "=r" (*regB0mid), [rB1mid] "=r" (*regB1mid), [rB2mid] "=r" (*regB2mid),
                           [rB0rgt] "=r" (*regB0rgt), [rB1rgt] "=r" (*regB1rgt), [rB2rgt] "=r" (*regB2rgt)
                        :  [matrixBcoord0lft]   "c" (subMatrixWStartInElements),                            
                           [strideTimesEight]   "c" ( 8*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD),
                           [strideTimesSixteen] "c" (16*LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_WORD),
#ifdef USE_ALU             
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00),
#endif                                         
                           [fiberOffset]      "r" (setup_info.b_globalOffsetInWords_malu)
                           
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
inline void mm_mad_N32K96_INT8_1x3_no_accum(float32 *regC_ptr,
                                   uint8 *regA0lft, uint8 *regA1lft, uint8 *regA2lft, uint4 *regB0lft, uint4 *regB1lft, uint4 *regB2lft,
                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid,
                                   uint8 *regA0rgt, uint8 *regA1rgt, uint8 *regA2rgt, uint4 *regB0rgt, uint4 *regB1rgt, uint4 *regB2rgt,
                                   mm_setup_info setup_info)
{
    short  linearIndex = get_local_id(0);
#ifdef USE_MALU
    __asm__ __volatile__(
                       "ALLOC_MALUM;"
                       "malu_wmma.m64n32.s8s32       m[0], (linear)(rmajor)%[rA0lft].s0, (folded)(cmajor)%[rB0lft].s0;"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1lft].s0, (folded)(cmajor)%[rB1lft].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2lft].s0, (folded)(cmajor)%[rB2lft].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0mid].s0, (folded)(cmajor)%[rB0mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1mid].s0, (folded)(cmajor)%[rB1mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2mid].s0, (folded)(cmajor)%[rB2mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0rgt].s0, (folded)(cmajor)%[rB0rgt].s0, m[0];"
#ifndef USE_ALU
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1rgt].s0, (folded)(cmajor)%[rB1rgt].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2rgt].s0, (folded)(cmajor)%[rB2rgt].s0, m[0];"
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "DEALLOC_MALUM;"
                       "(ss)nop;"
#else
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1rgt].s1, 1;" 
                       "DEALLOC_MALUM;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1rgt].s2, 1;"                      
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1rgt].s3, 1;"                  
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2rgt].s0, 1;"                      
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2rgt].s1, 1;"                         
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2rgt].s2, 1;"                  
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2rgt].s3, 1;"                
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
#endif
                        :  [rC] "=r" (*regC_ptr)
                        :  [rB0lft] "r" (*regB0lft), [rA0lft] "r" (*regA0lft), [rB1lft] "r" (*regB1lft), [rA1lft] "r" (*regA1lft), [rB2lft] "r" (*regB2lft), [rA2lft] "r" (*regA2lft),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0rgt] "r" (*regB0rgt), [rA0rgt] "r" (*regA0rgt), [rB1rgt] "r" (*regB1rgt), [rA1rgt] "r" (*regA1rgt), [rB2rgt] "r" (*regB2rgt), [rA2rgt] "r" (*regA2rgt),
                           [b_localOffsetInBytes]    "r" (setup_info.b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex),
                           [zero]                    "c" (0x00000000),
                           "malu_matrix_c_size"(MATRIX_C_ON_MALUMEM_IN_KBYTES)
                        : 
                       );
#else
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0lft].s0, 1;"           
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0lft].s1, 1;"                       
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 0>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0lft].s2, 1;"          
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0lft].s3, 1;"     
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 4>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!< 8>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<12>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<16>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1lft].s0, 1;"           
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1lft].s1, 1;"       
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<20>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<24>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x, %[zero],     ,    )
                       ALU_32_CYCLES_NO_ACC(%[rC].s!<28>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x, %[zero],     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1lft].s2, 1;"         
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1lft].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2lft].s0, 1;"         
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2lft].s1, 1;"               
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2lft].s2, 1;"         
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2lft].s3, 1;"       
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0mid].s0, 1;"             
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0mid].s1, 1;"           
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0rgt].s0, 1;"                    
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0rgt].s1, 1;"                          
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0rgt].s2, 1;"                 
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0rgt].s3, 1;"          
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1rgt].s0, 1;"              
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1rgt].s1, 1;"                 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1rgt].s2, 1;"                 
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1rgt].s3, 1;"           
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2rgt].s0, 1;"                  
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2rgt].s1, 1;"                       
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2rgt].s2, 1;"                 
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2rgt].s3, 1;"           
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
                        :  [rC] "=r" (*regC_ptr)
                        :  [rB0lft] "r" (*regB0lft), [rA0lft] "r" (*regA0lft), [rB1lft] "r" (*regB1lft), [rA1lft] "r" (*regA1lft), [rB2lft] "r" (*regB2lft), [rA2lft] "r" (*regA2lft),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0rgt] "r" (*regB0rgt), [rA0rgt] "r" (*regA0rgt), [rB1rgt] "r" (*regB1rgt), [rA1rgt] "r" (*regA1rgt), [rB2rgt] "r" (*regB2rgt), [rA2rgt] "r" (*regA2rgt),
                           [b_localOffsetInBytes]    "r" (setup_info.b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex),
                           [zero]                    "c" (0x00000000)
                        : 
                       );
#endif
}

inline void mm_mad_N32K96_INT8_1x3(float32 *regC_ptr,
                                   uint8 *regA0lft, uint8 *regA1lft, uint8 *regA2lft, uint4 *regB0lft, uint4 *regB1lft, uint4 *regB2lft,
                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint4 *regB0mid, uint4 *regB1mid, uint4 *regB2mid,
                                   uint8 *regA0rgt, uint8 *regA1rgt, uint8 *regA2rgt, uint4 *regB0rgt, uint4 *regB1rgt, uint4 *regB2rgt,
                                   mm_setup_info setup_info)
{
    short  linearIndex = get_local_id(0);
#ifdef USE_MALU
    __asm__ __volatile__(
                       "ALLOC_MALUM;"
                       "stm.m64n32.l.u32 m[0], (rmajor)%[rC].s!<0>;"
                       "(ss)nop;"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0lft].s0, (folded)(cmajor)%[rB0lft].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1lft].s0, (folded)(cmajor)%[rB1lft].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2lft].s0, (folded)(cmajor)%[rB2lft].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0mid].s0, (folded)(cmajor)%[rB0mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1mid].s0, (folded)(cmajor)%[rB1mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2mid].s0, (folded)(cmajor)%[rB2mid].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA0rgt].s0, (folded)(cmajor)%[rB0rgt].s0, m[0];"
#ifndef USE_ALU
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA1rgt].s0, (folded)(cmajor)%[rB1rgt].s0, m[0];"
                       "malu_wmma.accu.m64n32.s8s32  m[0], (linear)(rmajor)%[rA2rgt].s0, (folded)(cmajor)%[rB2rgt].s0, m[0];"
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "DEALLOC_MALUM;"
                       "(ss)nop;"
#else
                       "ldm.m64n32.l.u32 (rmajor)%[rC].s!<0>, m[0];"
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1rgt].s1, 1;" 
                       "DEALLOC_MALUM;"
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1rgt].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1rgt].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2rgt].s1, 1;"  
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2rgt].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2rgt].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
#endif
                        :  
                        :  [rC] "r" (*regC_ptr), [rB0lft] "r" (*regB0lft), [rA0lft] "r" (*regA0lft), [rB1lft] "r" (*regB1lft), [rA1lft] "r" (*regA1lft), [rB2lft] "r" (*regB2lft), [rA2lft] "r" (*regA2lft),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0rgt] "r" (*regB0rgt), [rA0rgt] "r" (*regA0rgt), [rB1rgt] "r" (*regB1rgt), [rA1rgt] "r" (*regA1rgt), [rB2rgt] "r" (*regB2rgt), [rA2rgt] "r" (*regA2rgt),
                           [b_localOffsetInBytes]    "r" (setup_info.b_localOffsetInBytes),
                           [linearIndex]             "r" (linearIndex),
                           [zero]                    "c" (0x00000000),
                           "malu_matrix_c_size"(MATRIX_C_ON_MALUMEM_IN_KBYTES)
                        : 
                       );
#else
    __asm__ __volatile__(
                       "mova  a0, %[linearIndex];" /* a0 is not used at all, but assigning a divergent value tricks the HW into locking the ALU for this wave */
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0lft].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0lft].s1, 1;"   
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0lft].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0lft].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1lft].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1lft].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0lft].s0, %[rA0lft].s1, %[rA0lft].s2, %[rA0lft].s3, %[rA0lft].s4, %[rA0lft].s5, %[rA0lft].s6, %[rA0lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1lft].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1lft].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2lft].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2lft].s1, 1;"   
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1lft].s0, %[rA1lft].s1, %[rA1lft].s2, %[rA1lft].s3, %[rA1lft].s4, %[rA1lft].s5, %[rA1lft].s6, %[rA1lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2lft].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2lft].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0mid].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0mid].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2lft].s0, %[rA2lft].s1, %[rA2lft].s2, %[rA2lft].s3, %[rA2lft].s4, %[rA2lft].s5, %[rA2lft].s6, %[rA2lft].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
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
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB0rgt].s1, 1;"        
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, %[rA2mid].s4, %[rA2mid].s5, %[rA2mid].s6, %[rA2mid].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0rgt].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB0rgt].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB1rgt].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA0rgt].s0, %[rA0rgt].s1, %[rA0rgt].s2, %[rA0rgt].s3, %[rA0rgt].s4, %[rA0rgt].s5, %[rA0rgt].s6, %[rA0rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1rgt].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB1rgt].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2rgt].s0, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_SECOND_256B)"], %[rB2rgt].s1, 1;" 
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA1rgt].s0, %[rA1rgt].s1, %[rA1rgt].s2, %[rA1rgt].s3, %[rA1rgt].s4, %[rA1rgt].s5, %[rA1rgt].s6, %[rA1rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 0>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 00.x, 16.x, 01.x, 17.x, 02.x, 18.x, 03.x, 19.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2rgt].s2, 1;"
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FOURTH_256B)"], %[rB2rgt].s3, 1;"
                       ALU_32_CYCLES(%[rC].s!< 4>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 04.x, 20.x, 05.x, 21.x, 06.x, 22.x, 07.x, 23.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!< 8>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 08.x, 24.x, 09.x, 25.x, 10.x, 26.x, 11.x, 27.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<12>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 12.x, 28.x, 13.x, 29.x, 14.x, 30.x, 15.x, 31.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<16>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 32.x, 48.x, 33.x, 49.x, 34.x, 50.x, 35.x, 51.x, (ss),    )
                       ALU_32_CYCLES(%[rC].s!<20>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 36.x, 52.x, 37.x, 53.x, 38.x, 54.x, 39.x, 55.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<24>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 40.x, 56.x, 41.x, 57.x, 42.x, 58.x, 43.x, 59.x,     ,    )
                       ALU_32_CYCLES(%[rC].s!<28>, %[rA2rgt].s0, %[rA2rgt].s1, %[rA2rgt].s2, %[rA2rgt].s3, %[rA2rgt].s4, %[rA2rgt].s5, %[rA2rgt].s6, %[rA2rgt].s7, 44.x, 60.x, 45.x, 61.x, 46.x, 62.x, 47.x, 63.x,     ,(ul))
                        : 
                        :  [rC] "r" (*regC_ptr), [rB0lft] "r" (*regB0lft), [rA0lft] "r" (*regA0lft), [rB1lft] "r" (*regB1lft), [rA1lft] "r" (*regA1lft), [rB2lft] "r" (*regB2lft), [rA2lft] "r" (*regA2lft),
                           [rB0mid] "r" (*regB0mid), [rA0mid] "r" (*regA0mid), [rB1mid] "r" (*regB1mid), [rA1mid] "r" (*regA1mid), [rB2mid] "r" (*regB2mid), [rA2mid] "r" (*regA2mid),
                           [rB0rgt] "r" (*regB0rgt), [rA0rgt] "r" (*regA0rgt), [rB1rgt] "r" (*regB1rgt), [rA1rgt] "r" (*regA1rgt), [rB2rgt] "r" (*regB2rgt), [rA2rgt] "r" (*regA2rgt),
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
#ifndef FP16_OUTPUT                        
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
                        "(rpt3)add.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     %[point_five];"
                        "(rpt3)add.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     %[point_five];"
                        "(rpt3)floor.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)floor.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)floor.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)floor.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)floor.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)floor.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)floor.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)floor.f           %[rC].s!<28>,  (r)%[rC].s!<28>;"                        
#else
                        "(rpt3)max.f             %[rC].s!< 0>,  (r)%[rC].s!< 0>,     0;"
                        "(rpt3)max.f             %[rC].s!< 4>,  (r)%[rC].s!< 4>,     0;"
                        "(rpt3)max.f             %[rC].s!< 8>,  (r)%[rC].s!< 8>,     0;"
                        "(rpt3)max.f             %[rC].s!<12>,  (r)%[rC].s!<12>,     0;"
                        "(rpt3)max.f             %[rC].s!<16>,  (r)%[rC].s!<16>,     0;"
                        "(rpt3)max.f             %[rC].s!<20>,  (r)%[rC].s!<20>,     0;"
                        "(rpt3)max.f             %[rC].s!<24>,  (r)%[rC].s!<24>,     0;"
                        "(rpt3)max.f             %[rC].s!<28>,  (r)%[rC].s!<28>,     0;"   
#endif                                             
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
inline void mm_prepare_store_c_N32(float32 *regC_ptr)
{
   __asm__ __volatile__(
#ifndef FP16_OUTPUT      
                        "(rpt3)cov.f32s32        %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32s32        %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32s32        %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32s32        %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.f32s32        %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        "(rpt3)cov.f32s32        %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.f32s32        %[rC].s!<28>,  (r)%[rC].s!<28>;"
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
#else
                        "(rpt3)cov.f32f16        %[rC].s!< 0>.lo,  (r)%[rC].s!< 0>;"
                        "(rpt3)cov.f32f16        %[rC].s!< 2>.lo,  (r)%[rC].s!< 4>;"
                        "(rpt3)cov.f32f16        %[rC].s!< 4>.lo,  (r)%[rC].s!< 8>;"
                        "(rpt3)cov.f32f16        %[rC].s!< 6>.lo,  (r)%[rC].s!<12>;"
                        "(rpt3)cov.f32f16        %[rC].s!< 8>.lo,  (r)%[rC].s!<16>;"
                        "(rpt3)cov.f32f16        %[rC].s!<10>.lo,  (r)%[rC].s!<20>;"
                        "(rpt3)cov.f32f16        %[rC].s!<12>.lo,  (r)%[rC].s!<24>;"
                        "(rpt3)cov.f32f16        %[rC].s!<14>.lo,  (r)%[rC].s!<28>;"
#endif                        

                        :  [rC] "+r" (*regC_ptr)
                        :
                        :  
                       );

}

inline void mm_store_c_N32(float32 *regC_ptr, uint subMatrixCStartInElements)
{
    size_t sub_block_id_m = get_local_id(0);
    uint c_coordInWords = (subMatrixCStartInElements + (sub_block_id_m * LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_WORD));
   __asm__ __volatile__(
#ifndef FP16_OUTPUT      
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!< 0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!< 4>,  %[c_coordInWords], 3;"                        
#else
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0           %[rC].s!< 0>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset4   %[rC].s!< 4>,  %[c_coordInWords], 3;"                         
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset8   %[rC].s!< 8>,  %[c_coordInWords], 3;"
                        "stib.untyped.u32.1d.4.unpacked.mode4.base0.offset12  %[rC].s!<12>,  %[c_coordInWords], 3;"
#endif                        
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

__attribute__((qcom_wave_pair_mode(1))) // 1=force single 2=force pair
__kernel void Conv_MxM_Baseline(__read_only    image1d_buffer_t matrix_A, // [IHxIW, IC]
                                __read_only    image1d_buffer_t matrix_W, // [OC, FHxFWxIC]
                                __global const float* restrict  matrix_B, // Bias
                                __global const float* restrict  matrix_S, // Scale
                                __read_write   image1d_buffer_t matrix_C, // [OHxOW, OC]
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
   // Now, in IC:96, we load each row of the 3x3 filter, execute 9 (3 columns and 3 blocks of K32 each) back-to-back wmma calls, and load the next row, ...

    __private uint4 regBlft[3], regBmid[3], regBrgt[3];
    __private uint8 regAlft[3], regAmid[3], regArgt[3];
    __private float32 regC;
    

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
         atomic_and(sync_control+sync_buffer_start_index+total_block_n+i, 0);
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
      uint matAcoord0mid = (input_row_id * input_width + input_col_id0) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord1mid = (input_row_id * input_width + input_col_id1) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord2mid = (input_row_id * input_width + input_col_id2) * (LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord0top = matAcoord0mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord1top = matAcoord1mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord2top = matAcoord2mid - DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);       
      uint matAcoord0bot = matAcoord0mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord1bot = matAcoord1mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES);
      uint matAcoord2bot = matAcoord2mid + DILATION*input_width*(LINE_STRIDE_MATRIX_A_IN_BYTES); 
      

      uint zfill0mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;


      // First row
      // First 96 ch
      // BARSYNC
      mm_load_b_N32K96_INT8_1x3( &regBlft[0], &regBlft[1], &regBlft[2],
                                 &regBmid[0], &regBmid[1], &regBmid[2],
                                 &regBrgt[0], &regBrgt[1], &regBrgt[2], (col * LINE_STRIDE_MATRIX_W_IN_BYTES / (4)), setup_info);

      mm_load_a_im2col_M64K96_INT8_1x3_zfill(&regAlft[0], &regAlft[1], &regAlft[2], matAcoord0top, zfill0top,
                                             &regAmid[0], &regAmid[1], &regAmid[2], matAcoord1top, zfill1top,
                                             &regArgt[0], &regArgt[1], &regArgt[2], matAcoord2top, zfill2top);

      // subMatrixWStartInElements = col * strideWinElements / (4/BYTES_PER_OPERAND) + 3*IC;
      // BARSET
      __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

      mm_mad_N32K96_INT8_1x3_no_accum(&regC, &regAlft[0], &regAlft[1], &regAlft[2], &regBlft[0], &regBlft[1], &regBlft[2],
                                    &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regArgt[0], &regArgt[1], &regArgt[2], &regBrgt[0], &regBrgt[1], &regBrgt[2], setup_info);


      // Second 96 ch
      // BARSYNC
      mm_load_b_N32K96_INT8_1x3( &regBlft[0], &regBlft[1], &regBlft[2],
                                 &regBmid[0], &regBmid[1], &regBmid[2],
                                 &regBrgt[0], &regBrgt[1], &regBrgt[2], (col * LINE_STRIDE_MATRIX_W_IN_BYTES / (4)) + 24, setup_info);

      mm_load_a_im2col_M64K96_INT8_1x3_zfill(&regAlft[0], &regAlft[1], &regAlft[2], matAcoord0top + 96, zfill0top,
                                             &regAmid[0], &regAmid[1], &regAmid[2], matAcoord1top + 96, zfill1top,
                                             &regArgt[0], &regArgt[1], &regArgt[2], matAcoord2top + 96, zfill2top);

      // subMatrixWStartInElements = col * strideWinElements / (4/BYTES_PER_OPERAND) + 3*IC;
      // BARSET
      __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

      mm_mad_N32K96_INT8_1x3(&regC, &regAlft[0], &regAlft[1], &regAlft[2], &regBlft[0], &regBlft[1], &regBlft[2],
                                    &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regArgt[0], &regArgt[1], &regArgt[2], &regBrgt[0], &regBrgt[1], &regBrgt[2], setup_info);


      // Second row
      #pragma unroll 1
      for (int i = 0; i < 2; i++) {
         // BARSYNC
         mm_load_b_N32K96_INT8_1x3( &regBlft[0], &regBlft[1], &regBlft[2],
                                    &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regBrgt[0], &regBrgt[1], &regBrgt[2], ((col * LINE_STRIDE_MATRIX_W_IN_BYTES + 3*IC) / (4)) + i*24, setup_info);

         mm_load_a_im2col_M64K96_INT8_1x3_zfill(&regAlft[0], &regAlft[1], &regAlft[2], matAcoord0mid + i*96, zfill0mid,
                                                &regAmid[0], &regAmid[1], &regAmid[2], matAcoord1mid + i*96, zfill1mid,
                                                &regArgt[0], &regArgt[1], &regArgt[2], matAcoord2mid + i*96, zfill2mid);

         // subMatrixWStartInElements = col * strideWinElements / (4/BYTES_PER_OPERAND) + 6*IC;
         // BARSET
         __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

         
         
         mm_mad_N32K96_INT8_1x3(&regC, &regAlft[0], &regAlft[1], &regAlft[2], &regBlft[0], &regBlft[1], &regBlft[2],
                                       &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                       &regArgt[0], &regArgt[1], &regArgt[2], &regBrgt[0], &regBrgt[1], &regBrgt[2], setup_info);

      }
      

      // Third row
      #pragma unroll 1
      for (int i = 0; i < 2; i++) {
         // BARSYNC
         mm_load_b_N32K96_INT8_1x3( &regBlft[0], &regBlft[1], &regBlft[2],
                                    &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regBrgt[0], &regBrgt[1], &regBrgt[2], ((col * LINE_STRIDE_MATRIX_W_IN_BYTES + 6*IC) / (4)) + i*24, setup_info);

                                    

         mm_load_a_im2col_M64K96_INT8_1x3_zfill(&regAlft[0], &regAlft[1], &regAlft[2], matAcoord0bot + i*96, zfill0bot,
                                                &regAmid[0], &regAmid[1], &regAmid[2], matAcoord1bot + i*96, zfill1bot,
                                                &regArgt[0], &regArgt[1], &regArgt[2], matAcoord2bot + i*96, zfill2bot);

         
         // BARSET
         __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

         mm_mad_N32K96_INT8_1x3(&regC, &regAlft[0], &regAlft[1], &regAlft[2], &regBlft[0], &regBlft[1], &regBlft[2],
                                       &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                       &regArgt[0], &regArgt[1], &regArgt[2], &regBrgt[0], &regBrgt[1], &regBrgt[2], setup_info);

      }
      
      if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);

      requant_N32(&regC, scaleConst, biasConst, max, min);
      mm_prepare_store_c_N32(&regC);

#ifndef FP16_OUTPUT      
      const uint subMatrixCStartInElements = (row * LINE_STRIDE_MATRIX_C_IN_BYTES + col)/BYTES_PER_RESULT;
#else
      const uint subMatrixCStartInElements = (row * LINE_STRIDE_MATRIX_C_IN_BYTES + col*BYTES_PER_HALF_WORD)/BYTES_PER_RESULT;
#endif      
      mm_store_c_N32(&regC, subMatrixCStartInElements);
      block_id_m = sub_group_shuffle(block_id_m, 0);
      // BARSYNC

   }
}
