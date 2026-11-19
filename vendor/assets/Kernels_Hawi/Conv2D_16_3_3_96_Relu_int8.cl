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
#define ALU_16_CYCLES(rC, rA0, rA1, rA2, rA3, b0, b1, b2, b3, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", (r)      "#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"
#define ALU_16_CYCLES_NO_ACC(rC, rA0, rA1, rA2, rA3, b0, b1, b2, b3, zero, ss, ul) \
                     #ss"(rpt3)dp4acc "#rC", (signed)(r)l"#b0", (signed)"#rA0", "#zero";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b1", (signed)"#rA1", (r)(last)"#rC";"\
                        "(rpt3)dp4acc "#rC", (signed)(r)l"#b2", (signed)"#rA2", (r)(last)"#rC";"\
                     #ul"(rpt3)dp4acc "#rC", (signed)(r)l"#b3", (signed)"#rA3", (r)(last)"#rC";"


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
inline void mm_load_a_im2col_M64K32_INT8_3x3_zfill(uint8 *regA0top, uint8 *regA1top, uint8 *regA2top, uint matAcoord0top, uint matAcoord1top, uint matAcoord2top, uint zfill0top, uint zfill1top, uint zfill2top,
                                                   uint8 *regA0mid, uint8 *regA1mid, uint8 *regA2mid, uint matAcoord0mid, uint matAcoord1mid, uint matAcoord2mid, uint zfill0mid, uint zfill1mid, uint zfill2mid,
                                                   uint8 *regA0bot, uint8 *regA1bot, uint8 *regA2bot, uint matAcoord0bot, uint matAcoord1bot, uint matAcoord2bot, uint zfill0bot, uint zfill1bot, uint zfill2bot)
{
    __asm__ __volatile__(

                        "(rpt3)mov.u32u32 %[rA0top].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA1top].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA2top].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA0mid].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA1mid].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA2mid].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA0bot].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA1bot].s4, 0;"
                        "(rpt3)mov.u32u32 %[rA2bot].s4, 0;"          
                        "cmps.u.gt p0, %[zfill0top], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0top].s0,  %[texCoord0top], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA0top].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1top], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1top].s0,  %[texCoord1top], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA1top].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2top], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2top].s0,  %[texCoord2top], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA2top].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0mid], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0mid].s0,  %[texCoord0mid], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA0mid].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1mid], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1mid].s0,  %[texCoord1mid], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA1mid].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2mid], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2mid].s0,  %[texCoord2mid], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA2mid].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill0bot], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA0bot].s0,  %[texCoord0bot], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA0bot].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill1bot], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA1bot].s0,  %[texCoord1bot], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA1bot].s0, %[zero];"
                        "prede;"
                        "cmps.u.gt p0, %[zfill2bot], 7; (ss)(rpt5)nop;"
                        "predf #2;"
                        "isam.v.1d.s2en.mode6.base0          (f32)(xyzw)      %[rA2bot].s0,  %[texCoord2bot], 16;" 
                        "predt #2;"
                        "(rpt3)    mov.u32u32 %[rA2bot].s0, %[zero];"
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
                           // [zero]      "c" (0x80808080)
                           [zero]      "c" (0x00000000)
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
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0top].s0, %[rB0top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0top].s2, %[rB0top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1top].s0, %[rB1top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1top].s2, %[rB1top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2top].s0, %[rB2top].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2top].s2, %[rB2top].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0mid].s0, %[rB0mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0mid].s2, %[rB0mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1mid].s0, %[rB1mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1mid].s2, %[rB1mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2mid].s0, %[rB2mid].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2mid].s2, %[rB2mid].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0bot].s0, %[rB0bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB0bot].s2, %[rB0bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 2;"
                        "(ss)nop;"
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
                        "cmps.u.gt p0, %[fiberIDmod8], 3; (ss)(rpt5)nop;"
#ifndef USE_ALU                        
                        "predf #36;"
#else
                        "predf #28;"
#endif                        
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
                        "predt #10;"
                        "(rpt3)mov.u32u32 %[rB0top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB0mid].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB0bot].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB1top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB1mid].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB1bot].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB2top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB2mid].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB2bot].s0, 0;"
                        "prede;"

#else
                        "predt #8;"
                        "(rpt3)mov.u32u32 %[rB0top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB0mid].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB0bot].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB1top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB1mid].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB2top].s0, 0;"
                        "(rpt3)mov.u32u32 %[rB2mid].s0, 0;"
                        "prede;"
                        "add.u %[rB1bot].s0, %[matrixBcoord1bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB1bot].s2, %[matrixBcoord1bot].x, %[b_globalOffsetInWords16];"
                        "add.u %[rB2bot].s0, %[matrixBcoord2bot].x, %[b_globalOffsetInWords00];"
                        "add.u %[rB2bot].s2, %[matrixBcoord2bot].x, %[b_globalOffsetInWords16];"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1bot].s0, %[rB1bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB1bot].s2, %[rB1bot].s2, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2bot].s0, %[rB2bot].s0, 2;"
                        "ldib.untyped.u32.1d.1.unpacked.mode4.base0 %[rB2bot].s2, %[rB2bot].s2, 2;"
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
                           [fiberIDmod8]      "r" ((sub_block_id_m&0x7)),
                           [b_globalOffsetInWords00] "r" (setup_info.b_globalOffsetInWords00),
                           [b_globalOffsetInWords16] "r" (setup_info.b_globalOffsetInWords16)
                        :  "p0.x"
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
                       "DEALLOC_MALUM;"
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1bot].s2, 1;"                    
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2bot].s0, 1;"                           
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"                     
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 44.x, 45.x, 46.x, 47.x,     ,(ul))
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
                       ALU_16_CYCLES_NO_ACC(%[rC].s!< 0>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 00.x, 01.x, 02.x, 03.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0top].s2, 1;"                     
                       ALU_16_CYCLES_NO_ACC(%[rC].s!< 4>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 04.x, 05.x, 06.x, 07.x, %[zero],     ,    )
                       ALU_16_CYCLES_NO_ACC(%[rC].s!< 8>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 08.x, 09.x, 10.x, 11.x, %[zero],     ,    )
                       ALU_16_CYCLES_NO_ACC(%[rC].s!<12>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 12.x, 13.x, 14.x, 15.x, %[zero],     ,    )
                       ALU_16_CYCLES_NO_ACC(%[rC].s!<16>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 32.x, 33.x, 34.x, 35.x, %[zero], (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1top].s0, 1;"                      
                       ALU_16_CYCLES_NO_ACC(%[rC].s!<20>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 36.x, 37.x, 38.x, 39.x, %[zero],     ,    )
                       ALU_16_CYCLES_NO_ACC(%[rC].s!<24>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 40.x, 41.x, 42.x, 43.x, %[zero],     ,    )
                       ALU_16_CYCLES_NO_ACC(%[rC].s!<28>, %[rA0top].s0, %[rA0top].s1, %[rA0top].s2, %[rA0top].s3, 44.x, 45.x, 46.x, 47.x, %[zero],     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1top].s2, 1;"                      
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2top].s0, 1;"                           
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA1top].s0, %[rA1top].s1, %[rA1top].s2, %[rA1top].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2top].s2, 1;"                       
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0mid].s0, 1;"                               
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA2top].s0, %[rA2top].s1, %[rA2top].s2, %[rA2top].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0mid].s2, 1;"                     
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1mid].s0, 1;"                  
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA0mid].s0, %[rA0mid].s1, %[rA0mid].s2, %[rA0mid].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1mid].s2, 1;"                   
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2mid].s0, 1;"                            
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA1mid].s0, %[rA1mid].s1, %[rA1mid].s2, %[rA1mid].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2mid].s2, 1;"                    
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB0bot].s0, 1;"                      
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA2mid].s0, %[rA2mid].s1, %[rA2mid].s2, %[rA2mid].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB0bot].s2, 1;"                      
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB1bot].s0, 1;"                   
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA0bot].s0, %[rA0bot].s1, %[rA0bot].s2, %[rA0bot].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB1bot].s2, 1;"                      
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_FIRST_256B) "], %[rB2bot].s0, 1;"                             
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA1bot].s0, %[rA1bot].s1, %[rA1bot].s2, %[rA1bot].s3, 44.x, 45.x, 46.x, 47.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 0>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 00.x, 01.x, 02.x, 03.x, (ss),    )
                       "stl.u32 l[%[b_localOffsetInBytes]+  "xstr(LM_THIRD_256B) "], %[rB2bot].s2, 1;"                    
                       ALU_16_CYCLES(%[rC].s!< 4>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 04.x, 05.x, 06.x, 07.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!< 8>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 08.x, 09.x, 10.x, 11.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<12>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 12.x, 13.x, 14.x, 15.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<16>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 32.x, 33.x, 34.x, 35.x, (ss),    )
                       ALU_16_CYCLES(%[rC].s!<20>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 36.x, 37.x, 38.x, 39.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<24>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 40.x, 41.x, 42.x, 43.x,     ,    )
                       ALU_16_CYCLES(%[rC].s!<28>, %[rA2bot].s0, %[rA2bot].s1, %[rA2bot].s2, %[rA2bot].s3, 44.x, 45.x, 46.x, 47.x,     ,(ul))
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
inline void requant_N32(float32 *regC_ptr, const float32 scaleConst, int32 biasConst, const float max, const float min, mm_setup_info setup_info)
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
                        // "(rpt3)rndaz.f           %[rC].s!< 0>,  (r)%[rC].s!< 0>;"
                        // "(rpt3)rndaz.f           %[rC].s!< 4>,  (r)%[rC].s!< 4>;"
                        // "(rpt3)rndaz.f           %[rC].s!< 8>,  (r)%[rC].s!< 8>;"
                        // "(rpt3)rndaz.f           %[rC].s!<12>,  (r)%[rC].s!<12>;"
                        // "(rpt3)rndaz.f           %[rC].s!<16>,  (r)%[rC].s!<16>;"
                        // "(rpt3)rndaz.f           %[rC].s!<20>,  (r)%[rC].s!<20>;"
                        // "(rpt3)rndaz.f           %[rC].s!<24>,  (r)%[rC].s!<24>;"
                        // "(rpt3)rndaz.f           %[rC].s!<28>,  (r)%[rC].s!<28>;"
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

}

//----------------------------------------------------------------------------//
// store  
//----------------------------------------------------------------------------//
inline void mm_prepare_store_c_N32(float32 *regC_ptr)
{
   __asm__ __volatile__(
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

    __private uint4 regBtop[3], regBmid[3], regBbot[3];
    __private uint8 regAtop[3], regAmid[3], regAbot[3];
    __private float32 regC;
    uint subMatrixCStartInElements;

    size_t block_id_n = get_global_id(2);
    uint  col = block_id_n * TILESIZE_N;

    size_t total_block_n = get_global_size(2);

    MM_SETUP(matrix_W)
    mm_setup_info setup_info = mm_setup();
    
    const uint strideWinElements = LINE_STRIDE_MATRIX_W_IN_BYTES/BYTES_PER_OPERAND;
    const uint strideCinElements = LINE_STRIDE_MATRIX_C_IN_BYTES/BYTES_PER_RESULT;
    uint subMatrixWStartInElements = col * strideWinElements; // B is Kfirst
    mm_load_b_N32K32_INT8_3x3(&regBtop[0], &regBtop[1], &regBtop[2], subMatrixWStartInElements+0*IC, subMatrixWStartInElements+1*IC, subMatrixWStartInElements+2*IC,
                              &regBmid[0], &regBmid[1], &regBmid[2], subMatrixWStartInElements+3*IC, subMatrixWStartInElements+4*IC, subMatrixWStartInElements+5*IC,
                              &regBbot[0], &regBbot[1], &regBbot[2], subMatrixWStartInElements+6*IC, subMatrixWStartInElements+7*IC, subMatrixWStartInElements+8*IC, setup_info);
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
      uint row = block_id_m * TILESIZE_M + TILE_OUTPUT_OFFSET;
      uint gidx_m = get_local_id(0) + TILESIZE_M * block_id_m;
      uint out_col_id = gidx_m % output_width;
      int out_row_id = gidx_m / output_width;
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

      uint zfill0mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2mid = (((input_row_id           ) < TILE_INPUT_OFFSET) || ((input_row_id           )>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2top = (((input_row_id - DILATION) < TILE_INPUT_OFFSET) || ((input_row_id - DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill0bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id0 < 0) || (input_col_id0 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill1bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id1 < 0) || (input_col_id1 >= input_width)) ? FULL_K_MASK : 0;
      uint zfill2bot = (((input_row_id + DILATION) < TILE_INPUT_OFFSET) || ((input_row_id + DILATION)>= input_height) || (input_col_id2 < 0) || (input_col_id2 >= input_width)) ? FULL_K_MASK : 0;

      // BARSYNC
      mm_load_a_im2col_M64K32_INT8_3x3_zfill(&regAtop[0], &regAtop[1], &regAtop[2], matAcoord0top, matAcoord1top, matAcoord2top, zfill0top, zfill1top, zfill2top,
                                             &regAmid[0], &regAmid[1], &regAmid[2], matAcoord0mid, matAcoord1mid, matAcoord2mid, zfill0mid, zfill1mid, zfill2mid,
                                             &regAbot[0], &regAbot[1], &regAbot[2], matAcoord0bot, matAcoord1bot, matAcoord2bot, zfill0bot, zfill1bot, zfill2bot);

      if (get_local_id(0) == 0) block_id_m = atomic_inc(sync_control+my_layer_index);
      // BARSET
      __asm__ __volatile__(PRIORITYRESET "(ss)(sy)nop;");

      mm_mad_N32K32_INT8_3x3(&regC, &regAtop[0], &regAtop[1], &regAtop[2], &regBtop[0], &regBtop[1], &regBtop[2],
                                    &regAmid[0], &regAmid[1], &regAmid[2], &regBmid[0], &regBmid[1], &regBmid[2],
                                    &regAbot[0], &regAbot[1], &regAbot[2], &regBbot[0], &regBbot[1], &regBbot[2], setup_info);

      requant_N32(&regC, scaleConst, biasConst, max, min, setup_info);
      mm_prepare_store_c_N32(&regC);
      subMatrixCStartInElements = row * strideCinElements + col/BYTES_PER_RESULT;
      mm_store_c_N32(&regC, subMatrixCStartInElements);
      block_id_m = sub_group_shuffle(block_id_m, 0);
      // BARSYNC

   }
}
