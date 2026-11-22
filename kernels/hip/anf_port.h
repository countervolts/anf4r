// anf4r: shared part of the OpenCL C ports of the compute passes (kernels/hip/*.cl), for native Windows where HIP cannot run SPIR-V
// (the same approach as m4r's shaders/glsl_port.h). A port reads the argument block runtime/pass.h's HipArgs describes through its
// single kernel argument: bindings in slot[] (buffer addresses, texture / sampler records), specialisation constants in spec[]
// (runtime values here), push constants in push[]. Arithmetic follows the GLSL: fma / min / max / floor as the hardware
// instructions, no contraction of separate multiplies and adds. No OpenCL library is linked.
// Each port defines LX, LY, LZ (its local size) before including this header.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma clang fp contract(off)

typedef struct { ulong slot[64]; int spec[64]; uint specSet[2]; uint pad[2]; uint push[32]; uint grid[4]; } Args;
// binding -> slot (runtime/pass.h hipSlot: 16 slots per HLSL register class of 128 bindings)
#define SLOT(b) (a->slot[((b) >> 7) * 16 + ((b) & 15)])

#ifndef LZ
#define LZ 1
#endif
#define ANF_KERNEL(entry) __attribute__((reqd_work_group_size(LX, LY, LZ))) kernel void entry(constant const Args* a)

#define LID_X ((uint)__builtin_amdgcn_workitem_id_x())
#define LID_Y ((uint)__builtin_amdgcn_workitem_id_y())
#define LID_Z ((uint)__builtin_amdgcn_workitem_id_z())
#define WG_X ((uint)__builtin_amdgcn_workgroup_id_x())
#define WG_Y ((uint)__builtin_amdgcn_workgroup_id_y())
#define WG_Z ((uint)__builtin_amdgcn_workgroup_id_z())
#define GID_X (WG_X * LX + LID_X)   // gl_GlobalInvocationID
#define GID_Y (WG_Y * LY + LID_Y)
#define GID_Z (WG_Z * LZ + LID_Z)
#define LIDX (LID_Z * (LX * LY) + LID_Y * LX + LID_X)   // gl_LocalInvocationIndex

#define SPEC(i) (a->spec[i])
#define SPECF(i) (as_float(a->spec[i]))
// a specialisation constant the host may leave unset: then its GLSL default, as in Vulkan
#define SPECD(i, def) (((a->specSet[(i) >> 5] >> ((i) & 31)) & 1u) ? a->spec[i] : (def))
#define SPECDF(i, def) (((a->specSet[(i) >> 5] >> ((i) & 31)) & 1u) ? as_float(a->spec[i]) : (def))
#define PUSHU(i) (a->push[i])
#define PUSHI(i) ((int)a->push[i])
#define PUSHF(i) (as_float(a->push[i]))
#define BUF(T, b) ((global T*)SLOT(b))
#define CBUF(T, b) ((global const T*)SLOT(b))

#define BARRIER() __builtin_amdgcn_s_barrier()
// barrier() with workgroup-shared memory: a fence around the hardware barrier, as GLSL's barrier() + memoryBarrierShared()
#define LDS_BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); \
                           __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)

// GLSL bitfieldExtract on int (sign-extending) and uint
static inline int bfe_i(int v, int off, int bits) { return (int)((uint)v << (32 - off - bits)) >> (32 - bits); }
static inline uint bfe_u(uint v, int off, int bits) { return (v >> off) & ((bits == 32) ? 0xffffffffu : ((1u << bits) - 1u)); }

// every g_* is overloaded (the GLSL built-ins of the translated passes, see below, take vectors too)
#define OV __attribute__((overloadable)) static inline
OV float g_min(float x, float y) { return __builtin_fminf(x, y); }
OV float g_max(float x, float y) { return __builtin_fmaxf(x, y); }
OV float g_clamp(float x, float lo, float hi) { return __builtin_fminf(__builtin_fmaxf(x, lo), hi); }
static inline int g_mini(int x, int y) { return x < y ? x : y; }
static inline int g_maxi(int x, int y) { return x > y ? x : y; }
static inline int g_clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
OV float g_fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
OV float g_floor(float x) { return __builtin_floorf(x); }
OV float g_ceil(float x) { return __builtin_ceilf(x); }
OV float g_abs(float x) { return __builtin_fabsf(x); }
OV float g_fract(float x) { return x - __builtin_floorf(x); }
// GLSL x / y as RADV lowers it for 32-bit floats (a multiply by the reciprocal); exact division where a pass writes `precise`
OV float g_div(float x, float y) { return x * __builtin_amdgcn_rcpf(y); }
OV float g_sqrt(float x) { return __builtin_amdgcn_sqrtf(x); }
OV float g_rsqrt(float x) { return __builtin_amdgcn_rsqf(x); }
OV float g_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
OV float g_log2(float x) { return __builtin_amdgcn_logf(x); }
OV float g_pow(float x, float y) { return __builtin_amdgcn_exp2f(y * __builtin_amdgcn_logf(x)); }
OV float g_exp(float x) { return __builtin_amdgcn_exp2f(x * 1.44269502f); }
OV float g_mix(float x, float y, float t) { return __builtin_fmaf(t, y - x, x); }   // RADV's flrp lowering (as m4r)

// float -> fp16 bits, round to nearest even (the passes' own f2h)
static inline uint f2h_rne(float f) {
    uint b = as_uint(f); uint s = (b >> 16) & 0x8000u; int e = (int)((b >> 23) & 0xffu) - 127 + 15; uint m = b & 0x7fffffu;
    if (((b >> 23) & 0xffu) == 255u) return s | 0x7c00u | (m != 0u ? 0x200u : 0u);
    if (e >= 31) return s | 0x7c00u;
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000u; int sh = 14 - e; uint r = m >> sh; uint rem = m & ((1u << sh) - 1u); uint hf = 1u << (sh - 1);
        if (rem > hf || (rem == hf && (r & 1u) != 0u)) ++r;
        return s | r;
    }
    uint r = ((uint)e << 10) | (m >> 13); uint rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) ++r;
    return s | r;
}
static inline float h2f(uint h) { return (float)as_half((ushort)(h & 0xffffu)); }
// GLSL unpackHalf2x16 / packHalf2x16 (the hardware conversion: round to nearest even)
static inline float2 unpackH2(uint v) { return (float2)((float)as_half((ushort)(v & 0xffffu)), (float)as_half((ushort)(v >> 16))); }
static inline uint packH2hw(float x, float y) { return (uint)as_ushort((half)x) | ((uint)as_ushort((half)y) << 16); }

// ---- images: texture records (sampled descriptor at +0, storage at +32), as m4r's nn_aux.cl --------------------------------
typedef __amdgpu_texture_t tex_t;
typedef constant const char* rec_t;
#define TEX(b) ((rec_t)SLOT(b))
#define SMP(b) ((rec_t)SLOT(b))
// texelFetch / imageLoad / imageStore by texture kind (float, I = itexture/iimage, U = utexture/uimage): the hardware moves dwords,
// the descriptor's format converts
OV float4 texelFetch(rec_t rec, int2 c, int lod) {
    return __builtin_amdgcn_image_load_2d_v4f32_i32(15, c.x, c.y, *(constant const tex_t*)rec, 0, 0);
}
OV int4 texelFetchI(rec_t rec, int2 c, int lod) { return as_int4(texelFetch(rec, c, lod)); }
OV uint4 texelFetchU(rec_t rec, int2 c, int lod) { return as_uint4(texelFetch(rec, c, lod)); }
OV float4 imageLoad(rec_t rec, int2 c) { return __builtin_amdgcn_image_load_2d_v4f32_i32(15, c.x, c.y, *(constant const tex_t*)(rec + 32), 0, 0); }
OV int4 imageLoadI(rec_t rec, int2 c) { return as_int4(imageLoad(rec, c)); }
OV uint4 imageLoadU(rec_t rec, int2 c) { return as_uint4(imageLoad(rec, c)); }
OV void imageStore(rec_t rec, int2 c, float4 v) {
    __builtin_amdgcn_image_store_2d_v4f32_i32(v, 15, c.x, c.y, *(constant const tex_t*)(rec + 32), 0, 0);
}
OV void imageStoreI(rec_t rec, int2 c, int4 v) { imageStore(rec, c, as_float4(v)); }
OV void imageStoreU(rec_t rec, int2 c, uint4 v) { imageStore(rec, c, as_float4(v)); }
// GLSL pack / unpack built-ins; packHalf2x16 as RADV lowers it (v_cvt_pkrtz: round toward zero)
OV float2 unpackHalf2x16(uint v) { return unpackH2(v); }
OV uint packHalf2x16(float2 v) { return as_uint(__builtin_amdgcn_cvt_pkrtz(v.x, v.y)); }
OV float4 unpackUnorm4x8(uint v) { return (float4)((float)(v & 255u), (float)((v >> 8) & 255u), (float)((v >> 16) & 255u), (float)(v >> 24)) * (1.0f / 255.0f); }
OV float2 unpackUnorm2x16(uint v) { return (float2)((float)(v & 65535u), (float)(v >> 16)) * (1.0f / 65535.0f); }
OV uint packUnorm4x8(float4 v) {
    const uint4 q = __builtin_convertvector(__builtin_elementwise_rint(__builtin_elementwise_min(__builtin_elementwise_max(v, (float4)(0.0f)), (float4)(1.0f)) * 255.0f), uint4);
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}
// textureLod(sampler2D(tex, smp), uv, lod): every lod the passes use is 0 (one mip level); texture() in a compute shader samples lod 0
OV float4 textureLod0(rec_t rec, rec_t smp, float2 uv, float lod) {
    return __builtin_amdgcn_image_sample_lz_2d_v4f32_f32(15, uv.x, uv.y, *(constant const tex_t*)rec, *(constant const int4*)smp, false, 0, 0);
}
OV float4 texture0(rec_t rec, rec_t smp, float2 uv) { return textureLod0(rec, smp, uv, 0.0f); }

// ---- GLSL built-ins for the translated passes (tools/glsl2cl.py): overloads by type, as GLSL resolves them ------------------
#define VEC_OPS(T, S)                                                                                                            \
    OV T g_min(T x, T y) { return x < y ? x : y; } OV T g_max(T x, T y) { return x > y ? x : y; }                                \
    OV T g_min(T x, S y) { return g_min(x, (T)(y)); } OV T g_max(T x, S y) { return g_max(x, (T)(y)); }                          \
    OV T g_clamp(T x, T lo, T hi) { return g_min(g_max(x, lo), hi); } OV T g_clamp(T x, S lo, S hi) { return g_clamp(x, (T)(lo), (T)(hi)); }
VEC_OPS(int2, int) VEC_OPS(int3, int) VEC_OPS(int4, int) VEC_OPS(uint2, uint) VEC_OPS(uint3, uint) VEC_OPS(uint4, uint)
OV uint g_min(uint x, uint y) { return x < y ? x : y; } OV uint g_max(uint x, uint y) { return x > y ? x : y; }
OV uint g_clamp(uint x, uint lo, uint hi) { return g_min(g_max(x, lo), hi); }
OV int g_min(int x, int y) { return x < y ? x : y; } OV int g_max(int x, int y) { return x > y ? x : y; }
OV int g_clamp(int x, int lo, int hi) { return g_min(g_max(x, lo), hi); }
#define FVEC_OPS(T)                                                                                                              \
    OV T g_min(T x, T y) { return __builtin_elementwise_min(x, y); } OV T g_max(T x, T y) { return __builtin_elementwise_max(x, y); } \
    OV T g_min(T x, float y) { return g_min(x, (T)(y)); } OV T g_max(T x, float y) { return g_max(x, (T)(y)); }                  \
    OV T g_clamp(T x, T lo, T hi) { return g_min(g_max(x, lo), hi); } OV T g_clamp(T x, float lo, float hi) { return g_clamp(x, (T)(lo), (T)(hi)); } \
    OV T g_floor(T x) { return __builtin_elementwise_floor(x); } OV T g_ceil(T x) { return __builtin_elementwise_ceil(x); }      \
    OV T g_abs(T x) { return __builtin_elementwise_abs(x); } OV T g_fract(T x) { return x - g_floor(x); }                         \
    OV T g_roundEven(T x) { return __builtin_elementwise_rint(x); } OV T g_trunc(T x) { return __builtin_elementwise_trunc(x); }   \
    OV T g_fma(T a, T b, T c) { return __builtin_elementwise_fma(a, b, c); }                                                      \
    OV T g_mix(T x, T y, T t) { return __builtin_elementwise_fma(t, y - x, x); } OV T g_mix(T x, T y, float t) { return g_mix(x, y, (T)(t)); }
FVEC_OPS(float2) FVEC_OPS(float3) FVEC_OPS(float4)
OV float g_roundEven(float x) { return __builtin_rintf(x); }
OV float g_trunc(float x) { return __builtin_truncf(x); }
OV int g_abs(int x) { return x < 0 ? -x : x; }
OV float g_dot(float2 x, float2 y) { return x.x * y.x + x.y * y.y; }
OV float g_dot(float3 x, float3 y) { return x.x * y.x + x.y * y.y + x.z * y.z; }
OV float g_dot(float4 x, float4 y) { return x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w; }
OV float g_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
OV float g_step(float e, float x) { return x < e ? 0.0f : 1.0f; }
OV float g_smoothstep(float e0, float e1, float x) { const float t = g_clamp(g_div(x - e0, e1 - e0), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
// GLSL's NMin / NMax (spvNMin / spvNMax: the non-NaN operand)
OV float spvNMin(float x, float y) { return __builtin_fminf(x, y); } OV float spvNMax(float x, float y) { return __builtin_fmaxf(x, y); }
// constructors: vecN(scalar) splats, vecN(...) concatenates, vecN(other vector type) converts componentwise
#define CTOR4(N, T) OV T##4 N(T x) { return (T##4)(x); } OV T##4 N(T x, T y, T z, T w) { return (T##4)(x, y, z, w); }          \
    OV T##4 N(T##2 a, T##2 b) { return (T##4)(a, b); } OV T##4 N(T##3 a, T w) { return (T##4)(a, w); } OV T##4 N(T##2 a, T z, T w) { return (T##4)(a, z, w); } \
    OV T##4 N(T x, T##3 b) { return (T##4)(x, b); } OV T##4 N(T x, T##2 b, T w) { return (T##4)(x, b, w); } OV T##4 N(T x, T y, T##2 b) { return (T##4)(x, y, b); }
#define CTOR3(N, T) OV T##3 N(T x) { return (T##3)(x); } OV T##3 N(T x, T y, T z) { return (T##3)(x, y, z); } OV T##3 N(T##2 a, T z) { return (T##3)(a, z); } \
    OV T##3 N(T x, T##2 b) { return (T##3)(x, b); } OV T##3 N(T##4 v) { return v.xyz; }
OV float2 vec2(float x) { return (float2)(x); } OV float2 vec2(float x, float y) { return (float2)(x, y); } OV float2 vec2(float2 v) { return v; }
OV float2 vec2(int2 v) { return __builtin_convertvector(v, float2); } OV float2 vec2(uint2 v) { return __builtin_convertvector(v, float2); }
OV float2 vec2(float4 v) { return v.xy; } OV float2 vec2(float3 v) { return v.xy; }
CTOR3(vec3, float) OV float3 vec3(int3 v) { return __builtin_convertvector(v, float3); } OV float3 vec3(uint3 v) { return __builtin_convertvector(v, float3); }
CTOR4(vec4, float) OV float4 vec4(int4 v) { return __builtin_convertvector(v, float4); } OV float4 vec4(uint4 v) { return __builtin_convertvector(v, float4); }
OV float4 vec4(float4 v) { return v; }
OV int2 ivec2(int x) { return (int2)(x); } OV int2 ivec2(int x, int y) { return (int2)(x, y); } OV int2 ivec2(int2 v) { return v; }
OV int2 ivec2(float2 v) { return __builtin_convertvector(v, int2); } OV int2 ivec2(uint2 v) { return __builtin_convertvector(v, int2); }
OV int2 ivec2(int4 v) { return v.xy; }
CTOR3(ivec3, int) CTOR4(ivec4, int) OV int4 ivec4(float4 v) { return __builtin_convertvector(v, int4); } OV int4 ivec4(uint4 v) { return as_int4(v); }
OV int4 ivec4(int4 v) { return v; }
OV uint2 uvec2(uint x) { return (uint2)(x); } OV uint2 uvec2(uint x, uint y) { return (uint2)(x, y); } OV uint2 uvec2(uint2 v) { return v; }
OV uint2 uvec2(int2 v) { return as_uint2(v); } OV uint2 uvec2(float2 v) { return __builtin_convertvector(v, uint2); } OV uint2 uvec2(uint4 v) { return v.xy; }
CTOR3(uvec3, uint) CTOR4(uvec4, uint) OV uint4 uvec4(int4 v) { return as_uint4(v); } OV uint4 uvec4(float4 v) { return __builtin_convertvector(v, uint4); }
OV uint4 uvec4(uint4 v) { return v; }

// GLSL vector relations (bvecN = intN with -1 for true, as OpenCL's vector comparisons), mix with a bvec mask (a select), bit casts
#define REL(T, R) OV R lessThan(T x, T y) { return x < y; } OV R lessThanEqual(T x, T y) { return x <= y; } \
    OV R greaterThan(T x, T y) { return x > y; } OV R greaterThanEqual(T x, T y) { return x >= y; } \
    OV R equal(T x, T y) { return x == y; } OV R notEqual(T x, T y) { return x != y; }
REL(float2, int2) REL(float3, int3) REL(float4, int4) REL(int2, int2) REL(int3, int3) REL(int4, int4) REL(uint2, int2) REL(uint3, int3) REL(uint4, int4)
OV int2 bvec2(bool x) { return (int2)(x ? -1 : 0); } OV int2 bvec2(bool x, bool y) { return (int2)(x ? -1 : 0, y ? -1 : 0); }
OV int4 bvec4(bool x) { return (int4)(x ? -1 : 0); }
OV bool any(int2 v) { return (v.x | v.y) != 0; } OV bool all(int2 v) { return (v.x & v.y) != 0; }
OV bool any(int3 v) { return (v.x | v.y | v.z) != 0; } OV bool all(int3 v) { return (v.x & v.y & v.z) != 0; }
OV bool any(int4 v) { return (v.x | v.y | v.z | v.w) != 0; } OV bool all(int4 v) { return (v.x & v.y & v.z & v.w) != 0; }
OV float2 g_mix(float2 x, float2 y, int2 m) { return m ? y : x; } OV float3 g_mix(float3 x, float3 y, int3 m) { return m ? y : x; }
OV float4 g_mix(float4 x, float4 y, int4 m) { return m ? y : x; } OV float g_mix(float x, float y, bool m) { return m ? y : x; }
OV int2 g_mix(int2 x, int2 y, int2 m) { return m ? y : x; } OV int4 g_mix(int4 x, int4 y, int4 m) { return m ? y : x; }
OV float2 spvNMin(float2 x, float2 y) { return __builtin_elementwise_min(x, y); } OV float2 spvNMax(float2 x, float2 y) { return __builtin_elementwise_max(x, y); }
OV float3 spvNMin(float3 x, float3 y) { return __builtin_elementwise_min(x, y); } OV float3 spvNMax(float3 x, float3 y) { return __builtin_elementwise_max(x, y); }
OV float4 spvNMin(float4 x, float4 y) { return __builtin_elementwise_min(x, y); } OV float4 spvNMax(float4 x, float4 y) { return __builtin_elementwise_max(x, y); }
OV uint floatBitsToUint(float x) { return as_uint(x); } OV uint2 floatBitsToUint(float2 x) { return as_uint2(x); } OV uint4 floatBitsToUint(float4 x) { return as_uint4(x); }
OV int floatBitsToInt(float x) { return as_int(x); } OV float uintBitsToFloat(uint x) { return as_float(x); } OV float intBitsToFloat(int x) { return as_float(x); }
OV float2 uintBitsToFloat(uint2 x) { return as_float2(x); } OV float4 uintBitsToFloat(uint4 x) { return as_float4(x); }
// atomics on shared / buffer memory (tools/glsl2cl.py passes the GLSL lvalue's address), relaxed as GLSL's default
#define ATOM(N, OP) OV uint N(local uint* p, uint v) { return OP(p, v, __ATOMIC_RELAXED); } OV uint N(global uint* p, uint v) { return OP(p, v, __ATOMIC_RELAXED); } \
    OV int N(local int* p, int v) { return OP(p, v, __ATOMIC_RELAXED); } OV int N(global int* p, int v) { return OP(p, v, __ATOMIC_RELAXED); }
ATOM(atomicMin, __atomic_fetch_min) ATOM(atomicMax, __atomic_fetch_max) ATOM(atomicAdd, __atomic_fetch_add) ATOM(atomicAnd, __atomic_fetch_and)
ATOM(atomicOr, __atomic_fetch_or) ATOM(atomicExchange, __atomic_exchange_n)
// textureSize(tex, 0): the size runtime/pass.cpp writes at +64 of the texture record
OV int2 textureSize(rec_t rec, int lod) { return (int2)(((constant const int*)(rec + 64))[0], ((constant const int*)(rec + 64))[1]); }
OV int2 imageSize(rec_t rec) { return textureSize(rec, 0); }
// per-component sqrt / inversesqrt / exp2 / log2 / pow / exp / division for vectors (the hardware instructions, as the scalars)
#define FVEC_UN(F) OV float2 F(float2 v) { return (float2)(F(v.x), F(v.y)); } OV float3 F(float3 v) { return (float3)(F(v.x), F(v.y), F(v.z)); } \
    OV float4 F(float4 v) { return (float4)(F(v.x), F(v.y), F(v.z), F(v.w)); }
#define FVEC_BIN(F) OV float2 F(float2 x, float2 y) { return (float2)(F(x.x, y.x), F(x.y, y.y)); } \
    OV float3 F(float3 x, float3 y) { return (float3)(F(x.x, y.x), F(x.y, y.y), F(x.z, y.z)); } \
    OV float4 F(float4 x, float4 y) { return (float4)(F(x.x, y.x), F(x.y, y.y), F(x.z, y.z), F(x.w, y.w)); }
FVEC_UN(g_sqrt) FVEC_UN(g_rsqrt) FVEC_UN(g_exp2) FVEC_UN(g_log2) FVEC_UN(g_exp) FVEC_BIN(g_pow) FVEC_BIN(g_div)
