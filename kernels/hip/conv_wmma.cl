// anf4r: OpenCL C port of kernels/conv_wmma.comp (int8 3x3 convolution as an implicit GEMM on the RDNA3 matrix units:
// v_wmma_i32_16x16x16_iu8). Bit-identical to fg_conv_i8: int32 accumulation is exact in any order, the epilogue is the same
// per-channel fma / clamp / rndaz. One wave32 per workgroup computes 32 output pixels (one row segment) x 32 output channels as
// 2 x 2 accumulator tiles, like the GLSL's STYLE 0 (its other variants are the same sums in another order: the runtime always
// builds variant 0 on HIP).
// WMMA operands (wave32, gfx11): lane l holds row (l & 15) of A (16 int8 of K) and column (l & 15) of B (16 int8 of K), lanes
// 16-31 repeating 0-15; accumulator element i of lane l is row 2 * i + (l >> 4), column l & 15. Each lane loads its own A row
// (a pixel's 16 channels, or the padding value) and B column (an output channel's 16 weights) straight from memory.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 32
#define LY 1
#include "anf_port.h"

#define IN_W SPECD(0, 1)
#define IN_H SPECD(1, 1)
#define ICW SPECD(2, 32)
#define INSTRIDE SPECD(3, 32)
#define OC SPECD(4, 32)
#define STRIDE SPECD(5, 1)
#define OUT_W SPECD(6, 1)
#define OUT_H SPECD(7, 1)
#define OUT_MODE SPECD(8, 0)
#define LO SPECD(9, -128)
#define HI SPECD(10, 127)
#define PADV SPECD(11, 0)
#define OUT_OFF SPECD(12, 0)
#define FLOW_ZP SPECD(13, 0)
#define MASK_ZP SPECD(14, 0)
#define RELU SPECD(15, 1)
#define FLOW_PITCH SPECD(16, 1)
#define MASK_PITCH SPECD(17, 1)

static inline float rndaz(float x) { return x >= 0.0f ? g_floor(x + 0.5f) : g_ceil(x - 0.5f); }

ANF_KERNEL(anf_main) {
    global const uchar* inp = CBUF(uchar, 0);
    global const uchar* wt = CBUF(uchar, 1);
    global const float* biasf = CBUF(float, 2);
    global const float* scale = CBUF(float, 3);
    const int ox0 = (int)WG_X * 32, oy = (int)WG_Y, n0 = (int)WG_Z * 32;
    const int l = (int)LID_X, r = l & 15;
    const int NKC = ICW / 16;
    const uint padw = (uint)(PADV & 255) * 0x01010101u;
    int8 c[2][2];
    for (int t = 0; t < 2; ++t) for (int j = 0; j < 2; ++j) c[t][j] = (int8)(0);
    for (int tap = 0; tap < 9; ++tap) {
        const int ky = tap / 3, kx = tap - ky * 3;
        const int iy = oy * STRIDE + ky - 1;
        for (int kc = 0; kc < NKC; ++kc) {
            int4 A[2], B[2];
            for (int t = 0; t < 2; ++t) {
                const int ix = (ox0 + t * 16 + r) * STRIDE + kx - 1;
                if (iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W) {
                    global const uint2* p = (global const uint2*)(inp + (size_t)(iy * IN_W + ix) * INSTRIDE + kc * 16);
                    const uint2 lo = p[0], hi = p[1];
                    A[t] = (int4)((int)lo.x, (int)lo.y, (int)hi.x, (int)hi.y);
                } else {
                    A[t] = (int4)((int)padw);
                }
            }
            for (int j = 0; j < 2; ++j)
                B[j] = *(global const int4*)(wt + (size_t)(n0 + j * 16 + r) * 9 * ICW + tap * ICW + kc * 16);
            for (int t = 0; t < 2; ++t)
                for (int j = 0; j < 2; ++j) c[t][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A[t], true, B[j], c[t][j], false);
        }
    }
    // epilogue: lane l holds channel n0 + j*16 + (l & 15) of pixels ox0 + t*16 + 2*i + (l >> 4)
    for (int t = 0; t < 2; ++t)
        for (int j = 0; j < 2; ++j) {
            const int cl = j * 16 + r, ch = n0 + cl;
            const float s = scale[ch], b = biasf[ch];
            for (int i = 0; i < 8; ++i) {
                const int ox = ox0 + t * 16 + 2 * i + (l >> 4);
                if (ox >= OUT_W) continue;
                const float f = g_fma(s, (float)c[t][j][i], b);
                const int pix = oy * OUT_W + ox;
                if (OUT_MODE == 2) {
                    const float lo = RELU != 0 ? (float)(j == 0 ? FLOW_ZP : MASK_ZP) : 0.0f;
                    if (j == 0) {
                        const int pos = cl >> 2, k = cl & 3;
                        const float g = g_min(g_max(rndaz(f), lo), 255.0f);
                        BUF(uchar, 6)[((2 * oy + (pos >> 1)) * FLOW_PITCH + 2 * ox + (pos & 1)) * 4 + k] = (uchar)(uint)g;
                    } else if (cl == 16) {
                        BUF(uchar, 7)[oy * MASK_PITCH + ox] = (uchar)(uint)g_min(g_max(rndaz(f), lo), 255.0f);
                    }
                } else if (OUT_MODE == 0) {
                    const float g = g_min(g_max(f, (float)LO), (float)HI);
                    BUF(uchar, 4)[(size_t)pix * OC + ch] = (uchar)(((int)rndaz(g) + OUT_OFF) & 255);
                } else {
                    BUF(ushort, 5)[(size_t)pix * OC + ch] = (ushort)f2h_rne(g_max(f, 0.0f));
                }
            }
        }
}
