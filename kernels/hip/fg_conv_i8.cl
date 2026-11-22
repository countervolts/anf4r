// anf4r: OpenCL C port of kernels/fg_conv_i8.comp (int8 NHWC 3x3 conv, per-channel requant). See the .comp for the math.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#include "anf_port.h"

#define IN_W SPECD(0, 1)
#define IN_H SPECD(1, 1)
#define IC SPECD(2, 4)
#define OC SPECD(3, 1)
#define STRIDE SPECD(4, 1)
#define OUT_W SPECD(5, 1)
#define OUT_H SPECD(6, 1)
#define OUT_MODE SPECD(7, 0)
#define LO SPECD(8, 0)
#define HI SPECD(9, 127)
#define PADV SPECD(10, 0)
#define OUT_OFF SPECD(11, 0)
#define INSTRIDE SPECD(12, 0)

static inline float rndaz(float x) { return x >= 0.0f ? g_floor(x + 0.5f) : g_ceil(x - 0.5f); }

static int convAcc(constant const Args* a, int ox, int oy, int oc) {
    global const uint* inp = CBUF(uint, 0);
    global const uint* wt = CBUF(uint, 1);
    int acc = 0;
    const int icw = IC / 4;
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = oy * STRIDE + ky - 1;
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = ox * STRIDE + kx - 1;
            if (iy < 0 || iy >= IN_H || ix < 0 || ix >= IN_W) {
                if (PADV != 0) {
                    const int wb = ((oc * 3 + ky) * 3 + kx) * icw;
                    for (int w = 0; w < icw; ++w) {
                        const int b = (int)wt[wb + w];
                        acc += PADV * (bfe_i(b, 0, 8) + bfe_i(b, 8, 8) + bfe_i(b, 16, 8) + bfe_i(b, 24, 8));
                    }
                }
                continue;
            }
            const int ib = (iy * IN_W + ix) * (INSTRIDE == 0 ? icw : INSTRIDE / 4);
            const int wb = ((oc * 3 + ky) * 3 + kx) * icw;
            for (int w = 0; w < icw; ++w) {
                const int x = (int)inp[ib + w], b = (int)wt[wb + w];
                acc += bfe_i(x, 0, 8) * bfe_i(b, 0, 8) + bfe_i(x, 8, 8) * bfe_i(b, 8, 8) + bfe_i(x, 16, 8) * bfe_i(b, 16, 8) +
                       bfe_i(x, 24, 8) * bfe_i(b, 24, 8);
            }
        }
    }
    return acc;
}

ANF_KERNEL(anf_main) {
    global const float* biasf = CBUF(float, 2);
    global const float* scale = CBUF(float, 3);
    const int ox = (int)GID_X, oy = (int)GID_Y, z = (int)GID_Z;
    if (ox >= OUT_W || oy >= OUT_H) return;
    if (OUT_MODE == 0) {
        if (z >= OC) return;
        float f = g_fma(scale[z], (float)convAcc(a, ox, oy, z), biasf[z]);
        f = g_min(g_max(f, (float)LO), (float)HI);
        BUF(char, 5)[(oy * OUT_W + ox) * OC + z] = (char)((int)rndaz(f) + OUT_OFF);
    } else {
        const int c0 = z * 2;
        if (c0 >= OC) return;
        float f0 = g_fma(scale[c0], (float)convAcc(a, ox, oy, c0), biasf[c0]);
        float f1 = g_fma(scale[c0 + 1], (float)convAcc(a, ox, oy, c0 + 1), biasf[c0 + 1]);
        f0 = g_max(f0, 0.0f); f1 = g_max(f1, 0.0f);
        BUF(uint, 4)[((oy * OUT_W + ox) * OC + c0) / 2] = f2h_rne(f0) | (f2h_rne(f1) << 16);
    }
}
