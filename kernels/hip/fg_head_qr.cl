// anf4r: OpenCL C port of kernels/fg_head_qr.comp (QR net head: int8 conv, requant to uint8 with the output zero-point).
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#include "anf_port.h"

#define IN_W SPECD(0, 1)
#define IN_H SPECD(1, 1)
#define MODE SPECD(2, 0)
#define OUT_STRIDE_W SPECD(3, 1)
#define FLOW_ZP SPECD(4, 0)
#define MASK_ZP SPECD(5, 0)
#define RELU SPECD(6, 1)

static inline float rndaz(float x) { return x >= 0.0f ? g_floor(x + 0.5f) : g_ceil(x - 0.5f); }

static int acc(constant const Args* a, int ox, int oy, int oc) {
    global const uint* inp = CBUF(uint, 0);
    global const uint* wt = CBUF(uint, 1);
    int s = 0;
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = oy + ky - 1;
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = ox + kx - 1;
            if (iy < 0 || iy >= IN_H || ix < 0 || ix >= IN_W) {   // pad = 0x80 (-128)
                const int wb = ((oc * 3 + ky) * 3 + kx) * 8;
                for (int w = 0; w < 8; ++w) { const int q = (int)wt[wb + w]; s -= 128 * (bfe_i(q, 0, 8) + bfe_i(q, 8, 8) + bfe_i(q, 16, 8) + bfe_i(q, 24, 8)); }
                continue;
            }
            const int ib = (iy * IN_W + ix) * 8, wb = ((oc * 3 + ky) * 3 + kx) * 8;
            for (int w = 0; w < 8; ++w) {
                const int p = (int)inp[ib + w], q = (int)wt[wb + w];
                s += bfe_i(p, 0, 8) * bfe_i(q, 0, 8) + bfe_i(p, 8, 8) * bfe_i(q, 8, 8) + bfe_i(p, 16, 8) * bfe_i(q, 16, 8) + bfe_i(p, 24, 8) * bfe_i(q, 24, 8);
            }
        }
    }
    return s;
}

static uchar q8(constant const Args* a, int ox, int oy, int oc, int zp) {
    float f = g_fma(CBUF(float, 3)[oc], (float)acc(a, ox, oy, oc), CBUF(float, 2)[oc]);
    f = rndaz(f);
    f = g_min(g_max(f, RELU != 0 ? (float)zp : 0.0f), 255.0f);
    return (uchar)(uint)f;
}

ANF_KERNEL(anf_main) {
    global uchar* outp = BUF(uchar, 4);
    const int x = (int)GID_X, y = (int)GID_Y, z = (int)GID_Z;
    if (x >= IN_W || y >= IN_H) return;
    if (MODE == 0) {
        if (z >= 4) return;
        const int px = x * 2 + (z & 1), py = y * 2 + (z >> 1);
        for (int k = 0; k < 4; ++k) outp[(py * OUT_STRIDE_W + px) * 4 + k] = q8(a, x, y, z * 4 + k, FLOW_ZP);
    } else {
        outp[y * OUT_STRIDE_W + x] = q8(a, x, y, 16, MASK_ZP);
    }
}
