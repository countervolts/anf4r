// anf4r: OpenCL C port of kernels/fg_head_lr.comp (LR net head: fp16 x fp16, fp32 sequential sums, flow depth-to-space + mask).
// SPDX-License-Identifier: GPL-3.0-only
#define LX 68
#define LY 1
#include "anf_port.h"

#define IN_W SPECD(0, 1)
#define IN_H SPECD(1, 1)

static inline float hval(global const uint* b, int idx) { const float2 h = unpackH2(b[idx >> 1]); return (idx & 1) ? h.y : h.x; }

static float conv(constant const Args* a, int ox, int oy, int oc) {
    global const uint* inp = CBUF(uint, 0);
    global const uint* wt = CBUF(uint, 1);
    float acc = 0.0f;
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = oy + ky - 1;
        if (iy < 0 || iy >= IN_H) continue;
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = ox + kx - 1;
            if (ix < 0 || ix >= IN_W) continue;
            const int ib = (iy * IN_W + ix) * 64, wb = ((oc * 3 + ky) * 3 + kx) * 64;
            for (int c = 0; c < 64; ++c) { const float p = hval(inp, ib + c) * hval(wt, wb + c); acc += p; }
        }
    }
    return acc + CBUF(float, 2)[oc];
}

ANF_KERNEL(anf_main) {
    local float res[68];
    const int lid = (int)LID_X, qi = lid / 17, c = lid - qi * 17;
    const int total = IN_W * IN_H, q = (int)WG_X * 4 + qi;
    const bool valid = q < total;
    const int x = valid ? q % IN_W : 0, y = valid ? q / IN_W : 0;
    res[lid] = valid ? conv(a, x, y, c) : 0.0f;
    LDS_BARRIER();
    if (valid && c < 16 && (c & 1) == 0) {
        const int pos = c >> 2, k = c & 3;
        const int px = x * 2 + (pos & 1), py = y * 2 + (pos >> 1);
        BUF(uint, 3)[((py * 2 * IN_W + px) * 4 + k) / 2] = f2h_rne(res[lid]) | (f2h_rne(res[lid + 1]) << 16);
    }
    if (c == 16 && (qi & 1) == 0 && valid)
        BUF(uint, 4)[q / 2] = f2h_rne(res[lid]) | (f2h_rne((q + 1 < total) ? res[lid + 17] : 0.0f) << 16);
}
