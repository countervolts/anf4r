#!/usr/bin/env python3
"""A plain temporal accumulation upscaler (jittered bilinear sample + neighbourhood-clamped exponential history) on a capture, for a
stability/sharpness yardstick next to ANF. usage: taau_reference.py <prefix> [alpha]"""
import sys, struct, numpy as np
pre = sys.argv[1]; alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9; K = 24
raw = open(pre + '_sr_in.bin', 'rb').read(); W, H, N = struct.unpack('<3i', raw[:12]); fs = 8 + 4 * (W * H * 3 + W * H + W * H * 2)
fr = []
for i in range(N):
    o = 12 + i * fs; j = struct.unpack('<2f', raw[o:o + 8]); o += 8
    fr.append((j, np.frombuffer(raw[o:o + 4 * W * H * 3], '<f4').reshape(H, W, 3)))
# jitters in the file are ANF's (y up); the image-coordinate sample offset is (jx, -jy)
HW, HH = 2 * W, 2 * H
xs = (np.arange(HW) + 0.5) / 2 - 0.5; ys = (np.arange(HH) + 0.5) / 2 - 0.5
def bil(img, u, v):
    u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001); x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int); fx = (u - x0)[..., None]; fy = (v - y0)[..., None]
    return (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy
hist = None; outs = []
for t, ((jx, jy), col) in enumerate(fr):
    sx, sy = jx, -jy
    U, V = np.meshgrid(xs - sx, ys - sy); cur = bil(col, U, V)
    # neighbourhood min/max from the 3x3 LR window around each HR pixel's source position
    mn = np.full_like(cur, 1e9); mx = np.full_like(cur, -1e9)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            n = bil(col, np.clip(np.round(U) + dx, 0, W - 1), np.clip(np.round(V) + dy, 0, H - 1)); mn = np.minimum(mn, n); mx = np.maximum(mx, n)
    hist = cur if hist is None else np.clip(hist, mn, mx) * alpha + cur * (1 - alpha)
    outs.append(hist.copy())
out = np.clip(np.stack(outs[-K:]), 0, 1)
fl = np.mean([np.abs(out[t] - out[t - 1]).mean() for t in range(1, K)])
g = out.mean(-1); lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean()
n8 = (K // 8) * 8; q = g[K - n8:].reshape(n8 // 8, 8, *g.shape[1:]); tv = q.reshape(n8, *g.shape[1:]).var(0); m = tv > 1e-5; leak = float((q.mean(0).var(0)[m] / tv[m]).mean())
print(f'plain TAAU (alpha {alpha}) flicker {fl*1e3:7.3f}e-3   sharpness {lap*1e3:7.3f}e-3   phase-leak {leak:.3f}')
