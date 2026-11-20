#!/usr/bin/env python3
"""Harness fixture (same file layout as m4r's tests/scene_motion.py / scene_fg.py) from the test image: a constant pan of the
image, Halton-jittered 2x-downscaled inputs, low-res motion vectors (pixels, previous - current), true in-between frames.
usage: make_hero_fixture.py <out dir> [frames] [step_x step_y (HR px per frame)]"""
import os, sys, numpy as np
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SX = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0; SY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
W, H = 1280, 720
os.makedirs(out, exist_ok=True)
img = np.fromfile('build/hero.rgb', dtype=np.uint8).reshape(2008, 2869, 3)
big = (img.astype(np.float32) / 255.0) ** 2.2
x0, y0 = (2869 - W) // 2 - 40, (2008 - H) // 2
def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0: f /= b; r += f * (i % b); i //= b
    return r - 0.5
def hr(t):   # HR frame with the content moved by t*(SX,SY): bilinear sample of the big image
    sx, sy = SX * t, SY * t
    X = x0 + np.arange(W)[None, :] - sx; Y = y0 + np.arange(H)[:, None] - sy
    ix, iy = np.floor(X).astype(int), np.floor(Y).astype(int); fx, fy = (X - ix)[..., None], (Y - iy)[..., None]
    g = lambda yy, xx: big[np.clip(yy, 0, 2007), np.clip(xx, 0, 2868)]
    return (g(iy, ix) * (1 - fx) + g(iy, ix + 1) * fx) * (1 - fy) + (g(iy + 1, ix) * (1 - fx) + g(iy + 1, ix + 1) * fx) * fy
def lr(t, jx, jy):   # 2x2 box prefilter of the HR frame sampled at 2*j offset (j = sample offset in LR px, image coords)
    h = hr(t)
    def sh(a, s, ax):
        i0 = int(np.floor(s)); k = s - i0; return np.roll(a, -i0, ax) * (1 - k) + np.roll(a, -i0 - 1, ax) * k
    return sh(sh(h, 2 * jx, 1), 2 * jy, 0).reshape(H // 2, 2, W // 2, 2, 3).mean(axis=(1, 3))
f16 = lambda a: a.astype('<f2')
for t in range(N):
    jx, jy = halton(t + 1, 2), halton(t + 1, 3)
    l = lr(t, jx, jy); rgba = np.concatenate([l, np.ones((H // 2, W // 2, 1), np.float32)], -1)
    f16(rgba).tofile(f'{out}/input-{t:02}.rgba16f')
    np.full((H // 2, W // 2), 0.5, '<f4').tofile(f'{out}/depth-{t:02}.r32f')
    mv = np.zeros((H // 2, W // 2, 2), np.float32); mv[..., 0] = -SX / 2; mv[..., 1] = -SY / 2
    f16(mv).tofile(f'{out}/mvlo-{t:02}.rg16f')
    f16(np.concatenate([np.zeros((H, W, 1), np.float32) + -SX, np.zeros((H, W, 1), np.float32) - SY], -1)).tofile(f'{out}/mv-{t:02}.rg16f')
    if t: f16(np.concatenate([hr(t - 0.5), np.ones((H, W, 1), np.float32)], -1)).tofile(f'{out}/mid-{t:02}.rgba16f')
f16(np.concatenate([hr(N - 1), np.ones((H, W, 1), np.float32)], -1)).tofile(f'{out}/reference.rgba16f')
print('wrote', out)
