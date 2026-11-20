#!/usr/bin/env python3
"""Offline SR demo/validation: static camera + Halton jitter over the SDK hero render, run build/cm/sr_run,
write PNGs and compare against bicubic upscaling of the same LR frames. Usage: sr_demo.py [LRW LRH [frames [model]]]"""
import sys, subprocess, numpy as np, struct
from PIL import Image
W = int(sys.argv[1]) if len(sys.argv) > 1 else 640
H = int(sys.argv[2]) if len(sys.argv) > 2 else 360
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
model = sys.argv[4] if len(sys.argv) > 4 else None
GAMMA = 2.2
import os
SX = float(os.environ.get('SIGN_X', 1)); SY = float(os.environ.get('SIGN_Y', 1))   # rendering-convention probe
img = np.fromfile('build/hero.rgb', dtype=np.uint8).reshape(2008, 2869, 3)
x0, y0 = (2869 - 2 * W) // 2, (2008 - 2 * H) // 2
hr = (img[y0:y0 + 2 * H, x0:x0 + 2 * W].astype(np.float32) / 255.0) ** GAMMA          # linear HR ground truth
def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0: f /= b; r += f * (i % b); i //= b
    return r
def render_lr(jx, jy):
    # LR pixel i samples HR position 2i+1+2*jx (x) and 2j+1-2*jy (y, ANF y-up jitter); 2x2 box prefilter
    sx, sy = SX * 2 * jx, -SY * 2 * jy
    def shift(a, s, axis):
        i0 = int(np.floor(s)); t = s - i0
        a0 = np.roll(a, -i0, axis=axis); a1 = np.roll(a, -i0 - 1, axis=axis)
        return a0 * (1 - t) + a1 * t
    s = shift(shift(hr, sx, 1), sy, 0)
    return s.reshape(H, 2, W, 2, 3).mean(axis=(1, 3))
frames, jit = [], []
for k in range(N):
    jx, jy = halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5
    frames.append(render_lr(jx, jy).astype(np.float32)); jit.append((jx, jy))
with open('build/sr_in.bin', 'wb') as f:
    f.write(struct.pack('<3i', W, H, N))
    for k in range(N):
        f.write(struct.pack('<2f', *jit[k])); f.write(frames[k].astype('<f4').tobytes())
        f.write(np.full((H, W), 0.5, '<f4').tobytes()); f.write(np.zeros((H, W, 2), '<f4').tobytes())
cmd = ['build/cm/sr_run', 'build/sr_in.bin', 'build/sr_out.bin'] + ([model] if model else [])
print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
out = np.fromfile('build/sr_out.bin', dtype='<f4').reshape(N, 2 * H, 2 * W, 3)
def disp(a): return np.clip(a, 0, 1) ** (1 / GAMMA)
def psnr(a, b, m=8): d = disp(a)[m:-m, m:-m] - disp(b)[m:-m, m:-m]; return 10 * np.log10(1.0 / max((d ** 2).mean(), 1e-12))
def bicubic(lr): return np.asarray(Image.fromarray((disp(lr) * 255 + .5).astype(np.uint8)).resize((2 * W, 2 * H), Image.BICUBIC), dtype=np.float32) / 255.0
base = [(bicubic(frames[k]) ** GAMMA) for k in (0, N - 1)]
def best_shift(o,r):
    b=(-1,0,0)
    for dy in range(-2,3):
        for dx in range(-2,3):
            v=psnr(np.roll(o,(dy,dx),(0,1)),r)
            if v>b[0] or b[0]==-1: b=(v,dy,dx)
    return b
print('best global shift (psnr,dy,dx):',best_shift(out[-1],hr))
print('frame  SR(dB)   bicubic-of-same-LR(dB)')
for k in range(N):
    print('%4d  %7.2f  %7.2f' % (k, psnr(out[k], hr), psnr(bicubic(frames[k]) ** GAMMA, hr)))
Image.fromarray((disp(out[-1]) * 255 + .5).astype(np.uint8)).save('build/sr_last.png')
Image.fromarray((disp(hr) * 255 + .5).astype(np.uint8)).save('build/sr_gt.png')
Image.fromarray((disp(base[1]) * 255 + .5).astype(np.uint8)).save('build/sr_bicubic.png')
print('saved build/sr_last.png (SR, last frame), sr_gt.png, sr_bicubic.png')
