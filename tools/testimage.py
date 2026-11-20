#!/usr/bin/env python3
"""Creates build/hero.rgb (2869x2008 raw RGB8), the scene the SR/FG tests and fixtures translate and jitter.
Default: a procedural image (fractal value noise + hard-edged shapes) so nothing third-party is needed.
HERO_IMAGE=/path/to/any/image.png uses your own picture instead (resized to 2869x2008). Quality numbers in docs/ were measured with a
real game render, so they differ on the procedural image. usage: testimage.py [out]"""
import os, sys
import numpy as np
from PIL import Image
out = sys.argv[1] if len(sys.argv) > 1 else 'build/hero.rgb'
W, H = 2869, 2008
if os.environ.get('HERO_IMAGE'):
    im = np.asarray(Image.open(os.environ['HERO_IMAGE']).convert('RGB').resize((W, H), Image.LANCZOS), np.uint8)
else:
    rng = np.random.default_rng(5); chans = []
    for c in range(3):
        acc = np.zeros((H, W), np.float32); amp, tot = 1.0, 0.0
        for o in range(8):
            cell = max(2, 256 >> o); gh, gw = H // cell + 2, W // cell + 2
            lat = rng.random((gh, gw)).astype(np.float32)
            ys = np.arange(H) / cell; xs = np.arange(W) / cell; y0 = ys.astype(int); x0 = xs.astype(int); fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
            fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
            a = lat[y0][:, x0]; b = lat[y0][:, x0 + 1]; cc = lat[y0 + 1][:, x0]; d = lat[y0 + 1][:, x0 + 1]
            acc += amp * ((a + (b - a) * fx) * (1 - fy) + (cc + (d - cc) * fx) * fy); tot += amp; amp *= 0.55
        chans.append(0.1 + 0.8 * acc / tot)
    img = np.stack(chans, -1)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(140):   # hard edges: rectangles and discs
        cx, cy, r = rng.integers(0, W), rng.integers(0, H), rng.integers(12, 90); col = rng.random(3)
        m = ((np.abs(xx - cx) < r) & (np.abs(yy - cy) < r * 0.7)) if i % 2 else (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r)
        img[m] = col
    im = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
os.makedirs(os.path.dirname(out) or '.', exist_ok=True); im.tofile(out); print('wrote', out, im.shape)
