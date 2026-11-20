#!/usr/bin/env python3
"""Decode an anf4r input capture (anf4r/data/dump/), print statistics of every input the game feeds SR, write viewable PNGs and
a sr_run input stream. usage: analyze_dump.py <dump dir> [out prefix]"""
import sys, struct, numpy as np
from PIL import Image
d = sys.argv[1].rstrip('/'); pre = sys.argv[2] if len(sys.argv) > 2 else 'build/dump'
rows = [l.split() for l in open(d + '/manifest.txt') if l.strip()]
def half(b, shape): return np.frombuffer(b, '<f2').astype(np.float32).reshape(shape)
def unpack_r11g11b10(w):
    def f(v, mb): e = (v >> mb) & 31; m = v & ((1 << mb) - 1); return np.where(e == 0, m * 2.0 ** -14 / (1 << mb), (1 + m / (1 << mb)) * 2.0 ** (e.astype(np.float32) - 15))
    return np.stack([f(w & 2047, 6), f((w >> 11) & 2047, 6), f((w >> 22) & 1023, 5)], -1).astype(np.float32)
def load_color(b, fmt, w, h):
    if fmt == 97: return half(b, (h, w, 4))[..., :3]
    if fmt == 122: return unpack_r11g11b10(np.frombuffer(b, '<u4').reshape(h, w))
    if fmt in (37, 43): a = np.frombuffer(b, np.uint8).reshape(h, w, 4)[..., :3] / 255.0; return a ** 2.2 if fmt == 43 else a
    if fmt == 44: return (np.frombuffer(b, np.uint8).reshape(h, w, 4)[..., 2::-1] / 255.0)
    if fmt == 64: v = np.frombuffer(b, '<u4').reshape(h, w); return np.stack([(v & 1023), (v >> 10) & 1023, (v >> 20) & 1023], -1) / 1023.0
    raise SystemExit(f'unsupported colour format {fmt}')
def load_depth(b, fmt, w, h):
    if fmt in (126, 100): return np.frombuffer(b, '<f4').reshape(h, w)
    if fmt in (125, 129): return (np.frombuffer(b, '<u4').reshape(h, w) & 0xFFFFFF) / 16777215.0
    raise SystemExit(f'unsupported depth format {fmt}')
def load_mv(b, fmt, w, h):
    if fmt == 83: return half(b, (h, w, 2))
    if fmt == 103: return np.frombuffer(b, '<f4').reshape(h, w, 2)
    raise SystemExit(f'unsupported motion format {fmt}')
disp = lambda a: (np.clip(a / (1 + a), 0, 1) ** (1 / 2.2) * 255 + .5).astype(np.uint8)
frames = []
print('frame  jitter(raw)       jitter(sent to ANF)  mvscale     reset depthRev | colour min/mean/p99/max | depth min/p50/max | mv|.| mean/p99 (px) nonzero')
for r in rows:
    i, cw, ch, cf, dw, dh, df, mw, mh, mf, cw_, ch_ = map(int, r[:12]); jx, jy, sx, sy, ajx, ajy = map(float, r[12:18]); reset, drev = int(r[18]), int(r[19]); ow, oh = int(r[20]), int(r[21])
    col = load_color(open(f'{d}/frame_{i:03d}.color.bin', 'rb').read(), cf, cw, ch)[:ch_, :cw_]
    dep = load_depth(open(f'{d}/frame_{i:03d}.depth.bin', 'rb').read(), df, dw, dh)[:ch_, :cw_]
    mv = load_mv(open(f'{d}/frame_{i:03d}.motion.bin', 'rb').read(), mf, mw, mh)[:ch_, :cw_]
    mvpx = np.abs(mv * np.array([sx, sy])).max(-1)
    print(f'{i:4d}  ({jx:+.3f},{jy:+.3f})  ({ajx:+.3f},{ajy:+.3f})    ({sx:g},{sy:g})  {reset}     {drev}      | {col.min():.3g}/{col.mean():.3g}/{np.percentile(col,99):.3g}/{col.max():.3g} | {dep.min():.4g}/{np.median(dep):.4g}/{dep.max():.4g} | {mvpx.mean():.3f}/{np.percentile(mvpx,99):.2f}  {100*(mvpx>1e-4).mean():.0f}%')
    frames.append((col.astype(np.float32), dep.astype(np.float32), mv.astype(np.float32), (jx, jy, sx, sy, ajx, ajy, reset, drev)))
    if i in (0, len(rows) - 1):
        Image.fromarray(disp(col)).save(f'{pre}_color_{i}.png')
        dv = (dep - dep.min()) / max(np.ptp(dep), 1e-9); Image.fromarray((np.sqrt(dv) * 255).astype(np.uint8)).save(f'{pre}_depth_{i}.png')
        m = np.clip(mv * np.array([sx, sy]) / 8 * .5 + .5, 0, 1); Image.fromarray((np.dstack([m, np.full(m.shape[:2], .5)]) * 255).astype(np.uint8)).save(f'{pre}_mv_{i}.png')
# sr_run stream: colour as-is, depth as-is, motion converted to uv units (curr - prev) as sr_run expects, jitter as sent to ANF
H, W = frames[0][0].shape[:2]
with open(f'{pre}_sr_in.bin', 'wb') as f:
    f.write(struct.pack('<3i', W, H, len(frames)))
    for col, dep, mv, (jx, jy, sx, sy, ajx, ajy, reset, drev) in frames:
        mvuv = -mv * np.array([sx, sy]) / np.array([W, H])
        d = dep if drev else 1 - dep           # sr_run assumes reverse-Z
        f.write(struct.pack('<2f', ajx, ajy)); f.write(col.astype('<f4').tobytes()); f.write(d.astype('<f4').tobytes()); f.write(mvuv.astype('<f4').tobytes())
print(f'wrote {pre}_sr_in.bin ({len(frames)} frames {W}x{H}); run: build/cm/sr_run {pre}_sr_in.bin {pre}_sr_out.bin')
