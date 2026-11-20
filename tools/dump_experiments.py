#!/usr/bin/env python3
"""Offline SR experiments on a game capture (tools/analyze_dump.py output <prefix>_sr_in.bin): jitter sign, colour encoding, MV sign.
Metrics on the last frames (static-ish camera): flicker = mean |out_t - out_{t-1}| in display space (lower = more stable), sharpness = mean |laplacian|.
usage: dump_experiments.py <prefix>"""
import sys, struct, subprocess, os, numpy as np
pre = sys.argv[1]; K = 24
raw = open(pre + '_sr_in.bin', 'rb').read(); W, H, N = struct.unpack('<3i', raw[:12]); fs = 8 + 4 * (W * H * 3 + W * H + W * H * 2)
frames = []
for i in range(N):
    o = 12 + i * fs; j = struct.unpack('<2f', raw[o:o + 8]); o += 8
    col = np.frombuffer(raw[o:o + 4 * W * H * 3], '<f4').reshape(H, W, 3); o += 4 * W * H * 3
    dep = np.frombuffer(raw[o:o + 4 * W * H], '<f4').reshape(H, W); o += 4 * W * H
    mv = np.frombuffer(raw[o:o + 8 * W * H], '<f4').reshape(H, W, 2); frames.append((j, col, dep, mv))
def run(name, jx=1, jy=1, enc='asis', mvs=1, extra_exp=1.0, depth_mode='asis', phase=0, model=None, pexp=1.0, stab=0.0):
    with open(pre + '_exp_in.bin', 'wb') as f:
        f.write(struct.pack('<3i', W, H, N))
        for (j, col, dep, mv) in frames:
            c = col
            if enc == 'srgb': c = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4)
            elif enc == 'g22': c = col ** 2.2
            c = c * extra_exp
            d = dep if depth_mode == 'asis' else (np.clip(dep * 20, 0, 1))      # 'stretch': spread the squashed reverse-Z depth
            f.write(struct.pack('<2f', j[0] * jx, j[1] * jy)); f.write(c.astype('<f4').tobytes()); f.write(d.astype('<f4').tobytes()); f.write((mv * mvs).astype('<f4').tobytes())
    env = dict(os.environ, ANF_SAVE_LAST=str(K), ANF_PHASE_MODE=str(phase), ANF_PREEXP=str(pexp), ANF_STAB=str(stab))
    subprocess.run(['build/cm/sr_run', pre + '_exp_in.bin', pre + '_exp_out.bin'] + ([model] if model else []), env=env, check=True, capture_output=True)
    out = np.fromfile(pre + '_exp_out.bin', '<f4').reshape(K, 2 * H, 2 * W, 3)
    if enc == 'srgb': out = np.where(out <= 0.0031308, out * 12.92, 1.055 * np.maximum(out, 0) ** (1 / 2.4) - 0.055)
    elif enc == 'g22': out = np.maximum(out, 0) ** (1 / 2.2)
    out = np.clip(out, 0, 1)
    fl = np.mean([np.abs(out[t] - out[t - 1]).mean() for t in range(1, K)])
    g = out.mean(-1); lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean()
    # bicubic-style reference flicker: the LR jittered frames themselves, nearest-upsampled
    # phase leakage: share of each pixel's temporal variance explained by the jitter phase (period 8) - jitter showing through in the output
    gg = out.mean(-1); n8 = (K // 8) * 8; q = gg[K - n8:].reshape(n8 // 8, 8, *gg.shape[1:]); pm = q.mean(0); tv = q.reshape(n8, *gg.shape[1:]).var(0); pv = pm.var(0)
    m = tv > 1e-5; leak = float((pv[m] / tv[m]).mean())
    print(f'{name:38s} flicker {fl*1000:7.3f}e-3   sharpness {lap*1000:7.3f}e-3   phase-leak {leak:.3f}')
    return out
lr = np.stack([f[1] for f in frames[-K:]]); lrf = np.mean([np.abs(lr[t] - lr[t - 1]).mean() for t in range(1, K)])
_g = lr.mean(-1); _q = _g.reshape(K // 8, 8, *_g.shape[1:]); _tv = _q.reshape(K, *_g.shape[1:]).var(0); _m = _tv > 1e-5; print('input LR phase-leak %.3f' % float((_q.mean(0).var(0)[_m] / _tv[_m]).mean()))
print(f'(input LR frames flicker {lrf*1000:.3f}e-3 for reference; capture {W}x{H}, {N} frames)')
if len(sys.argv) > 2 and sys.argv[2] == 'phases':
    for ph in (0, 8, 9, 10, 11): run(f'phase layout {ph}' + {0: ' (8x8 grid, current)', 8: ' (Halton2,3 idx 1..64)', 9: ' (Halton2,3 idx 0..63)', 10: ' (Halton3,2 idx 1..64)', 11: ' (Halton3,2 idx 0..63)'}[ph], phase=ph)
elif len(sys.argv) > 2 and sys.argv[2] == 'stab':
    for e in (1.0, 4.0):
        for b in (0.0, 0.5, 0.75, 0.875): run(f'pre_exposure {e}, stabiliser beta {b}', pexp=e, stab=b)
elif len(sys.argv) > 2 and sys.argv[2] == 'exposure':
    for e in (8.0, 16.0, 32.0, 64.0): run(f'pre_exposure {e}', pexp=e)
    for e in (4.0, 16.0): run(f'pre_exposure {e} + gamma2.2 decoded', pexp=e, enc='g22')
elif len(sys.argv) > 2 and sys.argv[2] == 'models':
    run('invpow2 model (current)'); run('prod-7p1 (float scales)', model='data/modelzoo/aiff-sr-v2s-prod-7p1')
elif len(sys.argv) > 2 and sys.argv[2] == 'signs':
    for jx in (1, -1):
        for jy in (1, -1): run(f'jitter x{jx:+d} y{jy:+d} (colour as-is)', jx, jy)
else:
    run('baseline (as sent)')
    run('colour as sRGB->linear', enc='srgb')
    run('colour as gamma2.2->linear', enc='g22')
    run('mv sign flipped', mvs=-1)
    run('depth stretched x20', depth_mode='stretch')
