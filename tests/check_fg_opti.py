#!/usr/bin/env python3
"""FG through OptiScaler under Proton: usage check_fg_opti.py <app dir> <frames> <fg fixture dir with mid-NN.rgba16f> [w h]
Checks the real frame is put back bit-exactly and reports how close each generated frame is to the true in-between frame
(vs. plain averaging of the neighbours and vs. repeating the current frame)."""
import re, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../../m4r/tests') if False else None
app, frames, mids = sys.argv[1].rstrip('/'), int(sys.argv[2]), sys.argv[3]
def srgb(x):
    x = np.clip(np.nan_to_num(x), 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
def psnr(a, b): return 10 * np.log10(1 / max(np.mean((srgb(a) - srgb(b)) ** 2), 1e-12))
log = open(app + '/harness.log').read()
presents = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r'frame (\d+): Present .* presents (\d+)', log)}
ld = lambda p: np.fromfile(p, dtype=np.float16).reshape(-1, 4)[:, :3].astype(np.float32)
twice = 0
for t in range(frames):
    p = presents.get(t)
    if p != 2: print(f'frame {t}: presents {p}'); continue
    twice += 1
    out, prev = ld(f'{app}/out/out-{t:02d}.rgba16f'), ld(f'{app}/out/out-{t - 1:02d}.rgba16f')
    real, gen = ld(f'{app}/out/bbk1-{t:02d}.rgba16f'), ld(f'{app}/out/bbk-{t:02d}.rgba16f')
    exact = np.array_equal(real.view(np.uint32), out.view(np.uint32))
    try: mid = ld(f'{mids}/mid-{t:02d}.rgba16f'); m = f'vs true mid: gen {psnr(gen, mid):.2f} dB, average {psnr((prev + out) / 2, mid):.2f}, repeat-current {psnr(out, mid):.2f}'
    except Exception: m = f'gen vs average {psnr(gen, (prev + out) / 2):.1f} dB, vs current {psnr(gen, out):.1f} dB'
    print(f'frame {t}: presented twice, real frame restored exactly: {exact} | {m}')
print(f'{twice} frames generated')
