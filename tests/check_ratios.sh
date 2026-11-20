#!/usr/bin/env bash
# SR at every upscaling ratio through OptiScaler under Proton (input 640x360): DLAA 1x, Quality 1.5x, Balanced 1.7x, Performance 2x, Ultra Performance 3x.
# usage: tests/check_ratios.sh PROTON_DIR OPTISCALER_DLL FIXTURE_DIR   (fixture reference is 1280x720)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for sz in 640x360 960x540 1088x612 1280x720 1920x1080; do
  W=${sz%x*}; H=${sz#*x}
  ANF_APP_DIR=$ROOT/build/harness-ratio timeout 250 "$ROOT/tests/run_optiscaler.sh" "$1" "$2" "$3" 20 640 360 $W $H >/dev/null 2>&1 || true
  python3 - "$ROOT/build/harness-ratio" "$3" $W $H <<'P'
import sys, numpy as np
from PIL import Image
app, fx, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def srgb(x):
    x = np.clip(np.nan_to_num(x), 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
ref = srgb(np.fromfile(fx + '/reference.rgba16f', np.float16).reshape(720, 1280, 4)[..., :3].astype(np.float32))
refr = np.asarray(Image.fromarray((ref * 255 + .5).astype(np.uint8)).resize((w, h), Image.LANCZOS), np.float32) / 255
try: o = srgb(np.fromfile(app + '/out/out-19.rgba16f', np.float16).reshape(h, w, 4)[..., :3].astype(np.float32))
except Exception as e: print(f'{w}x{h}: FAILED ({e})'); sys.exit()
inp = srgb(np.fromfile(fx + '/input-19.rgba16f', np.float16).reshape(360, 640, 4)[..., :3].astype(np.float32))
bil = np.asarray(Image.fromarray((inp * 255 + .5).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255
p = lambda a: 10 * np.log10(1 / max(np.mean((a - refr) ** 2), 1e-12))
print(f'{w}x{h} ({w / 640:.2f}x): ANF {p(o):.2f} dB | bilinear of the same input {p(bil):.2f} dB | nan {np.isnan(o).sum()}')
P
done
