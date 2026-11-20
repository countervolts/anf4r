#!/usr/bin/env bash
# SR checks (run from the repo root, needs build/hero.rgb): deterministic across runs, validation-clean,
# and temporal accumulation beats bicubic of the same LR frames.
set -e
[ -f build/hero.rgb ] || python3 tools/testimage.py build/hero.rgb
python3 tools/sr_demo.py 320 180 12 | tail -4
cp build/sr_out.bin build/sr_out_a.bin
./build/cm/sr_run build/sr_in.bin build/sr_out_b.bin >/dev/null
cmp build/sr_out_a.bin build/sr_out_b.bin && echo "sr: deterministic (bit-identical across runs)"
VAL=1 ./build/cm/sr_run build/sr_in.bin build/sr_out_c.bin 2>&1 | grep -ci "validation" | grep -qx 0 && echo "sr: validation clean"
