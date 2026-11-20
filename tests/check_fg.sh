#!/usr/bin/env bash
# FG checks: nets GPU==CPU bit-exact, full pipeline deterministic, validation-clean. Run from the repo root.
set -e
./build/cm/fgnet_test | tail -1
VAL=1 ./build/cm/fgnet_test 2>&1 | grep -ci "validation" | grep -qx 0 && echo "fgnet: validation clean"
[ -f build/hero.rgb ] || python3 tools/testimage.py build/hero.rgb
if [ -f build/hero.rgb ]; then
  ANF_HERO=build/hero.rgb ./build/cm/fg_test 1920 1080 2 0.7 | grep -E "deterministic|frame 1 vs"
  ANF_HERO=build/hero.rgb VAL=1 ./build/cm/fg_test 640 360 1 0.3 2>&1 | grep -ci "validation" | grep -qx 0 && echo "fg pipeline: validation clean"
fi
