#!/usr/bin/env bash
# anf4r: the conv network must produce bit-identical output across runs (no nondeterminism).
set -euo pipefail
cd "$(dirname "$0")/.."
for ph in 0 7 31 63; do
  out=$(./build/cm/model_test vendor/assets/modelzoo/aiff-sr-v2s-prod-7p1-invpow2 "$ph" 2>&1 | tail -1)
  echo "phase $ph: $out"
  [[ "$out" == MODEL\ OK* ]] || { echo "NONDETERMINISTIC or WRONG at phase $ph"; exit 1; }
done
echo "determinism OK"
