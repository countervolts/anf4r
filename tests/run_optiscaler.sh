#!/usr/bin/env bash
# End-to-end: the D3D12 harness (from m4r) calls OptiScaler's NGX D3D12 exports under Proton; OptiScaler's upscaler is ANF (anf4r.dll).
# usage: tests/run_optiscaler.sh PROTON_DIR OPTISCALER_DLL FIXTURE_DIR [FRAMES IN_W IN_H OUT_W OUT_H]
# env: ANF_HARNESS = path to m4r's d3d12_harness.exe (github.com/countervolts/m4r, tests/build_harness.sh), ANF_FLIP_JITTER=1, ANF_APP_DIR
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
[[ $# -ge 3 ]] || { sed -n 2,5p "$0" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"; OPTI="$(realpath "$2")"; FIX="$(realpath "$3")"
FRAMES="${4:-8}"; IW="${5:-640}"; IH="${6:-360}"; OW="${7:-1280}"; OH="${8:-720}"
HARNESS="${ANF_HARNESS:?set ANF_HARNESS to the path of m4r d3d12_harness.exe}"
APP="${ANF_APP_DIR:-$ROOT/build/harness-app}"; COMPAT="${ANF_COMPAT_DIR:-$ROOT/build/harness-compat}"
rm -rf "$APP"; mkdir -p "$APP/out" "$COMPAT"
cp -f "$HARNESS" "$APP/d3d12_harness.exe"; cp -f "$OPTI" "$APP/dxgi.dll"
python3 "$ROOT/scripts/install_anf.py" --out "$APP" >/dev/null
ln -s "$FIX" "$APP/fixture"
[[ -n "${ANF_STAB:-}" ]] && printf "stabilize = %s\npre_exposure = %s\n" "$ANF_STAB" "${ANF_PREEXP:-1}" > "$APP/anf4r/data/config.txt"
[[ -n "${ANF_DUMP:-}" ]] && echo "$ANF_DUMP" > "$APP/anf4r/data/dump.txt"   # capture the first N SR frames to anf4r/data/dump/
FGEN=false; FGIN=nofg; FGOUT=nofg
if [[ -n "${ANF_FG:-}" ]]; then FGEN=true; FGIN=upscaler; FGOUT=anf; export M4R_HARNESS_SWAPCHAIN=1; fi   # ANF_FG=1: also ANF frame generation (needs > 14 frames)
cat > "$APP/OptiScaler.ini" <<EOF
[Upscalers]
Dx12Upscaler=anf
[ANF]
FlipJitter=${ANF_FLIP_JITTER:-false}
[FrameGen]
Enabled=$FGEN
FGInput=$FGIN
FGOutput=$FGOUT
[Spoofing]
Dxgi=false
[Plugins]
LoadAsiPlugins=false
[Menu]
OverlayMenu=false
[Log]
LogToFile=true
LogLevel=${ANF_OPTI_LOGLEVEL:-1}
EOF
export ANF4R_HARNESS_REVERSED_DEPTH=1 M4R_HARNESS_REVERSED_DEPTH=1   # the fixtures use reversed depth
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$COMPAT" WINEDLLOVERRIDES="dxgi=n,b" WINEDEBUG="${WINEDEBUG:--all}"
cd "$APP"
"$PROTON_DIR/proton" run "$APP/d3d12_harness.exe" dxgi.dll fixture out "$FRAMES" "$IW" "$IH" "$OW" "$OH" || true
cat "$APP/harness.log" 2>/dev/null || true
