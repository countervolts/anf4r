#!/usr/bin/env bash
# Exports a clean tree for publishing: a fresh git repository with ONE commit and none of the private history,
# without vendor/ (Qualcomm material) and without the reverse-engineering tooling and notes.
# usage: scripts/make_oss_tree.sh <out dir> [--with-research]
#   --with-research  also include tools/extract_assets.py, elfmap.py, scan_blobs.py, tools/qemu-trace/ and docs/research-log.md
#                    (read NOTICE.md first: these concern Qualcomm's binary and its license restrictions)
set -euo pipefail
OUT="${1:?usage: make_oss_tree.sh <out dir> [--with-research]}"; shift || true
RESEARCH=0; [[ "${1:-}" == "--with-research" ]] && RESEARCH=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -e "$OUT" ]] && { echo "$OUT exists; refusing to overwrite" >&2; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
git ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    vendor/*) continue ;;
    docs/research-log.md|tools/extract_assets.py|tools/elfmap.py|tools/scan_blobs.py|tools/qemu-trace/*) [[ $RESEARCH == 1 ]] || continue ;;
  esac
  [[ -e "$f" && ! -L "$f" ]] || continue
  mkdir -p "$OUT/$(dirname "$f")"; cp -p "$f" "$OUT/$f"
done
cat > "$OUT/.gitignore" <<'G'
build/
build-*/
external/
vendor/
__pycache__/
*.pyc
*.o
*.a
*.spv
G
cd "$OUT"
git init -q -b main && git add -A
git commit -q -m "anf4r: Adreno Neural Fusion super resolution and frame generation on RDNA3" && echo "exported $(git ls-files | wc -l) files to $OUT (one commit, no history)"
