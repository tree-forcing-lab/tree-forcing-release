#!/bin/bash
# CARD_khop_v1 five-seed readout (section 4): khop_table.py on the KH / K1 / KD training tarballs, A2Z (j751-755), BASE0
# (j750), A2Z96 (j771) and both A2-6k reference sets (originals j520 j525 j530 j540 j542, re-trains j663 j664 j665 j540
# j666; pfx_table.py rc cells), then the section 0 verdict. Usage: bash run_khop.sh [results dir] [--json out.json]
# Missing tarballs stop the readout (it never reads a partial set).
set -euo pipefail
E=<tmp>
R=${1:-$E/results}; shift || true
t() { local f; f=$(ls "$R"/results_"$1"_*.tgz 2>/dev/null | grep -v '/results_smoke_' | head -n 1); [ -n "$f" ] || { echo "run_khop: no tarball for j$1 in $R" >&2; exit 1; }; echo "$f"; }
A=()
for s in 16 17 18 19 20; do
  case $s in 16) kh=756; k1=757; a2z=751 ;; *) kh=$((758 + s - 17)); k1=$((762 + s - 17)); a2z=$((752 + s - 17)) ;; esac
  kd=$((766 + s - 16))
  A+=(--run "KH:s$s=$(t $kh)" --run "K1:s$s=$(t $k1)" --run "KD:s$s=$(t $kd)" --run "A2Z:s$s=$(t $a2z)")
done
A+=(--run "BASE0:s0=$(t 750)" --run "A2Z96:s16=$(t 771)")
for p in 16:520 17:525 18:530 19:540 20:542; do A+=(--ref-orig "s${p%%:*}=$(t ${p##*:})"); done
for p in 16:663 17:664 18:665 19:540 20:666; do A+=(--ref-retrain "s${p%%:*}=$(t ${p##*:})"); done
python3 -I "$E/readout/khop_table.py" --khop-dir "$E/src/data/eval/khop" --hint-dir "$E/src/data/eval/hint" "${A[@]}" "$@"
