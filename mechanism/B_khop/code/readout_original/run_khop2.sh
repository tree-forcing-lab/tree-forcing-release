#!/bin/bash
# CARD_khop_v2 five-seed readout (section 4): khop2_table.py on the new D27 (j875-879), R3 (j880-884) and H3 (j885-889)
# tarballs plus the parent's KH (j756, j758-761), K1 (j757, j762-765), KD (j766-770), A2Z (j751-755), BASE0 (j750), A2Z96
# (j771) and both A2-6k reference sets (originals j520 j525 j530 j540 j542, re-trains j663 j664 j665 j540 j666), then the
# section 0 verdict, the branch (b) label and the GSM lines. The gate output is an input: when its last line is
# "GATE PASS (H3 dropped: ...)" no H3 tarball is read. A missing tarball stops the readout (it never reads a partial set).
# Usage: bash run_khop2.sh <gate output file> [results dir]
# Log: E/logs/khop2_readout_<date>.txt and .json.
set -euo pipefail
E=<tmp>
GATE=${1:?usage: run_khop2.sh <gate output file> [results dir]}
R=${2:-$E/results}
[ -f "$GATE" ] || { echo "run_khop2: no gate output $GATE" >&2; exit 1; }
LAST=$(grep -v '^[[:space:]]*$' "$GATE" | tail -n 1)
case "$LAST" in "GATE PASS"*) ;; *) echo "run_khop2: the gate did not pass ($LAST)" >&2; exit 1 ;; esac
H3=1; case "$LAST" in *"H3 dropped"*) H3=0 ;; esac
t() { local f; f=$(ls "$R"/results_"$1"_*.tgz 2>/dev/null | grep -v '/results_smoke_' | head -n 1); [ -n "$f" ] || { echo "run_khop2: no tarball for j$1 in $R" >&2; exit 1; }; echo "$f"; }
A=()
for s in 16 17 18 19 20; do
  case $s in 16) kh=756; k1=757; a2z=751 ;; *) kh=$((758 + s - 17)); k1=$((762 + s - 17)); a2z=$((752 + s - 17)) ;; esac
  i=$((s - 16))
  A+=(--run "D27:s$s=$(t $((875 + i)))" --run "R3:s$s=$(t $((880 + i)))")
  [ "$H3" = 1 ] && A+=(--run "H3:s$s=$(t $((885 + i)))")
  A+=(--run "KD:s$s=$(t $((766 + i)))" --run "KH:s$s=$(t $kh)" --run "K1:s$s=$(t $k1)" --run "A2Z:s$s=$(t $a2z)")
done
A+=(--run "BASE0:s0=$(t 750)" --run "A2Z96:s16=$(t 771)")
for p in 16:520 17:525 18:530 19:540 20:542; do A+=(--ref-orig "s${p%%:*}=$(t ${p##*:})"); done
for p in 16:663 17:664 18:665 19:540 20:666; do A+=(--ref-retrain "s${p%%:*}=$(t ${p##*:})"); done
mkdir -p "$E/logs"
OUT=$E/logs/khop2_readout_$(date +%Y%m%d)
python3 -I "$E/readout/khop2_table.py" --khop-dir "$E/src/data/eval/khop" --hint-dir "$E/src/data/eval/hint" \
  --gate-output "$GATE" --json "$OUT.json" "${A[@]}" | tee "$OUT.txt"
