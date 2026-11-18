#!/bin/bash
# CARD_khop_v4 five-seed readout (section 4): khop4_table.py on the 20 tarballs of the confirmed arm (CUR1: 975,
# 1029-1032; K1X1: 985, 1033-1036; CUR-a: 1025 (L) or 1027 (R), 1037-1040; K1X-a: 1026 (L) or 1028 (R), 1041-1044),
# both A2-6k reference sets (originals j520 j525 j530 j540 j542, re-trains j663 j664 j665 j540 j666), the recipe /
# context runs j980 and j990, the parent D27 (j875-879), and for arm R the parent KH (j756, j758-761) and K1 (j757,
# j762-765). The arm is read from the gate output's last line ("GATE PASS ARM R" / "GATE PASS ARM L"); khop4_table.py
# checks the whole gate output (every line, the 0.5 rule) again. The retry-walls file (one "<node> <hours>" line per
# failed execution of the smoke, pilot and confirmation DAGs, from the job event logs; empty when nothing retried) is
# required. A missing tarball, or two tarballs for one id, stops the readout. Ids are matched as results_<id>_*, so
# 4-digit ids never match a 3-digit id (results_975_ never matches results_9750_ or results_1975_). The 8 S2 ids of the
# confirmation hold two candidate envs; only the arm's TAG is accepted (results_<id>_4b_gsmvt_A2-khop4<cur|k1x><l|r>w*).
# Last, the measured family compute: readout/khop4_gpuh.py 1025 1049 <results dir> (every landed tarball of the family
# in the given results dir, smokes included; zone-aware CDT / CST walls, Amendment 3) plus the retry walls; a row with
# "n/a" (unreadable tarball, no log, no start / exit line, a zone other than CDT / CST) stops the readout.
# Each tarball lookup is its own assignment (p=$(t ...)), so a failed lookup stops the script under set -e (a compound
# array assignment takes the status of its last substitution only).
# Usage: bash run_khop4.sh <gate output file> <retry-walls file> [results dir]
# Log: E/logs/khop4_readout_<date>.txt and .json.
set -euo pipefail
E=${KHOP4_E:-<path>
GATE=${1:?usage: run_khop4.sh <gate output file> <retry-walls file> [results dir]}
WALLS=${2:?usage: run_khop4.sh <gate output file> <retry-walls file> [results dir]}
R=${3:-$E/results}
PY=${KHOP4_PY:-python3}
[ -f "$GATE" ] || { echo "run_khop4: no gate output $GATE" >&2; exit 1; }
[ -f "$WALLS" ] || { echo "run_khop4: no retry-walls file $WALLS (an empty file means no retry)" >&2; exit 1; }
LAST=$(grep -v '^[[:space:]]*$' "$GATE" | tail -n 1 || true)
case "$LAST" in
  "GATE PASS ARM R") ARM=R; a=r ;;
  "GATE PASS ARM L") ARM=L; a=l ;;
  *) echo "run_khop4: the gate did not pass ($LAST)" >&2; exit 1 ;;
esac
t() {   # t <id> [glob after the id]: exactly one landed tarball
  local n g=${2:-*}
  n=$(ls "$R"/results_"$1"_$g.tgz 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 1 ] || { echo "run_khop4: $n tarballs for j$1 ($g) in $R (need exactly 1)" >&2; exit 1; }
  local all
  all=$(ls "$R"/results_"$1"_*.tgz 2>/dev/null | wc -l | tr -d ' ')
  [ "$all" = 1 ] || { echo "run_khop4: $all tarballs for j$1 in $R (two landed tarballs for one id)" >&2; exit 1; }
  ls "$R"/results_"$1"_$g.tgz
}
case $ARM in R) PC=1027; PK=1028 ;; L) PC=1025; PK=1026 ;; esac
A=()
for s in 16 17 18 19 20; do
  i=$((s - 17))
  if [ "$s" = 16 ]; then c1=975; k1x1=985; c=$PC; k=$PK; else c1=$((1029 + i)); k1x1=$((1033 + i)); c=$((1037 + i)); k=$((1041 + i)); fi
  p=$(t $c1); A+=(--run "CUR1:s$s=$p")
  p=$(t $k1x1); A+=(--run "K1X1:s$s=$p")
  p=$(t $c "4b_gsmvt_A2-khop4cur${a}w*"); A+=(--run "CUR:s$s=$p")
  p=$(t $k "4b_gsmvt_A2-khop4k1x${a}w*"); A+=(--run "K1X:s$s=$p")
  p=$(t $((875 + s - 16))); A+=(--run "D27:s$s=$p")
  if [ "$ARM" = R ]; then
    case $s in 16) kh=756; k1=757 ;; *) kh=$((758 + s - 17)); k1=$((762 + s - 17)) ;; esac
    p=$(t $kh); A+=(--run "KH:s$s=$p")
    p=$(t $k1); A+=(--run "K1:s$s=$p")
  fi
done
p=$(t 980); A+=(--run "J980:s16=$p")
p=$(t 990); A+=(--run "J990:s16=$p")
for q in 16:520 17:525 18:530 19:540 20:542; do p=$(t ${q##*:}); A+=(--ref-orig "s${q%%:*}=$p"); done
for q in 16:663 17:664 18:665 19:540 20:666; do p=$(t ${q##*:}); A+=(--ref-retrain "s${q%%:*}=$p"); done
GPRC=0
GP=$("$PY" -I "$E/readout/khop4_gpuh.py" 1025 1049 "$R") || GPRC=$?
if [ "$GPRC" != 0 ] || printf '%s\n' "$GP" | grep -q ' n/a h '; then
  printf '%s\n' "$GP" >&2
  echo "run_khop4: khop4_gpuh.py 1025 1049 $R has a row without a wall (n/a) or failed (rc $GPRC): the compute line cannot be read" >&2
  exit 1
fi
mkdir -p "$E/logs"
OUT=${KHOP4_OUT:-$E/logs/khop4_readout_$(date +%Y%m%d)}
# Codex round 5: the table's stderr goes into the report too, and a failed readout ends the report with a READOUT
# FAILED line, so a saved report never shows a reading without the failure that stopped it.
set +e
{
  "$PY" -I "$E/readout/khop4_table.py" --khop-dir "$E/src/data/eval/khop" --hint-dir "$E/src/data/eval/hint" \
    --gate-output "$GATE" --retry-walls "$WALLS" --results-dir "$R" --json "$OUT.json" "${A[@]}" 2>&1 || exit $?
  echo "== measured family compute (khop4_gpuh.py 1025 1049 $R, landed tarballs incl. smokes, CDT / CST walls)"
  printf '%s\n' "$GP"
  awk 'NF == 2 { s += $2; n += 1 } END { printf "== retry walls (job event logs): %d failed executions, %.2f GPU-h\n", n, s }' "$WALLS"
} | tee "$OUT.txt"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" != 0 ]; then
  echo "== READOUT FAILED (khop4_table.py exit $rc): nothing in this file is a reading" | tee -a "$OUT.txt" >&2
  rm -f "$OUT.json"
fi
exit "$rc"
