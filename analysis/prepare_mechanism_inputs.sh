#!/usr/bin/env bash
# Builds the input folder of analysis/mechanism_check.py and analysis/mechanism_figure.py from mechanism/:
#   hintA/  hint sets of the hint test (mechanism/A_hint/data/hint)
#   hintB/  hint sets with serial-depth annotations (mechanism/B_khop/data/eval/hint)
#   s2/     predictions of the registered hint test, jobs 520-533 (mechanism/A_hint/data/stage2_preds.tgz)
#   khop/   the k-hop evaluation set khop_ca_fwd
#   bash analysis/prepare_mechanism_inputs.sh OUT_DIR
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); OUT=${1:?usage: prepare_mechanism_inputs.sh OUT_DIR}
mkdir -p "$OUT"/{hintA,hintB,s2,khop}
for f in "$ROOT"/mechanism/A_hint/data/hint/*.jsonl*; do b=$(basename "$f" .gz); zcat -f "$f" > "$OUT/hintA/$b"; done
for f in "$ROOT"/mechanism/B_khop/data/eval/hint/*.jsonl*; do b=$(basename "$f" .gz); zcat -f "$f" > "$OUT/hintB/$b"; done
tar xzf "$ROOT/mechanism/A_hint/data/stage2_preds.tgz" -C "$OUT/s2" 2>/dev/null
f=$(ls "$ROOT"/mechanism/B_khop/data/eval/khop/khop_ca_fwd.jsonl* | head -1); zcat -f "$f" > "$OUT/khop/khop_ca_fwd.jsonl"
echo "inputs in $OUT"
