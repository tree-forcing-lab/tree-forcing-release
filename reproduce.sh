#!/usr/bin/env bash
# One command: reproduce every number and data figure of the paper from the released run outputs.
#   bash reproduce.sh [path-or-URL-of-tree-forcing-runs.tar.gz]
# The archive defaults to $RUNS_ARCHIVE (a local file), else is downloaded from Zenodo (ZENODO_URL below). Outputs go to ./reproduced/; the last step checks
# the reproduced numbers against the paper (analysis/expected_numbers.json) and fails on any mismatch.
# Needs Python 3 with numpy and matplotlib ($PYTHON, default python3). CPU only; about an hour on two cores.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
PY=${PYTHON:-python3}
ZENODO_URL="https://zenodo.org/records/24381352/files/tree-forcing-runs.tar.gz"   # DOI 10.5281/zenodo.24381352
ARCHIVE=${1:-${RUNS_ARCHIVE:-$ZENODO_URL}}
if [ -z "$ARCHIVE" ]; then
  echo "usage: bash reproduce.sh PATH_OR_URL_OF_tree-forcing-runs.tar.gz   (or set RUNS_ARCHIVE)" >&2
  echo "The run outputs are not bundled with the repository; see README (Run outputs)." >&2
  exit 2
fi
case "$ARCHIVE" in
  http://*|https://*) echo "== downloading the run outputs"; curl -L --fail -o tree-forcing-runs.tar.gz "$ARCHIVE"; ARCHIVE=tree-forcing-runs.tar.gz ;;
esac
step() { echo; echo "== $1"; t0=$(date +%s); }
done_() { echo "   ($(( $(date +%s) - t0 )) s)"; }

step "checking the archive (sha256)"
want=$(cut -d' ' -f1 runs_archive.sha256); have=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
[ "$want" = "$have" ] || { echo "sha256 mismatch: expected $want, got $have" >&2; exit 1; }; done_
step "unpacking the run outputs to runs_archive/"
mkdir -p runs_archive; tar xzf "$ARCHIVE" -C runs_archive; done_
step "unpacking the data (md5 checked against data/MANIFEST.json)"
"$PY" scripts/unpack_data.py; done_

export RUNS="$ROOT/runs_archive/runs" RESTORED="$ROOT/runs_archive/restored_runs" OUT="$ROOT/reproduced"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
run() { local name=$1; shift; step "$name"; nice "$PY" "$@" > "$OUT/$name.md"; done_; }
run ledger              analysis/ledger.py            # main comparisons, transfer, GSM8K, per-seed table
run b23_readout         analysis/b23_readout.py       # proof-generation controls
run b25_readout         analysis/b25_readout.py       # 9B
run b26_b27_readout     analysis/b26_b27_readout.py   # DAIS-style control, FOLIO
run b28_readout         analysis/b28_readout.py       # node ablations, permuted control (6 seeds), GSM8K schedule
run zebra_readout       analysis/zebra_readout.py     # logic grid puzzles
run analysis_1117       analysis/analysis_1117.py     # K&K-Deep splits, BIG-Bench Hard by task, temperature scaling
run checks_1115         analysis/checks_1115.py       # matched-length transfer, coupling within puzzle size
run checks_1117         analysis/checks_1117.py       # held-out sets, pooled K&K-External
run extras              analysis/extras.py            # permuted control at six seeds on BBH, GSM8K placebo, written-reasoning references
run node_coupling       analysis/node_coupling.py     # node answers vs final answers
step "Figure 2 (content controls)"; nice "$PY" analysis/fig_content_data.py; nice "$PY" analysis/fig_content.py; done_
step "mechanism inputs"; bash analysis/prepare_mechanism_inputs.sh "$OUT/mechanism_inputs" > /dev/null; done_
run mechanism_check     analysis/mechanism_check.py "$OUT/mechanism_inputs"   # hint test, serial budget, GSM8K cost
step "Figure 3 (mechanism)"; FIG_OUT="$OUT/mechanism.pdf" nice "$PY" analysis/mechanism_figure.py "$OUT/mechanism_inputs"; done_

step "checking the reproduced numbers against the paper"
"$PY" analysis/check_numbers.py "$OUT"
