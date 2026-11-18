#!/bin/bash
# Rebuild the 18 hint sets from data/source with the recovered builder and compare md5 with data/hint/MANIFEST.json.
# Needs a Python with torch (cotdistill.student imports it at load time; the builder itself uses no GPU).
#   bash code/builder/rebuild.sh /path/to/python-with-torch      (run from the A_hint folder)
set -euo pipefail
PY=${1:?python with torch}
T=$(mktemp -d)
tar xzf code/repo_994ff50_full.tgz -C "$T"
PYTHONPATH="$T/repo/src" "$PY" code/builder/build_hint_evals.py --steps data/source/gsm8k_mc_v2_steps.jsonl \
  --x data/source/gsm8k_mc_v2_xprob.jsonl --out "$T/hint" > "$T/build.log"
python3 - "$T/hint" <<'PY'
import hashlib, json, sys
man = json.load(open("data/hint/MANIFEST.json"))["sets"]
bad = [s for s, v in man.items() if hashlib.md5(open(f"{sys.argv[1]}/{s}.jsonl", "rb").read()).hexdigest() != v["md5"]]
print("rebuilt sets byte-identical to the evaluated ones: %d / %d" % (len(man) - len(bad), len(man)), bad or "")
sys.exit(1 if bad else 0)
PY
