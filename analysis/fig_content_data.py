"""Data of the content-control figure (paper Figure 2): for each training variant, its K&K-Deep accuracy minus that of
answer-only training, paired by seed, each method at its selected epoch, on the items both runs of a seed hold.
Writes analysis/fig_content.json; analysis/fig_content.py draws it. RUNS=runs_archive/runs python3 analysis/fig_content_data.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import ledger as L  # noqa: E402

A = L.ARMS
A["Permuted 6"] = {**L.kk("GFPSR", "c6lr3ch", (13, 14, 15)), **L.kk("GFPSR", "c6lr3chv", (16, 17, 18))}
A["n3"] = L.kk("GFPS", "c6lr3chn3", range(13, 19), "student_kk_tf_n3")
A["n8"] = L.kk("GFPS", "c6lr3chn8", range(13, 19), "student_kk_tf_n8")
A["Verification"] = L.kk("GFPS", "c6lr3chvf", range(13, 19), "student_kk_tf_vrf")
A["Read-only"] = L.kk("GFPS", "c6lr3chrd", (13, 14, 15))
A["DAIS"] = L.kk("GFPS", "c6lr3chds", range(13, 19))
ROWS = [("Tree Forcing", "Tree"), ("DAIS-style", "DAIS"), ("Proof gen., tree batching", "Proof, tree path (rn)"),
        ("Eight nodes per puzzle", "n8"), ("Verification control", "Verification"), ("Three nodes per puzzle", "n3"),
        ("Proof generation", "A7 (writes proof)"), ("Permuted nodes", "Permuted 6"), ("Read nodes only", "Read-only")]
data = {}
for lab, key in ROWS:
    seeds = sorted(set(A[key]) & set(A["A2"]))
    ea, Ea = L.pick({s: A[key][s] for s in seeds})
    eb, Eb = L.pick({s: A["A2"][s] for s in seeds})
    d = []
    for s in seeds:
        ca = L.correct(A[key][s], "kkdeep", ea, Ea[s]); cb = L.correct(A["A2"][s], "kkdeep", eb, Eb[s])
        ids = [i for i in ca if i in cb]
        d.append(100 * float(np.mean([ca[i] - cb[i] for i in ids])))
    data[lab] = d
OUT = Path(__import__("os").environ.get("OUT", ROOT / "analysis"))
json.dump(data, open(OUT / "fig_content.json", "w"), indent=1)
print({k: round(float(np.mean(v)), 2) for k, v in data.items()})
