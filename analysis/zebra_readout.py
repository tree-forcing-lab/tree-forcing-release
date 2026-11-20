"""Logic grid (Zebra cell) comparison of the paper (Section 4.4): Tree Forcing 6 epochs vs answer-only 12 epochs,
seeds 13-18, each method's epoch by best seed-mean validation accuracy; in-range test set (zcin), larger grids (zcex)
and transfer to K&K. RUNS=runs_archive/runs python3 analysis/zebra_readout.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import ledger as L  # noqa: E402
M = "Qwen3.5-4B"; D = "student_zebra_cell_tf"
A = {"Z Tree": {s: f"GFPS-c6lr3chzt-{M}-{D}-s{s}" for s in range(13, 19)},
     "Z A2": {s: f"A2-c12lr3chzt-{M}-{D}-s{s}" for s in range(13, 19)}}
for a in A:
    print(a, "selected epoch", L.pick(A[a])[0])
print(L.HEAD)
for x in ("zcin", "zcex", "kk", "kkdeep", "xkk", "zebra"):
    L.compare("Z Tree", "Z A2", [x], A, items="common")
