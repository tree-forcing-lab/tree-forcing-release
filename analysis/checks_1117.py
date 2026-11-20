"""Checks 11-17: (5) content control with the re-run tree runs; (6) BBH / ECE of the controls vs answer-only;
(7) every held-out set the K&K runs were evaluated on; (8) K&K-External pooled with the re-runs."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import ledger as L
M = "Qwen3.5-4B"
A = L.ARMS
A["Tree rerun"] = L.kk("GFPS", "c6lr3cht", (13, 14, 15)); A["A2 rerun"] = L.kk("A2", "c12lr3cht", (13, 14, 15))
A["A2 s13-15"] = L.kk("A2", "c12lr3ch", (13, 14, 15))
print("## (5) content control\n" + L.HEAD)
L.compare("Tree rerun", "Shuffled", ["kkdeep"], A)
L.compare("Tree rerun", "Shuffled", ["kkdeep"], A, select=False, label=" (end, 2,000)", items="common")
L.compare("Tree s13-15", "Shuffled", ["kkdeep"], A, select=False, label=" (end, 2,000)", items="common")
L.compare("Shuffled", "A2 s13-15", ["kkdeep"], A, select=False, label=" (end, 2,000)", items="common")
L.compare("Shuffled", "A2 rerun", ["kkdeep"], A, select=False, label=" (end, 2,000)", items="common")
print("\n## (6) BBH and ECE: controls vs answer-only (end of training)\n" + L.HEAD)
L.compare("Shuffled", "A2 s13-15", ["bbh"], A, select=False, label=" (end)", items="common")
L.compare("Tree s13-15", "Shuffled", ["bbh"], A, select=False, label=" (end)", items="common")
L.compare("A7 (writes proof)", "A2", ["bbh"], A, select=False, label=" (end)", items="common")
L.compare("Proof, tree path (rn)", "A2", ["bbh"], A, select=False, label=" (end)", items="common")
L.compare("Tree", "A7 (writes proof)", ["bbh"], A, select=False, label=" (end)", items="common")
print("\n| arm | BBH ECE | BBH acc |\n|---|---|---|")
for a in ("A2", "A2 s13-15", "Tree", "Tree s13-15", "Shuffled", "A7 (writes proof)", "Proof, tree path (rn)", "DAIS-style" if "DAIS-style" in A else "A2"):
    print(f"| {a} | {np.mean([L.ece(n, 'bbh') for n in A[a].values()]):.1f} | {100 * np.mean([np.mean(list(L.correct(n, 'bbh').values())) for n in A[a].values()]):.1f} |")
print("\n## (7) every held-out set, Tree − A2, 6 seeds, end of training\n" + L.HEAD)
for x in ("bbh", "bbh500", "bbeh", "folio", "musr", "gsm8k", "jevbench", "jl", "policy", "sharc", "td", "diag", "diagA", "zebra"):
    try:
        L.compare("Tree", "A2", [x], A, select=False, label=" (end)", items="common")
    except Exception as e:
        print(f"| Tree − A2 | {x} | error {type(e).__name__}: {e} |")
print("\n## (8) K&K-External pooled: seeds 16-18 + batch-19 re-runs of 13-15 (selected epochs)\n" + L.HEAD)
A["Tree xkk6"] = {**L.kk("GFPS", "c6lr3chv", (16, 17, 18)), **L.kk("GFPS", "c6lr3cht", (13, 14, 15))}
A["A2 xkk6"] = {**L.kk("A2", "c12lr3chv", (16, 17, 18)), **L.kk("A2", "c12lr3cht", (13, 14, 15))}
L.compare("Tree xkk6", "A2 xkk6", ["xkk"], A)
L.compare("Tree xkk6", "A2 xkk6", ["xkk"], A, select=False, label=" (end)", items="common")
L.compare("Tree rerun", "A2 rerun", ["xkk"], A)
