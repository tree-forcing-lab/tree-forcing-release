"""Batch 28 readout (cloud/chtc/README_28.md §2, written before any batch-28 result; arms cd / placebo dropped before
any result). Report-only; wording rule: significant only if seed-t and two-level intervals both exclude 0."""
import sys
from pathlib import Path
import numpy as np, statistics as st
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import ledger as L
M = "Qwen3.5-4B"; A = L.ARMS
A["n3"] = L.kk("GFPS", "c6lr3chn3", range(13, 19), "student_kk_tf_n3")
A["n8"] = L.kk("GFPS", "c6lr3chn8", range(13, 19), "student_kk_tf_n8")
A["Verification"] = L.kk("GFPS", "c6lr3chvf", range(13, 19), "student_kk_tf_vrf")
A["Read-only"] = L.kk("GFPS", "c6lr3chrd", (13, 14, 15))
A["Permuted 6"] = {**L.kk("GFPSR", "c6lr3ch", (13, 14, 15)), **L.kk("GFPSR", "c6lr3chv", (16, 17, 18))}
for a in ("Tree", "A2", "n3", "n8", "Verification", "Read-only", "Permuted 6"):
    print(f"  {a}: selected epoch {L.pick(A[a])[0]}")
print("\n## K&K-Deep at the selected epochs (per-seed items)\n" + L.HEAD)
for a, b in (("Tree", "n3"), ("Tree", "n8"), ("n3", "A2"), ("n8", "A2"), ("n8", "n3"), ("Tree", "Verification"),
             ("Verification", "A2"), ("Tree s13-15", "Read-only"), ("Read-only", "A2"), ("Tree", "Permuted 6"), ("Permuted 6", "A2")):
    L.compare(a, b, ["kkdeep"], A)
print("\n## K&K-Deep at the end of training (all 2,000)\n" + L.HEAD)
for a, b in (("Tree", "n3"), ("Tree", "n8"), ("n3", "A2"), ("n8", "A2"), ("Tree", "Verification"), ("Verification", "A2"),
             ("Tree s13-15", "Read-only"), ("Read-only", "A2"), ("Tree", "Permuted 6"), ("Permuted 6", "A2")):
    L.compare(a, b, ["kkdeep"], A, select=False, label=" (end, 2,000)", items="common")
print("\n## kk / xkk (selected) and BBH (end)\n" + L.HEAD)
for a, b in (("n3", "A2"), ("n8", "A2"), ("Verification", "A2"), ("Tree", "Verification"), ("Tree", "Permuted 6")):
    L.compare(a, b, ["kk"], A)
A["n3 s16-18"] = L.kk("GFPS", "c6lr3chn3", (16, 17, 18), "student_kk_tf_n3"); A["vf s16-18"] = L.kk("GFPS", "c6lr3chvf", (16, 17, 18), "student_kk_tf_vrf")
A["n8 s16-18"] = L.kk("GFPS", "c6lr3chn8", (16, 17, 18), "student_kk_tf_n8")
for a in ("n3 s16-18", "n8 s16-18", "vf s16-18"):
    L.compare(a, "A2 s16-18", ["xkk"], A)
for a, b in (("n3", "A2"), ("n8", "A2"), ("Verification", "A2"), ("Tree", "Verification"), ("Tree", "n3"), ("Permuted 6", "A2"), ("Tree", "Permuted 6")):
    L.compare(a, b, ["bbh"], A, select=False, label=" (end)", items="common")
print("\n| arm | BBH ECE (end) | BBH acc | kkdeep ECE (selected) |\n|---|---|---|---|")
for a in ("A2", "Tree", "n3", "n8", "Verification", "Permuted 6"):
    e, E = L.pick(A[a])
    print(f"| {a} | {np.mean([L.ece(n, 'bbh') for n in A[a].values()]):.1f} | {100*np.mean([np.mean(list(L.correct(n, 'bbh').values())) for n in A[a].values()]):.1f} | "
          f"{np.mean([L.ece(n, 'kkdeep', e, E[s]) for s, n in A[a].items()]):.1f} |")
print("\n## GSM8K: Tree Forcing length by six-seed valx mean (1 vs 2 epochs, ties to 1)")
G = "student_gsm8k_tf_vt"
T1 = {s: f"GFPQ-c1lr3ch-{M}-{G}-s{s}" for s in range(13, 19)}
T2 = L.GSM["GSM Tree 2ep"]
vx = {k: st.mean(np.mean(list(L.correct(n, "valx").values())) for n in runs.values()) * 100 for k, runs in (("1", T1), ("2", T2))}
print(f"  valx six-seed mean: 1 epoch {vx['1']:.2f}, 2 epochs {vx['2']:.2f} -> pick {'1' if vx['1'] >= vx['2'] else '2'}")
L.GSM["GSM Tree 1ep"] = T1
print(L.HEAD)
for sets, lab in ((["gsm8kv2", "gsm1k"], " (original)"), (["gsm8kx", "gsm1kx"], " (cue-free)")):
    L.compare("GSM Tree 1ep", "GSM A2 1ep", sets, L.GSM, select=False, label=lab, items="common")
    L.compare("GSM Tree 1ep", "GSM Shuffled 2ep", sets, L.GSM, select=False, label=lab, items="common")
