"""Independent check of the second author's mechanism materials (PR #13, notes/2026-11-17-mechanism) from the prediction files.

(A) Stage 2 hint test (jobs 520-533, seeds 16-18; preds from A_hint/data/stage2_preds.tgz, extracted to
    SCRATCH/s2/<job>/): accuracy per arm and set, H = true_r1 - plain on clean items (no shown value equals an option)
    with n >= 4 (the registered primary), D = true_r1 - true_r4 on the items with n >= 5, the fixed-item curve,
    controls minus plain at r = 1, and Tree Forcing vs answer-only on the hint gain (paired by seed).
(B) Serial budget (k-hop): one-pass accuracy on khop_ca_fwd by hop count for answer-only weights evaluated on the
    k-hop sets (A2Z, jobs 751-755), the 27k-chain arm (D27) and the untrained model (750), from the run outputs (RUNS).
(C) GSM8K cost of chain training: D27 minus answer-only on gsm8kv2 and on the registered rc2 cell (items with whole-
    chain serial depth >= 4, the true-hint row with 2 remaining serial steps; the second author's cell definition), with the
    original answer-only runs (520 / 525 / 530 from the Stage 2 archive, 540 from runs/, 542 from git history) and the
    re-trained ones (the current runs/ folders: 663 / 664 / 665 / 666, and 540 for seed 19).
Statistic: scripts/paired_stats.paired_stats (seed-level t and two-level bootstrap with shared item indices;
"significant" only if both exclude 0).
  python3 analysis/mechanism_check.py SCRATCH > analysis/mechanism_check.md
"""
import collections
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paired_stats as P  # noqa: E402

SCR = Path(sys.argv[1])
RUNS = Path(__import__("os").environ.get("RUNS", ROOT / "runs_archive/runs"))
RESTORED = Path(__import__("os").environ.get("RESTORED", RUNS.parent / "restored_runs"))


def read(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def corr_rows(rows):
    return {r["item_id"]: int(r["labels"][int(np.argmax(r["probs"]))] == r["gold_label"]) for r in rows}


def stats(MA, MB):
    r = P.paired_stats(np.array(MA, float), np.array(MB, float), n_boot=10000, boot_seed=0)
    t = r["t"]
    tci = "-" if t["lo"] is None else f"[{100*t['lo']:+.1f}, {100*t['hi']:+.1f}]"
    return (f"{100*r['mean']:+.1f} (" + " / ".join(f"{100*x:+.1f}" for x in r["d"]) + f") seed-t {tci} "
            f"2L [{100*r['two_level'][0]:+.1f}, {100*r['two_level'][1]:+.1f}] {'sig' if r['significant'] else 'n.s.'}")


# ---------------------------------------------------------------- (A) Stage 2 hint test
ARMS = {"A2-c1": [520, 525, 530], "A2-c2": [521, 526, 531], "GFPQ-c2": [522, 527, 532], "GFPQR-c2": [523, 528, 533]}
SETS = (["plain", "full"] + [f"true_r{r}" for r in range(1, 7)] + [f"placebo_r{r}" for r in range(1, 5)] +
        ["other_r1", "other_r2", "wrongres_r1", "wrongres_r2", "wrongres_near_r1", "wrongres_near_r2"])
meta = {s: {r["item_id"]: r for r in read(SCR / "hintA" / f"hint_{s}.jsonl")} for s in SETS}
C = {j: {s: corr_rows(read(SCR / "s2" / str(j) / f"preds_hint_{s}.jsonl")) for s in SETS} for js in ARMS.values() for j in js}
for j in C:
    for s in SETS:
        assert set(C[j][s]) == set(meta[s]), (j, s)
print("# Mechanism check (independent recomputation from preds)\n")
print("## A. Stage 2 hint test (seeds 16-18)\n")
print("Accuracy (%), seed mean, on each set's own items:\n")
print("| arm | " + " | ".join(SETS) + " |\n|" + "---|" * (len(SETS) + 1))
for a, js in ARMS.items():
    print(f"| {a} | " + " | ".join(f"{100*st.mean(np.mean(list(C[j][s].values())) for j in js):.1f}" for s in SETS) + " |")
n_of = {i: meta["plain"][i]["n"] for i in meta["plain"]}
clean1 = {i for i, r in meta["true_r1"].items() if not r["shown_equals_option"]}
H_items = sorted(clean1 & {i for i, n in n_of.items() if n >= 4})
D_items = sorted(i for i in meta["true_r4"] if n_of[i] >= 5)
print(f"\nH items (clean, n >= 4): {len(H_items)}; D items (n >= 5): {len(D_items)}\n")
for a, js in ARMS.items():
    M = lambda s, items: [[C[j][s][i] for i in items] for j in js]
    print(f"- {a}: H = true_r1 - plain {stats(M('true_r1', H_items), M('plain', H_items))}")
    print(f"  {'':{len(a)}}  D = true_r1 - true_r4 {stats(M('true_r1', D_items), M('true_r4', D_items))}")
    allr1 = sorted(meta["true_r1"])
    print(f"  {'':{len(a)}}  true_r1 - plain (all 1,129) {stats(M('true_r1', allr1), M('plain', allr1))}")
    for c in ("other_r1", "wrongres_r1", "wrongres_near_r1", "placebo_r1", "placebo_r2"):
        items = sorted(set(meta[c]) & set(meta["plain"]))
        print(f"  {'':{len(a)}}  {c} - plain {stats(M(c, items), M('plain', items))}")
print("\nFixed-item curve on the D items (accuracy %, seed mean):\n")
print("| arm | plain | r1 | r2 | r3 | r4 | full |\n|---|---|---|---|---|---|---|")
for a, js in ARMS.items():
    print(f"| {a} | " + " | ".join(f"{100*st.mean(np.mean([C[j][s][i] for i in D_items]) for j in js):.1f}"
                                    for s in ("plain", "true_r1", "true_r2", "true_r3", "true_r4", "full")) + " |")
print("\nTree Forcing vs answer-only on the hint gain (per item: true_r1 - plain), paired by seed (s16 / s17 / s18):\n")
allr1 = sorted(meta["true_r1"])
gain = lambda js, items: [[C[j]["true_r1"][i] - C[j]["plain"][i] for i in items] for j in js]
for a, b in (("GFPQ-c2", "A2-c1"), ("GFPQ-c2", "A2-c2"), ("GFPQ-c2", "GFPQR-c2")):
    print(f"- {a} - {b}: gain, all 1,129 items {stats(gain(ARMS[a], allr1), gain(ARMS[b], allr1))}; "
          f"gain on H items {stats(gain(ARMS[a], H_items), gain(ARMS[b], H_items))}")
    for s in ("plain", "true_r1"):
        print(f"  {a} - {b} accuracy on {s} (all items) "
              f"{stats([[C[j][s][i] for i in allr1] for j in ARMS[a]], [[C[j][s][i] for i in allr1] for j in ARMS[b]])}")

# ---------------------------------------------------------------- (B) serial budget
print("\n## B. Serial budget: one-pass accuracy on khop_ca_fwd by hop count\n")
kmeta = {r["item_id"]: int(r["group"].rsplit("k", 1)[1]) for r in read(SCR / "khop" / "khop_ca_fwd.jsonl")}
KARMS = {"untrained (750)": ["750_4b_evkh_base"],
         "answer-only A2Z (751-755)": [f"{j}_4b_evkh_w{w}" for j, w in zip(range(751, 756), (663, 664, 665, 540, 666))],
         "27k chains D27 (875-879)": [f"A2-khopd27-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt-s{s}" for s in range(16, 21)]}
print("| arm | runs | " + " | ".join(f"k{k}" for k in range(1, 9)) + " |\n|---|---|" + "---|" * 8)
curve = {}
for a, runs in KARMS.items():
    cs = [corr_rows(read(RUNS / r / "preds_khop_ca_fwd.jsonl")) for r in runs]
    row = []
    for k in range(1, 9):
        per = [100 * np.mean([c[i] for i in c if kmeta[i] == k]) for c in cs]
        m = st.mean(per)
        h = (st.stdev(per) / len(per) ** 0.5 * {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}[len(per)]) if len(per) > 1 else None
        row.append((m, h)); curve.setdefault(a, []).append(m)
    print(f"| {a} | {len(runs)} | " + " | ".join(f"{m:.1f}" + (f" ±{h:.1f}" if h else "") for m, h in row) + " |")
print(f"\nItems per k: {dict(sorted(collections.Counter(kmeta.values()).items()))}; chance 20%; training covers k = 1-6.")
json.dump(curve, open(SCR / "khop_curve.json", "w"))

# ---------------------------------------------------------------- (C) GSM8K cost
print("\n## C. GSM8K cost of chain training (D27 minus answer-only)\n")
hintB = {s: {r["item_id"]: r for r in read(SCR / "hintB" / f"hint_{s}.jsonl")} for s in ["plain"] + [f"true_r{r}" for r in range(1, 7)]}
cells = {}
for i, d in hintB["plain"].items():
    if d["path"] >= 4:
        cells[i] = {}
for s in [f"true_r{r}" for r in range(1, 7)]:
    for i, d in hintB[s].items():
        if i in cells and d["r_path"] in (1, 2, 3, 4):
            cur = cells[i].get(d["r_path"])
            if cur is None or d["k"] > cur[1]["k"]:
                cells[i][d["r_path"]] = (s, d)
prim = sorted(i for i, c in cells.items() if 1 in c and 3 in c and not c[1][1]["shown_equals_option"] and not c[3][1]["shown_equals_option"])
print(f"primary items (path >= 4, clean rc1 and rc3 cells): {len(prim)}")


def git_preds(commit, rundir, setname):
    # an earlier version of an overwritten run folder, shipped in the runs archive as restored_runs/<run>@<commit>
    return corr_rows(read(RESTORED / f"{rundir}@{commit}" / f"preds_{setname}.jsonl"))


A2RUN = "A2-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt-s{}"
def load(src, s, setname):
    if src == "s2":
        return corr_rows(read(SCR / "s2" / {16: "520", 17: "525", 18: "530"}[s] / f"preds_{setname}.jsonl"))
    if src == "git542":
        return git_preds("c325c2c6", A2RUN.format(20), setname)
    return corr_rows(read(RUNS / src / f"preds_{setname}.jsonl"))


orig = {16: "s2", 17: "s2", 18: "s2", 19: A2RUN.format(19), 20: "git542"}
retrain = {s: A2RUN.format(s) for s in (16, 17, 18, 19, 20)}
d27 = {s: f"A2-khopd27-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt-s{s}" for s in range(16, 21)}


def rc_matrix(srcs, rc):
    M = []
    for s, src in srcs.items():
        sets = {cells[i][rc][0] for i in prim}
        cache = {x: load(src, s, f"hint_{x}") for x in sets}
        M.append([cache[cells[i][rc][0]][i] for i in prim])
    return M


for name, ref in (("original answer-only (520/525/530/540/542)", orig), ("re-trained answer-only (current runs/)", retrain)):
    for rc in (1, 2, 3):
        print(f"- D27 - {name}, rc{rc} (249 items): {stats(rc_matrix(d27, rc), rc_matrix(ref, rc))}")
    if name.startswith("re-trained"):
        ids = None
        A = [load(d27[s], s, "gsm8kv2") for s in range(16, 21)]; B = [load(retrain[s], s, "gsm8kv2") for s in range(16, 21)]
        ids = sorted(set.intersection(*[set(x) for x in A + B]))
        print(f"- D27 - {name}, gsm8kv2 ({len(ids)} items): {stats([[a[i] for i in ids] for a in A], [[b[i] for i in ids] for b in B])}")
