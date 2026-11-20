"""Numbers of the paper that build on the six-seed permuted control and the written-reasoning references:
(1) BIG-Bench Hard with the permuted control at six seeds: two-option vs more-option split, per-task accuracy,
 temperature scaling (T fitted on each run's K&K validation predictions);
(2) GSM8K Tree Forcing (1 epoch) vs the surface placebo (seeds 16-18) and the calibration of the GSM8K arms;
(3) written-reasoning references: generation accuracy of the proof-generation runs on K&K-Held-out and of the
 rationale-generation runs on the cue-free GSM8K sets.
 RUNS=runs_archive/runs python3 analysis/extras.py"""
import collections, json, math, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src")]
import ledger as L  # noqa: E402
from cotdistill.metrics import calibration  # noqa: E402
M = "Qwen3.5-4B"; A = L.ARMS; G = L.GSM
A["Permuted 6"] = {**L.kk("GFPSR", "c6lr3ch", (13, 14, 15)), **L.kk("GFPSR", "c6lr3chv", (16, 17, 18))}
G["GSM Tree 1ep"] = {s: f"GFPQ-c1lr3ch-{M}-student_gsm8k_tf_vt-s{s}" for s in range(13, 19)}
G["GSM Tree 1ep s16-18"] = {s: f"GFPQ-c1lr3ch-{M}-student_gsm8k_tf_vt-s{s}" for s in (16, 17, 18)}
bbh = {r["item_id"]: {"group": r["group"], "labels": r["labels"]} for r in L.rows(L.BASE_RUN, "bbh")}
two = lambda i: len(bbh[i]["labels"]) == 2


def sub(a, b, keep):
    seeds = sorted(set(A[a]) & set(A[b])); D = []
    for s in seeds:
        ca, cb = L.correct(A[a][s], "bbh"), L.correct(A[b][s], "bbh")
        D.append(np.array([ca[i] - cb[i] for i in ca if i in cb and keep(i)], float))
    r = L.per_seed_stats(D)
    return (f"| {a} − {b} | {{lab}} | {len(seeds)} | **{100 * r['mean']:+.2f}** | {', '.join(f'{100 * x:+.1f}' for x in r['d'])} | "
            f"[{100 * r['t']['lo']:+.2f}, {100 * r['t']['hi']:+.2f}] | [{100 * r['two_level'][0]:+.2f}, {100 * r['two_level'][1]:+.2f}] | "
            f"{'yes' if r['significant'] else 'no'} |")


print("## (1) BIG-Bench Hard with the permuted control at six seeds\n")
print("| comparison | subset | seeds | mean | per seed | seed-t | two-level | sig. |\n|---|---|---|---|---|---|---|---|")
for a, b in (("Permuted 6", "A2"), ("Tree", "Permuted 6")):
    for lab, keep in (("2 options", two), ("3+ options", lambda i: not two(i))):
        print(sub(a, b, keep).format(lab=lab))
print("\nper-task accuracy of the permuted control (six seeds, end of training):")
for t in sorted({v["group"] for v in bbh.values()}):
    acc = np.mean([np.mean([c for i, c in L.correct(n, "bbh").items() if bbh[i]["group"] == t]) for n in A["Permuted 6"].values()])
    print(f"  {t.split('/')[-1]}: {100 * acc:.1f}")


def pg(name, x):
    rr = [r for r in L.rows(name, x) if r.get("probs") and r["gold_label"] in r["labels"]]
    return [np.clip(np.array(r["probs"], float), 1e-12, 1) for r in rr], [r["labels"].index(r["gold_label"]) for r in rr]


def scale(P, T):
    o = []
    for p in P:
        z = np.log(p) / T; z -= z.max(); e = np.exp(z); o.append(e / e.sum())
    return o


nll = lambda P, g: -np.mean([math.log(max(p[k], 1e-12)) for p, k in zip(P, g)])
grid = np.exp(np.linspace(math.log(0.25), math.log(8), 120))
print("\ntemperature scaling, permuted control (six seeds):")
res = collections.defaultdict(list); Ts = []
for n in A["Permuted 6"].values():
    Pv, gv = pg(n, "val"); T = min(grid, key=lambda t: nll(scale(Pv, t), gv)); Ts.append(T)
    for x in ("bbh", "kkdeep"):
        P, g = pg(n, x); c0 = calibration([list(p) for p in P], g); c1 = calibration([list(p) for p in scale(P, T)], g)
        res[x].append((c0["ece"], c1["ece"], c0["nll"], c1["nll"], c0["brier"], c1["brier"]))
for x, v in res.items():
    m = np.mean(v, 0)
    print(f"  T {np.mean(Ts):.2f} {x}: ECE {100 * m[0]:.1f} -> {100 * m[1]:.1f}, NLL {m[2]:.3f} -> {m[3]:.3f}, Brier {m[4]:.3f} -> {m[5]:.3f}")

print("\n## (2) GSM8K: Tree Forcing (1 epoch) vs the surface placebo, and calibration\n" + L.HEAD)
for sets, lab in ((["gsm8kv2", "gsm1k"], " (original)"), (["gsm8kx", "gsm1kx"], " (cue-free)")):
    L.compare("GSM Tree 1ep s16-18", "GSM Placebo 2ep", sets, G, select=False, label=lab, items="common")
for x in ("gsm8kv2", "gsm8kx"):
    print(f"ECE {x}: answer-only 1 epoch {np.mean([L.ece(n, x) for n in G['GSM A2 1ep'].values()]):.1f}, "
          f"Tree Forcing 1 epoch {np.mean([L.ece(n, x) for n in G['GSM Tree 1ep'].values()]):.1f}")

print("\n## (3) written-reasoning references (generation evaluations of the same runs)\n")
for lab, runs, sets in (("K&K proof generation", A["A7 (writes proof)"], ("kk",)),
                        ("GSM8K rationale generation", {s: f"A7-c3lr3ch-{M}-student_gsm8k_tf_vt_rat-s{s}" for s in (13, 14, 15)}, ("gsm8kx", "gsm1kx"))):
    g = [json.load(open(L.rundir(n) / "metrics.json")).get("gen_eval", {}) for n in runs.values()]
    accs = {k: st.mean(x[k]["acc"] for x in g) for k in sets}
    toks = st.mean(st.mean(x[k]["mean_new_tokens"] for k in sets) for x in g)
    if len(sets) == 2:   # pooled over the two cue-free sets, weighted by their sizes
        n = {k: x[k].get("n") for x in g[:1] for k in sets}
        w = {"gsm8kx": 1317, "gsm1kx": 1205}
        accs["pooled"] = sum(accs[k] * w[k] for k in sets) / sum(w.values())
    print(f"{lab}: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in accs.items()) + f"; mean generated tokens {toks:.0f}")
