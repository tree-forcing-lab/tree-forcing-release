"""Zero-GPU analyses for the review (11-17), from saved predictions only.
(1) K&K-Deep split into in-range (4-10 inhabitants) and beyond (11-20).
(2) K&K-Deep by distractor difficulty: per item, the fewest statements any wrong option violates; and, for wrong answers,
    how many statements the chosen option violates (Tree vs answer-only).
(3) BBH by task type: two-option vs more-option tasks, and per task, for AO / TF / permuted / proof generation.
(4) Temperature scaling: per run, T fitted on its own K&K validation predictions (end of training, NLL), applied to BBH
    and K&K-Deep; ECE / NLL / Brier before and after."""
import collections, json, math, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "src")]
import ledger as L
from cotdistill.sources import _kk_formula, _kk_val
from cotdistill.metrics import calibration
A = L.ARMS

def subset_compare(a, b, setname, keep, label, select=True):
    # per-seed items restricted to keep(item_id)
    seeds = sorted(set(A[a]) & set(A[b]))
    ea, Ea = L.pick({s: A[a][s] for s in seeds}) if select else (None, {})
    eb, Eb = L.pick({s: A[b][s] for s in seeds}) if select else (None, {})
    D, accA, accB = [], [], []
    for s in seeds:
        ca = L.correct(A[a][s], setname, ea, Ea.get(s)); cb = L.correct(A[b][s], setname, eb, Eb.get(s))
        ids = [i for i in ca if i in cb and keep(i)]
        D.append(np.array([ca[i] - cb[i] for i in ids], float)); accA.append(np.mean([ca[i] for i in ids])); accB.append(np.mean([cb[i] for i in ids]))
    r = L.per_seed_stats(D)
    print(f"| {a} − {b} | {label} | {len(seeds)} | {100*np.mean(accA):.1f} / {100*np.mean(accB):.1f} | **{100*r['mean']:+.2f}** | "
          f"{', '.join(f'{100*x:+.1f}' for x in r['d'])} | [{100*r['t']['lo']:+.2f}, {100*r['t']['hi']:+.2f}] | "
          f"[{100*r['two_level'][0]:+.2f}, {100*r['two_level'][1]:+.2f}] | {'yes' if r['significant'] else 'no'} | {'/'.join(sorted({str(len(d)) for d in D}))} |")

H = "| comparison | subset | seeds | acc A / B | mean | per seed | seed-t | two-level | sig. | items |\n|---|---|---|---|---|---|---|---|---|---|"
deep = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "data/eval/kk_deep.jsonl")}
print("## (1) K&K-Deep in range vs beyond\n" + H)
for sel, lab in ((True, "selected"), (False, "end")):
    for name, keep in (("4-10 people", lambda i: deep[i]["depth"] <= 10), ("11-20 people", lambda i: deep[i]["depth"] >= 11)):
        subset_compare("Tree", "A2", "kkdeep", keep, f"{name} ({lab})", select=sel)

# (2) distractor difficulty
def parse_item(it):
    p = it["prompt"]
    stm = re.findall(r'^- (\w+) says: "(.*)"$', p, re.M)
    opts = {}
    for lab, body in re.findall(r"^([A-Z])\) (.*)$", p, re.M):
        if "None of the above" in body: continue
        opts[lab] = {nm: role == "knight" for nm, role in re.findall(r"(\w+): (knight|knave)", body)}
    people = set(nm for nm, _ in stm) | set(n for o in opts.values() for n in o)
    forms = [(sp, _kk_formula(s, people)) for sp, s in stm]
    return forms, opts
viol, fails = {}, 0
for i, it in deep.items():
    forms, opts = parse_item(it)
    if any(f is None for _, f in forms) or not opts: fails += 1; continue
    v = {lab: sum((_kk_val(f, r) != r[sp]) for sp, f in forms) for lab, r in opts.items() if all(sp in r for sp, _ in forms)}
    if v.get(it["gold_label"], 1) != 0: fails += 1; continue
    viol[i] = v
print(f"\n## (2) distractor difficulty (parsed {len(viol)} of {len(deep)} items; {fails} unparsed or gold not consistent)")
mins = {i: min(x for lab, x in v.items() if lab != deep[i]["gold_label"]) for i, v in viol.items()}
print("items by fewest statements violated by a wrong option:", sorted(collections.Counter(mins.values()).items()))
print(H)
for k, keep in (("hardest distractor violates 1", lambda i: mins.get(i) == 1), ("violates >= 2", lambda i: mins.get(i, 0) >= 2)):
    subset_compare("Tree", "A2", "kkdeep", keep, k, select=False)
print("\nwrong answers (end of training): statements violated by the chosen option, mean over runs")
for a in ("A2", "Tree"):
    cnt = collections.Counter(); tot = 0
    for s, n in A[a].items():
        for r in L.rows(n, "kkdeep"):
            pred = r["labels"][int(np.argmax(r["probs"]))]
            if r["item_id"] in viol and pred != r["gold_label"]:
                cnt[viol[r["item_id"]].get(pred, "none")] += 1; tot += 1
    print(f"  {a}: {tot / len(A[a]):.0f} wrong per run; " + ", ".join(f"{k}: {100 * v / tot:.0f}%" for k, v in sorted(cnt.items(), key=str)))

# (3) BBH by task type
print("\n## (3) BBH: two-option vs more-option tasks (end of training)\n" + H)
# task ("group") and option labels of every BIG-Bench Hard item, read from the prediction rows of the untrained model
# (each row carries them), so the BIG-Bench Hard evaluation file itself is not needed
bbh = {r["item_id"]: {"group": r["group"], "labels": r["labels"]} for r in L.rows(L.BASE_RUN, "bbh")}
two = lambda i: len(bbh[i]["labels"]) == 2
for a, b in (("Tree", "A2"), ("Tree s13-15", "Shuffled"), ("Shuffled", "A2 s13-15") if "A2 s13-15" in A else ("Shuffled", "A2")):
    pass
A["A2 s13-15"] = L.kk("A2", "c12lr3ch", (13, 14, 15))
for a, b in (("Tree", "A2"), ("Tree s13-15", "Shuffled"), ("Shuffled", "A2 s13-15"), ("A7 (writes proof)", "A2")):
    for lab, keep in (("2 options", two), ("3+ options", lambda i: not two(i))):
        subset_compare(a, b, "bbh", keep, lab, select=False)
print("\nper task, accuracy (%) at the end of training, mean over seeds: untrained / AO / TF / permuted(13-15) / proof gen.")
tasks = sorted({bbh[i]["group"] for i in bbh})
def task_acc(runs, t):
    return np.mean([np.mean([c for i, c in L.correct(n, "bbh").items() if bbh[i]["group"] == t]) for n in runs.values()]) * 100
base = {t: 100 * np.mean([c for i, c in L.correct(L.BASE_RUN, "bbh").items() if bbh[i]["group"] == t]) for t in tasks}
for t in tasks:
    print(f"  {t.split('/')[-1]:42s} n {sum(1 for i in bbh if bbh[i]['group'] == t):4d} | {base[t]:5.1f} {task_acc(A['A2'], t):5.1f} {task_acc(A['Tree'], t):5.1f} {task_acc(A['Shuffled'], t):5.1f} {task_acc(A['A7 (writes proof)'], t):5.1f}")

# (4) temperature scaling
def probs_gold(name, setname):
    rr = [r for r in L.rows(name, setname) if r.get("probs") and r["gold_label"] in r["labels"]]
    return [np.clip(np.array(r["probs"], float), 1e-12, 1) for r in rr], [r["labels"].index(r["gold_label"]) for r in rr]
def scale(P, T):
    out = []
    for p in P:
        z = np.log(p) / T; z -= z.max(); e = np.exp(z); out.append(e / e.sum())
    return out
def nll(P, g): return -np.mean([math.log(max(p[k], 1e-12)) for p, k in zip(P, g)])
def fit_T(P, g):
    grid = np.exp(np.linspace(math.log(0.25), math.log(8), 120)); return min(grid, key=lambda T: nll(scale(P, T), g))
print("\n## (4) temperature scaling: T fitted per run on its K&K val predictions (end), applied unchanged; mean over runs")
print("| arm | T | set | ECE before → after | NLL before → after | Brier before → after |\n|---|---|---|---|---|---|")
for a in ("A2", "Tree", "Shuffled", "A7 (writes proof)"):
    res = collections.defaultdict(list); Ts = []
    for s, n in A[a].items():
        Pv, gv = probs_gold(n, "val"); T = fit_T(Pv, gv); Ts.append(T)
        for x in ("bbh", "kkdeep"):
            P, g = probs_gold(n, x); c0 = calibration([list(p) for p in P], g); c1 = calibration([list(p) for p in scale(P, T)], g)
            res[x].append((c0["ece"], c1["ece"], c0["nll"], c1["nll"], c0["brier"], c1["brier"]))
    for x, v in res.items():
        m = np.mean(v, 0)
        print(f"| {a} | {np.mean(Ts):.2f} | {x} | {100*m[0]:.1f} → {100*m[1]:.1f} | {m[2]:.3f} → {m[3]:.3f} | {m[4]:.3f} → {m[5]:.3f} |")
