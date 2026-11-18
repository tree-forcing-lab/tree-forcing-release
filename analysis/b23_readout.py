"""Batch 23 readout (cloud/chtc/README_23.md section 3, written 2026-11-10 before any result).
Arms: A2 and GFPS = the 6-seed control pool (batch 16 s13-15 c12lr3ch / c6lr3ch, batch 21 s16-18 c12lr3chv / c6lr3chv);
A7 (c12lr3ch, data student_kk_tf_rat) and rn (GFPS c12lr3chrn, data student_kk_tf_rat_nonodes), seeds 13-18.
Epoch per arm: best seed-mean one-pass val over the arm's available runs, ties to fewer. One-pass kkdeep (2,000).
Primary: GFPS - rn, claim if mean >= +1.5, at least 5 of 6 seeds > 0, two-level lower bound > 0 (seeds resampled,
items resampled within the paired difference). Report: GFPS - A7, A7 - A2, rn - A2, rn - A7; kk; generation (A7, rn).
Interim when rn has fewer than 6 seeds (no claim).   python3 -I analysis/b23_readout.py runs_archive/runs"""
import json, math, random, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("RUNS", str(Path(__file__).resolve().parents[1] / "runs_archive/runs"))); S = (13, 14, 15, 16, 17, 18)
NAME = {"A2": lambda s: f"A2-c12lr3ch{'' if s <= 15 else 'v'}-Qwen3.5-4B-student_kk_tf-s{s}",
        "GFPS": lambda s: f"GFPS-c6lr3ch{'' if s <= 15 else 'v'}-Qwen3.5-4B-student_kk_tf-s{s}",
        "A7": lambda s: f"A7-c12lr3ch-Qwen3.5-4B-student_kk_tf_rat-s{s}",
        "rn": lambda s: f"GFPS-c12lr3chrn-Qwen3.5-4B-student_kk_tf_rat_nonodes-s{s}"}
def traj(n):
    m = json.load(open(R / n / "metrics.json"))
    p = {round(x["epoch"]): x for x in m["trajectory"]}; E = int(m["args"]["epochs"]); p[E] = m["eval"]; return p, E, m
def item_correct(n, e, E, setname):
    f = R / n / (f"preds_{setname}.jsonl" if e == E else f"mid/preds_{setname}_e{e}.jsonl")
    out = {}
    for line in open(f):
        r = json.loads(line); pr = r["probs"]
        out[r["item_id"]] = float(r["labels"][max(range(len(pr)), key=pr.__getitem__)] == r["gold_label"])
    return out
arms = {}
for a, fn in NAME.items():
    runs = {s: traj(fn(s)) for s in S if (R / fn(s) / "metrics.json").exists()}
    eps = sorted(set.intersection(*[set(v[0]) for v in runs.values()]))
    e = max(eps, key=lambda e: (round(st.mean(v[0][e]["val"]["acc"] for v in runs.values()), 6), -e))
    arms[a] = {"runs": runs, "e": e}
    acc = {k: round(st.mean(100 * v[0][e][k]["acc"] for v in runs.values()), 1) for k in ("val", "kk", "kkdeep")}
    print(f"{a:5s} seeds {sorted(runs)} epoch {e:2d}: {acc}")
def pair(b, a, setname="kkdeep", B=4000, need=None):
    seeds = sorted(set(arms[a]["runs"]) & set(arms[b]["runs"]))
    D = []
    for s in seeds:
        cb = item_correct(NAME[b](s), arms[b]["e"], arms[b]["runs"][s][1], setname)
        ca = item_correct(NAME[a](s), arms[a]["e"], arms[a]["runs"][s][1], setname)
        ids = sorted(set(ca) & set(cb)); D.append([cb[i] - ca[i] for i in ids])
    per = [100 * st.mean(d) for d in D]; m = st.mean(per)
    rng = random.Random(0); bs = []
    for _ in range(B):
        ss = [D[rng.randrange(len(D))] for _ in D]
        bs.append(st.mean(st.mean(d[rng.randrange(len(d))] for _ in range(len(d))) for d in ss))
    bs.sort(); lo, hi = 100 * bs[int(.025 * B)], 100 * bs[int(.975 * B)]
    sd = st.stdev(per) if len(per) > 1 else float("nan"); t = m / (sd / math.sqrt(len(per))) if len(per) > 1 and sd > 0 else float("nan")
    txt = f"{m:+.2f}  seeds {seeds}: {', '.join(f'{x:+.1f}' for x in per)}  t {t:.2f}  2L [{lo:+.2f}, {hi:+.2f}]"
    if need is not None:
        full = len(seeds) == 6
        ok = m >= need and sum(x > 0 for x in per) >= 5 and lo > 0
        txt += "  -> " + ("PASS" if ok and full else "not met" if full else "INTERIM (rn has %d of 6 seeds)" % len(seeds))
    return txt
print("\nPRIMARY GFPS - rn (kkdeep):", pair("GFPS", "rn", need=1.5))
for b, a in (("GFPS", "A7"), ("A7", "A2"), ("rn", "A2"), ("rn", "A7")):
    print(f"{b} - {a} (kkdeep): {pair(b, a)}")
print("GFPS - rn (kk):", pair("GFPS", "rn", "kk")); print("GFPS - A7 (kk):", pair("GFPS", "A7", "kk"))
for a in ("A7", "rn"):
    g = [v[2].get("gen_eval", {}) for v in arms[a]["runs"].values()]
    print(f"{a} generation (end of training): " + ", ".join(f"{k} {st.mean(100 * x[k]['acc'] for x in g):.1f}" for k in ("val", "kk"))
          + f"; tokens {st.mean(x['kk']['mean_new_tokens'] for x in g):.0f}")
