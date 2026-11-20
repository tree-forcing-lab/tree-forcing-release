"""Do the intermediate values that Tree Forcing installs reach the one-pass answer? (zero-GPU readout,

For every run: the end-of-training node probes (preds_subq.jsonl: yes/no questions on the val problems' tree nodes,
`correct` per question) and the end-of-training val answers (preds_val.jsonl). Per run:
 node_acc mean `correct` over all node questions
 coupling mean node accuracy on val items the run answered right minus on items it answered wrong
 pair_order share of (true, false) node pairs with the same stem where p_yes(true) > p_yes(false) (GSM8K value nodes)
 by_depth node accuracy by the node's depth in the tree (K&K: subqs node_depth; GSM8K: the step of the node)
Aggregated over the runs of an arm (mean, run SD) with a 95% item bootstrap (items resampled within each run, B=2000).
Run sets: K&K = the 6-seed control pool (s13-15 c12lr3ch / c6lr3ch, s16-18 c12lr3chv / c6lr3chv);
GSM8K = the runs at the epochs (A2 1, GFPQ 1, A7 3, rl 2; 3 seeds each).
Caveat: the node-probe prompt shows the answer options (student.py node-probe eval). Observational: no causal claim.
 python3 -I analysis/node_coupling.py runs_archive/runs"""
import collections
import glob
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("RUNS", str(Path(__file__).resolve().parents[1] / "runs_archive/runs")))
ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "K&K A2": [f"A2-c12lr3ch-Qwen3.5-4B-student_kk_tf-s{s}" for s in (13, 14, 15)] + [f"A2-c12lr3chv-Qwen3.5-4B-student_kk_tf-s{s}" for s in (16, 17, 18)],
    "K&K tree (GFPS)": [f"GFPS-c6lr3ch-Qwen3.5-4B-student_kk_tf-s{s}" for s in (13, 14, 15)] + [f"GFPS-c6lr3chv-Qwen3.5-4B-student_kk_tf-s{s}" for s in (16, 17, 18)],
    "GSM8K A2": [f"A2-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt_rat-s{s}" for s in (13, 14, 15)],
    "GSM8K tree (GFPQ)": [f"GFPQ-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt_rat-s{s}" for s in (13, 14, 15)],
    "GSM8K A7 (writes rationale)": [f"A7-c3lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt_rat-s{s}" for s in (13, 14, 15)],
    "GSM8K tree + rationale (rl)": [f"GFPQ-c2lr3chrl-Qwen3.5-4B-student_gsm8k_tf_vt_rat-s{s}" for s in (13, 14, 15)],
}
DEPTH_SRC = {"K&K": ROOT / "data/student_kk_tf/val.jsonl", "GSM8K": ROOT / "data/student_gsm8k_tf_vt_rat/val.jsonl"}


def depth_maps():
    out = {}
    for task, p in DEPTH_SRC.items():
        m = {}
        for line in open(p):
            it = json.loads(line)
            for n in it.get("subqs") or []:
                d = n.get("node_depth")
                if d is None:                                    # GSM8K value nodes: the step of the node (s<k>v / s<k>n)
                    mm = re.match(r"s(\d+)", str(n.get("id", "")))
                    d = int(mm.group(1)) if mm else None
                for q in (n.get("question"), n.get("opposite")):
                    if q:
                        m[(it["item_id"], q)] = d
        out[task] = m
    return out


def answers(f):
    out = {}
    for line in open(f):
        p = json.loads(line)
        pr = p.get("probs")
        if pr is None:
            continue
        pred = max(pr, key=pr.get) if isinstance(pr, dict) else p["labels"][max(range(len(pr)), key=pr.__getitem__)]
        out[p["item_id"]] = pred == p["gold_label"]
    return out


def load(run, dmap):
    rows = [json.loads(l) for l in open(R / run / "preds_subq.jsonl")]
    rows = [r for r in rows if r.get("truth_known") and r.get("p_yes") is not None]
    ans = answers(R / run / "preds_val.jsonl")
    per = collections.defaultdict(list)
    dep = collections.defaultdict(list)
    stems = collections.defaultdict(dict)
    for r in rows:
        per[r["item_id"]].append(r["correct"])
        dep[dmap.get((r["item_id"], r["question"]))].append(r["correct"])
        if "Is it " in r["question"]:
            stems[(r["item_id"], r["question"].rsplit("Is it ", 1)[0])][r["target"]] = r["p_yes"]
    pairs = [v[True] > v[False] for v in stems.values() if True in v and False in v]
    return per, ans, dep, pairs


def stat(per, ans, items):
    allc = [c for i in items for c in per[i]]
    right = [c for i in items if ans.get(i) for c in per[i]]
    wrong = [c for i in items if i in ans and not ans[i] for c in per[i]]
    return (100 * st.mean(allc), (100 * (st.mean(right) - st.mean(wrong))) if right and wrong else float("nan"),
            100 * st.mean(ans[i] for i in items if i in ans))


def main():
    dmaps = depth_maps()
    rng = random.Random(0)
    for arm, runs in ARMS.items():
        runs = [r for r in runs if (R / r / "preds_subq.jsonl").exists() and (R / r / "preds_val.jsonl").exists()]
        if not runs:
            print(f"{arm}: no runs"); continue
        task = arm.split()[0]
        data = [load(r, dmaps[task]) for r in runs]
        items = [sorted(i for i in d[0] if i in d[1]) for d in data]
        point = [stat(d[0], d[1], it) for d, it in zip(data, items)]
        boots = []
        for _ in range(2000):
            bs = []
            for d, it in zip(data, items):
                smp = [it[rng.randrange(len(it))] for _ in it]
                per = collections.defaultdict(list)
                ans = {}
                for k, i in enumerate(smp):
                    per[(i, k)] = d[0][i]
                    ans[(i, k)] = d[1][i]
                bs.append(stat(per, ans, list(per)))
            boots.append([st.mean(b[j] for b in bs) for j in range(3)])
        ci = [(sorted(b[j] for b in boots)[50], sorted(b[j] for b in boots)[1949]) for j in range(3)]
        m = [st.mean(p[j] for p in point) for j in range(3)]
        sd = [st.stdev(p[j] for p in point) if len(point) > 1 else float("nan") for j in range(3)]
        pairs = [100 * st.mean(d[3]) for d in data if d[3]]
        deps = collections.defaultdict(list)
        for d in data:
            for k, v in d[2].items():
                if k is not None and len(v) >= 30:
                    deps[k].append(100 * st.mean(v))
        dtxt = " ".join(f"d{k}:{st.mean(v):.0f}" for k, v in sorted(deps.items()) if len(v) == len(data))
        print(f"{arm:30s} runs {len(runs)} items {len(items[0])} | node acc {m[0]:.1f} [{ci[0][0]:.1f}, {ci[0][1]:.1f}] "
              f"(run SD {sd[0]:.1f}) | coupling (right - wrong) {m[1]:+.1f} [{ci[1][0]:+.1f}, {ci[1][1]:+.1f}] "
              f"(run SD {sd[1]:.1f}) | val acc {m[2]:.1f}" + (f" | pair order {st.mean(pairs):.1f}" if pairs else "")
              + f"\n{'':30s} by depth: {dtxt}")


if __name__ == "__main__":
    main()
