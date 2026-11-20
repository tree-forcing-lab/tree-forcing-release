"""Checks 11-15: (a) K&K transfer / calibration at MATCHED training length (separate runs, 4 and 8
epochs, seeds 13-15, end of training); (b) GSM8K one-epoch Tree Forcing (the cue-free validation pick on seeds 13-15)
vs one-epoch answer-only; (c) K&K node coupling within people-count strata of val."""
import collections, json, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "analysis")]
import os
sys.argv = [sys.argv[0], os.environ.get("RUNS", str(ROOT / "runs_archive/runs"))]
import ledger as L
M = "Qwen3.5-4B"
print("## (a) matched length, end of training, seeds 13-15\n" + L.HEAD)
arms = {}
for E in (4, 8):
    arms[f"A2 {E}ep"] = {s: f"A2-c{E}lr3ch-{M}-student_kk_tf-s{s}" for s in (13, 14, 15)}
    arms[f"Tree {E}ep"] = {s: f"GFPS-c{E}lr3ch-{M}-student_kk_tf-s{s}" for s in (13, 14, 15)}
for E in (4, 8):
    for x in ("bbh", "kkdeep", "gsm8k", "folio", "musr"):
        L.compare(f"Tree {E}ep", f"A2 {E}ep", [x], arms, select=False, label=" (end)", items="common")
print("\n| arm | set | ECE | acc |\n|---|---|---|---|")
for a, runs in arms.items():
    for x in ("bbh", "kkdeep", "gsm8k"):
        print(f"| {a} | {x} | {np.mean([L.ece(n, x) for n in runs.values()]):.1f} | {100 * np.mean([np.mean(list(L.correct(n, x).values())) for n in runs.values()]):.1f} |")
print("\n## (b) GSM8K one-epoch Tree Forcing vs one-epoch answer-only, seeds 13-15\n" + L.HEAD)
G = {"GSM Tree 1ep": {s: f"GFPQ-c1lr3ch-{M}-student_gsm8k_tf_vt-s{s}" for s in (13, 14, 15)},
     "GSM A2 1ep": {s: f"A2-c1lr3ch-{M}-student_gsm8k_tf_vt-s{s}" for s in (13, 14, 15)},
     "GSM Tree 2ep": {s: f"GFPQ-c2lr3ch-{M}-student_gsm8k_tf_vt-s{s}" for s in (13, 14, 15)}}
for sets, lab in ((["gsm8kv2", "gsm1k"], " (original)"), (["gsm8kx", "gsm1kx"], " (cue-free)"), (["valx"], " (cue-free val)")):
    L.compare("GSM Tree 1ep", "GSM A2 1ep", sets, G, select=False, label=lab, items="common")
    L.compare("GSM Tree 2ep", "GSM A2 1ep", sets, G, select=False, label=lab, items="common")
print("\n## (c) K&K coupling within people-count strata (val), mean over runs; stratum-weighted")
import node_coupling as NC
people = {json.loads(l)["item_id"]: json.loads(l).get("depth") for l in open(ROOT / "data/student_kk_tf/val.jsonl")}
dm = NC.depth_maps()["K&K"]
for arm in ("K&K A2", "K&K tree (GFPS)"):
    vals, accs = [], []
    for run in NC.ARMS[arm]:
        per, ans, _, _ = NC.load(run, dm)
        by = collections.defaultdict(list)
        for i in per:
            if i in ans: by[people.get(i)].append(i)
        cs, ws = [], []
        for p, items in by.items():
            r = [c for i in items if ans[i] for c in per[i]]; w = [c for i in items if not ans[i] for c in per[i]]
            if r and w: cs.append(100 * (st.mean(r) - st.mean(w))); ws.append(len(items))
        vals.append(np.average(cs, weights=ws)); accs.append(len(cs))
    print(f"  {arm}: within-stratum coupling {np.mean(vals):+.2f} (runs {', '.join(f'{v:+.1f}' for v in vals)}; strata used {accs})")
