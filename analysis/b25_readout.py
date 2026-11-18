"""Batch 25 readout (cloud/chtc/README_25.md section 4, written 2026-11-10 before any batch-25 result): K&K on
Qwen3.5-9B, GFPS 6 epochs vs A2 12 epochs, seeds 13-15, with the 4B control of the same recipe next to it.

Rule (README_21 section 3, the same for both models): per arm, the eval point (mid epochs 1..E-1 or the end) with
the best seed-mean val accuracy, ties to fewer epochs. Primary: kkdeep (2,000 items) GFPS - A2, seed-paired, seed t
and the two-level bootstrap of scripts/paired_stats.py. Gate "trees help at 9B": mean >= +1.5, every seed > 0,
two-level lower bound > 0. Also kk, xkk, zebra at the selected points; bbh500 / bbh at the end of training (they are
evaluated only then). The 4B side reads the batch-16 runs at the same seeds (13-15, tags c6lr3ch / c12lr3ch) and,
for reference, the 9-run control of analysis/b21_readout.py (printed by that script, not here).
  python3 analysis/b25_readout.py runs_archive/runs
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import paired_stats, top1  # noqa: E402

R = Path(sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("RUNS", str(ROOT / "runs_archive/runs")))
SEEDS = (13, 14, 15)
MID_SETS = ("kkdeep", "kk", "xkk", "zebra")
END_SETS = ("bbh500", "bbh")
MODELS = {"9B": ("Qwen3.5-9B", "c6lr3ch", "c12lr3ch"), "4B (batch 16)": ("Qwen3.5-4B", "c6lr3ch", "c12lr3ch")}


def run_dir(arm, tag, model, s):
    return R / f"{arm}-{tag}-{model}-student_kk_tf-s{s}"


def points(d: Path) -> dict:
    m = json.load(open(d / "metrics.json"))
    p = {float(x["epoch"]): x["val"]["acc"] for x in m["trajectory"]}
    p[float(m["args"]["epochs"])] = m["eval"]["val"]["acc"]
    return p, float(m["args"]["epochs"])


def preds_file(d: Path, name: str, ep: float, end: float) -> Path:
    if ep == end:
        return d / f"preds_{name}.jsonl"
    e = int(ep) if float(ep).is_integer() else ep
    return d / "mid" / f"preds_{name}_e{e}.jsonl"


def correct(d: Path, name: str, ep: float, end: float) -> dict:
    rows = [json.loads(l) for l in open(preds_file(d, name, ep, end)) if l.strip()]
    return {r["item_id"]: top1(r)[0] for r in rows}


def matrix(dirs, name, ep, end):
    cs = [correct(d, name, ep, end) for d in dirs]
    keys = sorted(cs[0])
    assert all(sorted(c) == keys for c in cs), f"{name}: item sets differ across runs"
    return np.array([[c[k] for k in keys] for c in cs])


for label, (model, tt, ta) in MODELS.items():
    dt = [run_dir("GFPS", tt, model, s) for s in SEEDS]
    da = [run_dir("A2", ta, model, s) for s in SEEDS]
    have = [d for d in dt + da if (d / "metrics.json").exists()]
    print(f"\n== {label}: {len(have)}/6 runs")
    if len(have) < 6:
        continue
    sel = {}
    for arm, ds in (("GFPS", dt), ("A2", da)):
        pts = [points(d) for d in ds]
        eps = sorted(set.intersection(*[set(p) for p, _ in pts]))
        e = max(eps, key=lambda e: (round(st.mean(p[e] for p, _ in pts), 6), -e))
        sel[arm] = (e, pts[0][1])
        print(f"  {arm}: picked epoch {e:g} (seed-mean val {100 * st.mean(p[e] for p, _ in pts):.1f})")
    for name in MID_SETS + END_SETS:
        et, ea = (sel["GFPS"][0], sel["A2"][0]) if name in MID_SETS else (sel["GFPS"][1], sel["A2"][1])
        try:
            cT, cA = matrix(dt, name, et, sel["GFPS"][1]), matrix(da, name, ea, sel["A2"][1])
        except FileNotFoundError as ex:
            print(f"  {name}: missing {ex.filename}")
            continue
        r = paired_stats(cT, cA)
        lo, hi = r["two_level"]
        gate = ""
        if name == "kkdeep":
            ok = r["mean"] >= 0.015 and all(x > 0 for x in r["d"]) and lo > 0
            gate = f"  GATE {'PASS' if ok else 'FAIL'} (mean >= +1.5, every seed > 0, 2L lower > 0)"
        print(f"  {name:7s} GFPS {100 * np.mean(r['acc_a']):.1f} A2 {100 * np.mean(r['acc_b']):.1f}  diff {100 * r['mean']:+.2f} "
              f"seeds {', '.join(f'{100 * x:+.1f}' for x in r['d'])}  2L [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
              f"seed-t p {r['t']['p'] if r['t']['p'] is None else round(r['t']['p'], 3)}{gate}")
