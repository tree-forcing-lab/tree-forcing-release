"""Paper ledger: every number the TMLR paper cites, recomputed with ONE rule and ONE statistic.

Selection (K&K arms): per arm, the eval point (mid epoch or end) with the best seed-mean val accuracy over the arm's
own seeds, ties to fewer epochs. Sets evaluated only at the end
of training (BBH, FOLIO, MuSR, GSM8K-MC transfer) are read at the end for both arms and labelled so. GSM8K arms are
fixed epoch counts chosen earlier by valx (best-vs-best, 10-19), read at the end.
Statistic: seed-level paired t interval (df = k - 1) and a two-level bootstrap (B = 10000, numpy seed 0); "sig." = both
intervals exclude 0. Items: "per-seed" (default for epoch-selected K&K rows, the estimator of the readouts
b21 / b23_readout.py): each seed's difference on the items both of its runs hold (earlier mid-epoch kkdeep preds hold
500 of the 2,000 items, later runs hold all 2,000); the bootstrap resamples seeds, then items within each drawn
seed. "common" (scripts/paired_stats.paired_stats): the items held by every run of both arms, shared indices.
 RUNS=runs_archive/runs python3 analysis/ledger.py > ledger.md"""
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
# Run outputs: RUNS (default runs_archive/runs) and RESTORED (default runs_archive/restored_runs), as unpacked from
# tree-forcing-runs.tar.zst.
R = Path(os.environ.get("RUNS", ROOT / "runs_archive/runs"))
import paired_stats as P  # noqa: E402


# Run folders that a later job with the same name overwrote are read from their FIRST completed run, restored from the
# results repository's git history (restored_runs/: the answer-only GSM8K runs of four seeds). Rule: the first completed run of each (configuration, seed) is the one we report.
RESTORED = Path(os.environ.get("RESTORED", R.parent / "restored_runs"))


def rundir(name):
    return RESTORED / name if (RESTORED / name / "metrics.json").exists() else R / name
M4 = "Qwen3.5-4B"


def kk(arm, tag, seeds, data="student_kk_tf"):
    return {s: f"{arm}-{tag}-{M4}-{data}-s{s}" for s in seeds}


ARMS = {
    "A2": {**kk("A2", "c12lr3ch", (13, 14, 15)), **kk("A2", "c12lr3chv", (16, 17, 18))},
    "Tree": {**kk("GFPS", "c6lr3ch", (13, 14, 15)), **kk("GFPS", "c6lr3chv", (16, 17, 18))},
    "A2 lr1.5": kk("A2", "c12lr1p5chv", range(13, 19)),
    "Tree lr1.5": kk("GFPS", "c6lr1p5chv", range(13, 19)),
    "A2 KL": kk("A2", "c12lr3chkl", range(13, 19)),
    "Tree KL": kk("GFPS", "c6lr3chkl", range(13, 19)),
    "Tree s13-15": kk("GFPS", "c6lr3ch", (13, 14, 15)),
    "Shuffled": kk("GFPSR", "c6lr3ch", (13, 14, 15)),
    "A7 (writes proof)": kk("A7", "c12lr3ch", range(13, 19), "student_kk_tf_rat"),
    "Proof, tree path (rn)": kk("GFPS", "c12lr3chrn", range(13, 19), "student_kk_tf_rat_nonodes"),
    "Tree s16-18": kk("GFPS", "c6lr3chv", (16, 17, 18)),
    "A2 s16-18": kk("A2", "c12lr3chv", (16, 17, 18)),
}
# Reruns of the same configs (tags c6lr3cht / c12lr3cht, seeds 13-15) as extra paired units 113-115
ARMS["Tree +b19"] = {**ARMS["Tree"], **{100 + s: n for s, n in kk("GFPS", "c6lr3cht", (13, 14, 15)).items()}}
ARMS["A2 +b19"] = {**ARMS["A2"], **{100 + s: n for s, n in kk("A2", "c12lr3cht", (13, 14, 15)).items()}}
BASE_RUN = "base-Qwen3.5-4B"
G = "student_gsm8k_tf_vt"
GSM = {
    "GSM A2 1ep": {s: f"A2-c1lr3ch-{M4}-{G}-s{s}" for s in range(13, 19)},
    "GSM Tree 2ep": {**{s: f"GFPQ-c2lr3ch-{M4}-{G}-s{s}" for s in (13, 14, 15)}, **{s: f"GFPQ-c2lr3-{M4}-{G}-s{s}" for s in (16, 17, 18)}},
    "GSM Shuffled 2ep": {**{s: f"GFPQR-c2lr3ch-{M4}-{G}-s{s}" for s in (13, 14, 15)}, **{s: f"GFPQR-c2lr3-{M4}-{G}-s{s}" for s in (16, 17, 18)}},
    "GSM Tree 2ep s16-18": {s: f"GFPQ-c2lr3-{M4}-{G}-s{s}" for s in (16, 17, 18)},
    "GSM Placebo 2ep": {s: f"GFPQX-c2lr3-{M4}-{G}-s{s}" for s in (16, 17, 18)},
}


def traj(name):
    m = json.load(open(rundir(name) / "metrics.json"))
    p = {round(x["epoch"]): x for x in m["trajectory"]}
    E = int(round(m["args"]["epochs"]))
    p[E] = m["eval"]
    return p, E


def pick(runs):
    T = {s: traj(n) for s, n in runs.items()}
    eps = sorted(set.intersection(*[set(v[0]) for v in T.values()]))
    e = max(eps, key=lambda e: (round(st.mean(v[0][e]["val"]["acc"] for v in T.values()), 6), -e))
    return e, {s: v[1] for s, v in T.items()}


def rows(name, setname, e=None, E=None):
    f = rundir(name) / (f"preds_{setname}.jsonl" if e is None or e == E else f"mid/preds_{setname}_e{e}.jsonl")
    return [json.loads(line) for line in open(f)]


def correct(name, setname, e=None, E=None):
    return {r["item_id"]: int(r["labels"][int(np.argmax(r["probs"]))] == r["gold_label"]) for r in rows(name, setname, e, E)}


def ece(name, setname, e=None, E=None):
    from cotdistill.metrics import calibration
    rr = [r for r in rows(name, setname, e, E) if r.get("probs") and r["gold_label"] in r["labels"]]
    return 100 * calibration([r["probs"] for r in rr], [r["labels"].index(r["gold_label"]) for r in rr], bins=15)["ece"]


TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def per_seed_stats(D, B=10000, seed=0):
    """D: list of per-seed item-difference arrays (A - B, 0/±1). Mean of per-seed means, seed-t, two-level bootstrap."""
    rng = np.random.default_rng(seed)
    per = np.array([d.mean() for d in D]); k = len(D); m = per.mean()
    sd = per.std(ddof=1) if k > 1 else float("nan")
    t = (m - TCRIT[k - 1] * sd / np.sqrt(k), m + TCRIT[k - 1] * sd / np.sqrt(k)) if k > 1 else (None, None)
    bs = np.empty(B)
    for b in range(B):
        si = rng.integers(0, k, k)
        bs[b] = np.mean([D[j][rng.integers(0, len(D[j]), len(D[j]))].mean() for j in si])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": m, "d": per, "t": {"lo": t[0], "hi": t[1]}, "two_level": (lo, hi),
            "significant": bool(k > 1 and not (lo <= 0 <= hi) and not (t[0] <= 0 <= t[1]))}


def compare(a, b, sets, arms, select=True, label="", items="per-seed"):
    A, B = arms[a], arms[b]
    seeds = sorted(set(A) & set(B))
    ea, Ea = pick({s: A[s] for s in seeds}) if select else (None, {})
    eb, Eb = pick({s: B[s] for s in seeds}) if select else (None, {})
    ca, cb = {}, {}
    for s in seeds:
        da, db = {}, {}
        for x in sets:
            for i, v in correct(A[s], x, ea, Ea.get(s)).items():
                da[f"{x}:{i}"] = v
            for i, v in correct(B[s], x, eb, Eb.get(s)).items():
                db[f"{x}:{i}"] = v
        ca[s], cb[s] = da, db
    if items == "per-seed":
        own = [sorted(set(ca[s]) & set(cb[s])) for s in seeds]
        D = [np.array([ca[s][i] - cb[s][i] for i in ids], float) for s, ids in zip(seeds, own)]
        r = per_seed_stats(D)
        accA = np.mean([np.mean([ca[s][i] for i in ids]) for s, ids in zip(seeds, own)])
        accB = np.mean([np.mean([cb[s][i] for i in ids]) for s, ids in zip(seeds, own)])
        nitems = "/".join(sorted({str(len(x)) for x in own}, key=int))
    else:
        ids = sorted(set.intersection(*[set(ca[s]) & set(cb[s]) for s in seeds]))
        MA = np.array([[ca[s][i] for i in ids] for s in seeds])
        MB = np.array([[cb[s][i] for i in ids] for s in seeds])
        r = P.paired_stats(MA, MB, n_boot=10000, boot_seed=0)
        accA, accB, nitems = MA.mean(), MB.mean(), str(len(ids))
    t = r["t"]
    tlo = "" if t["lo"] is None else f"[{100 * t['lo']:+.2f}, {100 * t['hi']:+.2f}]"
    print(f"| {a} − {b} | {'+'.join(sets)}{label} | {len(seeds)} ({','.join(map(str, seeds))}) | ep {ea}/{eb} | "
          f"{100 * accA:.1f} / {100 * accB:.1f} | **{100 * r['mean']:+.2f}** | "
          f"{', '.join(f'{100 * x:+.1f}' for x in r['d'])} | {tlo} | [{100 * r['two_level'][0]:+.2f}, {100 * r['two_level'][1]:+.2f}] | "
          f"{'yes' if r['significant'] else 'no'} | {nitems} |")


HEAD = ("| comparison (A − B) | test set | seeds | epochs A/B | acc A / B | mean | per seed | seed-t 95% | two-level 95% "
        "| sig. | items |\n|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    print("# Paper ledger (analysis/ledger.py)\n")
    print("## K&K (Qwen3.5-4B), one-pass readout, epochs selected on val\n" + HEAD)
    for a, b in (("Tree", "A2"), ("Tree lr1.5", "A2 lr1.5"), ("Tree KL", "A2 KL"), ("Tree s13-15", "Shuffled"),
                 ("Tree", "A7 (writes proof)"), ("A7 (writes proof)", "A2"), ("Tree", "Proof, tree path (rn)"),
                 ("Proof, tree path (rn)", "A2"), ("Shuffled", "A2")):
        compare(a, b, ["kkdeep"], ARMS)
    for a, b in (("Tree", "A2"), ("Tree s13-15", "Shuffled"), ("Tree", "A7 (writes proof)")):
        compare(a, b, ["kk"], ARMS)
    compare("Tree s16-18", "A2 s16-18", ["xkk"], ARMS)
    print("\n## K&K, kkdeep full 2,000 items at the end of training (robustness: no epoch selection)\n" + HEAD)
    for a, b in (("Tree", "A2"), ("Tree s13-15", "Shuffled"), ("Tree", "A7 (writes proof)"), ("A7 (writes proof)", "A2"),
                 ("Tree", "Proof, tree path (rn)"), ("Tree lr1.5", "A2 lr1.5"), ("Tree KL", "A2 KL")):
        compare(a, b, ["kkdeep"], ARMS, select=False, label=" (end, 2,000)", items="common")
    print("\n## K&K-trained models on other tasks (end of training, both arms)\n" + HEAD)
    for x in ("bbh500", "bbh", "folio", "musr", "gsm8k"):
        try:
            compare("Tree", "A2", [x], ARMS, select=False, label=" (end)", items="common")
        except FileNotFoundError as e:
            print(f"| Tree − A2 | {x} | missing: {e.filename} |")
    print("\n## Untrained Qwen3.5-4B (same one-pass readout; base-Qwen3.5-4B)\n| set | accuracy | items |\n|---|---|---|")
    for x in ("kkdeep", "kk", "bbh", "folio", "musr", "gsm8k"):
        c = correct(BASE_RUN, x)
        print(f"| {x} | {100 * np.mean(list(c.values())):.1f} | {len(c)} |")
    print("\n## Calibration: top-1 ECE (15 bins), mean over seeds (run SD); K&K kkdeep at the selected epoch, others at the end\n"
          "| arm | set | epoch | ECE | accuracy |\n|---|---|---|---|---|")
    for arm in ("A2", "Tree", "Shuffled", "A7 (writes proof)"):
        e, E = pick(ARMS[arm])
        for x, sel in (("kkdeep", True), ("bbh", False), ("folio", False), ("musr", False), ("gsm8k", False)):
            if arm in ("Shuffled", "A7 (writes proof)") and x != "kkdeep":
                continue
            v = [ece(n, x, e if sel else None, E[s]) for s, n in ARMS[arm].items()]
            a = [100 * np.mean(list(correct(n, x, e if sel else None, E[s]).values())) for s, n in ARMS[arm].items()]
            print(f"| {arm} | {x} | {e if sel else 'end'} | {np.mean(v):.1f} ({np.std(v, ddof=1):.1f}) | {np.mean(a):.1f} |")
    for x in ("kkdeep", "bbh", "folio", "musr", "gsm8k"):
        print(f"| untrained | {x} | - | {ece(BASE_RUN, x):.1f} | {100 * np.mean(list(correct(BASE_RUN, x).values())):.1f} |")
    for arm in ("GSM A2 1ep", "GSM Tree 2ep", "GSM Shuffled 2ep"):
        for x in ("gsm8kv2", "gsm8kx"):
            v = [ece(n, x) for n in GSM[arm].values()]
            print(f"| {arm} | {x} | end | {np.mean(v):.1f} ({np.std(v, ddof=1):.1f}) | |")
    print("\n## Reruns pooled (same configs, seeds 13-15 rerun as units 113-115; appendix)\n" + HEAD)
    compare("Tree +b19", "A2 +b19", ["kkdeep"], ARMS)
    compare("Tree +b19", "A2 +b19", ["kkdeep"], ARMS, select=False, label=" (end, 2,000)", items="common")
    print("\n## GSM8K (Qwen3.5-4B), tuned epochs, end of training\n" + HEAD)
    for sets, lab in ((["gsm8kv2", "gsm1k"], " (original)"), (["gsm8kx", "gsm1kx"], " (cue-free)")):
        compare("GSM Tree 2ep", "GSM A2 1ep", sets, GSM, select=False, label=lab, items="common")
        compare("GSM Tree 2ep", "GSM Shuffled 2ep", sets, GSM, select=False, label=lab, items="common")
        compare("GSM Shuffled 2ep", "GSM A2 1ep", sets, GSM, select=False, label=lab, items="common")
        compare("GSM Tree 2ep s16-18", "GSM Placebo 2ep", sets, GSM, select=False, label=lab, items="common")


if __name__ == "__main__":
    main()
