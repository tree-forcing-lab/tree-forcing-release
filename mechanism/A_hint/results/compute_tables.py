#!/usr/bin/env python3
"""Recompute every number of section A (hint experiments) from the preds files.

    python3 results/compute_tables.py --stage 1 [--runs <results repo>/runs]   # pilot, jobs 500-513
    python3 results/compute_tables.py --stage 2                                 # confirmation, jobs 520-533
    (from the A_hint folder; numpy needed; about 30 s each)

Inputs (relative to the A_hint folder):
  stage 1: <runs>/<job>/preds_hint_<set>.jsonl and metrics.json, the run folders runs/5xx_4b_hint_* of the results
           repo (default --runs ../../../runs, i.e. this folder placed at notes/<date>/A_hint in that repo), and
           data/job_logs/<job>.out for the accuracy self-test. data/results_tgz.md5 lists the original tarballs.
  stage 2: data/stage2_preds.tgz, <job>/{preds_hint_*,preds_valx,metrics.json,<job>.out} of the 12 Stage 2 jobs
           (taken from the job result tarballs; in the results repo the A2-c1 run folders were later overwritten by
           jobs 663-665, so the repo tree no longer holds them)
  data/hint/<set>.jsonl                 the 18 evaluated sets (item fields n, r, k, shown, shown_equals_option)
  data/source/gsm8k_mc_v2_steps.jsonl   worked steps, for the remaining critical path slice
Outputs (results/; stage 2 files carry the prefix stage2_ and add stage2_verdict.md):
  tables.md            all tables in markdown
  acc_per_seed.csv     model, job, seed, set, n_items, acc (points)
  contrasts.csv        within-model contrasts A - B, seed-paired, with seed-t and bootstrap intervals
  between_models.csv   arm - baseline on the same condition, seed-paired
  curves.csv           fixed-item curves acc(r) (cohorts n >= 5 and n >= 7), per seed and seed mean
  grid_n_r.csv         acc(true r) by item length n, seed mean per model
  path_slice.csv       true rows (items n >= 5) by remaining critical path vs plain items of that full path
  selftest.txt         the self-tests (preds vs hint files, accuracy vs the job logs)

Statistics: results/paired_stats.py is a byte-identical copy of the repo's scripts/paired_stats.py (md5 35738a91).
Its paired_stats() and top1() are imported, not re-implemented: per-seed difference, seed-level Student t (df = k-1),
item bootstrap with item indices shared across seeds, and the two-level bootstrap (seeds, then items). 10,000
draws, boot_seed 0, alpha 0.05. With one seed (base, A2-c12) there is no seed-t interval, and the two-level
interval equals an item bootstrap. The module imports cotdistill.metrics.calibration at load time (only used for
ECE, which is not computed here); a stub stands in for it so that no repo code is needed.
"""
import csv, io, json, os, re, sys, tarfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_m = types.ModuleType("cotdistill.metrics")
_m.calibration = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ECE not used here"))
sys.modules.setdefault("cotdistill", types.ModuleType("cotdistill"))
sys.modules["cotdistill.metrics"] = _m
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import paired_stats as PS  # noqa: E402

N_BOOT = 10000
SETS = ["hint_plain", "hint_full"] + [f"hint_true_r{r}" for r in range(1, 7)] + \
       [f"hint_{k}_r{r}" for k in ("other", "wrongres", "wrongres_near") for r in (1, 2)] + \
       [f"hint_placebo_r{r}" for r in range(1, 5)]
MODELS1 = [  # label, weights (training job), [(seed, eval job)]
    ("base", "Qwen3.5-4B (no fine-tuning)", [(None, "500_4b_hint_base")]),
    ("A2-c4", "83 / 84 / 85", [(12, "501_4b_hint_a2c4s12"), (13, "502_4b_hint_a2c4s13"), (14, "503_4b_hint_a2c4s14")]),
    ("A2-c12", "41", [(12, "504_4b_hint_a2c12s12")]),
    ("GFPQ-c4", "42 / 70 / 73", [(12, "505_4b_hint_gfpqc4s12"), (13, "506_4b_hint_gfpqc4s13"), (14, "507_4b_hint_gfpqc4s14")]),
    ("GFPQM-c4", "164 / 172 / 173", [(12, "508_4b_hint_gfpqmc4s12"), (13, "509_4b_hint_gfpqmc4s13"), (14, "510_4b_hint_gfpqmc4s14")]),
    ("GFPQMR-c4", "165 / 174 / 186", [(12, "511_4b_hint_gfpqmrc4s12"), (13, "512_4b_hint_gfpqmrc4s13"), (14, "513_4b_hint_gfpqmrc4s14")]),
]
BETWEEN1 = [(a, "A2-c4") for a in ("GFPQ-c4", "GFPQM-c4", "GFPQMR-c4")]
MODELS2 = [  # Stage 2: trained in the same job (hint sets in EXTRA_EVALS), final weights of the run
    ("A2-c1", "1 epoch", [(16, "520_4b_gsmvt_A2-c1lr3ch_s16_fsdp"), (17, "525_4b_gsmvt_A2-c1lr3ch_s17_fsdp"), (18, "530_4b_gsmvt_A2-c1lr3ch_s18_fsdp")]),
    ("A2-c2", "2 epochs", [(16, "521_4b_gsmvt_A2-c2lr3ch_s16_fsdp"), (17, "526_4b_gsmvt_A2-c2lr3ch_s17_fsdp"), (18, "531_4b_gsmvt_A2-c2lr3ch_s18_fsdp")]),
    ("GFPQ-c2", "2 epochs", [(16, "522_4b_gsmvt_GFPQ-c2lr3_s16_fsdp"), (17, "527_4b_gsmvt_GFPQ-c2lr3_s17_fsdp"), (18, "532_4b_gsmvt_GFPQ-c2lr3_s18_fsdp")]),
    ("GFPQR-c2", "2 epochs", [(16, "523_4b_gsmvt_GFPQR-c2lr3_s16_fsdp"), (17, "528_4b_gsmvt_GFPQR-c2lr3_s17_fsdp"), (18, "533_4b_gsmvt_GFPQR-c2lr3_s18_fsdp")]),
]
BETWEEN2 = [("GFPQ-c2", "GFPQR-c2"), ("GFPQ-c2", "A2-c1"), ("GFPQ-c2", "A2-c2"), ("GFPQR-c2", "A2-c1"), ("A2-c2", "A2-c1")]
MODELS, BETWEEN, PREFIX, STAGE, RUNS = MODELS1, BETWEEN1, "", 1, None


def read_jsonl(f):
    return [json.loads(l) for l in f if l.strip()]


def load_hints():
    return {s: {d["item_id"]: d for d in read_jsonl(open(os.path.join(ROOT, "data", "hint", s + ".jsonl")))} for s in SETS}


def _parse(rows):
    return {d["item_id"]: (PS.top1(d)[0], d["gold_label"]) for d in rows}


def _log_acc(lines):
    out = {}
    for line in lines:
        lm = re.match(r"(hint_[a-z0-9_]+) (\{.*\})\s*$", line)
        if lm:
            out[lm.group(1)] = json.loads(lm.group(2))
    return out


def load_run(job):
    """({set: {item_id: (correct, gold)}}, {set: log/metrics acc dict}, {set: metrics.json eval dict}, extra preds)."""
    preds, logacc, meta, extra = {}, {}, {}, {}
    if STAGE == 1:
        d = os.path.join(RUNS, job)
        for f in os.listdir(d):
            mm = re.fullmatch(r"preds_(hint_[a-z0-9_]+)\.jsonl", f)
            if mm:
                preds[mm.group(1)] = _parse(read_jsonl(open(os.path.join(d, f))))
        meta = json.load(open(os.path.join(d, "metrics.json")))["eval"]
        logacc = _log_acc(open(os.path.join(ROOT, "data", "job_logs", job + ".out")))
        return preds, logacc, meta, extra
    with tarfile.open(os.path.join(ROOT, "data", "stage2_preds.tgz")) as t:
        for m in t.getmembers():
            if not m.name.startswith(job.split("_")[0] + "/") or not m.isfile():
                continue
            f = m.name.split("/", 1)[1]
            body = io.TextIOWrapper(t.extractfile(m), encoding="utf-8")
            mm = re.fullmatch(r"preds_(hint_[a-z0-9_]+|valx)\.jsonl", f)
            if mm:
                (preds if mm.group(1) != "valx" else extra)[mm.group(1)] = _parse(read_jsonl(body))
            elif f == "metrics.json":
                meta = json.load(body)["eval"]
            elif f.endswith(".out"):
                logacc = _log_acc(body)
    return preds, logacc, meta, extra


def pts(x):
    return "-" if x is None else f"{100 * x:.1f}"


def dp(x):
    return "-" if x is None else f"{100 * x:+.1f}"


def ci(lo, hi):
    return "-" if lo is None or hi is None else f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"


def remaining_path(steps, k):
    """Longest dependency chain (a_from / b_from links) among the steps NOT shown (indices >= k); k = 0: the full
    critical path. Same definition as remaining_path() in the later builder version (v4, 2026-10-21 21:07)."""
    memo = {}

    def rp(j):
        if j not in memo:
            best = 0
            for key in ("a_from", "b_from"):
                src = steps[j].get(key)
                if src is not None and k <= int(src) < j:
                    best = max(best, rp(int(src)))
            memo[j] = best + 1
        return memo[j]
    return max((rp(j) for j in range(k, len(steps))), default=0)


def verdict(H, runs, mat):
    """The card's Stage 2 decisions (CARD_opc_sev6e_v4 sections 6-7, with the 2026-10-21 amendment), recomputed.
    Writes stage2_verdict.md. Bars: M1 full >= 90; M2 |placebo - plain| <= 3 pooled over r = 1..4 and <= 5 at each r;
    M3 / M3b / M4 true_r1 minus far-wrong / near-miss / other-item >= 10 (shared items); P2 H (clean, n >= 4)
    PASS = mean >= 15, every seed >= 10, seed t >= 4.30; NONE = mean < 5. P1 D = r1 - r4 on n >= 5: >= 15
    capacity-limited, < 5 not step-limited. P3 GFPQ-c2 - GFPQR-c2 on the pooled r in {1, 2} true rows: PASS = mean
    >= 1.5, all seeds > 0, t >= 4.30; closed = mean < 0.5; else between. Epoch: A2 epoch with the higher 3-seed valx
    mean (tie -> 1 epoch)."""
    seeds = [16, 17, 18]
    acc = lambda lab, sd, s, ids: sum(runs[(lab, sd)][s][i] for i in ids) / len(ids)
    out = ["# Stage 2 verdict (jobs 520-533), recomputed from preds", ""]
    vx = {lab: [acc(lab, sd, "valx", sorted(runs[(lab, sd)]["valx"])) for sd in seeds] for lab in ("A2-c1", "A2-c2")}
    m1, m2 = (sum(vx[x]) / 3 for x in ("A2-c1", "A2-c2"))
    pick = "A2-c1" if m1 >= m2 else "A2-c2"
    out += [f"Epoch rule (valx, 393 items): A2-c1 {100 * m1:.2f} (" + " / ".join(f"{100 * x:.2f}" for x in vx["A2-c1"]) +
            f"), A2-c2 {100 * m2:.2f} (" + " / ".join(f"{100 * x:.2f}" for x in vx["A2-c2"]) + f") -> the P2 student is {pick}.", ""]
    n4 = {i for i, d in H["hint_plain"].items() if d["n"] >= 4}
    clean = {i for i, d in H["hint_true_r1"].items() if not d["shown_equals_option"]} & n4
    n5 = sorted(i for i, d in H["hint_plain"].items() if d["n"] >= 5)
    n2 = sorted(i for i, d in H["hint_plain"].items() if d["n"] == 2)
    out += ["## Checks (seed means) and primary numbers per arm", "",
            "| arm | M1 full | M2 placebo-plain pooled (r1/r2/r3/r4) | M3 true-far | M3b true-near | M4 true-other | read (n=2) | gate | H clean n>=4 (per seed) | H seed-t 95% | H two-level 95% | D n>=5 (per seed) | D seed-t 95% |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for lab, _, _ in MODELS2:
        mean = lambda f: sum(f(sd) for sd in seeds) / 3
        full = mean(lambda sd: acc(lab, sd, "hint_full", sorted(H["hint_full"])))
        per_r, pooled_num, pooled_n = [], 0.0, 0
        for r in range(1, 5):
            ids = sorted(H[f"hint_placebo_r{r}"])
            d = mean(lambda sd: acc(lab, sd, f"hint_placebo_r{r}", ids) - acc(lab, sd, "hint_plain", ids))
            per_r.append(d); pooled_num += d * len(ids); pooled_n += len(ids)
        pooled = pooled_num / pooled_n
        def vs(ctrl):
            ids = sorted(set(H["hint_true_r1"]) & set(H[ctrl]))
            return mean(lambda sd: acc(lab, sd, "hint_true_r1", ids) - acc(lab, sd, ctrl, ids))
        m3, m3b, m4 = vs("hint_wrongres_r1"), vs("hint_wrongres_near_r1"), vs("hint_other_r1")
        read = mean(lambda sd: acc(lab, sd, "hint_true_r1", n2) - acc(lab, sd, "hint_plain", n2))
        gate = full >= 0.90 and abs(pooled) <= 0.03 + 1e-12 and all(abs(x) <= 0.05 + 1e-12 for x in per_r) and \
            m3 >= 0.10 and m3b >= 0.10 and m4 >= 0.10
        ids = sorted(clean)
        rh = PS.paired_stats(mat(lab, "hint_true_r1", ids)[0], mat(lab, "hint_plain", ids)[0], n_boot=N_BOOT, boot_seed=0)
        rd = PS.paired_stats(mat(lab, "hint_true_r1", n5)[0], mat(lab, "hint_true_r4", n5)[0], n_boot=N_BOOT, boot_seed=0)
        res[lab] = (gate, rh, rd)
        out.append(f"| {lab} | {pts(full)} | {dp(pooled)} ({' / '.join(dp(x) for x in per_r)}) | {dp(m3)} | {dp(m3b)} | {dp(m4)} | "
                   f"{dp(read)} | {'PASS' if gate else 'FAIL'} | {dp(rh['mean'])} ({' / '.join(dp(x) for x in rh['d'])}) | "
                   f"{ci(rh['t']['lo'], rh['t']['hi'])} | {ci(*rh['two_level'])} | {dp(rd['mean'])} ({' / '.join(dp(x) for x in rd['d'])}) | "
                   f"{ci(rd['t']['lo'], rd['t']['hi'])} |")
    gate, rh, rd = res[pick]
    H_ = rh["mean"]; tH = rh["t"]["t"]
    p2 = ("PASS" if H_ >= 0.15 and min(rh["d"]) >= 0.10 and tH is not None and tH >= 4.30 else "NONE" if H_ < 0.05 else "limited")
    p1 = "capacity-limited" if rd["mean"] >= 0.15 else "not step-limited" if rd["mean"] < 0.05 else "between"
    out += ["", f"## Decisions", "",
            f"- Gate on the P2 student {pick} (seed means): {'PASS' if gate else 'FAIL'}. P2 is read only if the gate passes.",
            f"- P2 (primary): H = {dp(H_)} on {rh['n_items']} clean items with n >= 4, seeds " + " / ".join(dp(x) for x in rh["d"]) +
            f", seed t {tH:.2f} (crit 4.30) -> **{p2 if gate else 'not read (gate failed)'}**.",
            f"- P1: D = acc(r1) - acc(r4) on {rd['n_items']} items with n >= 5 = {dp(rd['mean'])} -> {p1}."]
    for drop in (False, True):
        keys = [(f"hint_true_r{r}", i) for r in (1, 2) for i in sorted(H[f"hint_true_r{r}"])
                if not drop or not H[f"hint_true_r{r}"][i]["shown_equals_option"]]
        cA = np.array([[runs[("GFPQ-c2", sd)][s_][i] for s_, i in keys] for sd in seeds], dtype=np.int8)
        cB = np.array([[runs[("GFPQR-c2", sd)][s_][i] for s_, i in keys] for sd in seeds], dtype=np.int8)
        r3 = PS.paired_stats(cA, cB, n_boot=N_BOOT, boot_seed=0)
        t3 = r3["t"]["t"]
        v = "PASS" if r3["mean"] >= 0.015 and min(r3["d"]) > 0 and t3 >= 4.30 else "closed" if r3["mean"] < 0.005 else "between"
        out.append(f"- P3 (secondary) GFPQ-c2 - GFPQR-c2 on {len(keys)} r in {{1, 2}} true rows{' (clean only)' if drop else ''}: "
                   f"{' / '.join(dp(x) for x in r3['d'])}, mean {dp(r3['mean'])}, seed t {t3:.2f}, seed-t 95% "
                   f"{ci(r3['t']['lo'], r3['t']['hi'])}, two-level 95% {ci(*r3['two_level'])} -> {v}.")
    open(os.path.join(HERE, "stage2_verdict.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    H = load_hints()
    runs, selftest = {}, []
    for label, _, seeds in MODELS:
        for seed, job in seeds:
            preds, logacc, meta, extra = load_run(job)
            runs[(label, seed)] = {s: {i: c for i, (c, _) in v.items()} for s, v in list(preds.items()) + list(extra.items())}
            for s in SETS:
                ok_items = set(preds.get(s, {})) == set(H[s])
                ok_gold = all(preds[s][i][1] == H[s][i]["gold_label"] for i in H[s]) if ok_items else False
                acc = sum(c for c, _ in preds[s].values()) / len(preds[s]) if s in preds else None
                ok_meta = s in meta and acc is not None and abs(acc - meta[s]["acc"]) < 1e-12 and meta[s]["n"] == len(preds[s])
                # stage 2 logs print the hint sets only in metrics.json; stage 1 logs print them as lines
                ok_log = STAGE == 2 or (s in logacc and acc is not None and abs(acc - logacc[s]["acc"]) < 1e-12)
                if not (ok_items and ok_gold and ok_meta and ok_log):
                    selftest.append(f"FAIL {job} {s}: items {ok_items} gold {ok_gold} metrics-acc {ok_meta} log-acc {ok_log}")
            if STAGE == 2:
                va = extra["valx"]
                if abs(sum(c for c, _ in va.values()) / len(va) - meta["valx"]["acc"]) > 1e-12:
                    selftest.append(f"FAIL {job} valx: recomputed accuracy != metrics.json")
    n_checks = sum(len(s) for _, _, s in MODELS) * len(SETS)
    note = ("recomputed accuracy = metrics.json eval acc and n (to 1e-12)" +
            (" and = the 'acc' line of the job log" if STAGE == 1 else "; valx recomputed = metrics.json valx"))
    selftest.insert(0, f"{n_checks} (run, set) pairs checked: item ids = hint file, gold labels = hint file, {note}")
    selftest.append("ALL SELF-TESTS PASSED" if len(selftest) == 1 else "SELF-TESTS FAILED")
    open(os.path.join(HERE, PREFIX + "selftest.txt"), "w").write("\n".join(selftest) + "\n")
    if len(selftest) > 2:
        print("\n".join(selftest)); sys.exit(1)

    def mat(label, s, ids):
        seeds = [sd for sd, _ in dict((m[0], m[2]) for m in MODELS)[label]]
        return np.array([[runs[(label, sd)][s][i] for i in ids] for sd in seeds], dtype=np.int8), seeds

    md = (["# Section A: hint experiments, jobs 500-513 = Stage 1 PILOT (recomputed from preds)", "",
           "Accuracy in points. Seeds 12, 13, 14 for the 3-seed arms; one run for base and A2-c12."] if STAGE == 1 else
          ["# Section A: hint experiments, jobs 520-533 = Stage 2 registered CONFIRMATION (recomputed from preds)", "",
           "Accuracy in points. Seeds 16, 17, 18. Card verdict: stage2_verdict.md."]) + [
          "Every contrast is paired on the items present in both sets. Intervals: seed-level t (df 2) and the",
          "two-level bootstrap (seeds x items, 10,000 draws), algorithm of scripts/paired_stats.py.", ""]
    if STAGE == 2:
        verdict(H, runs, mat)

    # ---- Table 1: accuracy per model x set
    rows_csv = [["model", "eval_job", "seed", "set", "n_items", "acc_pts"]]
    md += ["## 1. Accuracy per model and set (seed mean; per-seed values in brackets)", "",
           "| set | n | " + " | ".join(m[0] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for s in SETS:
        cells = []
        for label, _, seeds in MODELS:
            accs = []
            for sd, job in seeds:
                v = runs[(label, sd)][s]
                a = sum(v.values()) / len(v)
                accs.append(a)
                rows_csv.append([label, job, sd if sd is not None else "", s, len(v), f"{100 * a:.4f}"])
            cells.append(pts(sum(accs) / len(accs)) + ("" if len(accs) == 1 else " (" + " / ".join(pts(a) for a in accs) + ")"))
        md.append(f"| {s} | {len(H[s])} | " + " | ".join(cells) + " |")
    with open(os.path.join(HERE, PREFIX + "acc_per_seed.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_csv)

    # ---- Table 2: within-model contrasts
    clean1 = {i for i, d in H["hint_true_r1"].items() if not d["shown_equals_option"]}
    n4 = {i for i, d in H["hint_plain"].items() if d["n"] >= 4}
    C = [("full - plain", "hint_full", "hint_plain", None, "all")]
    C += [(f"true_r{r} - plain", f"hint_true_r{r}", "hint_plain", None, f"items of true_r{r}") for r in range(1, 7)]
    C += [("true_r1 - plain (clean)", "hint_true_r1", "hint_plain", clean1, "no shown value equals an option"),
          ("H: true_r1 - plain (clean, n>=4)", "hint_true_r1", "hint_plain", clean1 & n4, "card primary H")]
    for k in ("other", "wrongres", "wrongres_near"):
        C += [(f"{k}_r{r} - plain", f"hint_{k}_r{r}", "hint_plain", None, f"items of {k}_r{r}") for r in (1, 2)]
    C += [(f"placebo_r{r} - plain", f"hint_placebo_r{r}", "hint_plain", None, f"items of placebo_r{r}") for r in range(1, 5)]
    for k in ("other", "wrongres", "wrongres_near", "placebo"):
        C += [(f"true_r{r} - {k}_r{r}", f"hint_true_r{r}", f"hint_{k}_r{r}", None, "shared items") for r in (1, 2)]
    n5 = {i for i, d in H["hint_plain"].items() if d["n"] >= 5}
    n7 = {i for i, d in H["hint_plain"].items() if d["n"] >= 7}
    C += [(f"[n>=5] true_r{r} - plain", f"hint_true_r{r}", "hint_plain", n5, "fixed items n>=5") for r in range(1, 5)]
    C += [("[n>=5] true_r1 - true_r4 (D)", "hint_true_r1", "hint_true_r4", n5, "fixed items n>=5")]
    C += [(f"[n>=7] true_r{r} - plain", f"hint_true_r{r}", "hint_plain", n7, "fixed items n>=7") for r in range(1, 7)]
    ccsv = [["model", "contrast", "set_a", "set_b", "items", "n_items", "k_seeds", "seeds", "d_per_seed", "mean",
             "seed_t_lo", "seed_t_hi", "t", "p", "item_boot_lo", "item_boot_hi", "two_level_lo", "two_level_hi", "significant", "acc_a_mean", "acc_b_mean"]]
    res = {}
    for label, _, seeds in MODELS:
        for name, a, b, filt, desc in C:
            ids = sorted(set(H[a]) & set(H[b]) & (filt if filt is not None else set(H[a])))
            cA, sd = mat(label, a, ids)
            cB, _ = mat(label, b, ids)
            r = PS.paired_stats(cA, cB, n_boot=N_BOOT, boot_seed=0)
            res[(label, name)] = r
            t = r["t"]
            ccsv.append([label, name, a, b, desc, r["n_items"], r["k"], ";".join("base" if x is None else str(x) for x in sd),
                         ";".join(f"{100 * x:.4f}" for x in r["d"]), f"{100 * r['mean']:.4f}"] +
                        [("" if t[x] is None else f"{100 * t[x]:.4f}") for x in ("lo", "hi")] +
                        [("" if t["t"] is None else f"{t['t']:.4f}"), ("" if t["p"] is None else f"{t['p']:.5f}")] +
                        [f"{100 * x:.4f}" for x in r["item_boot"] + r["two_level"]] + [int(r["significant"])] +
                        [f"{100 * float(np.mean(r['acc_a'])):.4f}", f"{100 * float(np.mean(r['acc_b'])):.4f}"])
    with open(os.path.join(HERE, PREFIX + "contrasts.csv"), "w", newline="") as f:
        csv.writer(f).writerows(ccsv)
    md += ["", "## 2. Within-model contrasts (A - B, paired on shared items)", "",
           "Per model: mean, per-seed values, seed-t 95% interval (3-seed arms), two-level bootstrap 95% interval.",
           "sig = paired_stats' own flag: both the seed-t and the two-level interval exclude 0 (never set with one seed).",
           "Rows marked [n>=5] / [n>=7] use one fixed item set for every r (the curve of section 4).", ""]
    for label, _, seeds in MODELS:
        md += [f"### {label}", "", "| contrast | items | mean | per seed | seed-t 95% | two-level 95% | sig |", "|---|---|---|---|---|---|---|"]
        for name, *_ in C:
            r = res[(label, name)]
            t = r["t"]
            md.append(f"| {name} | {r['n_items']} | {dp(r['mean'])} | " + " / ".join(dp(x) for x in r["d"]) +
                      f" | {ci(t['lo'], t['hi'])} | {ci(*r['two_level'])} | {'yes' if r['significant'] else ''} |")
        md.append("")

    # ---- Table 3: between models, seed-paired (seeds 12-14) against A2-c4
    bcsv = [["arm", "baseline", "quantity", "n_items", "d_seed1", "d_seed2", "d_seed3", "mean", "seed_t_lo", "seed_t_hi", "t", "p", "two_level_lo", "two_level_hi", "significant"]]
    md += ["## 3. Between models: arm - baseline, seed-paired (same seed with same seed)", "",
           "Gain = (true_r - plain) per item; the gain row is a difference of the two arms' gains.", "",
           "| arm - baseline | quantity | items | mean | per seed | seed-t 95% | two-level 95% | sig |", "|---|---|---|---|---|---|---|---|"]
    for arm, base in BETWEEN:
        for q in ("plain", "true_r1", "true_r2", "gain r1", "gain r2"):
            if q.startswith("gain"):
                r_ = q[-1]
                ids = sorted(H[f"hint_true_r{r_}"])
                cA = mat(arm, f"hint_true_r{r_}", ids)[0].astype(np.int16) - mat(arm, "hint_plain", ids)[0]
                cB = mat(base, f"hint_true_r{r_}", ids)[0].astype(np.int16) - mat(base, "hint_plain", ids)[0]
            else:
                ids = sorted(H[f"hint_{q}"])
                cA, cB = mat(arm, f"hint_{q}", ids)[0], mat(base, f"hint_{q}", ids)[0]
            r = PS.paired_stats(cA, cB, n_boot=N_BOOT, boot_seed=0)
            t = r["t"]
            md.append(f"| {arm} - {base} | {q} | {r['n_items']} | {dp(r['mean'])} | " + " / ".join(dp(x) for x in r["d"]) +
                      f" | {ci(t['lo'], t['hi'])} | {ci(*r['two_level'])} | {'yes' if r['significant'] else ''} |")
            bcsv.append([arm, base, q, r["n_items"]] + [f"{100 * x:.4f}" for x in r["d"]] + [f"{100 * r['mean']:.4f}",
                        f"{100 * t['lo']:.4f}", f"{100 * t['hi']:.4f}", f"{t['t']:.4f}", f"{t['p']:.5f}"] + [f"{100 * x:.4f}" for x in r["two_level"]] + [int(r["significant"])])
    with open(os.path.join(HERE, PREFIX + "between_models.csv"), "w", newline="") as f:
        csv.writer(f).writerows(bcsv)

    # ---- Table 4: fixed-item curves
    cur = [["model", "seed", "cohort", "n_items", "condition", "r", "acc_pts"]]
    md += ["", "## 4. Curves on fixed item sets (the same items at every r)", "",
           "The r sets shrink with r (1,129 / 877 / 576 / 332 / 158 / 75 items), so raw set accuracies mix r with item",
           "length. These curves hold the items fixed: n >= 5 (332 items, r = 1..4) and n >= 7 (75 items, r = 1..6).", ""]
    for cname, nmin, rmax in (("n>=5", 5, 4), ("n>=7", 7, 6)):
        ids = sorted(i for i, d in H["hint_plain"].items() if d["n"] >= nmin)
        md += [f"### Cohort {cname} ({len(ids)} items)", "", "| model | plain | " + " | ".join(f"r={r}" for r in range(1, rmax + 1)) +
               " | full |", "|---|---|" + "---|" * (rmax + 1)]
        for label, _, seeds in MODELS:
            vals = {}
            for cond, sname, r in [("plain", "hint_plain", "")] + [("true", f"hint_true_r{r}", r) for r in range(1, rmax + 1)] + [("full", "hint_full", 0)]:
                per = [sum(runs[(label, sd)][sname][i] for i in ids) / len(ids) for sd, _ in seeds]
                vals[(cond, r)] = per
                for (sd, _), a in zip(seeds, per):
                    cur.append([label, sd if sd is not None else "", cname, len(ids), cond, r, f"{100 * a:.4f}"])
                cur.append([label, "mean", cname, len(ids), cond, r, f"{100 * sum(per) / len(per):.4f}"])
            fmt = lambda per: pts(sum(per) / len(per)) + ("" if len(per) == 1 else " (" + "/".join(pts(a) for a in per) + ")")
            md.append(f"| {label} | {fmt(vals[('plain', '')])} | " + " | ".join(fmt(vals[('true', r)]) for r in range(1, rmax + 1)) +
                      f" | {fmt(vals[('full', 0)])} |")
        md.append("")
    with open(os.path.join(HERE, PREFIX + "curves.csv"), "w", newline="") as f:
        csv.writer(f).writerows(cur)

    # ---- Table 5: grid by n and r (seed mean)
    grid = [["model", "n", "condition", "r", "n_items", "acc_pts_seed_mean"]]
    md += ["## 5. Accuracy by item length n and steps left r (seed mean; cell = acc, items)", ""]
    nbin = lambda n: min(n, 7)
    for label, _, seeds in MODELS:
        md += [f"### {label}", "", "| n | plain | " + " | ".join(f"r={r}" for r in range(1, 7)) + " |", "|---|---|" + "---|" * 6]
        for nb in range(2, 8):
            cells = []
            for sname, r in [("hint_plain", "")] + [(f"hint_true_r{r}", r) for r in range(1, 7)]:
                ids = [i for i, d in H[sname].items() if nbin(d["n"]) == nb]
                if not ids:
                    cells.append("")
                    continue
                a = sum(sum(runs[(label, sd)][sname][i] for i in ids) / len(ids) for sd, _ in seeds) / len(seeds)
                cells.append(f"{pts(a)} ({len(ids)})")
                grid.append([label, f"{nb}" if nb < 7 else "7+", "plain" if r == "" else "true", r, len(ids), f"{100 * a:.4f}"])
            md.append(f"| {nb if nb < 7 else '7+'} | " + " | ".join(cells) + " |")
        md.append("")
    with open(os.path.join(HERE, PREFIX + "grid_n_r.csv"), "w", newline="") as f:
        csv.writer(f).writerows(grid)

    # ---- Table 6: remaining critical path (serial work left) on items n >= 5
    steps = {d["item_id"]: d["worked_steps"] for d in read_jsonl(open(os.path.join(ROOT, "data", "source", "gsm8k_mc_v2_steps.jsonl")))}
    big = {i for i, d in H["hint_plain"].items() if d["n"] >= 5}
    pcsv = [["model", "path_left", "true_rows", "acc_true_pts", "plain_items_with_that_full_path", "acc_plain_pts"]]
    md += ["## 6. True rows by remaining critical path (items n >= 5, r = 1..6 pooled) vs plain items with that full path", "",
           "r counts atomic steps left; some of them can run in parallel. path_left = longest a_from / b_from chain among",
           "the steps not shown. Plain column: all 1,129 plain items whose full critical path equals that length.", "",
           "| model | " + " | ".join(f"path {p}: true / plain" for p in (1, 2, 3, 4)) + " |", "|---|" + "---|" * 4]
    for label, _, seeds in MODELS:
        cells = []
        for p in (1, 2, 3, 4):
            tr = [(f"hint_true_r{r}", i) for r in range(1, 7) for i, d in H[f"hint_true_r{r}"].items()
                  if i in big and remaining_path(steps[i], d["k"]) == p]
            pl = [i for i in H["hint_plain"] if remaining_path(steps[i], 0) == p]
            at = sum(sum(runs[(label, sd)][s][i] for s, i in tr) / len(tr) for sd, _ in seeds) / len(seeds) if tr else None
            ap_ = sum(sum(runs[(label, sd)]["hint_plain"][i] for i in pl) / len(pl) for sd, _ in seeds) / len(seeds) if pl else None
            cells.append(f"{pts(at)} ({len(tr)}) / {pts(ap_)} ({len(pl)})")
            pcsv.append([label, p, len(tr), "" if at is None else f"{100 * at:.4f}", len(pl), "" if ap_ is None else f"{100 * ap_:.4f}"])
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    with open(os.path.join(HERE, PREFIX + "path_slice.csv"), "w", newline="") as f:
        csv.writer(f).writerows(pcsv)

    open(os.path.join(HERE, PREFIX + "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(selftest))
    print("wrote tables.md, acc_per_seed.csv, contrasts.csv, between_models.csv, curves.csv, grid_n_r.csv, path_slice.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--runs", default=os.path.join(ROOT, "..", "..", "..", "runs"))
    a = ap.parse_args()
    if a.stage == 2:
        MODELS, BETWEEN, PREFIX, STAGE = MODELS2, BETWEEN2, "stage2_", 2
    RUNS = os.path.abspath(a.runs)
    main()
