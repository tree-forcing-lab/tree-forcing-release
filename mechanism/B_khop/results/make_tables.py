#!/usr/bin/env python3
"""Step 2 of 2: every table and CSV of the k-hop section, from results/item_preds.json.gz and ../data only.

    cd B_khop/results && python3 -I make_tables.py

Needs numpy (for ../code/paired_stats.py). Writes into this folder:
  cells_per_seed.csv   one row per (arm, seed, set, k, subset): n items, accuracy (percent)
  cells_summary.csv    one row per (arm, set, k, subset): seed mean, SD, 95% seed-t interval, the seed values
  curve_ca_fwd.csv     the plot data: x = hops 1-8, one column per arm (seed mean), chance 20
  gsm_per_seed.csv     plain GSM sets and the rc1-rc4 cells per (arm, seed)
  gsm_summary.csv      the same as seed means
  contrasts.csv        seed-paired contrasts (A - B): mean, seed-t interval, two-level and item bootstrap intervals
  tables.md            the tables of the README, printed from the numbers above
Accuracy of a row = 1 when the arg max of the five option probabilities (one forward pass, the next-token
distribution after "Answer:") is the gold letter. Chance is 20% (five options, gold letters balanced per cell).
"""
import csv, gzip, json, math, os, sys

sys.dont_write_bytecode = True
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "generators"))
import numpy as np                      # noqa: E402
import types                            # noqa: E402
# paired_stats.py imports cotdistill.metrics.calibration at module level for its ECE extra. This script never calls
# the ECE code, so a stub module stands in for the repo package; paired_stats.paired_stats() does not use it.
_m = types.ModuleType("cotdistill.metrics")
_m.calibration = None
sys.modules.setdefault("cotdistill", types.ModuleType("cotdistill"))
sys.modules.setdefault("cotdistill.metrics", _m)
import paired_stats as PS              # noqa: E402  (copy of scripts/paired_stats.py, unchanged)
import pfx_table as PFX                 # noqa: E402  (rc cells: the 249 primary items)
import build_khop as BK                 # noqa: E402  (parse_prompt / query_hops / hard_item)

KHOP = ["khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev", "khop_cag_fwd",
        "khop_ca_fwd_qs", "khop_blind"]
PLAIN = ["gsm8kv2", "gsm8kx", "gsm8k", "gsm1k", "gsm1kx", "valx"]
SEEDS = [16, 17, 18, 19, 20]
# Arms in plot order. "steps" = optimizer updates of the run that produced the evaluated weights (read from the logs
# by extract_preds.py and checked below). Two-stage arms list stage 1 + stage 2.
ARMS = [
    ("BASE0", "Qwen3.5-4B, no training"),
    ("A2Z", "A2-6k answer-only GSM epoch (j663-666, j540 weights), zero chain rows"),
    ("A2Z96", "A2-96k answer-only GSM, zero chain rows (one seed)"),
    ("K1", "A2 epoch + 3,000 one-hop chain rows (KH statements, hop-1 query)"),
    ("KH", "A2 epoch + 3,000 chain rows, k 1-6"),
    ("H3", "A2 epoch + the 3,000 KH rows x 3"),
    ("KD", "A2 epoch + 9,000 chain rows"),
    ("R3", "A2 epoch + the 9,000 KD rows x 3"),
    ("D27", "A2 epoch + 27,000 chain rows"),
    ("CUR1", "stage 1 only: 27,000 chain rows, no GSM rows"),
    ("K1X1", "stage 1 only: the same 27,000 statements, hop-1 query"),
    ("CUR-R", "CUR1 weights, then A2 epoch + 3,000 KH replay rows"),
    ("K1X-R", "K1X1 weights, then A2 epoch + 3,000 K1 replay rows"),
    ("CUR-A2", "CUR1 weights (s16), then plain A2 epoch (khop3 j980)"),
    ("K1X-A2", "K1X1 weights (s16), then plain A2 epoch (khop3 j990)"),
    ("CUR-L", "CUR1 weights (s16), then A2 epoch at LR 7.5e-7 (khop4 arm L)"),
    ("K1X-L", "K1X1 weights (s16), then A2 epoch at LR 7.5e-7 (khop4 arm L)"),
]
EXPECT_STEPS = {"A2orig": [186], "A2re": [186], "K1": [279], "KH": [279], "H3": [467], "KD": [467], "R3": [1029],
                "D27": [1029], "CUR1": [844], "K1X1": [844], "CUR-R": [279], "K1X-R": [279], "CUR-A2": [186],
                "K1X-A2": [186], "CUR-L": [186], "K1X-L": [186]}
REF_RE = {16: "663", 17: "664", 18: "665", 19: "540", 20: "666"}   # A2 re-train set; s19 is j540 (as run_khop.sh)


def load():
    d = json.load(gzip.open(os.path.join(HERE, "item_preds.json.gz")))
    runs = defaultdict(dict)                     # arm -> seed -> job record
    for j, r in d["jobs"].items():
        if r["seed"] in runs[r["arm"]]:
            sys.exit(f"two jobs for {r['arm']} s{r['seed']}")
        runs[r["arm"]][r["seed"]] = dict(r, job=int(j))
    runs["A2re"][19] = runs["A2orig"][19]        # j540 serves both reference sets
    for arm, steps in EXPECT_STEPS.items():
        for s, r in runs[arm].items():
            got = [x["steps"] for x in r["stages"]]
            if got != steps:
                sys.exit(f"{arm} s{s} j{r['job']}: steps {got}, expected {steps}")
    return d, runs


def order_sensitive(row):
    """True when an order-free reader cannot single out the gold: rank the five chains at the queried depth by the
    signed sum of their + / - constants (every x2 / x3 step ignored); the item is order-sensitive when the option at
    the target's rank (ties: all tied ranks) is not exactly the gold. On these items a x step between + / - steps
    re-weights earlier constants, so the step order decides the answer (same rule as the khop2 adversarial check)."""
    P = BK.parse_prompt(row["prompt"])
    by = P["by"]
    depth = {}

    def dep(n):
        if n not in depth:
            depth[n] = 0 if by[n]["parent"] is None else 1 + dep(by[n]["parent"])
        return depth[n]

    qd = dep(P["query"])
    names = [s["name"] for s in P["stmts"] if dep(s["name"]) == qd]
    def ssum(n):
        x = 0
        while by[n]["parent"] is not None:
            o, c = by[n]["op"], by[n]["c"]
            x += c if o == "+" else -c if o == "-" else 0
            n = by[n]["parent"]
        return x
    keys = [ssum(n) for n in names]
    t = keys[names.index(P["query"])]
    lo, hi = sum(x < t for x in keys), sum(x <= t for x in keys)
    srt = sorted(P["options"])
    cand = {P["options"].index(srt[i]) for i in range(lo, hi)}
    return cand != {"ABCDE".index(row["gold_label"])}


def khop_rows():
    out = {}
    for s in KHOP:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "eval", "khop", s + ".jsonl")) if l.strip()]
        for r in rows:
            r["_alladd"] = None
            r["_hard"] = None
            r["_os"] = None
            if r["task"] in ("ca", "wp", "cag") and r["order"] != "blind":
                h = BK.query_hops(BK.parse_prompt(r["prompt"]))
                if h is not None:
                    r["_alladd"] = sum(o in ("+", "-") for o, _ in h[1]) == r["k"]
                r["_hard"] = BK.hard_item(r)
                if r["k"] >= 2 and not s.endswith("_qs"):
                    r["_os"] = order_sensitive(r)
        out[s] = rows
    return out


def correct_vec(d, rec, s):
    p, g = rec["picks"].get(s), d["sets"][s]["gold"]
    return None if p is None else np.array([int(a == b) for a, b in zip(p, g)], dtype=np.int8)


def rc_index(d):
    """{rc: [(set, row index)]} on the 249 primary items, rows as pfx_table.cells picks them."""
    h = PFX.load_hints(os.path.join(ROOT, "data", "eval", "hint"))
    cl = PFX.cells(h)
    prim = PFX.primary_items(cl)
    pos = {s: {i: n for n, i in enumerate(d["sets"][s]["ids"])} for s in ["hint_plain"] + PFX.TRUE}
    out = {rc: [(cl[i][rc][0], pos[cl[i][rc][0]][i]) for i in prim if rc in cl[i]] for rc in (1, 2, 3, 4)}
    out["plain249"] = [("hint_plain", pos["hint_plain"][i]) for i in prim]
    return out, len(prim)


def rc_vec(d, rec, idx):
    picks = rec["picks"]
    if not all(s in picks for s, _ in idx):
        return None
    return np.array([int(picks[s][n] == d["sets"][s]["gold"][n]) for s, n in idx], dtype=np.int8)


def tci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, None, None
    m = sum(xs) / n
    if n < 2:
        return m, None, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    tc = PS.t_ppf(0.975, n - 1)
    return m, sd, m - tc * sd / math.sqrt(n), m + tc * sd / math.sqrt(n)


def f1(x):
    return "" if x is None else f"{x:.1f}"


def main():
    d, runs = load()
    kr = khop_rows()
    rci, n_prim = rc_index(d)

    # ---------------- k-hop cells
    cellvec = {}            # (arm, seed, set, k, subset) -> 0/1 vector over the cell items (fixed item order)
    per_seed = []
    for arm, _ in ARMS:
        for seed, rec in sorted(runs[arm].items()):
            for s in KHOP:
                c = correct_vec(d, rec, s)
                if c is None:
                    continue
                rows = kr[s]
                for k in sorted({r["k"] for r in rows}):
                    for sub, keep in (("all", lambda r: True), ("alladd", lambda r: r["_alladd"] is True),
                                      ("hard", lambda r: r["_hard"] is True),
                                      ("ordersens", lambda r: r["_os"] is True)):
                        if sub != "all" and (s.startswith("khop_vb") or s == "khop_blind"):
                            continue
                        ix = [n for n, r in enumerate(rows) if r["k"] == k and keep(r)]
                        if not ix:
                            continue
                        v = c[ix]
                        cellvec[(arm, seed, s, k, sub)] = v
                        per_seed.append({"arm": arm, "seed": seed, "job": rec["job"], "set": s, "k": k, "subset": sub,
                                         "n_items": len(ix), "acc": round(100 * float(v.mean()), 4)})
    with open(os.path.join(HERE, "cells_per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)
    summ = defaultdict(dict)
    for r in per_seed:
        summ[(r["arm"], r["set"], r["k"], r["subset"], r["n_items"])][r["seed"]] = r["acc"]
    srows = []
    for (arm, s, k, sub, n), by in summ.items():
        m, sd, lo, hi = tci(list(by.values()))
        srows.append({"arm": arm, "set": s, "k": k, "subset": sub, "n_items": n, "n_seeds": len(by),
                      "mean": round(m, 4), "sd": "" if sd is None else round(sd, 4),
                      "ci_lo": "" if lo is None else round(lo, 4), "ci_hi": "" if hi is None else round(hi, 4),
                      "seeds": " ".join(f"s{a}:{b:.2f}" for a, b in sorted(by.items()))})
    with open(os.path.join(HERE, "cells_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    S = {(r["arm"], r["set"], r["k"], r["subset"]): r for r in srows}

    # plot CSV: (ca, fwd), all items
    plot_arms = [a for a, _ in ARMS if (a, "khop_ca_fwd", 1, "all") in S]
    with open(os.path.join(HERE, "curve_ca_fwd.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hops"] + [f"{a} (n={S[(a, 'khop_ca_fwd', 1, 'all')]['n_seeds']})" for a in plot_arms] + ["chance"])
        for k in range(1, 9):
            w.writerow([k] + [f"{S[(a, 'khop_ca_fwd', k, 'all')]['mean']:.2f}" for a in plot_arms] + ["20.00"])
    desc = dict(ARMS)
    with open(os.path.join(HERE, "curve_ca_fwd_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "setting", "hops", "trained_hop", "n_items", "n_seeds", "mean", "ci_lo", "ci_hi", "chance"])
        for a in plot_arms:
            for k in range(1, 9):
                r = S[(a, "khop_ca_fwd", k, "all")]
                w.writerow([a, desc[a], k, int(k <= 6), r["n_items"], r["n_seeds"], f"{r['mean']:.2f}",
                            "" if r["ci_lo"] == "" else f"{r['ci_lo']:.2f}", "" if r["ci_hi"] == "" else f"{r['ci_hi']:.2f}",
                            "20.00"])

    # ---------------- GSM: plain sets and rc cells
    gvec, grows = {}, []
    gsm_arms = ["A2orig", "A2re", "A2-96k"] + [a for a, _ in ARMS]
    for arm in gsm_arms:
        for seed, rec in sorted(runs.get(arm, {}).items()):
            row = {"arm": arm, "seed": seed, "job": rec["job"]}
            for s in PLAIN:
                c = correct_vec(d, rec, s)
                if c is not None:
                    gvec[(arm, seed, s)] = c
                    row[s] = round(100 * float(c.mean()), 4)
            for rc in ("plain249", 1, 2, 3, 4):
                c = rc_vec(d, rec, rci[rc])
                if c is not None:
                    key = rc if rc == "plain249" else f"rc{rc}"
                    gvec[(arm, seed, key)] = c
                    row[key] = round(100 * float(c.mean()), 4)
            if len(row) > 3:
                grows.append(row)
    gcols = ["arm", "seed", "job"] + PLAIN + ["plain249", "rc1", "rc2", "rc3", "rc4"]
    with open(os.path.join(HERE, "gsm_per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        w.writeheader()
        w.writerows(grows)
    gs = []
    for arm in gsm_arms:
        rr = [r for r in grows if r["arm"] == arm]
        if not rr:
            continue
        o = {"arm": arm, "n_seeds": len(rr), "seeds": " ".join(f"s{r['seed']}" for r in rr)}
        for c in gcols[3:]:
            xs = [r[c] for r in rr if c in r]
            o[c] = round(sum(xs) / len(xs), 4) if xs else ""
        gs.append(o)
    with open(os.path.join(HERE, "gsm_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "n_seeds", "seeds"] + gcols[3:])
        w.writeheader()
        w.writerows(gs)
    G = {r["arm"]: r for r in gs}

    # ---------------- contrasts (paired_stats.paired_stats: seeds 16-20, shared items)
    con = []
    def pair(label, a, b, key_a, key_b, what):
        seeds = [s for s in SEEDS if key_a(s) in vec and key_b(s) in vec]
        if len(seeds) < 2:
            return
        cA = np.stack([vec[key_a(s)] for s in seeds])
        cB = np.stack([vec[key_b(s)] for s in seeds])
        r = PS.paired_stats(cA, cB)
        t = r["t"]
        # s16 was the pilot seed of every round (and chose arm R in khop4): the same contrast on s17-20 alone
        s4 = [i for i, s in enumerate(seeds) if s != 16]
        t4 = PS.seed_t(np.asarray(r["d"])[s4]) if len(s4) >= 2 else None
        P4 = lambda x: "" if x is None else round(100 * x, 4)
        con.append({"family": label, "contrast": f"{a} - {b}", "cell": what, "n_seeds": r["k"], "n_items": r["n_items"],
                    "mean": round(100 * r["mean"], 4), "seed_t_lo": round(100 * t["lo"], 4),
                    "seed_t_hi": round(100 * t["hi"], 4), "t": "" if t["t"] is None else round(t["t"], 3),
                    "p": "" if t["p"] is None else round(t["p"], 5),
                    "two_level_lo": round(100 * r["two_level"][0], 4), "two_level_hi": round(100 * r["two_level"][1], 4),
                    "item_boot_lo": round(100 * r["item_boot"][0], 4), "item_boot_hi": round(100 * r["item_boot"][1], 4),
                    "both_exclude_0": r["significant"],
                    "mean_s17_20": P4(t4["mean"]) if t4 else "", "seed_t_lo_s17_20": P4(t4["lo"]) if t4 else "",
                    "seed_t_hi_s17_20": P4(t4["hi"]) if t4 else "",
                    "per_seed": " ".join(f"s{s}:{100 * x:+.2f}" for s, x in zip(seeds, r["d"]))})
    vec = cellvec
    kpairs = [("D27", "A2Z"), ("D27", "K1"), ("KH", "K1"), ("KH", "A2Z"), ("KD", "KH"), ("D27", "KD"), ("D27", "R3"),
              ("R3", "KD"), ("H3", "KH"), ("CUR1", "K1X1"), ("CUR-R", "K1X-R"), ("CUR-R", "CUR1"), ("CUR-R", "KH"),
              ("CUR-R", "D27")]
    for a, b in kpairs:
        for k in range(1, 9):
            pair("khop ca_fwd", a, b, lambda s: (a, s, "khop_ca_fwd", k, "all"), lambda s: (b, s, "khop_ca_fwd", k, "all"),
                 f"ca_fwd k{k}")
        pair("khop ca_fwd order-sensitive", a, b, lambda s: (a, s, "khop_ca_fwd", 3, "ordersens"),
             lambda s: (b, s, "khop_ca_fwd", 3, "ordersens"), "ca_fwd k3 order-sensitive")
        pair("khop ca_fwd all-+/-", a, b, lambda s: (a, s, "khop_ca_fwd", 3, "alladd"),
             lambda s: (b, s, "khop_ca_fwd", 3, "alladd"), "ca_fwd k3 all-+/-")
    for a, b in [("D27", "A2Z"), ("KH", "K1"), ("CUR-R", "K1X-R")]:
        for s_ in ("khop_ca_rev", "khop_wp_fwd", "khop_vb_fwd", "khop_cag_fwd"):
            for k in (2, 3, 4):
                pair(f"khop {s_[5:]}", a, b, lambda s: (a, s, s_, k, "all"), lambda s: (b, s, s_, k, "all"),
                     f"{s_[5:]} k{k}")
    vec = gvec
    gpairs = [("A2orig", "A2re")] + [(x, r) for x in ("KH", "K1", "KD", "H3", "R3", "D27", "CUR-R", "K1X-R")
                                       for r in ("A2orig", "A2re")] + \
             [("CUR-R", "K1X-R"), ("CUR-R", "KH"), ("K1X-R", "K1"), ("D27", "KH"), ("KH", "K1")]
    for a, b in gpairs:
        for c in ["rc1", "rc2", "rc3", "rc4", "plain249"] + PLAIN:
            pair("gsm", a, b, lambda s: (a, s, c), lambda s: (b, s, c), c)
    with open(os.path.join(HERE, "contrasts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(con[0]))
        w.writeheader()
        w.writerows(con)
    C = {(r["contrast"], r["cell"]): r for r in con}

    # ---------------- tables.md
    L = []
    def table(head, rows):
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for r in rows:
            L.append("| " + " | ".join(str(x) for x in r) + " |")
        L.append("")
    L.append("### Table 1. One-pass accuracy on (ca, fwd) by hops, seed mean (percent; chance 20)\n")
    L.append("Trained hop range k = 1-6 for every chain arm; k = 7, 8 are not in any training file (100 items each).\n")
    rows = []
    for a, desc in ARMS:
        if (a, "khop_ca_fwd", 1, "all") not in S:
            continue
        n = S[(a, "khop_ca_fwd", 1, "all")]["n_seeds"]
        rows.append([a, n] + [f1(S[(a, "khop_ca_fwd", k, "all")]["mean"]) for k in range(1, 9)] +
                    [f1(S.get((a, "khop_blind", 3, "all"), {}).get("mean"))])
    table(["arm", "seeds"] + [f"k{k}" for k in range(1, 9)] + ["blind k3"], rows)
    L.append("### Table 2. 95% seed-t intervals on (ca, fwd), five-seed arms\n")
    rows = []
    for a, _ in ARMS:
        r1 = S.get((a, "khop_ca_fwd", 1, "all"))
        if not r1 or r1["n_seeds"] != 5:
            continue
        rows.append([a] + [f"{f1(S[(a, 'khop_ca_fwd', k, 'all')]['mean'])} [{f1(S[(a, 'khop_ca_fwd', k, 'all')]['ci_lo'])}, "
                           f"{f1(S[(a, 'khop_ca_fwd', k, 'all')]['ci_hi'])}]" for k in (2, 3, 4, 6, 8)])
    table(["arm", "k2", "k3", "k4", "k6", "k8"], rows)
    L.append("### Table 3. Per-seed (ca, fwd) values for the arms in the first author's reading\n")
    rows = []
    for a in ("A2Z", "K1", "KH", "KD", "D27", "CUR-R"):
        for k in (1, 2, 3, 4, 6, 8):
            r = S[(a, "khop_ca_fwd", k, "all")]
            rows.append([a, f"k{k}", f1(r["mean"]), r["seeds"]])
    table(["arm", "cell", "mean", "per seed"], rows)
    L.append("### Table 4. (ca, fwd) subsets, seed mean (items in brackets)\n")
    L.append("alladd = every step of the queried chain is + / - (an order-free signed sum is exact); hard = no "
             "drop-one-constant estimate puts the gold among its nearest options (build_khop.hard_item); ordersens = an "
             "order-free signed-sum reader does not single out the gold (a x step re-weights earlier constants).\n")
    rows = []
    for a in ("A2Z", "K1", "KH", "KD", "D27", "CUR1", "CUR-R"):
        for sub in ("alladd", "hard", "ordersens"):
            rows.append([a, sub] + [f"{f1(S[(a, 'khop_ca_fwd', k, sub)]['mean'])} ({S[(a, 'khop_ca_fwd', k, sub)]['n_items']})"
                                    if (a, "khop_ca_fwd", k, sub) in S else "" for k in range(1, 9)])
    table(["arm", "subset"] + [f"k{k} (n)" for k in range(1, 9)], rows)
    L.append("### Table 5. Other eval sets at k = 3, seed mean\n")
    rows = []
    sets5 = ["khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev", "khop_cag_fwd", "khop_ca_fwd_qs"]
    for a, _ in ARMS:
        if (a, "khop_ca_fwd", 1, "all") not in S:
            continue
        rows.append([a] + [f1(S.get((a, s, 3, "all"), {}).get("mean")) for s in sets5] +
                    [f1(S.get((a, "khop_blind", 3, "all"), {}).get("mean"))])
    table(["arm"] + [s[5:] for s in sets5] + ["blind"], rows)
    L.append(f"### Table 6. GSM, seed mean (percent). rc cells on the {n_prim} primary items (pfx_table.py)\n")
    rows = []
    for a in ["A2orig", "A2re", "A2-96k"] + [x for x, _ in ARMS]:
        if a in G:
            g = G[a]
            rows.append([a, g["n_seeds"]] + [f1(g[c]) if g[c] != "" else "" for c in PLAIN + ["plain249", "rc1", "rc2", "rc3", "rc4"]])
    table(["arm", "seeds"] + PLAIN + ["plain249", "rc1", "rc2", "rc3", "rc4"], rows)
    L.append("### Table 7. Key contrasts (A - B, points; seed-paired, n = 5 seeds)\n")
    L.append("seed-t = 95% Student-t interval on the 5 per-seed differences; two-level = bootstrap over seeds then items "
             "(paired_stats.py, 10,000 draws, boot seed 0/1).\n")
    keys = [("KH - K1", "ca_fwd k2"), ("KH - K1", "ca_fwd k3"), ("KD - KH", "ca_fwd k3"), ("D27 - KD", "ca_fwd k3"),
            ("D27 - R3", "ca_fwd k3"), ("R3 - KD", "ca_fwd k3"), ("D27 - A2Z", "ca_fwd k3"), ("D27 - K1", "ca_fwd k3"),
            ("D27 - A2Z", "ca_fwd k4"), ("D27 - A2Z", "ca_fwd k6"), ("D27 - A2Z", "ca_fwd k8"),
            ("D27 - A2Z", "ca_fwd k3 all-+/-"), ("D27 - A2Z", "ca_fwd k3 order-sensitive"), ("CUR1 - K1X1", "ca_fwd k3"), ("CUR-R - K1X-R", "ca_fwd k3"),
            ("CUR-R - CUR1", "ca_fwd k3"), ("CUR-R - CUR1", "ca_fwd k4"),
            ("D27 - A2orig", "rc2"), ("D27 - A2re", "rc2"), ("D27 - A2orig", "gsm8kv2"), ("D27 - A2orig", "gsm8kx"),
            ("KH - A2orig", "rc2"), ("CUR-R - A2orig", "rc2"), ("CUR-R - A2re", "rc2"), ("CUR-R - A2orig", "rc3"),
            ("CUR-R - A2orig", "gsm8kv2"), ("CUR-R - A2orig", "gsm8kx"), ("CUR-R - A2orig", "gsm8k"),
            ("CUR-R - A2re", "gsm8kv2"), ("K1X-R - A2orig", "rc2"), ("CUR-R - K1X-R", "rc2"), ("CUR-R - KH", "rc2"),
            ("A2orig - A2re", "rc2")]
    rows = []
    for k in keys:
        r = C.get(k)
        if r:
            s4 = (f"{r['mean_s17_20']:+.1f} [{r['seed_t_lo_s17_20']:+.1f}, {r['seed_t_hi_s17_20']:+.1f}]"
                  if r["mean_s17_20"] != "" else "")
            rows.append([r["contrast"], r["cell"], r["n_items"], f"{r['mean']:+.1f}", f"[{r['seed_t_lo']:+.1f}, {r['seed_t_hi']:+.1f}]",
                         f"[{r['two_level_lo']:+.1f}, {r['two_level_hi']:+.1f}]", "yes" if r["both_exclude_0"] else "no",
                         s4, r["per_seed"]])
    table(["contrast", "cell", "items", "mean", "seed-t 95%", "two-level 95%", "both exclude 0", "s17-20 only (seed-t)",
           "per seed"], rows)
    L.append("### Table 8. Runs and update counts (from the job logs)\n")
    rows = []
    for a in ["A2orig", "A2re"] + [x for x, _ in ARMS]:
        for s, r in sorted(runs.get(a, {}).items()):
            if a == "A2re" and s == 19:
                continue
            rows.append([a, s, r["job"], r["run_dir"] or "", "+".join(str(x["steps"]) for x in r["stages"]) or "eval only",
                         r.get("lr") if r.get("lr") is not None else "", (r.get("extra_train") or "").replace("data/", ""),
                         (r.get("init_model") or "").replace("models/", "")])
    table(["arm", "seed", "job", "run dir (results repo runs/)", "updates", "LR", "extra train file", "init / evaluated model"], rows)
    open(os.path.join(HERE, "tables.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
