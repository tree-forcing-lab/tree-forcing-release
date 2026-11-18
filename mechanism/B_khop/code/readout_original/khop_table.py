#!/usr/bin/env python3
"""CARD_khop_v1 readout: the k-hop curves, the installation lines I1-I4, N1, the GSM rc cells and the verdict of section 0.
Standard library, Python 3.9 (pfx_table.py for the GSM rc cells; build_khop.parse_prompt for the binding share).

    python3 -I khop_table.py --khop-dir <E>/src/data/eval/khop --hint-dir <E>/src/data/eval/hint \\
        --run KH:s16=<756 tgz> ... --run K1:s16=<757 tgz> ... --run KD:s16=<766 tgz> ... \\
        --run A2Z:s16=<751 tgz> ... --run BASE0:s0=<750 tgz> --run A2Z96:s16=<771 tgz> \\
        --ref-orig s16=<520 tgz> ... --ref-retrain s16=<663 tgz> ... [--json out.json]

A run is a results tarball or a directory; its khop preds are found by name (preds_khop_*.jsonl anywhere inside), its
metrics.json and logs/<name>.out too. Cells: accuracy of arm X on set khop_<task>_<order> at chain length k, one value
per seed; m = seed mean; CI = m +/- t x SD / sqrt(n) (t = 2.78 at n = 5; TCRIT for other n). SOLVED: m >= 0.80 and CI
lower >= 0.70; FAILED: CI upper < 0.80; k* = the largest k <= 6 with cells 1..k all SOLVED. rc2_ref / rc3_ref: per seed,
the rc cell of the set (KH, A2 originals, A2 re-trains) with the highest seed mean on that cell (best-A2 rule; G2 =
a_KH(ca, fwd, 2) - rc2_ref, G3 at k = 3). Readings in the card's order: MANIPULATION FAILED (I1-I4), PASS, NEGATIVE,
NULL, BETWEEN (verdict()). N1 (plain not down by more than 2 points) uses the better A2 set per plain metric.
Reported, no verdict: KH - K1, KH - A2Z, fwd - rev at k = 2, 3 (paired by seed), wp / vb k*, G3, ca-gsm, extrapolation
k = 7, 8, BASE0 / A2Z96 curves, and the nearest-chain share of wrong picks (chance about 0.25 per item: the share of the
four distractors whose last statement is nearest to the target's last statement).
Amendment 2 (build critic finding 1): the hard subset of a ca / wp / cag cell = the items where every constant of the
target chain is needed (build_khop.hard_item: no drop-one-constant estimate puts the gold among its nearest options).
Reported per run and cell (Amendment 3: reported only, no longer a verdict input). Also reported per run: the agreement
of the run's picks with the chase-sign magnitude policy (the option nearest the target's root with x2 / x3 applied and
every +c / -c at the mean constant).
Amendment 3 (FIXROUND_CRITIC_khop.md finding 1): the five chains share one root and one op sequence, so a x2 / x3 step
moves all five options the same way and never decides the answer; only the + / - steps (n_add of an item) are composed.
- The word "composes" needs m(a_KH(ca, fwd, 3 | n_add = 3)) >= 0.80, the five-seed mean on the items of (ca, fwd, 3)
  whose three steps are all + / - (48 items in the data; (wp, fwd, 3 | n_add = 3), 67 items, printed beside it). Else
  the reading is "PASS (approximate)". The full-cell SOLVED / FAILED bars, k* and G2 stay the decision lines.
- The claim names the number of + / - steps composed, not "k given operations" (claim(), printed as "claim:").
- G2 on the all-+/- items of (ca, fwd, 2) (63 items) is printed with its CI: a reported line, not a decision line.
- Every ca / wp / cag curve is also printed split by n_add, and the all-+/- curve (n_add = k) on its own line.
- Two rank probes are printed as reference lines per cell (data only) and as agreement with each run's picks:
  rank_sum (the option whose rank among the five equals the rank of the target chain's signed sum of + / - constants,
  every x step ignored) and rank_h1 (the same with the signed first constant only). Ties: uniform over the tied ranks.
  On an all-+/- item the signed sum is the exact value, so rank_sum is 1.00 there by construction.
"""
import argparse, io, json, math, os, re, sys, tarfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
E = os.path.dirname(HERE)
TCRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57}
SETS = ["khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev", "khop_cag_fwd",
        "khop_ca_fwd_qs", "khop_blind"]
ROWS = {"khop_ca_fwd": 1100, "khop_ca_rev": 1100, "khop_wp_fwd": 1100, "khop_wp_rev": 1100, "khop_vb_fwd": 1100,
        "khop_vb_rev": 1100, "khop_cag_fwd": 900, "khop_ca_fwd_qs": 900, "khop_blind": 324}
CELLS6 = [(t, o) for t in ("ca", "wp", "vb") for o in ("fwd", "rev")]
PLAIN = ["gsm8kx", "gsm1kx", "gsm8kv2", "valx"]
BAR = {"solved_m": 0.80, "solved_lo": 0.70, "I1": 0.90, "I2": 0.85, "I3": 0.30, "G2": 0.15, "KD": 0.05, "N1": 0.02,
       "alladd": 0.80}
I4_ROWS = {"KH": 3000, "K1": 3000, "KD": 9000}


def read_jsonl(b):
    return [json.loads(l) for l in io.BytesIO(b) if l.strip()]


def load_run(path):
    """{"preds": {set: {item_id: (correct, label, probs)}}, "metrics": dict or None, "log": str, "path": path, "all_preds":
    every preds_<name>.jsonl name}."""
    out = {"preds": {}, "metrics": None, "log": "", "path": path, "all_preds": []}
    def take(name, rows):
        d = {}
        for r in rows:
            k = max(range(len(r["probs"])), key=r["probs"].__getitem__)
            d[r["item_id"]] = (int(r["labels"][k] == r["gold_label"]), r["labels"][k], r["probs"])
        out["preds"][name] = d
    def visit(name, data):
        a = re.search(r"(?:^|/)preds_([A-Za-z0-9_@.-]+)\.jsonl$", name)
        if a:
            out["all_preds"].append(a.group(1))
        m = re.search(r"(?:^|/)preds_(khop_[a-z_]+)\.jsonl$", name)
        if m:
            take(m.group(1), read_jsonl(data()))
        elif name.endswith("/metrics.json") or name == "metrics.json":
            out["metrics"] = json.loads(data())
        elif re.search(r"(?:^|/)logs/[^/]+\.out$", name):
            out["log"] += data().decode("utf-8", "replace")
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                visit(os.path.relpath(p, path), lambda p=p: open(p, "rb").read())
    else:
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.isfile():
                    visit(m.name, lambda m=m: t.extractfile(m).read())
    return out


def load_khop(khop_dir):
    """{set: [rows]} of the khop eval files."""
    return {s: [json.loads(l) for l in open(os.path.join(khop_dir, s + ".jsonl")) if l.strip()]
            for s in SETS if os.path.exists(os.path.join(khop_dir, s + ".jsonl"))}


def acc(run, s, ids):
    p = run["preds"].get(s)
    if p is None:
        return None
    xs = [p[i][0] for i in ids if i in p]
    return sum(xs) / len(xs) if xs else None


def by_k(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["k"]].append(r["item_id"])
    return d


def curve(run, khop, s):
    """{k: accuracy} of one run on one set."""
    return {k: acc(run, s, ids) for k, ids in sorted(by_k(khop.get(s, [])).items())}


def qs_pair(run, khop, k=1):
    """Share of ca fwd items at chain length k where the item and its -qs twin are both right."""
    a, q = run["preds"].get("khop_ca_fwd"), run["preds"].get("khop_ca_fwd_qs")
    if a is None or q is None:
        return None
    ids = [r["item_id"] for r in khop.get("khop_ca_fwd", []) if r["k"] == k and r["item_id"] + "-qs" in q and r["item_id"] in a]
    return sum(a[i][0] * q[i + "-qs"][0] for i in ids) / len(ids) if ids else None


def blind(run, khop):
    return acc(run, "khop_blind", [r["item_id"] for r in khop.get("khop_blind", [])])


_BK = []


def build_khop():
    """scripts/build_khop.py of E/src_khop/repo (the generator's parser and D1 lines), loaded once."""
    if not _BK:
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_khop", os.path.join(E, "src_khop/repo/scripts/build_khop.py"))
        B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(B)
        _BK.append(B)
    return _BK[0]


_HARD = {}


def hard_ids(khop, s):
    """{k: [item_id]} of the hard rows of set s (build_khop.hard_item True); {} for vb and blind sets."""
    if s not in _HARD:
        B = build_khop()
        d = defaultdict(list)
        for r in khop.get(s, []):
            if B.hard_item(r):
                d[r["k"]].append(r["item_id"])
        _HARD[s] = dict(d)
    return _HARD[s]


def hard_curve(run, khop, s):
    """{k: accuracy on the hard rows} of one run on one set."""
    return {k: acc(run, s, ids) for k, ids in sorted(hard_ids(khop, s).items())}


def n_add_of(row):
    """Number of + / - steps of the queried chain (None for vb / blind rows)."""
    B = build_khop()
    h = B.query_hops(B.parse_prompt(row["prompt"]))
    return None if h is None else sum(o in ("+", "-") for o, _ in h[1])


_SPLIT = {}


def split_ids(khop, s):
    """{k: {n_add: [item_id]}} of set s; {} for vb and blind sets."""
    if s not in _SPLIT:
        d = defaultdict(lambda: defaultdict(list))
        for r in khop.get(s, []):
            na = n_add_of(r)
            if na is not None:
                d[r["k"]][na].append(r["item_id"])
        _SPLIT[s] = {k: dict(sorted(v.items())) for k, v in sorted(d.items())}
    return _SPLIT[s]


def alladd_ids(khop, s):
    """{k: [item_id]} of the rows of set s whose k steps are all + / - (n_add = k)."""
    return {k: v[k] for k, v in split_ids(khop, s).items() if k in v}


def split_curve(run, khop, s):
    """{k: {n_add: accuracy}} of one run on one set."""
    return {k: {na: acc(run, s, ids) for na, ids in v.items()} for k, v in split_ids(khop, s).items()}


def alladd_curve(run, khop, s):
    """{k: accuracy on the n_add = k rows} of one run on one set."""
    return {k: acc(run, s, ids) for k, ids in alladd_ids(khop, s).items()}


def chain_keys(P):
    """(target chain index, [[(kind, const)] per chain from hop 1 to the queried depth], [option index per chain]) of a
    parsed ca / wp / cag prompt; None for vb / blind. The five chains are the five names at the queried depth."""
    B = build_khop()
    if B.query_hops(P) is None:
        return None
    depth = {}

    def dep(n):
        if n not in depth:
            par = P["by"][n]["parent"]
            depth[n] = 0 if par is None else 1 + dep(par)
        return depth[n]

    qd = dep(P["query"])
    names = [st["name"] for st in P["stmts"] if dep(st["name"]) == qd]
    ops = []
    for n in names:
        o = []
        while P["by"][n]["op"] != "root":
            o.append((P["by"][n]["op"], P["by"][n]["c"]))
            n = P["by"][n]["parent"]
        ops.append(o[::-1])
    opt = [P["options"].index(P["values"][n]) for n in names]
    return names.index(P["query"]), ops, opt


def _signed(o, c):
    return (c if o == "+" else -c) if o in ("+", "-") else 0


RANK_KEYS = {"rank_sum": lambda ops: sum(_signed(o, c) for o, c in ops),
             "rank_h1": lambda ops: _signed(*ops[0])}


def rank_probe(row, probe):
    """(expected accuracy, candidate option indices) of a rank probe on one row (critic probe.py by_rank: the options
    whose rank equals the target's rank on the key; ties among keys give every tied rank). None for vb / blind."""
    B = build_khop()
    P = B.parse_prompt(row["prompt"])
    ck = chain_keys(P)
    if ck is None:
        return None
    tgt, ops, _ = ck
    keys = [RANK_KEYS[probe](o) for o in ops]
    srt = sorted(P["options"])
    lo = sum(x < keys[tgt] for x in keys)
    hi = sum(x <= keys[tgt] for x in keys)
    cand = [P["options"].index(srt[i]) for i in range(lo, hi)]
    gold = "ABCDE".index(row["gold_label"])
    return (1.0 / len(cand) if gold in cand else 0.0), cand


_PROBE = {}


def probe_ref(khop, s, probe):
    """{k: (accuracy on the full cell, n, accuracy on the n_add = k rows, n)} of a rank probe on set s (data only)."""
    if (s, probe) not in _PROBE:
        full, add = defaultdict(list), defaultdict(list)
        for r in khop.get(s, []):
            x = rank_probe(r, probe)
            if x is None:
                continue
            full[r["k"]].append(x[0])
            if n_add_of(r) == r["k"]:
                add[r["k"]].append(x[0])
        m = lambda xs: sum(xs) / len(xs) if xs else None
        _PROBE[(s, probe)] = {k: (m(full[k]), len(full[k]), m(add[k]), len(add[k])) for k in sorted(full)}
    return _PROBE[(s, probe)]


def probe_agree(run, khop, s, probe, ks=(2, 3)):
    """(share of the run's picks equal to the probe's pick, n) over the rows of cells ks where the probe has a unique
    candidate."""
    p = run["preds"].get(s)
    if p is None:
        return None
    n = same = 0
    for r in khop.get(s, []):
        if r["k"] not in ks or r["item_id"] not in p:
            continue
        x = rank_probe(r, probe)
        if x is None or len(x[1]) != 1:
            continue
        n += 1
        same += r["labels"][x[1][0]] == p[r["item_id"]][1]
    return (same / n, n) if n else None


def chase_agree(run, khop, s, ks=(2, 3)):
    """(share of the run's picks equal to the chase-sign pick, n) over the rows of cells ks where the chase-sign policy
    has a unique nearest option."""
    B = build_khop()
    p = run["preds"].get(s)
    if p is None:
        return None
    n = same = 0
    for r in khop.get(s, []):
        if r["k"] not in ks or r["item_id"] not in p:
            continue
        P = B.parse_prompt(r["prompt"])
        h = B.query_hops(P)
        t = B.task_of(r)
        if h is None or t not in B.RANGES:
            continue
        x = B.chase_est(h[0], h[1], set(), (B.RANGES[t][4] + B.RANGES[t][5]) / 2)
        d = [abs(o - x) for o in P["options"]]
        near = [i for i in range(5) if d[i] == min(d)]
        if len(near) != 1:
            continue
        n += 1
        same += r["labels"][near[0]] == p[r["item_id"]][1]
    return (same / n, n) if n else None


def nearest_share(run, khop, s, ks=(2, 3)):
    """(observed share of wrong picks that are a nearest chain, expected share under a uniform wrong pick, n wrong)."""
    B = build_khop()
    p = run["preds"].get(s)
    if p is None:
        return None
    obs = exp = n = 0
    for r in khop.get(s, []):
        if r["k"] not in ks or r["item_id"] not in p or p[r["item_id"]][0]:
            continue
        P = B.parse_prompt(r["prompt"])
        last = {}
        for i, st in enumerate(P["stmts"]):
            last[P["chain_of"][st["name"]]] = max(last.get(P["chain_of"][st["name"]], -1), i)
        tgt = P["chain_of"][P["query"]]
        dist = {c: abs(last[c] - last[tgt]) for c in last if c != tgt}
        near = {c for c, d in dist.items() if d == min(dist.values())}
        # the chain of the picked option: the chain whose value at the queried depth equals it
        depth = {}
        for st in P["stmts"]:
            depth[st["name"]] = 0 if st["parent"] is None else depth.get(st["parent"], 0) + 1
        qd = depth[P["query"]]
        val_chain = {P["values"][n_]: P["chain_of"][n_] for n_ in P["values"] if depth[n_] == qd}
        picked = P["options"][r["labels"].index(p[r["item_id"]][1])]
        c = val_chain.get(picked)
        n += 1
        obs += c in near
        exp += len(near) / 4
    return (obs / n, exp / n, n) if n else None


def stats(xs):
    """(mean, sd, lo, hi, n) with the t CI; xs without None."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if not n:
        return None
    m = sum(xs) / n
    if n < 2:
        return (m, float("nan"), float("nan"), float("nan"), n)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = TCRIT.get(n, 2.0) * sd / math.sqrt(n)
    return (m, sd, m - h, m + h, n)


def solved(st):
    return st is not None and st[4] >= 2 and st[0] >= BAR["solved_m"] and st[2] >= BAR["solved_lo"]


def failed(st):
    return st is not None and st[4] >= 2 and st[3] < BAR["solved_m"]


def kstar(cell_stats):
    """cell_stats: {k: stats}. The largest k <= 6 with cells 1..k all SOLVED (0 when k = 1 is not)."""
    k = 0
    while k < 6 and solved(cell_stats.get(k + 1)):
        k += 1
    return k


def paired(a, b, seeds):
    """Per-seed a - b over the seeds both have."""
    return [a[s] - b[s] for s in seeds if a.get(s) is not None and b.get(s) is not None]


def ci_text(st):
    return f"m {st[0]:.3f} CI [{st[2]:.3f}, {st[3]:.3f}] (n seeds {st[4]})" if st else "missing"


def claim(reading, d):
    """The claim sentence of a PASS / PASS (approximate) reading (Amendment 3 item 1: it names the number of + / -
    steps composed); None for the other readings (their text is in the card, section 0 items 3-5)."""
    n3 = d.get("n_add3")
    nt = f"the {n3} items" if n3 else "the items"
    st = stats(list((d.get("kh_add3") or {}).values()))
    x = f"{st[0]:.3f}" if st else "missing"
    if reading == "PASS":
        return ("one pass of Qwen3.5-4B composes 3 + / - steps on small integers in reading order (3-step chains whose "
                f"x2 / x3 steps are shared by all five options; on {nt} of (ca, fwd, 3) where all three steps are + / - "
                f"the mean is {x}), while the GSM rc2 cell stays at least 15 points lower; the rc2 ceiling is not a "
                "limit on composing given + / - steps")
    if reading == "PASS (approximate)":
        return (f"one pass of Qwen3.5-4B selects the value of a 3-step chain in reading order; on {nt} where all "
                f"three steps are + / - the mean is {x}")
    return None


def verdict(d):
    """The section 0 readings in order. d: {"I": {"I1": bool, ...}, "kh": {k: {seed: acc}} (ca fwd), "kd": same,
    "g2": {seed: G2}, "kh_add3": {seed: acc on the all-+/- rows of (ca, fwd, 3)} (Amendment 3: the "composes" line),
    and printed only: "wp_add3" (same on (wp, fwd, 3)), "g2_add2" ({seed: G2 on the all-+/- rows of (ca, fwd, 2)}),
    "kh_hard" ({seed: acc on the hard rows of (ca, fwd, 3)}), "n_add3" / "n_add3_wp" / "n_add2" (item counts)}.
    Returns (reading, detail lines); a PASS / PASS (approximate) reading ends with a "claim: ..." line."""
    lines = []
    bad = [k for k, v in d["I"].items() if not v]
    if bad:
        return "MANIPULATION FAILED", [f"installation lines failing: {', '.join(sorted(bad))}"]
    seeds = sorted(d["kh"].get(1, {}))
    kh = {k: stats(list(v.values())) for k, v in d["kh"].items()}
    kd = {k: stats(list(v.values())) for k, v in d.get("kd", {}).items()}
    ks = kstar(kh)
    g2 = stats(list(d["g2"].values()))
    lines.append(f"k*(KH, ca, fwd) = {ks}; G2 m {g2[0]:+.3f} CI [{g2[2]:+.3f}, {g2[3]:+.3f}]" if g2 else "G2 missing")
    g2a = stats(list((d.get("g2_add2") or {}).values()))
    lines.append(f"G2 on the all-+/- items of (ca, fwd, 2) (n items {d.get('n_add2', '-')}, reported, not a decision "
                 f"line): " + (f"m {g2a[0]:+.3f} CI [{g2a[2]:+.3f}, {g2a[3]:+.3f}]" if g2a else "missing"))
    if ks >= 3 and g2 and g2[0] >= BAR["G2"] and g2[2] > 0:
        ad = stats(list((d.get("kh_add3") or {}).values()))
        lines.append(f"all-+/- subset (ca, fwd, 3) (n items {d.get('n_add3', '-')}): {ci_text(ad)}; bar m >= "
                     f"{BAR['alladd']:.2f} for \"composes\"")
        lines.append(f"all-+/- subset (wp, fwd, 3) (n items {d.get('n_add3_wp', '-')}, reported): "
                     f"{ci_text(stats(list((d.get('wp_add3') or {}).values())))}")
        lines.append(f"hard subset (ca, fwd, 3) (reported only since Amendment 3): "
                     f"{ci_text(stats(list((d.get('kh_hard') or {}).values())))}")
        r = "PASS (approximate)" if not ad or ad[0] < BAR["alladd"] else "PASS"
        return r, lines + [f"claim: {claim(r, d)}"]
    kd3 = paired(d.get("kd", {}).get(3, {}), d["kh"].get(3, {}), seeds)
    dd = stats(kd3)
    if failed(kh.get(3)) and failed(kd.get(3)) and dd and dd[3] < BAR["KD"]:
        lines.append(f"KD - KH at (ca, fwd, 3) m {dd[0]:+.3f} CI upper {dd[3]:+.3f} < +{BAR['KD']}")
        if failed(kh.get(2)) and failed(kd.get(2)):
            lines.append("cell (ca, fwd, 2) FAILED under KH and KD: the claim is k* <= 1")
        return "NEGATIVE", lines
    if g2 and g2[2] <= 0 <= g2[3]:
        return "NULL", lines
    why = []
    if not solved(kh.get(3)) and not failed(kh.get(3)):
        why.append("cell (ca, fwd, 3) neither SOLVED nor FAILED")
    if ks >= 3 and g2 and g2[2] > 0 and g2[0] < BAR["G2"]:
        why.append("k* >= 3 and G2 CI lower > 0 but m(G2) < 0.15")
    if failed(kh.get(3)) and solved(kd.get(3)):
        why.append("KH fails k = 3 and KD solves it: data-limited at 3,000 rows")
    elif failed(kh.get(3)) and failed(kd.get(3)) and dd and dd[3] >= BAR["KD"]:
        why.append("KH and KD both fail k = 3, KD - KH CI upper >= +0.05: data- or schedule-limited")
    return "BETWEEN", lines + why


def rc_cells(path, hint_dir):
    """{rc: accuracy} of a run on the 249 primary items (pfx_table.py)."""
    import pfx_table as P
    run = P.load_run(path)
    h = P.load_hints(hint_dir)
    cl = P.cells(h)
    prim = P.primary_items(cl)
    return {rc: P.acc_rows(run, [(cl[i][rc][0], i) for i in prim if rc in cl[i]])[0] for rc in (1, 2, 3, 4)}


def plain_acc(run, name):
    m = run.get("metrics") or {}
    return ((m.get("eval") or {}).get(name) or {}).get("acc")


def i4(run, arm):
    """I4: the log's extra-train line with the arm's row count, and metrics.json extra_train agrees."""
    n = I4_ROWS[arm]
    m = re.findall(r"^extra-train: (\d+) rows from (\S+)", run["log"], re.M)
    mx = (run.get("metrics") or {}).get("extra_train") or {}
    return len(m) == 1 and int(m[0][0]) == n and mx == {m[0][1]: n}


def parse_runs(specs):
    """--run ARM:s<seed>=<path> -> {arm: {seed: path}}."""
    out = defaultdict(dict)
    for s in specs:
        arm, rest = s.split(":", 1)
        seed, path = rest.split("=", 1)
        out[arm][int(seed.lstrip("s"))] = path
    return out


def parse_refs(specs):
    out = {}
    for s in specs or []:
        seed, path = s.split("=", 1)
        out[int(seed.lstrip("s"))] = path
    return out


def pts(x):
    return "   -  " if x is None else "%5.1f" % (100 * x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--hint-dir", default=os.path.join(E, "src/data/eval/hint"))
    ap.add_argument("--run", action="append", default=[], help="ARM:s<seed>=<tgz or dir>")
    ap.add_argument("--ref-orig", action="append", default=[], help="s<seed>=<A2-6k original tgz> (j520 ...)")
    ap.add_argument("--ref-retrain", action="append", default=[], help="s<seed>=<A2-6k re-train tgz> (j663 ...)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    khop = load_khop(a.khop_dir)
    specs = parse_runs(a.run)
    runs = {arm: {s: load_run(p) for s, p in d.items()} for arm, d in specs.items()}
    out = {"curves": {}, "I": {}, "lines": []}
    pr = lambda *x: (print(*x), out["lines"].append(" ".join(str(y) for y in x)))
    # curves per arm and set
    for arm, d in runs.items():
        for s in SETS:
            if s == "khop_blind":
                continue
            for seed, run in sorted(d.items()):
                c = curve(run, khop, s)
                out["curves"].setdefault(arm, {}).setdefault(s, {})[seed] = c
                pr(f"{arm:6s} s{seed} {s:15s} " + " ".join(f"k{k} {pts(v)}" for k, v in c.items()))
                if s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_cag_fwd"):
                    hc = hard_curve(run, khop, s)
                    out.setdefault("hard", {}).setdefault(arm, {}).setdefault(s, {})[seed] = hc
                    n_h = {k: len(v) for k, v in hard_ids(khop, s).items()}
                    pr(f"{arm:6s} s{seed} {s + ' hard':20s} " + " ".join(f"k{k} {pts(v)} (n {n_h[k]})" for k, v in hc.items()))
                if s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_cag_fwd", "khop_ca_fwd_qs"):
                    sc, ac = split_curve(run, khop, s), alladd_curve(run, khop, s)
                    out.setdefault("split", {}).setdefault(arm, {}).setdefault(s, {})[seed] = sc
                    out.setdefault("alladd", {}).setdefault(arm, {}).setdefault(s, {})[seed] = ac
                    sp = split_ids(khop, s)
                    pr(f"{arm:6s} s{seed} {s + ' by n+/-':20s} " + " | ".join(
                        f"k{k} " + " ".join(f"n{na} {pts(v)} ({len(sp[k][na])})" for na, v in c.items()) for k, c in sc.items()))
                    pr(f"{arm:6s} s{seed} {s + ' all +/-':20s} " + " ".join(
                        f"k{k} {pts(v)} (n {len(sp[k][k])})" for k, v in ac.items()))
        for seed, run in sorted(d.items()):
            ns = nearest_share(run, khop, "khop_ca_fwd") if run["preds"].get("khop_ca_fwd") else None
            ca_ = chase_agree(run, khop, "khop_ca_fwd") if run["preds"].get("khop_ca_fwd") else None
            ag = {pb: probe_agree(run, khop, "khop_ca_fwd", pb) if run["preds"].get("khop_ca_fwd") else None
                  for pb in RANK_KEYS}
            pr(f"{arm:6s} s{seed} blind {pts(blind(run, khop))} | qs pair k1 {pts(qs_pair(run, khop))} | nearest-chain "
               f"share of wrong ca fwd picks k2-3 {'-' if not ns else '%.3f (uniform %.3f, n %d)' % ns} | chase-sign "
               f"agreement ca fwd k2-3 {'-' if not ca_ else '%.3f (n %d)' % ca_} | " + " | ".join(
                   f"{pb} agreement ca fwd k2-3 {'-' if not x else '%.3f (n %d)' % x}" for pb, x in ag.items()))
    # rank probe reference lines (data only, Amendment 3 item 4)
    for s in ("khop_ca_fwd", "khop_wp_fwd", "khop_cag_fwd"):
        for pb in RANK_KEYS:
            ref = probe_ref(khop, s, pb)
            out.setdefault("probe_ref", {}).setdefault(s, {})[pb] = ref
            pr(f"probe {pb:8s} {s:12s} " + " ".join(
                f"k{k} {pts(f)} (all +/- {pts(a)}, n {na})" for k, (f, _, a, na) in ref.items()))
    cell = lambda arm, s, k: {seed: out["curves"].get(arm, {}).get(s, {}).get(seed, {}).get(k) for seed in runs.get(arm, {})}
    # installation lines
    I = out["I"]
    if "KH" in runs and "K1" in runs:
        I["I1"] = all(stats(list(cell(arm, f"khop_{t}_{o}", 1).values())) and
                      stats(list(cell(arm, f"khop_{t}_{o}", 1).values()))[0] >= BAR["I1"] for arm in ("KH", "K1") for t, o in CELLS6)
        q = stats([qs_pair(r, khop) for r in runs["KH"].values()])
        I["I2"] = bool(q) and q[0] >= BAR["I2"]
        I["I3"] = all((stats([blind(r, khop) for r in runs[arm].values()]) or (1,))[0] <= BAR["I3"]
                      for arm in ("KH", "K1", "A2Z") if arm in runs)
        I["I4"] = all(i4(r, arm) for arm in ("KH", "K1", "KD") if arm in runs for r in runs[arm].values())
        pr("installation:", json.dumps(I))
    # GSM side: rc cells and plain sets
    orig, ret = parse_refs(a.ref_orig), parse_refs(a.ref_retrain)
    rc = {"orig": {s: rc_cells(p, a.hint_dir) for s, p in orig.items()},
          "retrain": {s: rc_cells(p, a.hint_dir) for s, p in ret.items()}}
    for arm in ("KH", "K1", "KD"):
        if arm in specs:
            rc[arm] = {s: rc_cells(p, a.hint_dir) for s, p in specs[arm].items()}
    for name, d in rc.items():
        for s, c in sorted(d.items()):
            pr(f"rc cells {name:8s} s{s} " + " ".join(f"rc{k} {pts(v)}" for k, v in c.items()))
    def best(rck, cands):
        means = {n: stats([rc[n][s][rck] for s in rc[n]]) for n in cands if rc.get(n)}
        means = {n: st[0] for n, st in means.items() if st}       # a run without hint preds is no candidate
        return max(means, key=means.get) if means else None
    if "KH" in runs and rc.get("orig") and rc.get("retrain"):
        b2, b3 = best(2, ("KH", "orig", "retrain")), best(3, ("KH", "orig", "retrain"))
        g2 = {s: cell("KH", "khop_ca_fwd", 2)[s] - rc[b2][s][2] for s in runs["KH"] if s in rc[b2] and cell("KH", "khop_ca_fwd", 2)[s] is not None}
        g3 = {s: cell("KH", "khop_ca_fwd", 3)[s] - rc[b3][s][3] for s in runs["KH"] if s in rc[b3] and cell("KH", "khop_ca_fwd", 3)[s] is not None}
        pr(f"rc2_ref = {b2}, rc3_ref = {b3}; G3 {stats(list(g3.values()))}")
        kh = {k: cell("KH", "khop_ca_fwd", k) for k in range(1, 9)}
        kd = {k: cell("KD", "khop_ca_fwd", k) for k in range(1, 9)} if "KD" in runs else {}
        kh_hard = {s_: (out.get("hard", {}).get("KH", {}).get("khop_ca_fwd", {}).get(s_) or {}).get(3) for s_ in runs["KH"]}
        kh_hard = {s_: v_ for s_, v_ in kh_hard.items() if v_ is not None}
        add = lambda s, k: {s_: v_ for s_, v_ in {s_: (out.get("alladd", {}).get("KH", {}).get(s, {}).get(s_) or {}).get(k)
                                                  for s_ in runs["KH"]}.items() if v_ is not None}
        a2 = add("khop_ca_fwd", 2)
        g2_add2 = {s: a2[s] - rc[b2][s][2] for s in a2 if s in rc[b2]}
        nadd = lambda s, k: len(alladd_ids(khop, s).get(k, []))
        pr(f"G2 all +/- (ca, fwd, 2) per seed: " + json.dumps({s: round(x, 4) for s, x in sorted(g2_add2.items())}))
        v, lines = verdict({"I": I, "kh": kh, "kd": kd, "g2": g2, "kh_hard": kh_hard, "kh_add3": add("khop_ca_fwd", 3),
                            "wp_add3": add("khop_wp_fwd", 3), "g2_add2": g2_add2, "n_add3": nadd("khop_ca_fwd", 3),
                            "n_add3_wp": nadd("khop_wp_fwd", 3), "n_add2": nadd("khop_ca_fwd", 2)})
        for l in lines:
            pr(l)
        # N1: plain not down (best A2 set per metric)
        refs = {"orig": {s: load_run(p) for s, p in orig.items()}, "retrain": {s: load_run(p) for s, p in ret.items()}}
        n1 = {}
        for pname in PLAIN:
            rs = [stats([plain_acc(r, pname) for r in refs[n].values()]) for n in refs]
            rm = max((x[0] for x in rs if x), default=None)
            ks_ = stats([plain_acc(r, pname) for r in runs["KH"].values()])
            km = ks_[0] if ks_ else None
            n1[pname] = (km, rm, km is not None and rm is not None and km >= rm - BAR["N1"])
        pr("N1 plain (KH mean, best A2 mean, ok):", json.dumps(n1))
        if not all(x[2] for x in n1.values()):
            v += " (plain cost)"
        out["verdict"] = v
        pr(f"VERDICT: {v}")
    # secondary contrasts (reported)
    for k in (2, 3):
        for x, y, s1, s2 in (("KH", "K1", "khop_ca_fwd", "khop_ca_fwd"), ("KH", "A2Z", "khop_ca_fwd", "khop_ca_fwd"),
                             ("KH", "KH", "khop_ca_fwd", "khop_ca_rev"), ("KH", "K1", "khop_ca_rev", "khop_ca_rev")):
            if x in runs and y in runs:
                st = stats(paired(cell(x, s1, k), cell(y, s2, k), sorted(runs[x])))
                if st:
                    pr(f"{x} {s1} - {y} {s2} at k={k}: m {st[0]:+.3f} CI [{st[2]:+.3f}, {st[3]:+.3f}] n {st[4]}")
    for t in ("wp", "vb"):
        for o in ("fwd", "rev"):
            if "KH" in runs:
                pr(f"k*(KH, {t}, {o}) = {kstar({k: stats(list(cell('KH', f'khop_{t}_{o}', k).values())) for k in range(1, 7)})}")
    if "KH" in runs:
        pr(f"k*(KH, ca, rev) = {kstar({k: stats(list(cell('KH', 'khop_ca_rev', k).values())) for k in range(1, 7)})}")
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
