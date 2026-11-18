#!/usr/bin/env python3
"""CARD_khop_v2 readout: the dose curves, the installation lines I1-I5, the section 0 verdict (data dose D27 - KD), the
branch (b) label (rows versus updates, pooled rule of Amendment 1), the log-dose slope, N1, the plain cost of the dose
and the GSM rc transfer / harm sentences. Standard library, Python 3.9; khop_table.py primitives by import (load_run,
curve, alladd_curve, alladd_ids, qs_pair, blind, stats, solved, failed, kstar, paired, rc_cells, probe_agree,
plain_acc); khop_table.py is not edited.

    python3 -I khop2_table.py --khop-dir <E>/src/data/eval/khop --hint-dir <E>/src/data/eval/hint \\
        --run D27:s16=<875 tgz> ... --run R3:s16=<880 tgz> ... --run H3:s16=<885 tgz> ... \\
        --run KD:s16=<766 tgz> ... --run KH:s16=<756 tgz> ... [--run K1:... --run A2Z:... --run BASE0:s0=...] \\
        --ref-orig s16=<520 tgz> ... --ref-retrain s16=<663 tgz> ... [--gate-output gate.txt] [--json out.json]

Cells: a_X(t, o, k) = accuracy of arm X on khop_<t>_<o> at chain length k, one value per seed; m = seed mean; CI = m
+/- t x SD / sqrt(n) (t 2.78 at n = 5). SOLVED: m >= 0.80 and CI lower >= 0.70; FAILED: CI upper < 0.80; k* = the
largest k <= 6 with cells 1..k SOLVED. D_dose = a_D27(ca, fwd, 3) - a_KD(ca, fwd, 3), paired per seed.
Readings, first that applies (verdict()): MANIPULATION FAILED (I1-I5), PASS / PASS (approximate) (k*(D27) >= 3; the
word "composes" needs the all-+/- subset of (ca, fwd, 3) at m >= 0.80), NEGATIVE (D27 FAILED at (ca, fwd, 3) and D_dose
CI upper < +0.05), NULL (D_dose CI includes 0), BETWEEN with the sub-labels in the card's order.
Branch (b) (branch_b()): R27 = D27 - R3, R9 = KD - H3, U9 = R3 - KD, U3 = H3 - KH at (ca, fwd, 3); rows_pool = (R27 +
R9) / 2 and upd_pool = (U9 + U3) / 2 per seed; rows = rows_pool CI lower > 0, upd = upd_pool CI lower > 0; BOTH / ROWS /
UPDATES / UNRESOLVED. Without H3 (gate "GATE PASS (H3 dropped ...)"): rows_pool = R27, upd_pool = U9, "(one level only)".
Exact decisions (Codex fix round, 2026-10-26): every reading, label and I-line is decided on exact rationals. The cells of
the decision arms are Fraction(correct, items) from the preds (xacc), and the CI tests are squared rational forms with t as
the decimal of TCRIT (lo >= L <=> m - L >= 0 and (m - L)^2 n >= t^2 var), so a float rounding (Python 3.9 sum, a - b + c - d)
never moves a bar. Floats are printing only. A decision arm without exactly seeds 16-20, a preds file that misses an item
of its set, or a NaN / None value stops the readout (UNREADABLE); no denominator shrinks. The same holds on the GSM side
(Codex fix round 5): the nine khop files must have the CARD_khop_v1 1.2 row counts, and every decision arm, KD, KH and
both A2-6k sets need all four plain metrics as finite numbers at seeds 16-20, so N1, best-A2 and the plain cost of the
dose always read five seeds. N1 and the pooled plain-cost answer are decided on the exact values too.
GSM side (reported): best-A2 rule per metric (the A2-6k set with the higher five-seed mean); N1 plain not down more than
2 points; D27 - KD per plain set and pooled; per (arm, rc1..rc4) X - best-A2 and X - KD with the three-way sentence.
"""
import argparse, json, math, os, re, sys
from collections import defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402
import khop_pilot_gate as G  # noqa: E402  (wall_hours)

E = os.path.dirname(HERE)
NEW = ("D27", "R3", "H3")
BAR = {"I1": 0.90, "I2": 0.85, "I3": 0.30, "I5": 0.85, "neg": 0.05, "N1": 0.02, "alladd": 0.80}
# arm -> (extra file, chain rows, total training rows, final step)
I4_SPEC = {"D27": ("data/eval/khop2/train_kd27.jsonl", 27000, 32923, 1029),
           "R3": ("data/eval/khop2/train_kd_x3.jsonl", 27000, 32923, 1029),
           "H3": ("data/eval/khop2/train_kh_x3.jsonl", 9000, 14923, 467)}
SEEDS5 = (16, 17, 18, 19, 20)
# exact bars of the decisions (BAR stays for printing); Fraction("0.80") is 4/5, Fraction(0.80) is not
XBAR = {"solved_m": Fraction("0.80"), "solved_lo": Fraction("0.70"), "I1": Fraction("0.90"), "I2": Fraction("0.85"),
        "I3": Fraction("0.30"), "I5": Fraction("0.85"), "neg": Fraction("0.05"), "alladd": Fraction("0.80"),
        "N1": Fraction("0.02")}
LABEL_TEXT = {"BOTH": "both distinct rows and repetition", "ROWS": "distinct rows (distinct rows are the variable)",
              "UNRESOLVED": "an unresolved mix (rows and updates not separated)"}


def unreadable(msg):
    raise SystemExit(f"khop2_table: UNREADABLE: {msg} (the readout never reads a partial or broken input)")


def q(x):
    """The exact value of one per-seed number. Fraction / int pass; a float (test fixtures, a contrast of count ratios)
    is taken as the ratio of integers it rounds from (limit_denominator(10**6), checked to 1e-12: distinct ratios with
    denominators <= 10**6 lie >= 1e-12 apart, float error is about 1e-16). None, bool, NaN and inf stop the readout."""
    if isinstance(x, Fraction) or (isinstance(x, int) and not isinstance(x, bool)):
        return Fraction(x)
    if isinstance(x, float) and math.isfinite(x):
        f = Fraction(x).limit_denominator(10 ** 6)
        if abs(float(f) - x) <= 1e-12:
            return f
    unreadable(f"value {x!r} is not a finite count ratio")


class XS:
    """Exact five-seed statistics of the card: m, var = sum (x - m)^2 / (n - 1), t = TCRIT[n] as its decimal. The CI
    m +/- t sqrt(var / n) is compared in squared form, so lo >= L, lo > L and hi < U are exact."""

    def __init__(self, xs):
        xs = [q(x) for x in xs]
        self.n = n = len(xs)
        self.m = sum(xs, Fraction(0)) / n if n else None
        self.var = sum(((x - self.m) ** 2 for x in xs), Fraction(0)) / (n - 1) if n >= 2 else None
        self.t2 = Fraction(str(T.TCRIT.get(n, 2.0))) ** 2

    def ok(self):
        return self.n >= 2

    def mean_ge(self, b):
        return self.n >= 1 and self.m >= b

    def lo_ge(self, b):
        d = self.m - b if self.ok() else None
        return self.ok() and d >= 0 and d * d * self.n >= self.t2 * self.var

    def lo_gt(self, b):
        d = self.m - b if self.ok() else None
        return self.ok() and d > 0 and d * d * self.n > self.t2 * self.var

    def hi_lt(self, b):
        e = b - self.m if self.ok() else None
        return self.ok() and e > 0 and e * e * self.n > self.t2 * self.var


def xsolved(xs):
    x = XS(xs)
    return x.ok() and x.m >= XBAR["solved_m"] and x.lo_ge(XBAR["solved_lo"])


def xfailed(xs):
    x = XS(xs)
    return x.ok() and x.hi_lt(XBAR["solved_m"])


def xkstar(cells):
    """cells: {k: {seed: value}}. The largest k <= 6 with cells 1..k all SOLVED, exact."""
    k = 0
    while k < 6 and k + 1 in cells and xsolved(list(cells[k + 1].values())):
        k += 1
    return k


def fstats(xs):
    """T.stats on floats of the same values (printing only)."""
    return T.stats([float(x) for x in xs if x is not None])


def need_seeds(d, what):
    """d: {seed: value}; a decision cell must hold exactly seeds 16-20 with finite values."""
    if sorted(d) != list(SEEDS5):
        unreadable(f"{what} has seeds {sorted(d)}, the card reads exactly {list(SEEDS5)}")
    for v in d.values():
        q(v)


def xacc(run, s, ids, what):
    """Fraction(correct, len(ids)) of one run on one set: every id must be in the run's preds of that set."""
    p = run["preds"].get(s)
    if p is None:
        unreadable(f"{what}: no preds_{s}.jsonl in {run.get('path')}")
    miss = [i for i in ids if i not in p]
    if miss or not ids:
        unreadable(f"{what}: preds_{s} of {run.get('path')} misses {len(miss)} of {len(ids)} items")
    return Fraction(sum(int(p[i][0]) for i in ids), len(ids))


def xqs_pair(run, khop, what, k=1):
    """Exact I2: share of the ca fwd k items whose -qs twin (in the qs set) is also right; every pair must be read."""
    twins = {r["item_id"] for r in khop.get("khop_ca_fwd_qs", [])}
    ids = [r["item_id"] for r in khop.get("khop_ca_fwd", []) if r["k"] == k and r["item_id"] + "-qs" in twins]
    xacc(run, "khop_ca_fwd", ids, what)          # both sides of every pair must be read
    xacc(run, "khop_ca_fwd_qs", [i + "-qs" for i in ids], what)
    pa, pq = run["preds"]["khop_ca_fwd"], run["preds"]["khop_ca_fwd_qs"]
    return Fraction(sum(int(pa[i][0]) * int(pq[i + "-qs"][0]) for i in ids), len(ids))


# the A2-6k reference sets of card 2.4 (j540 is s19 in both)
REF_JOBS = {"orig": {16: "520", 17: "525", 18: "530", 19: "540", 20: "542"},
            "retrain": {16: "663", 17: "664", 18: "665", 19: "540", 20: "666"}}
RC_N = {1: 249, 2: 249, 3: 249, 4: 128}     # Amendment (Codex fix round): rc4 on the 128 primary items with an rc4 cell


def xrc_cells(path, hint_dir, what):
    """{rc: Fraction} of a run on the primary items (pfx_table.py: cells, primary_items): rc1-rc3 on all 249, rc4 on the
    128 that have an rc4 cell. Every item of each population must be in the run's preds, else UNREADABLE (T.rc_cells
    dropped missing items)."""
    import pfx_table as P
    run, cl = P.load_run(path), P.cells(P.load_hints(hint_dir))
    prim = P.primary_items(cl)
    out = {}
    for rc in (1, 2, 3, 4):
        rows = [(cl[i][rc][0], i) for i in prim if rc in cl[i]]
        if len(rows) != RC_N[rc]:
            unreadable(f"rc{rc} population has {len(rows)} items, the card reads {RC_N[rc]}")
        miss = [i for st, i in rows if st not in run or i not in run[st]]
        if miss:
            unreadable(f"{what}: {path} misses {len(miss)} of the {len(rows)} rc{rc} items")
        out[rc] = Fraction(sum(int(run[st][i][0]) for st, i in rows), len(rows))
    return out


def check_seed(run, declared, what):
    """The seed and order_seed in the run's metrics.json args must equal the seed of its command-line label, so a wrong
    tarball path (one run under two seeds, a pilot under s17) never enters a paired contrast (Codex fix round 3)."""
    a = (run.get("metrics") or {}).get("args") or {}
    if a.get("seed") != declared or a.get("order_seed") != declared:
        unreadable(f"{what}: {run.get('path')} records seed {a.get('seed')} / order_seed {a.get('order_seed')}, "
                   f"the label says s{declared}")


def i4(run, arm):
    """I4: one extra-train line with the arm's file and row count, the total line, the final step (metrics history and
    the log's last step line), and metrics.json extra_train agrees."""
    path, n, tot, step = I4_SPEC[arm]
    log, m = run["log"], run.get("metrics") or {}
    ex = re.findall(r"^extra-train: (\d+) rows from (\S+)", log, re.M)
    tl = re.findall(r"^extra-train: total training rows (\d+)", log, re.M)
    ls = [int(x) for x in re.findall(r'^\{"step": (\d+),', log, re.M)]
    h = m.get("history") or []
    return (ex == [(str(n), path)] and tl == [str(tot)] and ls[-1:] == [step] and bool(h) and h[-1].get("step") == step
            and m.get("extra_train") == {path: n})


def ci_s(st, pts=True):
    if not st:
        return "missing"
    f = (lambda x: f"{100 * float(x):+.1f}") if pts else (lambda x: f"{float(x):.3f}")
    return f"m {f(st[0])} CI [{f(st[2])}, {f(st[3])}] (n {st[4]})"


def seeds_of(*ds):
    return sorted(set().union(*[set(d) for d in ds if d]))


def pool(a, b):
    """Per-seed (a + b) / 2 over the seeds both have."""
    return {s: (a[s] + b[s]) / 2 for s in a if s in b}


def pdiff(a, b):
    """Per-seed a - b over the seeds both have, as a dict."""
    return {s: a[s] - b[s] for s in sorted(a) if s in b and a[s] is not None and b[s] is not None}


def branch_b(c, have_h3=True, what="k = 3"):
    """c: {arm: {seed: acc}} at one k for D27, R3, KD, H3, KH (each exactly seeds 16-20, else UNREADABLE). Returns
    (label, label text, contrasts) with contrasts {name: {seed: float diff}} for R27, R9, U9, U3, rows_pool, upd_pool
    (Amendment 1 finding 4); the label is decided on the exact contrasts."""
    for a in ("D27", "R3", "KD", "KH") + (("H3",) if have_h3 else ()):
        need_seeds(c.get(a, {}), f"branch (b) {what} cell of {a}")
    c = {a: {s: q(v) for s, v in d.items()} for a, d in c.items() if a != "H3" or have_h3}
    con = {"R27": pdiff(c.get("D27", {}), c.get("R3", {})), "U9": pdiff(c.get("R3", {}), c.get("KD", {}))}
    if have_h3:
        con["R9"] = pdiff(c.get("KD", {}), c.get("H3", {}))
        con["U3"] = pdiff(c.get("H3", {}), c.get("KH", {}))
        con["rows_pool"], con["upd_pool"] = pool(con["R27"], con["R9"]), pool(con["U9"], con["U3"])
    else:
        con["rows_pool"], con["upd_pool"] = dict(con["R27"]), dict(con["U9"])
    rows, upd = XS(con["rows_pool"].values()).lo_gt(0), XS(con["upd_pool"].values()).lo_gt(0)
    rp = fstats(con["rows_pool"].values())
    con = {n: {s: float(x) for s, x in d.items()} for n, d in con.items()}
    lab = "BOTH" if rows and upd else "ROWS" if rows else "UPDATES" if upd else "UNRESOLVED"
    txt = LABEL_TEXT.get(lab) or ("repetition gains; no detectable effect of distinct rows (rows_pool CI upper "
                                  f"{'-' if not rp else '%+.1f' % (100 * rp[3])})")
    if not have_h3:
        lab, txt = lab + " (one level only)", txt + " (one level only)"
    return lab, txt, con


def claim(reading, d, lab, txt):
    st = fstats(d["cells"]["D27"].get(3, {}).values())
    ad = fstats((d.get("add3") or {}).values())
    word = "data" if lab.split()[0] in ("ROWS", "BOTH") else "training"
    x = f"{st[0]:.3f} [{st[2]:.3f}, {st[3]:.3f}]" if st else "missing"
    a = f"{ad[0]:.3f}" if ad else "missing"
    n = d.get("n_add3", 48)
    head = (f"3 hops are reachable in one pass of Qwen3.5-4B at this scale with enough chain {word}: at 27,000 chain rows "
            "(1 epoch, 1,029 updates, LR 3e-6) one pass ")
    if reading == "PASS":
        body = (f"composes 3 + / - steps and selects the value of a 3-step symbolic chain at m = {x} (on the {n} "
                f"all-+/- items the mean is {a})")
    else:
        body = f"selects the value of a 3-step chain at m = {x}; on the {n} all-+/- items the mean is {a}"
    return head + body + f"; the gain comes from {txt}"


def verdict(d):
    """The section 0 readings in order. d: {"I": {"I1": bool, ...}, "cells": {arm: {k: {seed: acc}}} on (ca, fwd) for
    D27, R3, H3, KD, KH, "add3": {seed: a_D27(ca, fwd, 3 | all +/-)}, "n_add3", "have_h3"}.
    Returns (reading, sub-label or None, lines, branch (b) label)."""
    c, have_h3 = d["cells"], d.get("have_h3", True)
    if sorted(d["I"]) != ["I1", "I2", "I3", "I4", "I5"] or not all(isinstance(v, bool) for v in d["I"].values()):
        unreadable(f"installation lines {d['I']}")
    for k in (1, 2, 3):                          # the readings need k* >= 3 or not, and the k = 2 cell
        need_seeds(c.get("D27", {}).get(k, {}), f"(ca, fwd, {k}) of D27")
    c = {a: {k: {s: q(v) for s, v in x.items()} for k, x in ks_.items()} for a, ks_ in c.items()}
    lab, txt, con = branch_b({a: c.get(a, {}).get(3, {}) for a in ("D27", "R3", "KD", "H3", "KH")}, have_h3)
    lines = []
    bad = [k for k, v in d["I"].items() if not v]
    if bad:
        return "MANIPULATION FAILED", None, [f"installation lines failing: {', '.join(sorted(bad))}; no dose reading"], lab
    st = {a: {k: fstats(v.values()) for k, v in c.get(a, {}).items()} for a in c}
    xv = lambda a, k: list(c.get(a, {}).get(k, {}).values())
    d27 = st.get("D27", {})
    ks = xkstar(c["D27"])
    ddx = list(pdiff(c["D27"].get(3, {}), c.get("KD", {}).get(3, {})).values())
    dd, DD = fstats(ddx), XS(ddx)
    lines.append(f"k*(D27, ca, fwd) = {ks}; (ca, fwd, 3) D27 {ci_s(d27.get(3), False)}; D_dose (D27 - KD at (ca, fwd, 3), "
                 f"27,000 rows at 1,029 updates against 9,000 rows at 467 updates) {ci_s(dd)}")
    if ks >= 3:
        need_seeds(d.get("add3") or {}, "the all-+/- subset (ca, fwd, 3 | n_add 3) of D27")
        ad = fstats(d["add3"].values())
        lines.append(f"all-+/- subset (ca, fwd, 3) under D27 (n items {d.get('n_add3', '-')}): {ci_s(ad, False)}; bar m >= "
                     f"{BAR['alladd']:.2f} for \"composes\"")
        r = "PASS" if XS(d["add3"].values()).mean_ge(XBAR["alladd"]) else "PASS (approximate)"
        return r, None, lines + [f"claim: {claim(r, d, lab, txt)}"], lab
    if xfailed(xv("D27", 3)) and DD.hi_lt(XBAR["neg"]):
        s = ("claim: at 1 epoch and LR 3e-6, 3,000 to 27,000 chain rows do not make one pass of Qwen3.5-4B select the "
             "value of a 3-step chain reliably, and the k = 3 cell is flat from 9,000 rows at 467 updates to 27,000 rows "
             "at 1,029 updates; capacity reading at this schedule")
        if xfailed(xv("R3", 3)):
            s += ", and 3 repeats of 9,000 rows do not help either"
        return "NEGATIVE", None, lines + [s], lab
    if DD.ok() and not DD.lo_gt(0) and not DD.hi_lt(0):
        return "NULL", None, lines + [
            "claim: no detectable dose effect from 9,000 rows at 467 updates to 27,000 rows at 1,029 updates, and the "
            f"data do not exclude a gain of {100 * dd[3]:.1f} points"], lab
    r27x = list(pdiff(c["D27"].get(3, {}), c.get("R3", {}).get(3, {})).values())
    r27 = fstats(r27x)
    if DD.lo_gt(0) and not xsolved(xv("D27", 3)):
        sub = ("still data-limited at 27,000 rows and 1,029 updates" if XS(r27x).lo_gt(0) else
               "still dose-limited at 27,000 rows and 1,029 updates (rows and updates not separated; see branch (b))")
    elif xsolved(xv("R3", 3)) and not xsolved(xv("D27", 3)):
        sub = "reachable with repetition, not with fresh rows"
    elif DD.hi_lt(0) and not xfailed(xv("D27", 3)):
        sub = "lower at the higher dose"
    elif xsolved(xv("D27", 3)) and not xsolved(xv("D27", 2)):
        sub = "k = 3 solved, k = 2 not"
    else:
        sub = "unresolved"
    return "BETWEEN", sub, lines + [f"R27 (D27 - R3 at (ca, fwd, 3)) {ci_s(r27)}"], lab


def gsm_sentence(j, st, n1_ok, xs=None):
    """The Amendment 1 (finding 2) three-way sentence for one (arm, rc<j>) from X - best-A2 stats (printing); with xs
    (the per-seed X - best-A2 values) the CI side is decided exactly."""
    if not st or st[4] < 2:
        return f"rc{j}: missing"
    st = tuple(float(v) for v in st[:4]) + (st[4],)
    x = f"{100 * st[0]:+.1f} [{100 * st[2]:+.1f}, {100 * st[3]:+.1f}]"
    X = XS(xs) if xs is not None else None
    if (X.lo_gt(0) if X else st[2] > 0):
        return (f"chain training raises GSM rc{j} by {100 * st[0]:.1f} points (X - best-A2 {x})" if n1_ok else
                f"no transfer sentence for rc{j} (plain cost; X - best-A2 {x})")
    if (X.hi_lt(0) if X else st[3] < 0):
        return f"chain training lowers GSM rc{j} by {-100 * st[0]:.1f} points (X - best-A2 {x})"
    return f"no detectable change of GSM rc{j} at this dose (CI {100 * st[2]:+.1f}, {100 * st[3]:+.1f})"


def ls_slope(ys, xs=(0, 1, 2)):
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def log_dose_slope(kh, kd, d27):
    """Per seed: the least-squares slope of a(ca, fwd, 3) on log3(distinct rows) over KH, KD, D27 (points 0, 1, 2)."""
    return {s: ls_slope([kh[s], kd[s], d27[s]]) for s in sorted(d27) if s in kh and s in kd}


def best_set(per_set, key):
    """The A2-6k set ("orig" / "retrain") with the higher seed mean on key (per_set: {set: {seed: {key: v}}}), on the
    exact values; every set must hold seeds 16-20 with the key (else UNREADABLE: a missing value never moves the
    choice; Codex fix round 5)."""
    if not per_set:
        unreadable(f"no reference set for {key}")
    means = {}
    for n, d in per_set.items():
        need_seeds({s: v.get(key) for s, v in d.items()}, f"best-A2 set {n} on {key}")
        means[n] = XS([v[key] for v in d.values()]).m
    return max(means, key=means.get)


def gate_h3(path):
    if not path:
        return True
    last = [l.strip() for l in open(path).read().splitlines() if l.strip()][-1:]
    if not last or not last[0].startswith("GATE PASS"):
        raise SystemExit(f"khop2_table: the gate output {path} does not end with GATE PASS ({last})")
    return "H3 dropped" not in last[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--hint-dir", default=os.path.join(E, "src/data/eval/hint"))
    ap.add_argument("--run", action="append", default=[], help="ARM:s<seed>=<tgz or dir>")
    ap.add_argument("--ref-orig", action="append", default=[], help="s<seed>=<A2-6k original tgz> (j520 ...)")
    ap.add_argument("--ref-retrain", action="append", default=[], help="s<seed>=<A2-6k re-train tgz> (j663 ...)")
    ap.add_argument("--gate-output", default="", help="the pilot gate output; 'GATE PASS (H3 dropped ...)' drops H3")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    have_h3 = gate_h3(a.gate_output)
    khop = T.load_khop(a.khop_dir)
    for s in T.SETS:                             # xacc reads the ids of these files: a short file shrinks every cell
        if len(khop.get(s, [])) != T.ROWS[s]:
            unreadable(f"{s} has {len(khop.get(s, []))} rows in --khop-dir {a.khop_dir}, CARD_khop_v1 1.2 has {T.ROWS[s]}")
    specs = T.parse_runs(a.run)
    if not have_h3:
        specs.pop("H3", None)
    need = ("D27", "R3", "KD", "KH") + (("H3",) if have_h3 else ())
    miss = [x for x in need if x not in specs]
    if miss:
        raise SystemExit(f"khop2_table: no runs for {miss} (a partial set is never read)")
    if not (a.hint_dir and os.path.isfile(os.path.join(a.hint_dir, "hint_plain.jsonl"))):
        unreadable(f"no hint files in --hint-dir {a.hint_dir!r}: G2 / G3 and the rc1-rc4 lines need them")
    for x in need:
        if sorted(specs[x]) != list(SEEDS5):
            unreadable(f"arm {x} has seeds {sorted(specs[x])}, the card reads exactly {list(SEEDS5)}")
    runs = {arm: {s: T.load_run(p) for s, p in d.items()} for arm, d in specs.items()}
    for x in need:
        for sd, r in runs[x].items():
            check_seed(r, sd, f"arm {x} s{sd}")
    out = {"curves": {}, "I": {}, "lines": [], "have_h3": have_h3}
    pr = lambda *x: (print(*x), out["lines"].append(" ".join(str(y) for y in x)))
    pr(f"H3 {'in the readout' if have_h3 else 'dropped by the pilot gate (P7b): branch (b) reads R27 and U9 only'}")
    for arm, d in runs.items():
        for s in T.SETS:
            if s == "khop_blind":
                continue
            for seed, run in sorted(d.items()):
                c = T.curve(run, khop, s)
                out["curves"].setdefault(arm, {}).setdefault(s, {})[seed] = c
                pr(f"{arm:6s} s{seed} {s:15s} " + " ".join(f"k{k} {T.pts(v)}" for k, v in c.items()))
    # exact cells of the decision arms: Fraction(correct, items), every item of the set read (else UNREADABLE)
    byk = {s: T.by_k(khop.get(s, [])) for s in T.SETS}
    dec_sets = {x: ([s for s in T.SETS if s != "khop_blind"] if x in NEW else ["khop_ca_fwd"]) for x in need}
    xc = {x: {s: {k: {sd: xacc(r, s, ids, f"{x} s{sd} {s} k{k}") for sd, r in sorted(runs[x].items())}
                  for k, ids in sorted(byk[s].items())} for s in dec_sets[x]} for x in need}
    xcell = lambda arm, s, k: xc[arm][s][k]
    st_cell = lambda arm, s, k: fstats(xcell(arm, s, k).values())     # printing, on the exact complete cells
    # five-seed curves, (ca, fwd), k = 1..8 (2.1)
    for arm in ("D27", "R3", "H3", "KD", "KH"):
        if arm in runs:
            pr(f"curve {arm:4s} (ca, fwd) " + " | ".join(f"k{k} {ci_s(st_cell(arm, 'khop_ca_fwd', k), False)}"
                                                       for k in range(1, 9)))
    # installation lines (2.5)
    I = out["I"]
    arms_new = [x for x in NEW if x in runs]
    blind_ids = [r["item_id"] for r in khop.get("khop_blind", [])]
    xqs = {x: [xqs_pair(r, khop, f"{x} s{sd} qs pair") for sd, r in sorted(runs[x].items())] for x in arms_new}
    xbl = {x: [xacc(r, "khop_blind", blind_ids, f"{x} s{sd} blind") for sd, r in sorted(runs[x].items())]
           for x in arms_new}
    I["I1"] = all(XS(xcell(x, f"khop_{t}_{o}", 1).values()).mean_ge(XBAR["I1"]) for x in arms_new for t, o in T.CELLS6)
    I["I2"] = all(XS(xqs[x]).mean_ge(XBAR["I2"]) for x in arms_new)
    I["I3"] = all(XS(xbl[x]).m <= XBAR["I3"] for x in arms_new)
    i4r = {x: [i4(r, x) for _, r in sorted(runs[x].items())] for x in arms_new}
    I["I4"] = all(all(v) for v in i4r.values())
    I["I5"] = all(XS(xcell(x, "khop_ca_fwd", 2).values()).mean_ge(XBAR["I5"]) for x in arms_new)
    pr("installation:", json.dumps(I))
    for x in arms_new:
        pr(f"installation {x}: I1 min k1 cell " + " ".join(f"{t}/{o} {ci_s(st_cell(x, f'khop_{t}_{o}', 1), False)}"
                                                         for t, o in T.CELLS6)
           + f" | I2 qs pair {ci_s(fstats(xqs[x]), False)}"
           + f" | I3 blind {ci_s(fstats(xbl[x]), False)}"
           + f" | I4 {i4r[x]}"
           + f" | I5 (ca, fwd, 2) {ci_s(st_cell(x, 'khop_ca_fwd', 2), False)}")
    # verdict (section 0)
    add = {}
    for arm in runs:
        for seed, run in runs[arm].items():
            if run["preds"].get("khop_ca_fwd"):
                add.setdefault(arm, {})[seed] = T.alladd_curve(run, khop, "khop_ca_fwd")
    ids3 = T.alladd_ids(khop, "khop_ca_fwd").get(3, [])
    add3 = {sd: xacc(r, "khop_ca_fwd", ids3, f"D27 s{sd} all-+/- k3") for sd, r in sorted(runs["D27"].items())}
    cells = {arm: {k: xcell(arm, "khop_ca_fwd", k) for k in range(1, 9)} for arm in need}
    v, sub, lines, lab = verdict({"I": I, "cells": cells, "add3": add3, "have_h3": have_h3,
                                  "n_add3": len(T.alladd_ids(khop, "khop_ca_fwd").get(3, []))})
    for l in lines:
        pr(l)
    out["verdict"], out["sublabel"] = v, sub
    pr(f"VERDICT: {v}" + (f" ({sub})" if sub else ""))
    # branch (b) (2.2), at k = 3 (decision line) and k = 4 (reported)
    for k in (3, 4):
        lab_k, txt_k, con = branch_b({x: cells.get(x, {}).get(k, {}) for x in ("D27", "R3", "KD", "H3", "KH")}, have_h3,
                                     f"k = {k}")
        for n in ("R27", "R9", "U9", "U3", "rows_pool", "upd_pool"):
            if n in con:
                pr(f"branch (b) k={k} {n:9s} {ci_s(fstats(con[n].values()))} per seed "
                   + json.dumps({s: round(100 * x, 1) for s, x in sorted(con[n].items())}))
        if k == 3:
            out["label"], out["contrasts"] = lab_k, {n: dict(x) for n, x in con.items()}
            pr(f"BRANCH (b): {lab_k}: {txt_k}" + ("" if v != "MANIPULATION FAILED" else " (not read: MANIPULATION FAILED)"))
        else:
            pr(f"branch (b) at k = 4 (reported): {lab_k}")
    # secondary (2.3)
    order = ("D27", "R3", "H3", "KD", "KH", "K1", "A2Z", "BASE0", "A2Z96")
    for arm in [x for x in order if x in runs] + sorted(x for x in runs if x not in order):    # every arm (2.3)
        a3 = T.stats([v_.get(3) for v_ in add.get(arm, {}).values()])
        w3 = T.stats([T.alladd_curve(r, khop, "khop_wp_fwd").get(3) for r in runs[arm].values()])
        pr(f"all-+/- {arm:4s} (ca, fwd, 3 | n_add 3, n {len(T.alladd_ids(khop, 'khop_ca_fwd').get(3, []))}) "
           f"{ci_s(a3, False)} | (wp, fwd, 3 | n_add 3, n {len(T.alladd_ids(khop, 'khop_wp_fwd').get(3, []))}) {ci_s(w3, False)}")
        for pb in T.RANK_KEYS:
            for k in (2, 3):
                ag = [T.probe_agree(r, khop, "khop_ca_fwd", pb, ks=(k,)) for r in runs[arm].values()]
                pr(f"probe {pb:8s} agreement {arm:4s} ca fwd k{k} {ci_s(T.stats([x[0] for x in ag if x]), False)}")
    if all(x in cells for x in ("KH", "KD", "D27")):
        sl = {s: float(x) for s, x in log_dose_slope(cells["KH"][3], cells["KD"][3], cells["D27"][3]).items()}
        out["log_dose_slope"] = sl
        pr(f"log-dose slope of a(ca, fwd, 3) on log3(distinct rows) (KH, KD, D27; reported, mixes rows and updates): "
           f"{ci_s(T.stats(list(sl.values())))} points per 3x rows")
    for arm in arms_new:
        ks_ = {f"{t}/{o}": xkstar({k: xcell(arm, f"khop_{t}_{o}", k) for k in range(1, 7)})
               for t in ("ca", "wp", "vb") for o in ("fwd", "rev")}
        ks_["cag/fwd"] = xkstar({k: xcell(arm, "khop_cag_fwd", k) for k in range(1, 7)})
        pr(f"k* {arm}: {json.dumps(ks_)}")
        for t in ("ca", "wp", "vb", "cag"):
            for o in (("fwd", "rev") if t != "cag" else ("fwd",)):
                pr(f"a({arm}, {t}, {o}, 2..4) " + " ".join(f"k{k} {ci_s(st_cell(arm, f'khop_{t}_{o}', k), False)}"
                                                         for k in (2, 3, 4)))
        for t in ("ca", "wp", "vb"):
            for k in (2, 3):
                fr = fstats(pdiff(xcell(arm, f"khop_{t}_fwd", k), xcell(arm, f"khop_{t}_rev", k)).values())
                pr(f"fwd - rev {arm} {t} k={k}: {ci_s(fr)}")
    # GSM side
    orig, ret = T.parse_refs(a.ref_orig), T.parse_refs(a.ref_retrain)
    refs = {"orig": {s: T.load_run(p) for s, p in orig.items()}, "retrain": {s: T.load_run(p) for s, p in ret.items()}}
    for n, d in refs.items():
        if sorted(d) != list(SEEDS5):
            unreadable(f"reference set {n} has seeds {sorted(d)}, the card reads exactly {list(SEEDS5)}")
        for sd, r in d.items():
            check_seed(r, sd, f"reference {n} s{sd}")
            j = REF_JOBS[n][sd]
            if not re.search(rf"(^|_){j}(_|\.tgz$)", os.path.basename(str(r.get("path")))):
                unreadable(f"reference {n} s{sd} is {r.get('path')}, the card names j{j}")
    have_hint = bool(a.hint_dir) and os.path.isdir(a.hint_dir)
    rc = {}
    if have_hint:
        for n, d in (("orig", orig), ("retrain", ret)):
            rc[n] = {s: xrc_cells(p, a.hint_dir, f"{n} s{s}") for s, p in d.items()}
        for arm in ("D27", "R3", "H3", "KD", "KH"):
            if arm in specs:
                rc[arm] = {s: xrc_cells(p, a.hint_dir, f"{arm} s{s}") for s, p in specs[arm].items()}
        pr("rc populations (primary items, pfx_table.py): " + ", ".join(f"rc{k} {n}" for k, n in RC_N.items())
           + " (rc4: the primary items with an rc4 cell; Amendment, Codex fix round)")
        for name, d in rc.items():
            for s, c in sorted(d.items()):
                pr(f"rc cells {name:8s} s{s} " + " ".join(f"rc{k} {T.pts(x)}" for k, x in c.items()))
    plain = {n: {s: {p: T.plain_acc(r, p) for p in T.PLAIN} for s, r in d.items()} for n, d in refs.items()}
    for arm in ("D27", "R3", "H3", "KD", "KH"):
        if arm in runs:
            plain[arm] = {s: {p: T.plain_acc(r, p) for p in T.PLAIN} for s, r in runs[arm].items()}
    # every plain metric of every arm and reference that a GSM line reads: a finite number at each of seeds 16-20, else
    # UNREADABLE (Codex fix round 5, finding 2: a missing value made N1 a four-seed mean against five and t 3.18)
    for n, d in plain.items():
        for sd, v in sorted(d.items()):
            for p in T.PLAIN:
                if not isinstance(v[p], float) or not math.isfinite(v[p]):
                    unreadable(f"{n} s{sd} {p}: plain accuracy {v[p]!r} in {(runs.get(n) or refs.get(n))[sd].get('path')}")
                q(v[p])
    get = lambda src, key: {s: v[key] for s, v in src.items()}
    # G2 / G3 for D27 and R3 (reference candidates: both A2-6k sets and the arm itself)
    if have_hint:
        for arm in ("D27", "R3"):
            for k in (2, 3):
                cand = {n: rc[n] for n in ("orig", "retrain", arm) if rc.get(n)}
                b = best_set(cand, k)
                if b:
                    g = pdiff(xcell(arm, "khop_ca_fwd", k), get(rc[b], k))
                    pr(f"G{k} {arm} = a({arm}, ca, fwd, {k}) - rc{k}_ref (ref {b}): {ci_s(T.stats(list(g.values())))}")
    # N1 (2.6)
    bestp = {p: best_set({n: plain[n] for n in ("orig", "retrain")}, p) for p in T.PLAIN}
    n1 = {}
    for arm in arms_new:
        n1[arm] = {}
        for p in T.PLAIN:
            xm, rm = XS(get(plain[arm], p).values()).m, XS(get(plain[bestp[p]], p).values()).m
            n1[arm][p] = (float(xm), float(rm), xm >= rm - XBAR["N1"])
            dref = T.stats(list(pdiff(get(plain[arm], p), get(plain[bestp[p]], p)).values()))
            dkd = T.stats(list(pdiff(get(plain[arm], p), get(plain["KD"], p)).values()))
            pr(f"plain {arm} {p:8s} mean {T.pts(n1[arm][p][0])} | - best-A2 ({bestp[p]}) {ci_s(dref)} | - KD {ci_s(dkd)}")
        ok_all = all(x[2] for x in n1[arm].values())
        pr(f"N1 {arm}: {'holds' if ok_all else 'plain cost'} " + json.dumps(n1[arm]))
    out["N1"] = n1
    # plain cost of the dose: D27 - KD per plain set and pooled
    per = {p: pdiff(get(plain["D27"], p), get(plain["KD"], p)) for p in T.PLAIN}
    pooled = {s: sum(per[p][s] for p in T.PLAIN) / len(T.PLAIN) for s in seeds_of(*per.values())
              if all(s in per[p] for p in T.PLAIN)}
    ps = T.stats(list(pooled.values()))
    for p in T.PLAIN:
        pr(f"plain cost of the dose D27 - KD {p:8s} {ci_s(T.stats(list(per[p].values())))}")
    PX = XS([sum((q(plain["D27"][s][p]) - q(plain["KD"][s][p]) for p in T.PLAIN), Fraction(0)) / len(T.PLAIN)
             for s in SEEDS5])
    ans = ("yes" if PX.hi_lt(0) else "no detectable cost" if not PX.lo_gt(0) else
           "no cost (pooled D27 - KD CI lower > 0)")
    pr(f"plain cost of the dose, pooled over the four plain sets: {ci_s(ps)}; does more chain data hurt plain? {ans}")
    # transfer / harm sentences (2.6), rc1..rc4
    if have_hint:
        out["gsm"] = {}
        for j in (1, 2, 3, 4):
            b = best_set({n: rc[n] for n in ("orig", "retrain")}, j)
            for arm in ("KD", "KH") + tuple(arms_new):
                dx = T.stats(list(pdiff(get(rc[arm], j), get(rc[b], j)).values()))
                if arm in ("KD", "KH"):
                    pr(f"rc{j} {arm} - best-A2 ({b}) {ci_s(dx)} (parent, reported)")
                    continue
                dk = T.stats(list(pdiff(get(rc[arm], j), get(rc["KD"], j)).values()))
                xs = list(pdiff({s: q(v) for s, v in get(rc[arm], j).items()},
                                {s: q(v) for s, v in get(rc[b], j).items()}).values())
                s_ = gsm_sentence(j, dx, all(x[2] for x in n1[arm].values()), xs)
                out["gsm"].setdefault(arm, {})[j] = s_
                pr(f"rc{j} {arm} - best-A2 ({b}) {ci_s(dx)} | {arm} - KD {ci_s(dk)} | {arm}: {s_}")
    # timing and memory per job
    for arm in arms_new:
        for s, r in sorted(runs[arm].items()):
            h = (r["metrics"] or {}).get("history") or []
            sps = [x.get("s_per_step") for x in h if x.get("s_per_step")]
            w = G.wall_hours(r["log"])
            pr(f"timing {arm} s{s}: wall {'-' if w is None else '%.2f h' % w}, s_per_step "
               f"{'-' if not sps else '%.2f' % (sum(sps) / len(sps))}, peak_gib {max([x.get('peak_gib') or 0 for x in h] + [0]):.1f}")
    pr("note: the owner's 2-epoch branch runs as R3 / H3 (repetition in the file, matched updates); the literal EPOCHS=2 "
       "arms need a new card and a new id range")
    pr("note (card 1.3, cross-bundle pairing): D27 - KD, R3 - KD, KD - H3 and H3 - KH pair the new jobs (bundle x1khop2) "
       "with the parent's tarballs (bundle x1khop); same seeds, env keys and trainer (D5), but host and GPU type differ "
       "per job in both rounds, as they did inside CARD_khop_v1")
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
