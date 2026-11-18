#!/usr/bin/env python3
"""PFX card readout (CARD_pfx_v1.md): one-pass capacity by REMAINING SERIAL DEPTH within item, for a reader trained
on worked-step prefixes (PFX) and the answer-only student (A2). Standard library, Python 3.9.

    python3 -I pfx_table.py --hint-dir data/eval/hint --run PFX-c1:s16=results_540_...tgz ... --run A2-c1:s16=...
        [--pair PFX-c1,A2-c1]

Cells: for every item with whole-chain serial depth cp >= 4 (plain row field "path"), the rc = 1, 2, 3 cells are the
true rows whose remaining serial depth r_path equals rc, taking the row with the LARGEST k (fewest remaining
operations). Primary items: cp >= 4 with CLEAN rc = 1 and rc = 3 cells (no shown value equals an option).
Per run: D = acc(rc = 1 cell) - acc(rc = 3 cell) on the primary items; D_placebo on the placebo rows at the same r;
M3' = true - far-wrong and M4' = true - other at the rc = 3 cell (same r); W = within item at rc = 3, the row with the
fewest operations minus the row with the most (clean pairs); H = acc(rc = 1) - acc(plain); the curve by rc; V - O
(true minus plan at the same r) at both cells; D4 = acc(rc = 1) - acc(rc = 4) on the items with cp >= 5 and clean rc = 1
and rc = 4 cells (the theory's boundary), with the rc 3 -> 4 drop on the same items; the OPC gates (M1, M2 at r 1..5,
M3, M3b, M4 at r = 1); the read check (n = 2 items, informational: the A2 reference reader sits at +4.5); the PFX
training-time plain share from metrics.json "si" (0.45..0.55).
Per arm: seed mean D, every seed, paired t (df n-1), PASS / NONE / BETWEEN; c = the largest rc within 5 points of
rc = 1 on the seed-mean curve; D4 seed mean and seeds. Per arm family X-c1 / X-c2: the dose contrast D(c1) - D(c2)
(>= 5: the slope still moves with training). Per --pair a,b: S = (D_b - D_a) / D_b per seed, plain parity.
Missing sets print WAITING.
"""
import argparse, io, json, math, os, re, tarfile

TRUE = [f"hint_true_r{r}" for r in (1, 2, 3, 4, 5, 6)]
SETS = ["hint_plain", "hint_full"] + TRUE + [f"hint_{k}_r{r}" for k in ("placebo", "other", "wrongres", "wrongres_near", "plan") for r in (1, 2, 3, 4, 5)]
BAR = {"D_pass": 0.15, "D_seed": 0.05, "D_none": 0.05, "M1": 0.90, "M2_pool": 0.03, "M2_r": 0.05, "M2_spread": 0.05,
       "M3": 0.10, "M3b": 0.10, "M4": 0.10, "M3p": 0.05, "M3bp": 0.05, "M4p": 0.05, "Dpl": 0.05, "READ": 0.05, "PARITY": 0.03,
       "C_WITHIN": 0.05, "W_SERIAL": 0.05, "W_OPS": 0.10, "MIX_LO": 0.45, "MIX_HI": 0.55}
TCRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57}
EPS = 1e-9


def read_jsonl(f):
    return [json.loads(l) for l in (f if not isinstance(f, bytes) else io.BytesIO(f)) if l.strip()]


def load_run(path):
    """{set: {item_id: (correct, predicted label)}} plus "_si": the metrics.json si log (list) if present."""
    out = {}
    def take(name, rows):
        out[name] = {}
        for d in rows:
            k = max(range(len(d["probs"])), key=d["probs"].__getitem__)
            out[name][d["item_id"]] = (int(d["labels"][k] == d["gold_label"]), d["labels"][k])
    if os.path.isdir(path):
        for s in SETS:
            p = os.path.join(path, f"preds_{s}.jsonl")
            if os.path.exists(p):
                take(s, read_jsonl(open(p)))
        mp = os.path.join(path, "metrics.json")
        if os.path.exists(mp):
            out["_si"] = json.load(open(mp)).get("si")
        return out
    with tarfile.open(path) as t:
        for m in t.getmembers():
            mm = re.search(r"/preds_(hint_[a-z0-9_]+)\.jsonl$", m.name)
            if mm and mm.group(1) in SETS:
                take(mm.group(1), read_jsonl(t.extractfile(m).read()))
            elif m.name.endswith("/metrics.json"):
                out["_si"] = json.load(t.extractfile(m)).get("si")
    return out


def load_hints(hint_dir):
    h = {}
    for s in SETS:
        p = os.path.join(hint_dir, s + ".jsonl")
        if os.path.exists(p):
            h[s] = {d["item_id"]: d for d in read_jsonl(open(p))}
    return h


def cells(hints):
    """{item_id: {"cp": path, rc: (set name, row)}} for items with cp >= 4; the cell row = largest k at that r_path."""
    out = {}
    for iid, d in hints["hint_plain"].items():
        if d["path"] >= 4:
            out[iid] = {"cp": d["path"], "n": d["n"]}
    for s in TRUE:
        for iid, d in hints.get(s, {}).items():
            if iid in out and d["r_path"] in (1, 2, 3, 4):
                cur = out[iid].get(d["r_path"])
                if cur is None or d["k"] > cur[1]["k"]:
                    out[iid][d["r_path"]] = (s, d)
    return out


def primary_items(cl):
    return sorted(i for i, c in cl.items() if 1 in c and 3 in c and not c[1][1]["shown_equals_option"] and not c[3][1]["shown_equals_option"])


def d4_items(cl):
    """The theory's boundary population: cp >= 5 with clean rc = 1 and rc = 4 cells."""
    return sorted(i for i, c in cl.items() if c["cp"] >= 5 and 1 in c and 4 in c and not c[1][1]["shown_equals_option"] and not c[4][1]["shown_equals_option"])


def acc_rows(run, rows):
    """rows: list of (set, item_id). Mean correctness over the rows present in the run, or (None, 0)."""
    xs = [run[s][i][0] for s, i in rows if s in run and i in run[s]]
    return (sum(xs) / len(xs) if xs else None), len(xs)


def pts(x):
    return "  -  " if x is None else "%5.1f" % (100 * x)


def dpts(x):
    return "   -  " if x is None else "%+6.1f" % (100 * x)


def sub(a, b):
    return None if a is None or b is None else a - b


def t_stat(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, (m / (sd / math.sqrt(len(xs))) if sd else float("inf"))


def per_run(run, hints, cl, prim, label, d4=()):
    o = {"label": label}
    print(f"=== {label} ===")
    # curve by rc on the primary items (cells), plain, H
    curve = {}
    for rc in (1, 2, 3, 4):
        rows = [(cl[i][rc][0], i) for i in prim if rc in cl[i]]
        curve[rc] = acc_rows(run, rows)
    plain = acc_rows(run, [("hint_plain", i) for i in prim])
    o["D"] = sub(curve[1][0], curve[3][0])
    o["H"] = sub(curve[1][0], plain[0])
    o["curve"] = {rc: curve[rc][0] for rc in curve}
    o["plain_all"] = acc_rows(run, [("hint_plain", i) for i in hints["hint_plain"]])[0]
    print("primary items %d (cp >= 4, clean rc1 and rc3 cells): plain %s | rc=1 %s (%d) rc=2 %s (%d) rc=3 %s (%d) rc=4 %s (%d) | D = rc1 - rc3 %s | H = rc1 - plain %s" % (
        len(prim), pts(plain[0]), pts(curve[1][0]), curve[1][1], pts(curve[2][0]), curve[2][1], pts(curve[3][0]), curve[3][1], pts(curve[4][0]), curve[4][1], dpts(o["D"]), dpts(o["H"])))
    # same-r controls at the two cells: placebo (D_placebo), far-wrong and other at rc = 3 (M3', M4'), plan (V - O)
    def ctrl(kind, rc):
        rows, trues = [], []
        for i in prim:
            s, d = cl[i][rc]
            cs = f"hint_{kind}_r{d['r']}"
            if cs in run and i in run[cs]:
                rows.append((cs, i)); trues.append((s, i))
        return acc_rows(run, rows), acc_rows(run, trues)
    pl1, tr1 = ctrl("placebo", 1); pl3, tr3 = ctrl("placebo", 3)
    o["D_placebo"] = None if None in (pl1[0], pl3[0]) else pl1[0] - pl3[0]
    fw3, tf3 = ctrl("wrongres", 3); nr3, tn3 = ctrl("wrongres_near", 3); ot3, to3 = ctrl("other", 3); pn1, tp1 = ctrl("plan", 1); pn3, tp3 = ctrl("plan", 3)
    o["M3p"] = sub(tf3[0], fw3[0]); o["M3bp"] = sub(tn3[0], nr3[0]); o["M4p"] = sub(to3[0], ot3[0])
    print("D_placebo (placebo rows at the cells' r) %s [%d / %d rows] | at rc=3: true - far-wrong (M3') %s (%d), true - near-miss (M3b') %s (%d), true - other (M4') %s (%d) | V - O: rc=1 %s, rc=3 %s; O - plain rc=1 %s" % (
        dpts(o["D_placebo"]), pl1[1], pl3[1], dpts(o["M3p"]), fw3[1], dpts(o["M3bp"]), nr3[1], dpts(o["M4p"]), ot3[1], dpts(sub(tp1[0], pn1[0])), dpts(sub(tp3[0], pn3[0])), dpts(sub(pn1[0], plain[0]))))
    # W: within item at rc = 3, two true rows with different r (fewest ops vs most), clean pairs
    w = []
    for i in cl:
        rows = [(s, d) for s in TRUE for iid, d in hints.get(s, {}).items() if iid == i and d["r_path"] == 3 and not d["shown_equals_option"]]
        if len(rows) >= 2:
            lo, hi = min(rows, key=lambda t: t[1]["r"]), max(rows, key=lambda t: t[1]["r"])
            if lo[0] in run and hi[0] in run and i in run[lo[0]] and i in run[hi[0]]:
                w.append(run[lo[0]][i][0] - run[hi[0]][i][0])
    o["W"] = (sum(w) / len(w)) if w else None
    print("W (within item at rc=3, fewest ops - most ops, %d clean pairs) %s" % (len(w), dpts(o["W"])))
    # D4 on the boundary population (cp >= 5, clean rc1 and rc4 cells); the rc 3 -> 4 drop on the same items
    a1 = acc_rows(run, [(cl[i][1][0], i) for i in d4]); a4 = acc_rows(run, [(cl[i][4][0], i) for i in d4])
    a3 = acc_rows(run, [(cl[i][3][0], i) for i in d4 if 3 in cl[i]]); a4b = acc_rows(run, [(cl[i][4][0], i) for i in d4 if 3 in cl[i]])
    o["D4"] = sub(a1[0], a4[0]); o["d34"] = sub(a3[0], a4b[0]); o["D69"] = sub(a1[0], a3[0])
    p1 = acc_rows(run, [(f"hint_placebo_r{cl[i][1][1]['r']}", i) for i in d4]); p4 = acc_rows(run, [(f"hint_placebo_r{cl[i][4][1]['r']}", i) for i in d4])
    o["D4_placebo"] = None if None in (p1[0], p4[0]) else p1[0] - p4[0]
    print("D4 (boundary, %d items cp >= 5 with clean rc1 and rc4 cells): rc=1 %s rc=3 %s rc=4 %s | D on these %s | D4 %s | rc 3 -> 4 drop %s (%d items) | D4_placebo %s [%d / %d rows]" % (
        len(d4), pts(a1[0]), pts(a3[0]), pts(a4[0]), dpts(o["D69"]), dpts(o["D4"]), dpts(o["d34"]), a3[1], dpts(o["D4_placebo"]), p1[1], p4[1]))
    # OPC gates
    ids1 = set(hints["hint_true_r1"])
    def accs(s, ids):
        return acc_rows(run, [(s, i) for i in ids])[0]
    full = accs("hint_full", hints["hint_full"])
    o["M1"] = None if full is None else full >= BAR["M1"]
    m2 = []
    pool_d = pool_n = 0.0
    for r in (1, 2, 3, 4, 5):
        s = f"hint_placebo_r{r}"
        ids = set(hints.get(s, {}))
        pb, pl = accs(s, ids), accs("hint_plain", ids)
        m2.append(sub(pb, pl))
        if pb is not None and pl is not None:
            pool_d += (pb - pl) * len(ids); pool_n += len(ids)
    o["M2_pool"] = pool_d / pool_n if pool_n else None
    o["M2"] = None if None in m2 or o["M2_pool"] is None else (abs(o["M2_pool"]) <= BAR["M2_pool"] and all(abs(x) <= BAR["M2_r"] for x in m2) and max(m2) - min(m2) <= BAR["M2_spread"])
    shared = lambda s: ids1 & set(hints.get(s, {}))
    o["M3"] = sub(accs("hint_true_r1", shared("hint_wrongres_r1")), accs("hint_wrongres_r1", shared("hint_wrongres_r1")))
    o["M3b"] = sub(accs("hint_true_r1", shared("hint_wrongres_near_r1")), accs("hint_wrongres_near_r1", shared("hint_wrongres_near_r1")))
    o["M4"] = sub(accs("hint_true_r1", shared("hint_other_r1")), accs("hint_other_r1", shared("hint_other_r1")))
    n2 = {i for i, d in hints["hint_plain"].items() if d["n"] == 2}
    o["read"] = sub(accs("hint_true_r1", n2), accs("hint_plain", n2))
    print("gates: M1 full %s | M2 placebo-plain pooled %s, r1..r5 %s (spread %s) | M3 %s M3b %s M4 %s at r=1 | read (n=2) %s" % (
        pts(full), dpts(o["M2_pool"]), " ".join(dpts(x) for x in m2), dpts(None if None in m2 else max(m2) - min(m2)), dpts(o["M3"]), dpts(o["M3b"]), dpts(o["M4"]), dpts(o["read"])))
    # PFX training-time plain share
    si = run.get("_si")
    if not si and label.startswith("PFX"):
        print("si log: MISSING on a PFX run -> the readout refuses the run (mix gate FAIL)")
    if si:
        shares = [p.get("si_mix_plain", 0) / p["si_items"] for p in si if p.get("mix") and p.get("si_items")]
        o["mix_share"] = shares
        print("si log: mix passes %d, plain share per pass %s -> %s" % (len(shares), " ".join("%.3f" % x for x in shares),
              "ok" if shares and all(BAR["MIX_LO"] <= x <= BAR["MIX_HI"] for x in shares) else "OUT OF 0.45..0.55 (readout refuses the run)" if shares else "no mix passes"))
    return o


def gate_ok(o):
    checks = {"M1": o["M1"], "M2": o["M2"], "M3": None if o["M3"] is None else o["M3"] >= BAR["M3"], "M3b": None if o["M3b"] is None else o["M3b"] >= BAR["M3b"],
              "M4": None if o["M4"] is None else o["M4"] >= BAR["M4"], "M3p": None if o["M3p"] is None else o["M3p"] >= BAR["M3p"],
              "M3bp": None if o["M3bp"] is None else o["M3bp"] >= BAR["M3bp"],
              "M4p": None if o["M4p"] is None else o["M4p"] >= BAR["M4p"], "Dpl": None if o["D_placebo"] is None else abs(o["D_placebo"]) <= BAR["Dpl"]}
    # the read check (n = 2 items) is informational: the A2 reference reader sits at +4.5 with H +35
    if o["label"].startswith("PFX"):   # the mixture gate belongs to the PFX arms only; a missing si log fails it (the :1072 / :1104 failure)
        share = o.get("mix_share") or []
        checks["mix"] = bool(share) and all(BAR["MIX_LO"] <= x <= BAR["MIX_HI"] for x in share)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-dir", required=True)
    ap.add_argument("--run", action="append", default=[], help="<arm>:<seed>=<results tgz or dir>")
    ap.add_argument("--pair", action="append", default=[], help="<arm a>,<arm b>: S = (D_b - D_a) / D_b per seed and plain parity")
    a = ap.parse_args()
    hints = load_hints(a.hint_dir)
    cl = cells(hints)
    prim = primary_items(cl)
    d4 = d4_items(cl)
    print("population cp >= 4: %d items; primary (clean rc1 and rc3 cells): %d; boundary D4 population (cp >= 5, clean rc1 and rc4): %d" % (len(cl), len(prim), len(d4)))
    runs, arms = {}, {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        o = per_run(load_run(path), hints, cl, prim, label, d4)
        runs[label] = o
        arm, seed = label.split(":")
        arms.setdefault(arm, {})[seed] = o
    print("=== arms: PRIMARY D = acc(rc=1) - acc(rc=3) on the primary items; PASS mean >= 15, every seed >= 5, t >= t_crit(df n-1), gates; NONE mean < 5 ===")
    for arm, by in arms.items():
        seeds = sorted(by)
        D = [by[s]["D"] for s in seeds if by[s]["D"] is not None]
        if len(D) != len(seeds) or not D:
            print("%-8s WAITING (D missing on some seed)" % arm); continue
        m, t = t_stat(D)
        crit = TCRIT.get(len(D), 2.0)
        gates = {k: all(gate_ok(by[s]).get(k) for s in seeds) if all(gate_ok(by[s]).get(k) is not None for s in seeds) else None for k in gate_ok(by[seeds[0]])}
        g_ok = all(v for v in gates.values() if v is not None) and None not in gates.values()
        verdict = "PASS" if (m >= BAR["D_pass"] - EPS and min(D) >= BAR["D_seed"] and t >= crit and g_ok) else "NONE" if m < BAR["D_none"] else "BETWEEN"
        if m >= BAR["D_pass"] - EPS and min(D) >= BAR["D_seed"] and t >= crit and not g_ok:
            verdict = "D passes but a gate fails: " + ",".join(k for k, v in gates.items() if not v)
        cur = {rc: [by[s]["curve"].get(rc) for s in seeds] for rc in (1, 2, 3, 4)}
        cm = {rc: (sum(v) / len(v) if v and None not in v else None) for rc, v in cur.items()}
        c = max((rc for rc in (1, 2, 3, 4) if cm[rc] is not None and cm[1] is not None and cm[1] - cm[rc] <= BAR["C_WITHIN"]), default=None)
        W = [by[s]["W"] for s in seeds if by[s]["W"] is not None]
        wm = sum(W) / len(W) if W else None
        wl = "-" if wm is None else "serial depth" if abs(wm) <= BAR["W_SERIAL"] else "operation count" if wm >= BAR["W_OPS"] else "between"
        H = [by[s]["H"] for s in seeds if by[s]["H"] is not None]
        D4 = [by[s]["D4"] for s in seeds if by[s]["D4"] is not None]
        d4m = sum(D4) / len(D4) if D4 else None
        d4l = "-" if d4m is None else "boundary at 4 (>= 15)" if d4m >= BAR["D_pass"] - EPS else "no boundary up to 4 (< 5)" if d4m < BAR["D_none"] else "undetermined (5..15)"
        d4p = [by[s]["D4_placebo"] for s in seeds if by[s]["D4_placebo"] is not None]
        if d4p and abs(sum(d4p) / len(d4p)) > BAR["Dpl"]:
            d4l += " BUT |D4_placebo| %.1f > 5: length, not depth" % (100 * abs(sum(d4p) / len(d4p)))
        rd = [by[s]["read"] for s in seeds if by[s]["read"] is not None]
        print("%-8s seeds %s: D %s [%s] t %.2f (crit %.2f, n %d) -> %s | curve rc1..4 %s | c = %s | H %s | W %s (%s) | D4 %s [%s] %s | read (info) %s | gates %s" % (
            arm, ",".join(seeds), dpts(m), " ".join(dpts(x).strip() for x in D), t, crit, len(D), verdict,
            " ".join(pts(cm[rc]).strip() for rc in (1, 2, 3, 4)), c, dpts(sum(H) / len(H)) if H else "-", dpts(wm), wl,
            dpts(d4m), " ".join(dpts(x).strip() for x in D4), d4l, dpts(sum(rd) / len(rd)) if rd else "-",
            " ".join("%s:%s" % (k, {True: "ok", False: "FAIL", None: "?"}[v]) for k, v in gates.items())))
        arm_D = locals().setdefault("arm_D", {}); arm_D[arm] = m
    # dose contrast per arm family: X-c1 vs X-c2 on the seed means
    fam = {}
    for arm, by in arms.items():
        if arm.endswith("-c1") or arm.endswith("-c2"):
            D = [by[s]["D"] for s in sorted(by) if by[s]["D"] is not None]
            if D and len(D) == len(by):
                fam.setdefault(arm[:-3], {})[arm[-2:]] = sum(D) / len(D)
    for x, d in fam.items():
        if "c1" in d and "c2" in d:
            dd = d["c1"] - d["c2"]
            print("dose %s: D(c1) - D(c2) %s -> %s" % (x, dpts(dd), "slope still moves with training (>= 5): no architectural wording" if dd >= BAR["D_none"] - EPS else "dose-stable (< 5)"))
    for pair in a.pair:
        x, y = pair.split(",")
        seeds = sorted(set(arms.get(x, {})) & set(arms.get(y, {})))
        S, par = [], []
        for s in seeds:
            dx, dy = arms[x][s]["D"], arms[y][s]["D"]
            if dx is not None and dy:
                S.append((dy - dx) / dy)
            if arms[x][s]["plain_all"] is not None and arms[y][s]["plain_all"] is not None:
                par.append(arms[x][s]["plain_all"] - arms[y][s]["plain_all"])
        if not seeds:
            print("pair %s vs %s: WAITING" % (x, y)); continue
        pm = sum(par) / len(par) if par else None
        print("pair %s vs %s (seeds %s): S = (D_%s - D_%s)/D_%s per seed %s mean %s | plain parity %s - %s %s -> %s" % (
            x, y, ",".join(seeds), y, x, y, " ".join("%.2f" % v for v in S), ("%.2f" % (sum(S) / len(S))) if S else "-", x, y, dpts(pm),
            "-" if pm is None else ("within 3" if abs(pm) <= BAR["PARITY"] else "OUTSIDE 3: read H against the other arm's plain too")))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main()
