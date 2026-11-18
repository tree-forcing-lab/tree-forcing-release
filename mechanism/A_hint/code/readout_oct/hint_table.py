#!/usr/bin/env python3
"""OPC readout (card CARD_opc_sev6e_v1.md sections 5 and 7) from eval-only results. Standard library.

    python3 -I hint_table.py --hint-dir data/eval/hint --run A2-c4:s12=results_340_4b_hint_a2c4s12.tgz \
        --run A2-c4:s13=... [--pair GFPQM-c4,GFPQMR-c4] [--pair GFPQ-c4,A2-c4]

A run is a results tgz (runs/<name>/preds_<set>.jsonl inside) or a directory with the preds files. Run labels are
<arm>:<seed>. Per run: accuracy per set, the capacity curve acc(r), M1-M3, H and D. Across seeds of one arm: the seed
mean of H and D (P1, P2). For each --pair a,b: the seed-paired difference on the r in {1, 2} true variants (P3) with
the df = n-1 t. The sensitivity row drops variants whose shown value equals an option. Missing sets print WAITING.
"""
import argparse, io, json, math, os, re, sys, tarfile

SETS = ["hint_plain", "hint_full"] + [f"hint_true_r{r}" for r in (1, 2, 3, 4, 5, 6)] + \
       [f"hint_{k}_r{r}" for r in (1, 2) for k in ("other", "wrongres", "wrongres_near")] + [f"hint_placebo_r{r}" for r in (1, 2, 3, 4)]
BAR = {"M1_full": 0.90, "M2_ctrl": 0.03, "M2_ctrl_r": 0.05, "M3_wrong": 0.10, "M3b_near": 0.10, "M4_other": 0.10, "READ": 0.05, "READ_LOW": 0.02, "H_big": 0.15, "H_small": 0.05, "H_seed": 0.10,
       "D_big": 0.15, "D_small": 0.05, "P3_pass": 0.015, "P3_fail": 0.005}
TCRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}


def read_jsonl(f):
    return [json.loads(l) for l in (f if not isinstance(f, bytes) else io.BytesIO(f)) if l.strip()]


def load_run(path):
    """{set: {item_id: correct}} from a results tgz or a run directory."""
    out = {}
    if os.path.isdir(path):
        for s in SETS:
            p = os.path.join(path, f"preds_{s}.jsonl")
            if os.path.exists(p):
                out[s] = {}
                for d in read_jsonl(open(p)):
                    k = max(range(len(d["probs"])), key=d["probs"].__getitem__)
                    out[s][d["item_id"]] = int(d["labels"][k] == d["gold_label"])
                    out.setdefault("_pred", {}).setdefault(s, {})[d["item_id"]] = d["labels"][k]
        return out
    with tarfile.open(path) as t:
        for m in t.getmembers():
            mm = re.search(r"/preds_(hint_[a-z0-9_]+)\.jsonl$", m.name)
            if mm and mm.group(1) in SETS:
                out[mm.group(1)] = {}
                for d in read_jsonl(t.extractfile(m).read()):
                    k = max(range(len(d["probs"])), key=d["probs"].__getitem__)
                    out[mm.group(1)][d["item_id"]] = int(d["labels"][k] == d["gold_label"])
                    out.setdefault("_pred", {}).setdefault(mm.group(1), {})[d["item_id"]] = d["labels"][k]
    return out


def option_text_by_label(prompt):
    tail = prompt.rpartition("\nOptions:\n")[2]
    return {ln.split(") ", 1)[0].strip(): ln.split(") ", 1)[1].strip() for ln in tail.split("\n") if ") " in ln}


def copy_rate(run, hints, s):
    """Share of predictions that pick the option equal to the LAST shown value, on the rows where that value is an
    option text (the copy misreading's own measure). (rate, n rows) or (None, 0)."""
    preds = run.get("_pred", {}).get(s)
    if not preds:
        return None, 0
    hit = n = 0
    for iid, d in hints[s].items():
        last = d["shown"][-1].rpartition("=")[2].strip() if d["shown"] else None
        ot = option_text_by_label(d["prompt"])
        if last is None or last not in ot.values() or iid not in preds:
            continue
        n += 1
        hit += int(ot.get(preds[iid]) == last)
    return (hit / n if n else None), n


def load_hints(hint_dir):
    """{set: {item_id: row}} with n, r, kind, shown_equals_option."""
    return {s: {d["item_id"]: d for d in read_jsonl(open(os.path.join(hint_dir, s + ".jsonl")))} for s in SETS}


def acc(run, hints, s, ids=None, drop_eq=False):
    if s not in run:
        return None, 0
    rows = [(i, c) for i, c in run[s].items() if (ids is None or i in ids) and (not drop_eq or not hints[s][i]["shown_equals_option"])]
    return (sum(c for _, c in rows) / len(rows) if rows else None), len(rows)


def ids_with(hints, s, pred):
    return {i for i, d in hints[s].items() if pred(d["n"])}


def pts(x):
    return "  -  " if x is None else "%5.1f" % (100 * x)


def dpts(x):
    return "   -  " if x is None else "%+6.1f" % (100 * x)


def t_stat(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, (m / (sd / math.sqrt(len(xs))) if sd else float("inf"))


def per_run(run, hints, label):
    out = {"label": label}
    print(f"=== {label} ===")
    print("set            acc   n    | drop shown=option: acc   n")
    for s in SETS:
        a, n = acc(run, hints, s)
        b, m = acc(run, hints, s, drop_eq=True)
        print(f"{s:14s} {pts(a)} {n:5d} | {pts(b)} {m:5d}" if a is not None else f"{s:14s} WAITING")
    # capacity curve on items with n >= 5 (all r = 1..4 exist for them)
    big = ids_with(hints, "hint_plain", lambda n: n >= 5)
    curve = {r: acc(run, hints, f"hint_true_r{r}", big)[0] for r in (1, 2, 3, 4)}
    plain5 = acc(run, hints, "hint_plain", big)[0]
    out["D"] = None if curve[1] is None or curve[4] is None else curve[1] - curve[4]
    print("P1 capacity (items n >= 5, %d): plain %s | r=1 %s r=2 %s r=3 %s r=4 %s | D = acc(r1) - acc(r4) %s" % (
        len(big), pts(plain5), pts(curve[1]), pts(curve[2]), pts(curve[3]), pts(curve[4]), dpts(out["D"])))
    # shift check: acc(r | n >= r + 3) vs plain acc of items with n = r
    for r in (1, 2, 3):
        a = acc(run, hints, f"hint_true_r{r}", ids_with(hints, "hint_plain", lambda n, r=r: n >= r + 3))[0]
        b = acc(run, hints, "hint_plain", ids_with(hints, "hint_plain", lambda n, r=r: n == r))[0]
        print("   shift r=%d: acc(r=%d | n>=%d) %s vs plain acc(n=%d) %s" % (r, r, r + 3, pts(a), r, pts(b)))
    # capacity by REMAINING CRITICAL PATH (links a_from / b_from): true rows of items n >= 5 pooled over r, sliced by the
    # serial work left (r_path), against plain items whose full path equals that length (the shift check by path)
    if "r_path" in next(iter(hints["hint_true_r1"].values())):
        byp, plainp = {}, {}
        for r in (1, 2, 3, 4, 5, 6):
            sname = f"hint_true_r{r}"
            for iid, c in run.get(sname, {}).items():
                d = hints[sname][iid]
                if d["n"] >= 5:
                    a = byp.setdefault(d["r_path"], [0, 0]); a[0] += c; a[1] += 1
        for iid, c in run.get("hint_plain", {}).items():
            d = hints["hint_plain"][iid]
            a = plainp.setdefault(d["path"], [0, 0]); a[0] += c; a[1] += 1
        cells = ["path %d: true %s (%d) vs plain-with-that-path %s (%d)" % (
            pth, pts(byp[pth][0] / byp[pth][1]) if pth in byp else "  -  ", byp.get(pth, [0, 0])[1],
            pts(plainp[pth][0] / plainp[pth][1]) if pth in plainp else "  -  ", plainp.get(pth, [0, 0])[1]) for pth in (1, 2, 3, 4)]
        out["D_path"] = None if 1 not in byp or 4 not in byp else byp[1][0] / byp[1][1] - byp[4][0] / byp[4][1]
        print("P1' capacity by remaining critical path (true rows, items n >= 5): " + " | ".join(cells) + " | D_path %s" % dpts(out["D_path"]))
    # headroom on items n >= 4. PRIMARY H is on the clean variants (no shown value equals an option), plain restricted
    # to the same items; the unfiltered H is reported next to it.
    n4 = ids_with(hints, "hint_plain", lambda n: n >= 4)
    clean4 = {i for i in n4 if i in hints["hint_true_r1"] and not hints["hint_true_r1"][i]["shown_equals_option"]}
    h1c, h0c = acc(run, hints, "hint_true_r1", clean4)[0], acc(run, hints, "hint_plain", clean4)[0]
    out["H"] = None if h1c is None or h0c is None else h1c - h0c
    h1, h0 = acc(run, hints, "hint_true_r1", n4)[0], acc(run, hints, "hint_plain", n4)[0]
    out["H_all"] = None if h1 is None or h0 is None else h1 - h0
    hw = acc(run, hints, "hint_wrongres_r1", n4)[0]
    print("P2 headroom, clean variants (items n >= 4 with no shown value = option, %d): plain %s true r=1 %s -> H %s | all %d items: H %s; true - wrongres at r=1 %s" % (
        len(clean4), pts(h0c), pts(h1c), dpts(out["H"]), len(n4), dpts(out["H_all"]), dpts(None if h1 is None or hw is None else h1 - hw)))
    # manipulation checks. M1 full >= 90 (the block is read at all). M2 length / position: |placebo - plain| <= 3 on
    # the placebo rows pooled over r = 1..4 AND <= 5 at every single r (hard stop if either fails). M3 value use:
    # true - wrongres (consistent far-wrong chain) >= 10 at r = 1. M4 value distraction: true - other >= 10 at r = 1.
    full = acc(run, hints, "hint_full")[0]
    out["M1"] = None if full is None else full >= BAR["M1_full"]
    m2, pooled_diff, pooled_n = [], 0.0, 0
    for r in (1, 2, 3, 4):
        ids_r = set(hints[f"hint_placebo_r{r}"])
        pb, plr = acc(run, hints, f"hint_placebo_r{r}", ids_r)[0], acc(run, hints, "hint_plain", ids_r)[0]
        m2.append(None if None in (pb, plr) else pb - plr)
        if None not in (pb, plr):
            pooled_diff += (pb - plr) * len(ids_r)
            pooled_n += len(ids_r)
    out["M2_pooled"] = pooled_diff / pooled_n if pooled_n else None
    out["M2_r"] = m2
    out["M2"] = None if None in m2 else (abs(out["M2_pooled"]) <= BAR["M2_ctrl"] and all(abs(x) <= BAR["M2_ctrl_r"] for x in m2))
    # M3 and M4 compare true r = 1 with each control on the ITEMS PRESENT IN BOTH sets (the far chain drops 18 items)
    ids_w = set(hints["hint_true_r1"]) & set(hints["hint_wrongres_r1"])
    ids_o = set(hints["hint_true_r1"]) & set(hints["hint_other_r1"])
    tr1w, wr1 = acc(run, hints, "hint_true_r1", ids_w)[0], acc(run, hints, "hint_wrongres_r1", ids_w)[0]
    tr1, ot1 = acc(run, hints, "hint_true_r1", ids_o)[0], acc(run, hints, "hint_other_r1", ids_o)[0]
    out["M3"] = None if None in (tr1w, wr1) else (tr1w - wr1) >= BAR["M3_wrong"]
    out["M4"] = None if None in (tr1, ot1) else (tr1 - ot1) >= BAR["M4_other"]
    out["M3_d"] = None if None in (tr1w, wr1) else tr1w - wr1
    out["M4_d"] = None if None in (tr1, ot1) else tr1 - ot1
    out["full"] = full
    ids1 = ids_w
    print("M1 full %s (bar 90) -> %s | M2 placebo-plain pooled %s (bar +-3), at r1..r4 %s (bar +-5 each) -> %s" % (
        pts(full), out["M1"], dpts(out["M2_pooled"]), " ".join(dpts(x) for x in m2), out["M2"]))
    ids_n = set(hints["hint_true_r1"]) & set(hints["hint_wrongres_near_r1"])
    tr1n, wn1 = acc(run, hints, "hint_true_r1", ids_n)[0], acc(run, hints, "hint_wrongres_near_r1", ids_n)[0]
    out["M3b_d"] = None if None in (tr1n, wn1) else tr1n - wn1
    out["M3b"] = None if out["M3b_d"] is None else out["M3b_d"] >= BAR["M3b_near"]
    print("M3 true-wrongres(far chain) at r1 %s on %d shared items (bar 10) -> %s | M3b true-wrongres_near (plausible chain) %s (bar 10) -> %s | M4 true-other at r1 %s on %d shared items (bar 10) -> %s" % (
        dpts(out["M3_d"]), len(ids_w), out["M3"], dpts(out["M3b_d"]), out["M3b"], dpts(out["M4_d"]), len(ids_o), out["M4"]))
    cr = []
    for s in ("hint_true_r1", "hint_true_r2", "hint_other_r1"):
        rate, n = copy_rate(run, hints, s)
        cr.append("%s %s (n %d)" % (s, "-" if rate is None else "%.1f%%" % (100 * rate), n))
        out["copy_" + s] = rate
    print("copy rate (predictions = the last shown value where it is an option): " + "; ".join(cr))
    # read check (format): items with n = 2 at r = 1 (one step shown, one left) vs their plain accuracy
    n2 = ids_with(hints, "hint_plain", lambda n: n == 2)
    a2, b2 = acc(run, hints, "hint_true_r1", n2)[0], acc(run, hints, "hint_plain", n2)[0]
    out["read_n2"] = None if None in (a2, b2) else a2 - b2
    rv = "WAITING" if out["read_n2"] is None else "reads (>= +5)" if out["read_n2"] >= BAR["READ"] else "does not read (< +2)" if out["read_n2"] < BAR["READ_LOW"] else "weak (+2 .. +5)"
    print("read check (n = 2 items, %d): true r=1 %s vs plain %s -> %s -> %s" % (len(n2), pts(a2), pts(b2), dpts(out["read_n2"]), rv))
    out["r12"] = {}
    for drop in (False, True):
        xs = []
        for r in (1, 2):
            run_s = run.get(f"hint_true_r{r}", {})
            xs += [c for i, c in run_s.items() if not drop or not hints[f"hint_true_r{r}"][i]["shown_equals_option"]]
        out["r12"][drop] = (sum(xs) / len(xs)) if xs else None
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-dir", required=True)
    ap.add_argument("--run", action="append", default=[], help="<arm>:<seed>=<results tgz or dir>")
    ap.add_argument("--pair", action="append", default=[], help="<arm a>,<arm b>: seed-paired a - b on r in {1,2} (P3)")
    a = ap.parse_args()
    hints = load_hints(a.hint_dir)
    runs = {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        runs[label] = per_run(load_run(path), hints, label)
    arms = {}
    for label, o in runs.items():
        arm, seed = label.split(":")
        arms.setdefault(arm, {})[seed] = o
    print("=== P2 (primary) and P1 by arm. P2 PASS: seed mean H >= 15, every seed >= 10, paired t of H >= t_crit (df n-1); NONE: mean < 5. P1: D >= 15 capacity-limited, < 5 not ===")
    for arm, by in arms.items():
        H = [o["H"] for o in by.values() if o["H"] is not None]
        Dd = [o["D"] for o in by.values() if o["D"] is not None]
        if H:
            mH, tH = t_stat(H)
            crit = TCRIT.get(len(H), 2.0)
            hv = "H %s [%s] t %.2f (crit %.2f, n %d) -> %s" % (dpts(mH), " ".join(dpts(x).strip() for x in H), tH, crit, len(H),
                 "PASS (big headroom)" if mH >= BAR["H_big"] and min(H) >= BAR["H_seed"] and tH >= crit else "NONE" if mH < BAR["H_small"] else "limited / undetermined")
        else:
            hv = "H -"
        dv = "D %s (%s)" % (dpts(sum(Dd) / len(Dd)) if Dd else "-", "capacity-limited" if Dd and sum(Dd) / len(Dd) >= BAR["D_big"] else "not step-limited" if Dd and sum(Dd) / len(Dd) < BAR["D_small"] else "between")
        m = ["M%s %s" % (i, all(o["M%s" % i] for o in by.values()) if all(o["M%s" % i] is not None for o in by.values()) else "WAITING") for i in ("1", "2", "3", "3b", "4")]
        print("%-10s seeds %s: %s; %s; per-run all: %s" % (arm, ",".join(sorted(by)), hv, dv, " ".join(m)))
        # the pre-registered gate for 3-seed arms: SEED MEANS of full, placebo - plain (pooled and per r), true - wrongres,
        # true - near, true - other, read check; single models use their own run above
        os_ = list(by.values())
        def smean(key):
            xs = [o[key] for o in os_ if o.get(key) is not None]
            return (sum(xs) / len(xs)) if len(xs) == len(os_) and xs else None
        full_m, p_m, w_m, n_m, o_m, r_m = (smean(k) for k in ("full", "M2_pooled", "M3_d", "M3b_d", "M4_d", "read_n2"))
        per_r = []
        for i in range(4):
            xs = [o["M2_r"][i] for o in os_ if o.get("M2_r") and o["M2_r"][i] is not None]
            per_r.append((sum(xs) / len(xs)) if len(xs) == len(os_) and xs else None)
        g = {"M1": None if full_m is None else full_m >= BAR["M1_full"],
             "M2": None if p_m is None or None in per_r else (abs(p_m) <= BAR["M2_ctrl"] and all(abs(x) <= BAR["M2_ctrl_r"] for x in per_r)),
             "M3": None if w_m is None else w_m >= BAR["M3_wrong"], "M3b": None if n_m is None else n_m >= BAR["M3b_near"],
             "M4": None if o_m is None else o_m >= BAR["M4_other"]}
        rv = "WAITING" if r_m is None else "reads" if r_m >= BAR["READ"] else "does not read" if r_m < BAR["READ_LOW"] else "weak"
        print("%-10s seed-mean gate (n %d): full %s M1 %s | placebo-plain pooled %s per-r %s M2 %s | true-wrongres %s M3 %s | true-near %s M3b %s | true-other %s M4 %s | read %s (%s) -> gate %s" % (
            arm, len(os_), pts(full_m), g["M1"], dpts(p_m), " ".join(dpts(x) for x in per_r), g["M2"], dpts(w_m), g["M3"], dpts(n_m), g["M3b"], dpts(o_m), g["M4"], dpts(r_m), rv,
            "WAITING" if None in g.values() else ("PASS" if all(g.values()) else "FAIL")))
    for pair in a.pair:
        x, y = pair.split(",")
        for drop in (False, True):
            seeds = sorted(set(arms.get(x, {})) & set(arms.get(y, {})))
            ds = [arms[x][s]["r12"][drop] - arms[y][s]["r12"][drop] for s in seeds if arms[x][s]["r12"][drop] is not None and arms[y][s]["r12"][drop] is not None]
            if len(ds) < 2:
                print("P3 %s - %s%s: WAITING (seeds %s)" % (x, y, " [drop shown=option]" if drop else "", seeds))
                continue
            m, t = t_stat(ds)
            crit = TCRIT.get(len(ds), 2.0)
            verdict = "PASS" if m >= BAR["P3_pass"] and min(ds) > 0 and t >= crit else "FAIL (closed)" if m < BAR["P3_fail"] else "between"
            print("P3 %s - %s%s on r in {1,2} (secondary; exploratory if any seed is 12): %s | mean %s | t %.2f (crit %.2f, n %d) -> %s" % (
                x, y, " [drop shown=option]" if drop else "", " ".join(dpts(d).strip() for d in ds), dpts(m), t, crit, len(ds), verdict))


if __name__ == "__main__":
    main()
