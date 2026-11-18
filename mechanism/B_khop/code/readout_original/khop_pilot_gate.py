#!/usr/bin/env python3
"""CARD_khop_v1 pilot gate (section 3, lines P0-P9). Deterministic; exit 0 = GATE PASS, 1 = GATE FAIL.

    python3 -I khop_pilot_gate.py --base <j750 tgz> --a2z s16=<j751 tgz> --kh <j756 tgz> --k1 <j757 tgz> \\
        --ref s16=<results_520 tgz>,<results_663 tgz> [--report <khop dir>/build_report.json] [--khop-dir <dir>]

Lines (FAIL on any):
- P0 build_report.json: d_pass with no D-line failure, D3 zero shares, D4 longest prompt < 1536 tokens; the md5 of the 9
  eval and 3 train files in --khop-dir equal the report; and the D1 / D2 numbers of the report re-checked here against
  the card's bars (fix round, build critic finding 3): every barred line (BARRED on every file, EVAL_BARRED on the eval
  sets) present in every cell, <= 0.33 in every cell of >= 100 rows and <= 0.25 in every pooled cell (khop_blind
  excepted), gold shares in [0.17, 0.23] in every cell of >= 100 rows and in the khop_blind task pools.
- P1 every pilot tarball (750, 751, 756, 757) has metrics.json and every khop preds file with the 1.2 row counts; 756 /
  757 also have every GSM and hint preds file of the 663 env (val kk jl jevbench td bbeh + its EXTRA_EVALS, subq).
- P2 756 / 757 logs: one "extra-train: 3000 rows from data/eval/khop/train_kh.jsonl" (757: train_k1.jsonl) line,
  "extra-train: total training rows 8923", the last history loss finite.
- P3 (I1 at n = 1) a_KH(t, o, 1) >= 0.85 and a_K1(t, o, 1) >= 0.85 on the six (task, order) cells of ca / wp / vb.
- P4 (I2 at n = 1) KH query-swap pair accuracy at k = 1 >= 0.80.
- P5 (I3 at n = 1) blind accuracy <= 0.32 for 756, 757 and 751.
- P6 756 and 757 gsm8kx and valx accuracy >= max(j520, j663) - 0.04 (metrics.json eval).
- P7 wall time of 756 and 757 (first "== <name> on" line to "== exit" line of the log) <= 1.6 h each.
- P7b projected confirmation GPU-h <= 25 (fix round, build critic finding 2): 8 x t + 5 x (14923 / 8923) x t + 5 x e,
  t = mean wall of 756 / 757 (KH / K1 jobs; KD scaled by its row count), e = max(0.2, mean wall of 750 / 751).
- P8 750 and 751 landed with "== exit 0".
- P9 dead training: FAIL when |a_KH(ca, fwd, k) - a_A2Z(ca, fwd, k)| < 0.03 at both k = 2 and k = 3 AND a_KH(ca, fwd, 3)
  < 0.80 (756 vs 751); the report then says "new card for the schedule".
Then INFO lines (no decision): every curve, error mix, order contrast, rc cells of 756 / 757 and the references, timing,
peak memory. Output ends with "GATE PASS" or "GATE FAIL: <lines>".
"""
import argparse, hashlib, json, math, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402

E = os.path.dirname(HERE)
ENV663 = os.path.join(E, "stage/condor/jobs_x1/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.env")
GSM_BASE = ["val", "kk", "jl", "jevbench", "td", "bbeh", "subq"]
FILES12 = T.SETS + ["train_kh", "train_k1", "train_kd"]
BAR = {"P3": 0.85, "P4": 0.80, "P5": 0.32, "P6": 0.04, "P7_h": 1.6, "P7b_gpuh": 25.0, "P9_d": 0.03, "P9_acc": 0.80,
       "D4": 1536, "D1_cell": 0.33, "D1_pooled": 0.25, "D2_lo": 0.17, "D2_hi": 0.23}
# CARD_khop_v1 2.4 D1 (Amendment 2): the barred lines, fixed here independently of the generator
BARRED = ("max", "min", "median", "closest_lit", "first_lit", "last_lit", "unwritten", "qop_pair", "qop_consistent",
          "one_op_parent", "chase_mult", "chase_sign")
EVAL_BARRED = ("qop_avoid",)
KD_SCALE = 14923 / 8923
TRAIN_ROWS = 8923


def gsm_sets(env_path=ENV663):
    """The preds names the 663 env writes: the trainer's fixed sets, its EXTRA_EVALS names, subq."""
    txt = open(env_path).read()
    m = re.search(r"^EXTRA_EVALS=(.*)$", txt, re.M)
    names = [kv.split("=", 1)[0] for kv in m.group(1).replace("\\ ", " ").split() if "=" in kv] if m else []
    return GSM_BASE + names


def wall_hours(log):
    """Hours from the "== <name> on <host> <date>" line to the "== exit <rc> <date>" line (None if either is missing)."""
    a = re.search(r"^== \S+ on \S+ (\w{3} \w{3} +\d+ [\d:]{8}) \w+ (\d{4})", log, re.M)
    b = re.search(r"^== exit -?\d+ (\w{3} \w{3} +\d+ [\d:]{8}) \w+ (\d{4})", log, re.M)
    if not a or not b:
        return None
    f = lambda m: time.mktime(time.strptime(f"{m.group(1)} {m.group(2)}", "%a %b %d %H:%M:%S %Y"))
    return (f(b) - f(a)) / 3600


def exit_code(log):
    m = re.findall(r"^== exit (-?\d+)", log, re.M)
    return int(m[-1]) if m else None


def d_bar_violations(rep):
    """The D1 / D2 numbers of build_report.json that break the card's bars (P0); [] = all within the bars."""
    bad = []
    d1, d2 = rep.get("d1_baselines") or {}, rep.get("gold_shares") or {}
    for f in FILES12:
        if f not in d1 or f not in d2:
            bad.append(f"{f}: no D1 / D2 entry")
            continue
        lines_ = BARRED + (() if f.startswith("train") else EVAL_BARRED)
        for c, v in d1[f].items():
            pooled = c.startswith("pooled")
            if f == "khop_blind" or (not pooled and v.get("n", 0) < 100):
                continue
            bar = BAR["D1_pooled"] if pooled else BAR["D1_cell"]
            for b in lines_:
                x = v.get(b)
                if x is None or x > bar:
                    bad.append(f"D1 {f} {c} {b} {x}")
        for c, g in d2[f].items():
            n = (d1[f].get(c) or {}).get("n", 0)
            if (f == "khop_blind" and c.startswith("pooled/")) or (f != "khop_blind" and n >= 100):
                if len(g) != 5 or not all(BAR["D2_lo"] <= x <= BAR["D2_hi"] for x in g.values()):
                    bad.append(f"D2 {f} {c} {g}")
    return bad


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def gate(a):
    lines, fails = [], []
    def line(name, ok, detail):
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)
    khop = T.load_khop(a.khop_dir)
    rep = json.load(open(a.report)) if os.path.exists(a.report) else None
    # P0
    if rep is None:
        line("P0", False, f"no build report {a.report}")
    else:
        d3 = rep.get("d3_disjointness", {})
        d4 = max((c["max"] for s in (rep.get("d4_tokens") or {}).values() for c in s.values()), default=None)
        bad_md5 = [f for f in FILES12 if not os.path.exists(os.path.join(a.khop_dir, f + ".jsonl")) or
                   md5(os.path.join(a.khop_dir, f + ".jsonl")) != rep["md5"].get(f + ".jsonl")]
        bars = d_bar_violations(rep)
        ok = (rep.get("d_pass") is True and not rep.get("d_fail") and d3.get("shared_prompt_hashes") == 0 and
              d3.get("shared_value_keys") == 0 and d4 is not None and d4 < BAR["D4"] and not bad_md5 and not bars)
        line("P0", ok, f"d_pass {rep.get('d_pass')} d_fail {len(rep.get('d_fail') or [])} D3 {d3.get('shared_prompt_hashes')}/"
             f"{d3.get('shared_value_keys')} D4 longest {d4} md5 mismatches {bad_md5} D1 / D2 re-check "
             f"{'ok' if not bars else bars[:4]}")
    runs = {"750": T.load_run(a.base), "751": T.load_run(a.a2z.split("=", 1)[1]), "756": T.load_run(a.kh),
            "757": T.load_run(a.k1)}
    # P1
    gsm = gsm_sets(a.env663)
    for j, r in runs.items():
        miss = [s for s in T.SETS if len(r["preds"].get(s, {})) != T.ROWS[s]]
        extra = ""
        if j in ("756", "757"):
            have = set(r.get("all_preds", []))
            gm = [g for g in gsm if g not in have]
            miss += gm
            extra = f", GSM / hint sets {len(gsm) - len(gm)}/{len(gsm)}"
        line(f"P1 j{j}", r["metrics"] is not None and not miss,
             f"metrics.json {'yes' if r['metrics'] is not None else 'MISSING'}, khop sets "
             f"{sum(len(r['preds'].get(s, {})) == T.ROWS[s] for s in T.SETS)}/9{extra}; missing {miss[:6]}")
    # P2
    for j, f in (("756", "train_kh"), ("757", "train_k1")):
        log = runs[j]["log"]
        m = re.findall(r"^extra-train: (\d+) rows from (\S+)", log, re.M)
        tot = re.findall(r"^extra-train: total training rows (\d+)", log, re.M)
        h = ((runs[j]["metrics"] or {}).get("history") or [])
        loss = h[-1].get("loss") if h else None
        ok = (len(m) == 1 and m[0] == ("3000", f"data/eval/khop/{f}.jsonl") and tot == [str(TRAIN_ROWS)] and
              loss is not None and math.isfinite(loss))
        line(f"P2 j{j}", ok, f"extra-train lines {m}, total {tot}, final loss {loss}")
    # P3
    cur = lambda j, s: T.curve(runs[j], khop, s)
    vals = {(j, t, o): cur(j, f"khop_{t}_{o}").get(1) for j in ("756", "757") for t, o in T.CELLS6}
    line("P3", all(v is not None and v >= BAR["P3"] for v in vals.values()),
         " ".join(f"{'KH' if j == '756' else 'K1'}/{t}/{o} {T.pts(v)}" for (j, t, o), v in vals.items()))
    # P4
    q = T.qs_pair(runs["756"], khop)
    line("P4", q is not None and q >= BAR["P4"], f"KH query-swap pair accuracy k=1 {T.pts(q)}")
    # P5
    bl = {j: T.blind(runs[j], khop) for j in ("756", "757", "751")}
    line("P5", all(v is not None and v <= BAR["P5"] for v in bl.values()), " ".join(f"j{j} {T.pts(v)}" for j, v in bl.items()))
    # P6
    refs = [T.load_run(p) for p in a.ref.split("=", 1)[1].split(",")]
    det, ok6 = [], True
    for s in ("gsm8kx", "valx"):
        rv = [T.plain_acc(r, s) for r in refs]
        rm = max(v for v in rv if v is not None) if any(v is not None for v in rv) else None
        for j in ("756", "757"):
            v = T.plain_acc(runs[j], s)
            ok = v is not None and rm is not None and v >= rm - BAR["P6"]
            ok6 &= ok
            det.append(f"j{j} {s} {T.pts(v)} (ref {T.pts(rm)})")
    line("P6", ok6, " ".join(det))
    # P7
    wh = {j: wall_hours(runs[j]["log"]) for j in ("756", "757")}
    line("P7", all(v is not None and v <= BAR["P7_h"] for v in wh.values()),
         " ".join(f"j{j} {'-' if v is None else '%.2f h' % v}" for j, v in wh.items()))
    # P7b
    ev = [v for v in (wall_hours(runs[j]["log"]) for j in ("750", "751")) if v is not None]
    e_h = max(0.2, sum(ev) / len(ev)) if ev else 0.2
    if None in wh.values():
        line("P7b", False, "wall time of 756 / 757 missing")
    else:
        t = (wh["756"] + wh["757"]) / 2
        proj = 8 * t + 5 * KD_SCALE * t + 5 * e_h
        line("P7b", proj <= BAR["P7b_gpuh"], f"projected confirmation {proj:.1f} GPU-h (t {t:.2f} h, eval {e_h:.2f} h, "
             f"cap {BAR['P7b_gpuh']:.0f})")
    # P8
    ex = {j: exit_code(runs[j]["log"]) for j in ("750", "751")}
    line("P8", all(v == 0 for v in ex.values()), " ".join(f"j{j} exit {v}" for j, v in ex.items()))
    # P9
    ckh, ca2z = cur("756", "khop_ca_fwd"), cur("751", "khop_ca_fwd")
    d = {k: (None if ckh.get(k) is None or ca2z.get(k) is None else ckh[k] - ca2z[k]) for k in (2, 3)}
    dead = (all(v is not None and abs(v) < BAR["P9_d"] for v in d.values()) and ckh.get(3) is not None
            and ckh[3] < BAR["P9_acc"])
    line("P9", not dead and None not in d.values(),
         f"KH - A2Z ca fwd k2 {T.pts(d[2]) if d[2] is not None else '-'} k3 {T.pts(d[3]) if d[3] is not None else '-'}, "
         f"KH k3 {T.pts(ckh.get(3))}" + ("; dead training: new card for the schedule" if dead else ""))
    # INFO
    info = []
    for j, lab in (("750", "BASE0"), ("751", "A2Z"), ("756", "KH"), ("757", "K1")):
        for s in T.SETS:
            if s != "khop_blind":
                info.append(f"INFO {lab} {s:15s} " + " ".join(f"k{k} {T.pts(v)}" for k, v in cur(j, s).items()))
    for j in ("756", "757"):
        h = (runs[j]["metrics"] or {}).get("history") or []
        info.append(f"INFO j{j} peak_gib {max([x.get('peak_gib') or 0 for x in h] + [0]):.1f} wall {wh[j]}")
        for k in (2, 3):
            f_, r_ = cur(j, "khop_ca_fwd").get(k), cur(j, "khop_ca_rev").get(k)
            info.append(f"INFO j{j} ca fwd - rev k{k} {T.pts(None if None in (f_, r_) else f_ - r_)}")
        ns = T.nearest_share(runs[j], khop, "khop_ca_fwd")
        info.append(f"INFO j{j} nearest-chain share of wrong ca fwd picks k2-3 {ns}")
    if a.hint_dir and os.path.isdir(a.hint_dir):
        for j, p in (("756", a.kh), ("757", a.k1)):
            try:
                info.append(f"INFO j{j} rc cells {T.rc_cells(p, a.hint_dir)}")
            except Exception as e:                     # INFO only
                info.append(f"INFO j{j} rc cells unavailable ({e})")
    return lines, fails, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--a2z", required=True, help="s16=<j751 tgz>")
    ap.add_argument("--kh", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--ref", required=True, help="s16=<j520 tgz>,<j663 tgz>")
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--report", default="")
    ap.add_argument("--env663", default=ENV663)
    ap.add_argument("--hint-dir", default=os.path.join(E, "src/data/eval/hint"))
    a = ap.parse_args(argv)
    a.report = a.report or os.path.join(a.khop_dir, "build_report.json")
    lines, fails, info = gate(a)
    for l in lines + info:
        print(l)
    print("GATE PASS" if not fails else "GATE FAIL: " + ", ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
