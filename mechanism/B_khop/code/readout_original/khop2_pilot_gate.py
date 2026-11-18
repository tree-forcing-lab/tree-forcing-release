#!/usr/bin/env python3
"""CARD_khop_v2 pilot gate (section 3, lines P0-P9 with P7b). Deterministic; exit 0 = GATE PASS (both forms), 1 = FAIL.

    python3 -I khop2_pilot_gate.py --d27 <j875 tgz> --r3 <j880 tgz> \\
        --kd s16=<E>/results/results_766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.tgz \\
        --ref s16=<results_520 tgz>,<results_663 tgz> \\
        --report <E>/src/data/eval/khop2/build_report.json --parent-report <E>/src/data/eval/khop/build_report.json \\
        [--smoke-d27 <smoke_875 tgz>] [--smoke-r3 <smoke_880 tgz>] [--results-dir <E>/results]

The smoke tarballs default to results_smoke_875_*.tgz / results_smoke_880_*.tgz in --results-dir (P8 fails without them).
Lines (FAIL on any):
- P0 data: the khop2 report has d_pass true and no d_fail; the md5 of the 3 new files in --khop2-dir equal that report;
  the md5 of the 12 parent files in --khop-dir equal the parent report AND the CARD_khop_v1 Amendment 2 table (PARENT_MD5
  below), and the khop2 report's parent_md5 equals them; the gate re-checks every barred D1 value and every D2 share of
  the khop2 report itself (d_bar_violations, its own list of the 12 barred lines; the 4 stems are training files, so
  qop_avoid is reported, not barred); D3 shared counts 0; D6 nested and copy flags true; 18,000 new distinct value keys.
- P1 each pilot tarball has metrics.json, the nine khop preds files with the CARD_khop_v1 1.2 row counts and exactly the
  item ids of the eval files, and every GSM and hint preds file of the 663 env (its fixed sets and its EXTRA_EVALS line).
- P2 (I4 at n = 1) j875 / j880: one "extra-train: 27000 rows from data/eval/khop2/<train_kd27 | train_kd_x3>.jsonl" line,
  "extra-train: total training rows 32923", final step 1,029 (metrics history and the log's last step line), final loss
  finite, metrics.json extra_train = {that path: 27000}.
- P3 (I1 at n = 1) a(t, o, 1) >= 0.85 on the six (task, order) cells of ca / wp / vb, j875 and j880.
- P4 (I2 at n = 1) query-swap pair accuracy at k = 1 >= 0.80, j875 and j880.
- P5 (I3 at n = 1) blind accuracy <= 0.32, j875 and j880.
- P6 plain catastrophe line (Amendment 1 finding 1): gsm8kx and gsm8kv2 of j875 and j880 >= max(j520, j663) - 0.10.
  valx and gsm1kx are INFO. Both references must be finite on the line's metric, else P6 fails (no one-reference bar).
Missing, partial, NaN or non-numeric inputs fail their line; no line drops a value and decides on the rest.
- P7 wall time of j875 and j880 each <= 2.9 h.
- P7b projected confirmation: t = mean wall of j875 / j880; main = 8 t; h3 = 5 x max(1.11, 0.521 t); main + h3 <= 25:
  PASS; else main <= 25: "GATE PASS (H3 dropped: projection <main + h3> GPU-h)", no H3 node; else FAIL.
- P8 smoke_875, smoke_880, j875 and j880 landed with "== exit 0".
- P9 (I5 at n = 1) a(ca, fwd, 2) >= 0.80, j875 and j880. No gate line compares D27 or R3 with KD at k >= 3.
INFO lines (no decision): every curve, a(ca, fwd, 3) of j875 / j880 against j766 (KD s16), the all-+/- subsets, probe
agreement, GSM rc cells (pfx_table.py), valx / gsm1kx, timing and peak memory.
Output ends with "GATE PASS", "GATE PASS (H3 dropped: projection <x> GPU-h)" or "GATE FAIL: <lines>".
Every message to the second author about this family says that the owner's 2-epoch branch runs as R3 / H3 (repetition in the file)
and that the literal EPOCHS=2 arms need a new card and id range; the gate prints that note before its last line.
"""
import argparse, glob, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402
import khop_pilot_gate as G  # noqa: E402  (wall_hours, exit_code, gsm_sets)

E = os.path.dirname(HERE)
BAR = {"P3": 0.85, "P4": 0.80, "P5": 0.32, "P6": 0.10, "P7_h": 2.9, "cap": 25.0, "h3_min_h": 1.11, "h3_ratio": 0.521,
       "P9": 0.80, "D1_cell": 0.33, "D1_pooled": 0.25, "D2_lo": 0.17, "D2_hi": 0.23}
# CARD_khop_v1 2.4 D1 (Amendment 2): the 12 barred lines, fixed here independently of the generator
BARRED = ("max", "min", "median", "closest_lit", "first_lit", "last_lit", "unwritten", "qop_pair", "qop_consistent",
          "one_op_parent", "chase_mult", "chase_sign")
NEW_FILES = ("train_kd27", "train_kd_x3", "train_kh_x3")
D_STEMS = NEW_FILES + ("train_kd27_new",)
# CARD_khop_v1 Amendment 2 table (E/src_khop/BUILD_REPORT.md, "Data (rebuilt 2026-10-26 00:22 CDT)")
PARENT_MD5 = {"khop_ca_fwd": "171f097c2f0f55ae4505d37365d95491", "khop_ca_rev": "2f2bc7b6fc1cfd45f24451741f0e4105",
              "khop_wp_fwd": "c58c93ffa4d3912cfdbfd848f6eb9fb9", "khop_wp_rev": "bdad8780cc64e94e3fcdd0148833a4d7",
              "khop_vb_fwd": "046c6ad382d66c02f1315ce7f8c7577d", "khop_vb_rev": "2279c09f4adb666196bdbbe636b53646",
              "khop_cag_fwd": "3e79c1b5233e3db7dc23927e3066eba5", "khop_ca_fwd_qs": "38cf266ea1490f0c42f6378baa85f4e4",
              "khop_blind": "b58f7284e1b1d88197a44d0b54eba2ae", "train_kh": "a16b1520f0fb8b24ffd8fa57ef696de0",
              "train_k1": "4bb15f66c8d8843013eb0410c6a87e48", "train_kd": "2ed4ecc377e4638e35b12e8b22e37461"}
PILOT = {"875": ("D27", "train_kd27"), "880": ("R3", "train_kd_x3")}
CHAIN_ROWS, TOTAL_ROWS, STEPS, NEW_ROWS = 27000, 32923, 1029, 18000
CELLS_D1 = tuple(f"{t}/k{k}" for t in ("ca", "wp", "vb") for k in range(1, 7)) + tuple(f"pooled_k1-6/{t}"
                                                                                   for t in ("ca", "wp", "vb"))
NOTE = ("NOTE the owner's 2-epoch branch runs as R3 / H3 (3 repeats inside the chain file, matched updates); the literal "
        "EPOCHS=2 arms need a new card and a new id range.")


def num(x):
    """x when it is a finite int / float (not bool), else None: a missing, NaN or non-numeric value never passes a line."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def d_bar_violations(rep, stems=D_STEMS):
    """The D1 / D2 numbers of the khop2 build_report.json that break the card's bars; [] = all within the bars. Every
    stem is a training file: the 12 BARRED lines in every cell of >= 100 rows (<= 0.33) and every pooled cell (<= 0.25),
    gold shares in [0.17, 0.23] in every cell of >= 100 rows. Every (task, k) cell of ca / wp / vb at k = 1..6 and the
    three pooled cells must be in the report with a row count n, and every non-pooled cell of >= 100 rows needs its D2
    entry: a missing cell, n or value, or a NaN, is a violation (Codex fix round 5; the old loop read only the cells the
    report had)."""
    bad = []
    d1, d2 = rep.get("d1_baselines") or {}, rep.get("gold_shares") or {}
    for f in stems:
        if f not in d1 or f not in d2:
            bad.append(f"{f}: no D1 / D2 entry")
            continue
        for c in CELLS_D1:
            v = d1[f].get(c)
            n = (v or {}).get("n")
            if v is None or not isinstance(n, int) or isinstance(n, bool):
                bad.append(f"D1 {f} {c}: cell or row count missing")
                continue
            pooled = c.startswith("pooled")
            if not pooled and n < 100:
                continue
            bar = BAR["D1_pooled"] if pooled else BAR["D1_cell"]
            for b in BARRED:
                x = num(v.get(b))
                if x is None or x > bar:
                    bad.append(f"D1 {f} {c} {b} {v.get(b)}")
            if not pooled:
                g = d2[f].get(c)
                if not isinstance(g, dict) or len(g) != 5 or \
                        not all(num(x) is not None and BAR["D2_lo"] <= x <= BAR["D2_hi"] for x in g.values()):
                    bad.append(f"D2 {f} {c} {g}")
    return bad


def p7b(t):
    """(mode, main, h3, total): mode "full" (main + h3 <= 25), "no_h3" (only main <= 25) or "fail"."""
    main = 8 * t
    h3 = 5 * max(BAR["h3_min_h"], BAR["h3_ratio"] * t)
    tot = main + h3
    return ("full" if tot <= BAR["cap"] else "no_h3" if main <= BAR["cap"] else "fail"), main, h3, tot


def log_steps(log):
    return [int(x) for x in re.findall(r'^\{"step": (\d+),', log, re.M)]


def find_smoke(results_dir, jid):
    f = sorted(glob.glob(os.path.join(results_dir, f"results_smoke_{jid}_*.tgz")))
    return f[0] if f else None


def gate(a):
    lines, fails, info = [], [], []

    def line(name, ok, detail):
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(name)
    khop = T.load_khop(a.khop_dir)
    # P0
    rep = json.load(open(a.report)) if os.path.exists(a.report) else None
    prep = json.load(open(a.parent_report)) if os.path.exists(a.parent_report) else None
    if rep is None or prep is None:
        line("P0", False, f"no build report ({a.report} {'ok' if rep else 'MISSING'}, {a.parent_report} "
             f"{'ok' if prep else 'MISSING'})")
    else:
        d3 = rep.get("d3_disjointness") or {}
        bad_new = [f for f in NEW_FILES if not os.path.exists(os.path.join(a.khop2_dir, f + ".jsonl")) or
                   md5(os.path.join(a.khop2_dir, f + ".jsonl")) != (rep.get("md5") or {}).get(f + ".jsonl")]
        bad_par = []
        for f, want in PARENT_MD5.items():
            p = os.path.join(a.khop_dir, f + ".jsonl")
            got = md5(p) if os.path.exists(p) else None
            if got != want or (prep.get("md5") or {}).get(f + ".jsonl") != want or \
                    (rep.get("parent_md5") or {}).get(f + ".jsonl") != want:
                bad_par.append(f)
        bars = d_bar_violations(rep)
        d3_ok = all((d3.get(s) or {}).get("shared_prompt_hashes") == 0 and (d3.get(s) or {}).get("shared_value_keys") == 0
                    for s in NEW_FILES)
        d6_ok = rep.get("nested") is True and rep.get("copies_ok") == {"train_kd_x3": True, "train_kh_x3": True}
        ok = (rep.get("d_pass") is True and not rep.get("d_fail") and not bad_new and not bad_par and not bars and d3_ok
              and d6_ok and rep.get("new_distinct_keys") == NEW_ROWS)
        line("P0", ok, f"d_pass {rep.get('d_pass')} d_fail {len(rep.get('d_fail') or [])} new md5 mismatches {bad_new} "
             f"parent md5 mismatches {bad_par} D1 / D2 re-check {'ok' if not bars else bars[:4]} D3 "
             f"{'0' if d3_ok else 'NONZERO'} D6 nested {rep.get('nested')} copies {rep.get('copies_ok')} new distinct "
             f"keys {rep.get('new_distinct_keys')}")
    runs = {"875": T.load_run(a.d27), "880": T.load_run(a.r3)}
    # the pilots and their references are s16 (card section 3): the seed metrics.json records must say so, so a wrong
    # tarball path never passes the gate (Codex fix round 3)
    rseed = lambda r: tuple(((r.get("metrics") or {}).get("args") or {}).get(k) for k in ("seed", "order_seed"))
    # P1
    # every GSM / hint preds file of the 663 env: the trainer's fixed sets plus its EXTRA_EVALS; an env without the
    # EXTRA_EVALS line gives the fixed sets only, so P1 would check 7 of the 55 files (Codex fix round 5)
    gsm = G.gsm_sets(a.env663)
    env_ok = len(gsm) > len(G.GSM_BASE)
    # P3-P9 read khop_table.acc / qs_pair, which drop ids absent from the preds: each preds file must hold exactly the
    # item ids of its eval file (row count alone let a foreign id shrink a denominator; Codex fix round 5)
    want_ids = {s: {x["item_id"] for x in khop.get(s, [])} for s in T.SETS}
    for j, r in runs.items():
        miss = [s for s in T.SETS if len(r["preds"].get(s, {})) != T.ROWS[s] or len(want_ids[s]) != T.ROWS[s]]
        idbad = [s for s in T.SETS if s not in miss and set(r["preds"][s]) != want_ids[s]]
        have = set(r.get("all_preds", []))
        gm = [g for g in gsm if g not in have]
        line(f"P1 j{j}", r["metrics"] is not None and not miss and not idbad and env_ok and not gm and rseed(r) == (16, 16),
             f"metrics.json {'yes' if r['metrics'] is not None else 'MISSING'}, recorded seed / order_seed "
             f"{rseed(r)} (need (16, 16)), khop sets "
             f"{sum(s not in miss and s not in idbad for s in T.SETS)}/9 (row counts off {miss[:3]}, ids differ "
             f"{idbad[:3]}), GSM / hint sets {len(gsm) - len(gm)}/{len(gsm)}"
             f"{'' if env_ok else ' (env663 has no EXTRA_EVALS line: ' + str(a.env663) + ')'}; missing {gm[:6]}")
    # P2
    for j, (arm, f) in PILOT.items():
        log, m = runs[j]["log"], runs[j]["metrics"] or {}
        path = f"data/eval/khop2/{f}.jsonl"
        ex = re.findall(r"^extra-train: (\d+) rows from (\S+)", log, re.M)
        tot = re.findall(r"^extra-train: total training rows (\d+)", log, re.M)
        h = m.get("history") or []
        loss, step = (h[-1].get("loss"), h[-1].get("step")) if h else (None, None)
        ls = log_steps(log)
        ok = (ex == [(str(CHAIN_ROWS), path)] and tot == [str(TOTAL_ROWS)] and step == STEPS and ls[-1:] == [STEPS]
              and loss is not None and math.isfinite(loss) and m.get("extra_train") == {path: CHAIN_ROWS})
        line(f"P2 j{j}", ok, f"extra-train lines {ex}, total {tot}, final step {step} (log {ls[-1:]}), final loss {loss}, "
             f"metrics extra_train {m.get('extra_train')}")
    cur = lambda j, s: T.curve(runs[j], khop, s)
    # P3
    vals = {(j, t, o): cur(j, f"khop_{t}_{o}").get(1) for j in PILOT for t, o in T.CELLS6}
    line("P3", all(v is not None and v >= BAR["P3"] for v in vals.values()),
         " ".join(f"{PILOT[j][0]}/{t}/{o} {T.pts(v)}" for (j, t, o), v in vals.items()))
    # P4
    q = {j: T.qs_pair(runs[j], khop) for j in PILOT}
    line("P4", all(v is not None and v >= BAR["P4"] for v in q.values()),
         " ".join(f"j{j} query-swap pair k=1 {T.pts(v)}" for j, v in q.items()))
    # P5
    bl = {j: T.blind(runs[j], khop) for j in PILOT}
    line("P5", all(v is not None and v <= BAR["P5"] for v in bl.values()), " ".join(f"j{j} {T.pts(v)}" for j, v in bl.items()))
    # P6
    ref_lab, ref_paths = a.ref.split("=", 1)
    refs = [T.load_run(p) for p in ref_paths.split(",")]
    ref_seeds = [rseed(r) for r in refs]
    # the two references are j520 (original A2-6k) and j663 (re-train), in that order and distinct files: the bar is
    # max(j520, j663), so the same file twice must not pass (Codex fix round 4)
    rl = ref_paths.split(",")
    ids_ok = (len(rl) == 2 and len({os.path.realpath(p) for p in rl}) == 2
              and all(re.search(rf"(^|_){j}(_|\.tgz$)", os.path.basename(p)) for j, p in zip(REF_JOBS, rl)))
    ok6 = ref_lab == "s16" and ids_ok and all(x == (16, 16) for x in ref_seeds)
    det = [f"refs {ref_lab} recorded seed / order_seed {ref_seeds} (need s16, two refs at (16, 16)); j520 / j663 "
           f"distinct files {'ok' if ids_ok else 'NO'} ({[os.path.basename(p) for p in rl]})"]
    for s in ("gsm8kx", "gsm8kv2", "valx", "gsm1kx"):
        # the bar is max(j520, j663) of card 2.4: both must be finite numbers, else no bar (P6 fails); dropping a missing
        # or NaN reference first made a one-reference bar (Codex fix round 5, finding 1)
        rv = [num(T.plain_acc(r, s)) for r in refs]
        rm = max(rv) if len(rv) == 2 and None not in rv else None
        rtxt = "/".join(T.pts(x) if x is not None else "MISSING" for x in rv)
        for j in PILOT:
            v = num(T.plain_acc(runs[j], s))
            if s in ("gsm8kx", "gsm8kv2"):
                ok = v is not None and rm is not None and v >= rm - BAR["P6"]
                ok6 &= ok
                det.append(f"j{j} {s} {T.pts(v) if v is not None else 'MISSING'} (j520 / j663 {rtxt}, bar "
                           f"{T.pts(None if rm is None else rm - BAR['P6'])})")
            else:
                info.append(f"INFO P6 j{j} {s} {T.pts(v)} (j520 / j663 {rtxt})")
    line("P6", ok6, " ".join(det))
    # P7, P7b
    wh = {j: G.wall_hours(runs[j]["log"]) for j in PILOT}
    line("P7", all(v is not None and v <= BAR["P7_h"] for v in wh.values()),
         " ".join(f"j{j} {'-' if v is None else '%.2f h' % v}" for j, v in wh.items()) + f" (bar {BAR['P7_h']} h)")
    mode = None
    if None in wh.values():
        line("P7b", False, "wall time of j875 / j880 missing")
    else:
        t = (wh["875"] + wh["880"]) / 2
        mode, main, h3, tot = p7b(t)
        line("P7b", mode != "fail", f"t {t:.2f} h: main (D27, R3 s17-20) {main:.1f} + H3 s16-20 {h3:.1f} = {tot:.1f} "
             f"GPU-h (cap {BAR['cap']:.0f}) -> {'all nodes' if mode == 'full' else 'H3 dropped' if mode == 'no_h3' else 'FAIL'}")
    # P8
    sm = {"smoke_875": a.smoke_d27 or find_smoke(a.results_dir, 875), "smoke_880": a.smoke_r3 or find_smoke(a.results_dir, 880)}
    ex = {k: (G.exit_code(T.load_run(p)["log"]) if p and os.path.exists(p) else None) for k, p in sm.items()}
    ex.update({f"j{j}": G.exit_code(runs[j]["log"]) for j in PILOT})
    line("P8", all(v == 0 for v in ex.values()), " ".join(f"{k} exit {v}" for k, v in ex.items()))
    # P9
    c2 = {j: cur(j, "khop_ca_fwd").get(2) for j in PILOT}
    line("P9", all(v is not None and v >= BAR["P9"] for v in c2.values()),
         " ".join(f"j{j} a(ca, fwd, 2) {T.pts(v)}" for j, v in c2.items()))
    # INFO
    kd = T.load_run(a.kd.split("=", 1)[1]) if a.kd else None
    if kd:
        info.append(f"INFO KD s16 recorded seed / order_seed {rseed(kd)}")
    info_runs = dict(runs, **({"766": kd} if kd else {}))
    lab = {"875": "D27", "880": "R3", "766": "KD"}
    for j, r in info_runs.items():
        for s in T.SETS:
            if s != "khop_blind":
                info.append(f"INFO {lab[j]:3s} {s:15s} " + " ".join(f"k{k} {T.pts(v)}" for k, v in T.curve(r, khop, s).items()))
        ad = T.alladd_curve(r, khop, "khop_ca_fwd") if r["preds"].get("khop_ca_fwd") else {}
        info.append(f"INFO {lab[j]:3s} khop_ca_fwd all +/- " + " ".join(f"k{k} {T.pts(v)}" for k, v in ad.items()))
        for pb in T.RANK_KEYS:
            for k in (2, 3):
                x = T.probe_agree(r, khop, "khop_ca_fwd", pb, ks=(k,)) if r["preds"].get("khop_ca_fwd") else None
                info.append(f"INFO {lab[j]:3s} {pb} agreement ca fwd k{k} {'-' if not x else '%.3f (n %d)' % x}")
    if kd:
        k3 = T.curve(kd, khop, "khop_ca_fwd").get(3)
        for j in PILOT:
            v = cur(j, "khop_ca_fwd").get(3)
            info.append(f"INFO {lab[j]} - KD s16 at (ca, fwd, 3): {T.pts(None if None in (v, k3) else v - k3)} "
                        f"({T.pts(v)} vs {T.pts(k3)})")
    for j in PILOT:
        h = (runs[j]["metrics"] or {}).get("history") or []
        sps = [x.get("s_per_step") for x in h if x.get("s_per_step")]
        info.append(f"INFO j{j} wall {'-' if wh[j] is None else '%.2f' % wh[j]} h, s_per_step mean {sum(sps) / len(sps) if sps else float('nan'):.2f}, "
                    f"peak_gib {max([x.get('peak_gib') or 0 for x in h] + [0]):.1f}")
    if a.hint_dir and os.path.isdir(a.hint_dir):
        for j, p in (("875", a.d27), ("880", a.r3)):
            try:
                info.append(f"INFO j{j} rc cells {T.rc_cells(p, a.hint_dir)}")
            except Exception as e:                     # INFO only
                info.append(f"INFO j{j} rc cells unavailable ({e})")
    return lines, fails, info, mode, (p7b((wh["875"] + wh["880"]) / 2)[3] if None not in wh.values() else None)


REF_JOBS = ("520", "663")


def final_line(fails, mode, tot):
    if fails:
        return "GATE FAIL: " + ", ".join(fails)
    if mode == "no_h3":
        return f"GATE PASS (H3 dropped: projection {tot:.1f} GPU-h)"
    return "GATE PASS"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d27", required=True, help="j875 tarball")
    ap.add_argument("--r3", required=True, help="j880 tarball")
    ap.add_argument("--kd", default="", help="s16=<j766 tgz> (INFO only)")
    ap.add_argument("--ref", required=True, help="s16=<j520 tgz>,<j663 tgz>")
    ap.add_argument("--smoke-d27", default="")
    ap.add_argument("--smoke-r3", default="")
    ap.add_argument("--results-dir", default=os.path.join(E, "results"))
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--khop2-dir", default=os.path.join(E, "src/data/eval/khop2"))
    ap.add_argument("--report", default="")
    ap.add_argument("--parent-report", default="")
    ap.add_argument("--env663", default=G.ENV663)
    ap.add_argument("--hint-dir", default=os.path.join(E, "src/data/eval/hint"))
    a = ap.parse_args(argv)
    a.report = a.report or os.path.join(a.khop2_dir, "build_report.json")
    a.parent_report = a.parent_report or os.path.join(a.khop_dir, "build_report.json")
    lines, fails, info, mode, tot = gate(a)
    for l in lines + info:
        print(l)
    print(NOTE)
    print(final_line(fails, mode, tot))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
