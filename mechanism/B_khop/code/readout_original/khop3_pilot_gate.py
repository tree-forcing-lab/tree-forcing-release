#!/usr/bin/env python3
"""CARD_khop_v3 pilot gate (section 3, lines P0-P11 with P7b and P7c). Deterministic; exit 0 = GATE PASS, 1 = GATE FAIL.

    python3 -I khop3_pilot_gate.py --cur1 <j975 tgz> --cur <j980 tgz> --k1x1 <j985 tgz> --k1x <j990 tgz> \\
        --smokes <smoke_975 tgz>,<smoke_985 tgz>,<smoke_995 tgz> \\
        --ref s16=<E>/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,<E>/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz \\
        --report <E>/src/data/eval/khop3/build_report.json --khop2-report <E>/src/data/eval/khop2/build_report.json \\
        --parent-report <E>/src/data/eval/khop/build_report.json --retry-walls <file>

--retry-walls is required: one "<node> <hours>" line per failed execution of a pilot node (from the DAGs' nodes.log /
job event logs on the AP), an empty file when no node retried. The gate refuses to run without it.
Every line fails closed on a missing, partial or non-finite input; no line drops a value and decides on the rest. Every
accuracy is an exact count (Fraction(correct, items)); the bars are rationals.
- P0 data: D1 (c27 train md5 = train_kd27 = 92545dd4..., both val md5 = 9f55fec5..., the 12 khop files = the Amendment 2
  table = the khop report, train_kd27 = the khop2 report) and D2 (the khop3 report: d_pass, 27,000 K1X rows, replay
  27,000 / 27,000 and 3,000 / 3,000, K1X rows 0..2,999 = train_k1, every K1X check count 0, eval hash overlap 0, item_ids
  distinct, every gold share of every cell in [0.18, 0.22], token total within 1% in tokens) re-checked from the report,
  and the md5 of the 4 files on disk equal to the report; each dir holds exactly train.jsonl and val.jsonl.
- P1 metrics.json in each pilot tarball with seed / order_seed (16, 16); S1 tarballs: the nine khop preds files with
  exactly the eval files' item ids and the four plain sets; S2 tarballs: the nine khop files and every GSM / hint preds
  file of the 663 env (its fixed sets and its EXTRA_EVALS). Codex round 1 finding 1: khop3_table.load checks every set
  of the job's metrics.json args.eval (each preds file holds exactly the eval file's item ids, finite probs) and that
  args.eval lists khop3_table.S1_NEED (S1) or khop3_table.s2_need() (S2); a failure makes P1 FAIL with the reason as
  INFO. The references j520 / j663 get the same check (P6 FAIL).
- P2 (I4 at n = 1) the S1 and S2 log lines of khop3_table.i4_s1 / i4_s2 for j975, j980, j985, j990.
- P3 a_980(t, o, 1) >= 0.85 on the six (task, order) cells. P4 query-swap pair at k = 1 >= 0.80 for j980.
- P5 blind <= 0.32 for j980 and j975.
- P6 reference = mean(j520, j663) at s16 (two distinct files named j520 then j663, both recorded (16, 16)): CUR (j980)
  gsm8kx and gsm8kv2 >= reference - 3.0 points, K1X (j990) >= reference - 5.0. valx and gsm1kx INFO.
- P7 wall j975, j985 <= 2.0 h; j980, j990 <= 0.9 h. P7c measured pilot total (3 smokes + 4 jobs + every retry wall)
  <= 6.0 GPU-h. P7b projected confirmation 4 x (975 + 980 + 985 + 990) <= 25 GPU-h (printed; never fails under P7).
- P8 smoke_975, smoke_985, smoke_995, j975, j980, j985, j990 landed with "== exit 0"; the smoke_995 log has the
  "== weights ... 663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tar -> models/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp/model" line.
- P9 a_975(ca, fwd, 3) >= 0.78, a_980(ca, fwd, 2) >= 0.85, a_980(ca, fwd, 3) >= 0.78; when j975 passes and j980 fails
  the failing line reads "P9 (installed, then lost at s16)".
- P10 a_980(ca, fwd, 3) - a_990(ca, fwd, 3) >= 0.15.
- P11 (I7 at n = 1) the j985 loss mean of the step lines with step >= 760 <= 1.0 (at least 5 lines, all finite), and
  a_985(t, o, 1) >= 0.97 on each of the six (task, order) cells.
No gate line reads rc1..rc4 or any hint set of the 249 primary items, and the gate prints none of them. INFO lines: the
curves, timing, peak memory, tar sizes. Output ends with "GATE PASS" or "GATE FAIL: <lines>".
"""
import argparse, hashlib, json, math, os, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402
import khop_pilot_gate as G  # noqa: E402  (wall_hours, exit_code, gsm_sets)
import khop2_pilot_gate as G2  # noqa: E402  (PARENT_MD5)
import khop3_table as K3  # noqa: E402

E = os.path.dirname(HERE)
F = Fraction
BAR = {"P3": F("0.85"), "P4": F("0.80"), "P5": F("0.32"), "P6_cur": F(3, 100), "P6_k1x": F(5, 100), "P7_s1": F(2),
       "P7_s2": F("0.9"), "P7c": F(6), "P7b": F(25), "P9_k3": F("0.78"), "P9_k2": F("0.85"), "P10": F("0.15"),
       "P11_loss": F(1), "P11_k1": F("0.97")}
KD27_MD5, VAL_MD5 = "92545dd42ca451c39e0e7c104aa78d1e", "9f55fec541246e41401a7b5fa6c5510a"
K1X_CHECKS = ("statement_mismatch", "query_not_hop1", "gold_not_hop1_value", "option_rule_fail", "vb_options_not_roots",
              "row_rule_errors", "teacher_not_onehot", "keyset_differs", "id_group_depth_differs")
SHARE_CELLS = ["all"] + [f"{t}/k{k}" for t in ("ca", "vb", "wp") for k in range(1, 7)]
DIRS = ("student_khop3_c27", "student_khop3_k1x")
W663 = "663_4b_gsmvt_A2-c1lr3ch_s16_fsdp"
REF_JOBS = ("520", "663")
PILOT = {"975": "CUR1", "980": "CUR", "985": "K1X1", "990": "K1X"}
SMOKES = (975, 985, 995)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def p0(a):
    """(ok, detail) of P0."""
    bad = []
    rep = json.load(open(a.report)) if os.path.exists(a.report) else None
    rep2 = json.load(open(a.khop2_report)) if os.path.exists(a.khop2_report) else None
    prep = json.load(open(a.parent_report)) if os.path.exists(a.parent_report) else None
    if rep is None or rep2 is None or prep is None:
        return False, f"report missing (khop3 {rep is not None}, khop2 {rep2 is not None}, khop {prep is not None})"
    # D1: files on disk
    kd = os.path.join(a.khop2_dir, "train_kd27.jsonl")
    if not os.path.exists(kd) or md5(kd) != KD27_MD5 or (rep2.get("md5") or {}).get("train_kd27.jsonl") != KD27_MD5:
        bad.append("train_kd27 md5")
    for f, want in G2.PARENT_MD5.items():
        p = os.path.join(a.khop_dir, f + ".jsonl")
        if not os.path.exists(p) or md5(p) != want or (prep.get("md5") or {}).get(f + ".jsonl") != want or \
                (rep.get("parent_md5") or {}).get(f + ".jsonl") != want:
            bad.append(f"parent {f}")
    rmd5 = rep.get("md5") or {}
    want_files = {f"{d}/{f}" for d in DIRS for f in ("train.jsonl", "val.jsonl")}
    if set(rmd5) != want_files:
        bad.append(f"report md5 keys {sorted(rmd5)}")
    for f in sorted(want_files):
        p = os.path.join(a.data_root, f)
        if not os.path.exists(p) or md5(p) != rmd5.get(f):
            bad.append(f"{f} md5 differs from the report")
    if rmd5.get("student_khop3_c27/train.jsonl") != KD27_MD5:
        bad.append("c27 train md5 != train_kd27")
    if any(rmd5.get(f"{d}/val.jsonl") != VAL_MD5 for d in DIRS):
        bad.append("val md5")
    for d in DIRS:
        p = os.path.join(a.data_root, d)
        if not os.path.isdir(p) or sorted(os.listdir(p)) != ["train.jsonl", "val.jsonl"]:
            bad.append(f"{d} file set {sorted(os.listdir(p)) if os.path.isdir(p) else 'missing'}")
    # D2: report lines
    rows = rep.get("rows") or {}
    rp = rep.get("replay") or {}
    ck = rep.get("k1x_checks") or {}
    tt = rep.get("token_totals") or {}
    if rep.get("d_pass") is not True or rep.get("d_fail"):
        bad.append(f"d_pass {rep.get('d_pass')} d_fail {rep.get('d_fail')}")
    if rows.get("student_khop3_k1x/train.jsonl") != 27000 or rows.get("student_khop3_c27/train.jsonl") != 27000:
        bad.append(f"rows {rows}")
    if (rp.get("kd27_rows_equal"), rp.get("kd27_rows"), rp.get("k1_rows_equal"), rp.get("k1_rows")) != (27000, 27000,
                                                                                                         3000, 3000):
        bad.append(f"replay {rp}")
    if rep.get("k1x_rows_0_2999_equal_train_k1") is not True:
        bad.append("K1X rows 0..2,999 != train_k1")
    for x in K1X_CHECKS:
        if ck.get(x) != 0:
            bad.append(f"{x} {ck.get(x)!r}")
    if rep.get("eval_hash_overlap") != {"k1x": 0, "c27": 0}:
        bad.append(f"eval hash overlap {rep.get('eval_hash_overlap')}")
    if rep.get("item_ids_distinct") is not True:
        bad.append("item_ids")
    sh = ck.get("gold_shares") or {}
    for c in SHARE_CELLS:
        v = sh.get(c)
        if not isinstance(v, dict) or sorted(v) != list("ABCDE") or \
                not all(K3.finite(x) and F("0.18") <= F(str(x)) <= F("0.22") for x in v.values()):
            bad.append(f"gold share {c} {v}")
    if set(sh) != set(SHARE_CELLS):
        bad.append(f"gold share cells {sorted(sh)}")
    n_k, n_x = tt.get("kd27"), tt.get("k1x")
    if not (isinstance(n_k, int) and isinstance(n_x, int) and n_k > 0 and abs(n_x - n_k) * 100 <= n_k
            and tt.get("unit") == "tokens"):
        bad.append(f"token totals {tt}")
    return not bad, ("D1 / D2 ok" if not bad else "; ".join(bad[:8]))


def read_retry(path):
    """[(node, hours as an exact Fraction of the decimal)]; SystemExit when the file is absent (the gate refuses to run),
    None when a line is malformed (P7c fails)."""
    if not path or not os.path.exists(path):
        raise SystemExit(f"khop3_pilot_gate: --retry-walls <file> is required (empty when no node retried): {path!r}")
    out = []
    for l in open(path).read().splitlines():
        if not l.strip():
            continue
        p = l.split()
        try:
            h = F(p[1]) if len(p) == 2 and re.match(r"^[0-9]+(\.[0-9]+)?$", p[1]) else None
        except (ValueError, ZeroDivisionError):
            h = None
        if h is None or not re.match(r"^[A-Za-z0-9_.:-]+$", p[0]):
            return None
        out.append((p[0], h))
    return out


def wall(log):
    """The wall of a log in hours, exact: G.wall_hours is a whole number of seconds over 3600 (None when missing)."""
    h = G.wall_hours(log)
    return None if h is None else F(round(h * 3600), 3600)


def gate(a):
    lines, fails, info = [], [], []

    def line(name, ok, detail, label=None):
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fails.append(label or name)
    retry = read_retry(a.retry_walls)
    khop = T.load_khop(a.khop_dir)
    khop_ok = all(len(khop.get(s, [])) == T.ROWS[s] for s in T.SETS)
    ok, det = p0(a)
    line("P0", ok, det)
    paths = {"975": a.cur1, "980": a.cur, "985": a.k1x1, "990": a.k1x}
    runs = {}
    for j, p in paths.items():
        try:
            runs[j] = K3.load(p, f"j{j}", K3.S1_NEED if PILOT[j] in K3.S1_DATA else K3.s2_need(a.env663))
        except SystemExit as e:
            runs[j] = None
            info.append(f"INFO j{j} not loaded: {e}")
    rseed = lambda r: tuple(((r.get("metrics") or {}).get("args") or {}).get(k) for k in ("seed", "order_seed"))
    want_ids = {s: {x["item_id"] for x in khop.get(s, [])} for s in T.SETS}
    gsm = G.gsm_sets(a.env663)
    env_ok = len(gsm) > len(G.GSM_BASE)
    # P1
    for j, r in runs.items():
        if r is None:
            line(f"P1 j{j}", False, f"tarball {paths[j]} missing or unreadable")
            continue
        miss = [s for s in T.SETS if not khop_ok or set(r["preds"].get(s, {})) != want_ids[s]]
        have = set(r.get("all_preds", []))
        need = list(K3.PLAIN) if PILOT[j] in K3.S1_DATA else gsm
        gm = [g for g in need if g not in have]
        ok = not miss and not gm and rseed(r) == (16, 16) and (env_ok or PILOT[j] in K3.S1_DATA)
        line(f"P1 j{j}", ok, f"{PILOT[j]}: seed / order_seed {rseed(r)} (need (16, 16)); khop sets "
             f"{9 - len(miss)}/9 with the eval ids (off {miss[:3]}); {'plain' if PILOT[j] in K3.S1_DATA else 'GSM / hint'} "
             f"sets {len(need) - len(gm)}/{len(need)} (missing {gm[:5]})" + ("" if env_ok else " env663 has no EXTRA_EVALS"))
    ld = lambda j: runs.get(j) is not None
    # P2
    for j in PILOT:
        if not ld(j):
            line(f"P2 j{j}", False, "no run")
            continue
        try:
            if PILOT[j] in K3.S1_DATA:
                ok, det = K3.i4_s1(runs[j], PILOT[j], 16)
            else:
                w = K3.run_name(runs[str(int(j) - 5)]) if ld(str(int(j) - 5)) else None
                ok, det = K3.i4_s2(runs[j], w, 16) if w else (False, "its S1 run is missing")
        except SystemExit as e:
            ok, det = False, str(e)
        line(f"P2 j{j}", ok, det)

    def cell(j, s, k):
        """Exact a_j(s, k) or None (missing run / partial preds)."""
        if not ld(j) or not khop_ok:
            return None
        ids = [r["item_id"] for r in khop[s] if r["k"] == k]
        p = runs[j]["preds"].get(s) or {}
        if not ids or any(i not in p for i in ids):
            return None
        return F(sum(int(p[i][0]) for i in ids), len(ids))
    pv = lambda x: "MISSING" if x is None else f"{100 * float(x):.1f}"
    ge = lambda x, b: x is not None and x >= b
    # P3
    v3 = {(t, o): cell("980", f"khop_{t}_{o}", 1) for t, o in T.CELLS6}
    line("P3", all(ge(v, BAR["P3"]) for v in v3.values()), " ".join(f"{t}/{o} {pv(v)}" for (t, o), v in v3.items())
         + " (bar 85.0)")
    # P4
    try:
        q4 = K3.K2.xqs_pair(runs["980"], khop, "j980 qs pair") if ld("980") and khop_ok else None
    except SystemExit:
        q4 = None
    line("P4", ge(q4, BAR["P4"]), f"j980 query-swap pair k=1 {pv(q4)} (bar 80.0)")
    # P5
    bl = {}
    for j in ("980", "975"):
        try:
            bl[j] = K3.K2.xacc(runs[j], "khop_blind", [r["item_id"] for r in khop["khop_blind"]], f"j{j} blind") \
                if ld(j) and khop_ok else None
        except SystemExit:
            bl[j] = None
    line("P5", all(v is not None and v <= BAR["P5"] for v in bl.values()),
         " ".join(f"j{j} {pv(v)}" for j, v in bl.items()) + " (bar <= 32.0)")
    # P6
    try:
        lab, rp = a.ref.split("=", 1)
        rl = rp.split(",")
    except ValueError:
        lab, rl = None, []
    ids_ok = (lab == "s16" and len(rl) == 2 and len({os.path.realpath(p) for p in rl}) == 2
              and all(re.search(rf"(^|_){j}(_|\.tgz$)", os.path.basename(p)) for j, p in zip(REF_JOBS, rl)))
    refs = []
    for p in rl:
        try:
            refs.append(K3.load(p, "P6 reference"))
        except SystemExit:
            refs.append(None)
    ref_ok = ids_ok and len(refs) == 2 and all(r is not None and rseed(r) == (16, 16) for r in refs)
    det = [f"refs {lab} {[os.path.basename(p) for p in rl]} (need j520 then j663, distinct, seeds (16, 16)): "
           f"{'ok' if ref_ok else 'NO'}"]
    ok6 = ref_ok
    for s in K3.PLAIN:
        # the item ids of j520's file are the canonical set: j663 and the pilots must hold exactly these PLAIN_N ids
        canon = {i for i, _ in refs[0]["plain"].get(s, [])} if refs and refs[0] else set()
        rv = []
        for r in refs:
            try:
                rv.append(K3.xplain(r, s, canon, "P6 reference") if r and canon else None)
            except SystemExit:
                rv.append(None)
        ref = (rv[0] + rv[1]) / 2 if len(rv) == 2 and None not in rv else None
        for j, b in (("980", BAR["P6_cur"]), ("990", BAR["P6_k1x"])):
            try:
                v = K3.xplain(runs[j], s, canon, f"j{j}") if ld(j) and canon else None
            except SystemExit:
                v = None
            if s in ("gsm8kx", "gsm8kv2"):
                ok = v is not None and ref is not None and v >= ref - b
                ok6 &= ok
                det.append(f"j{j} {s} {pv(v)} (reference mean(j520, j663) {pv(ref)}, bar {pv(None if ref is None else ref - b)})")
            else:
                info.append(f"INFO P6 j{j} {s} {pv(v)} (reference {pv(ref)})")
    line("P6", ok6, " ".join(det))
    # P7, P7c, P7b
    wh = {j: (wall(runs[j]["log"]) if ld(j) else None) for j in PILOT}
    bars7 = {"975": BAR["P7_s1"], "985": BAR["P7_s1"], "980": BAR["P7_s2"], "990": BAR["P7_s2"]}
    line("P7", all(wh[j] is not None and wh[j] <= bars7[j] for j in PILOT),
         " ".join(f"j{j} {'MISSING' if wh[j] is None else '%.2f h' % wh[j]} (bar {float(bars7[j])})" for j in PILOT))
    sm = {}
    for n, p in zip(SMOKES, (a.smokes or "").split(",") if a.smokes else []):
        sm[n] = p
    smr = {}
    for n in SMOKES:
        p = sm.get(n)
        ok_name = bool(p) and re.match(rf"^results_smoke_{n}_", os.path.basename(p)) is not None
        smr[n] = None
        if ok_name and os.path.exists(p):
            try:
                smr[n] = T.load_run(p)
            except Exception:
                smr[n] = None
    sw = {n: (wall(r["log"]) if r else None) for n, r in smr.items()}
    if retry is None:
        line("P7c", False, f"retry-walls file {a.retry_walls} has a malformed line (need '<node> <hours>')")
    else:
        parts = list(wh.values()) + list(sw.values())
        okc = None not in parts and sum(parts) + sum(h for _, h in retry) <= BAR["P7c"]
        tot = sum(x for x in parts if x is not None) + sum(h for _, h in retry)
        line("P7c", okc, f"smokes {' '.join('-' if v is None else '%.2f' % v for v in sw.values())} + jobs "
             f"{' '.join('-' if v is None else '%.2f' % v for v in wh.values())} + retries {len(retry)} "
             f"({float(sum(h for _, h in retry)):.2f} h) = {float(tot):.2f} GPU-h (cap {float(BAR['P7c'])}"
             + ("" if None not in parts else "; a wall is MISSING") + ")")
    if None in wh.values():
        line("P7b", False, "a pilot wall is missing")
    else:
        proj = 4 * sum(wh.values())
        line("P7b", proj <= BAR["P7b"], f"projected confirmation 4 x {float(sum(wh.values())):.2f} = {float(proj):.2f} "
             "GPU-h (cap 25)")
    # P8
    ex = {f"smoke_{n}": (G.exit_code(r["log"]) if r else None) for n, r in smr.items()}
    ex.update({f"j{j}": (G.exit_code(runs[j]["log"]) if ld(j) else None) for j in PILOT})
    w995 = len(K3.weights_import(smr[995]["log"], W663)) == 1 if smr.get(995) else False
    line("P8", all(v == 0 for v in ex.values()) and w995,
         " ".join(f"{k} exit {v}" for k, v in ex.items()) + f"; smoke_995 weights line {'ok' if w995 else 'MISSING'}")
    # P9
    a975, a980_2, a980_3 = cell("975", "khop_ca_fwd", 3), cell("980", "khop_ca_fwd", 2), cell("980", "khop_ca_fwd", 3)
    ok975 = ge(a975, BAR["P9_k3"])
    ok980 = ge(a980_2, BAR["P9_k2"]) and ge(a980_3, BAR["P9_k3"])
    lost = ok975 and not ok980
    line("P9", ok975 and ok980, f"a_975(ca, fwd, 3) {pv(a975)} (bar 78.0), a_980(ca, fwd, 2) {pv(a980_2)} (bar 85.0), "
         f"a_980(ca, fwd, 3) {pv(a980_3)} (bar 78.0)" + (" (installed, then lost at s16)" if lost else ""),
         label="P9 (installed, then lost at s16)" if lost else None)
    # P10
    a990_3 = cell("990", "khop_ca_fwd", 3)
    d10 = None if None in (a980_3, a990_3) else a980_3 - a990_3
    line("P10", ge(d10, BAR["P10"]), f"a_980(ca, fwd, 3) - a_990(ca, fwd, 3) {pv(d10)} (bar 15.0)")
    # P11
    lm = None
    if ld("985"):
        try:
            lm = K3.i7_loss(runs["985"]["log"])
        except SystemExit as e:
            info.append(f"INFO P11 loss lines unreadable: {e}")
    k1 = {(t, o): cell("985", f"khop_{t}_{o}", 1) for t, o in T.CELLS6}
    ok11 = lm is not None and lm[0] <= BAR["P11_loss"] and all(ge(v, BAR["P11_k1"]) for v in k1.values())
    line("P11", ok11, f"j985 loss mean (step >= 760) {'MISSING' if lm is None else '%.3f (n %d)' % (float(lm[0]), lm[1])}"
         " (bar <= 1.0); k=1 cells " + " ".join(f"{t}/{o} {pv(v)}" for (t, o), v in k1.items()) + " (bar 97.0)")
    # INFO (no rc / hint line)
    for j in PILOT:
        if not ld(j):
            continue
        for s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_vb_fwd", "khop_cag_fwd"):
            info.append(f"INFO {PILOT[j]:4s} {s:13s} " + " ".join(f"k{k} {pv(cell(j, s, k))}" for k in range(1, 9)))
        h = (runs[j]["metrics"] or {}).get("history") or []
        sps = [x.get("s_per_step") for x in h if K3.finite(x.get("s_per_step"))]
        tar = K3.weights_export(runs[j]["log"], K3.run_name(runs[j]) or "") if PILOT[j] in K3.S1_DATA else []
        info.append(f"INFO j{j} wall {'-' if wh[j] is None else '%.2f' % wh[j]} h, s_per_step "
                    f"{'-' if not sps else '%.2f' % (sum(sps) / len(sps))}, peak_gib "
                    f"{max([x.get('peak_gib') or 0 for x in h] + [0]):.1f}" + (f", weights tar {tar[0]}" if tar else ""))
    return lines, fails, info


def final_line(fails):
    return "GATE FAIL: " + "; ".join(fails) if fails else "GATE PASS"      # "; ": the P9 label holds a comma


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cur1", required=True, help="j975 tarball")
    ap.add_argument("--cur", required=True, help="j980 tarball")
    ap.add_argument("--k1x1", required=True, help="j985 tarball")
    ap.add_argument("--k1x", required=True, help="j990 tarball")
    ap.add_argument("--smokes", required=True, help="<smoke_975 tgz>,<smoke_985 tgz>,<smoke_995 tgz>")
    ap.add_argument("--ref", required=True, help="s16=<j520 tgz>,<j663 tgz>")
    ap.add_argument("--retry-walls", required=True)
    ap.add_argument("--report", default=os.path.join(E, "src/data/eval/khop3/build_report.json"))
    ap.add_argument("--khop2-report", default=os.path.join(E, "src/data/eval/khop2/build_report.json"))
    ap.add_argument("--parent-report", default=os.path.join(E, "src/data/eval/khop/build_report.json"))
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--khop2-dir", default=os.path.join(E, "src/data/eval/khop2"))
    ap.add_argument("--data-root", default=os.path.join(E, "src/data"))
    ap.add_argument("--env663", default=G.ENV663)
    a = ap.parse_args(argv)
    lines, fails, info = gate(a)
    for l in lines + info:
        print(l)
    print(final_line(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
