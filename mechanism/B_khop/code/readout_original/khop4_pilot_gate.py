#!/usr/bin/env python3
"""CARD_khop_v4 pilot gate (section 3, seed 16). Deterministic; exit 0 only for "GATE PASS ARM R" / "GATE PASS ARM L".

    python3 -I khop4_pilot_gate.py --s1 <975 tgz> <985 tgz> --curl <1025 tgz> --k1xl <1026 tgz> --curr <1027 tgz> \\
        --k1xr <1028 tgz> --smokes <smoke_1025 tgz>,<smoke_1026 tgz>,<smoke_1027 tgz>,<smoke_1028 tgz> \\
        --ref s16=<E>/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,<E>/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz \\
        --recipe-ref <980 tgz> <990 tgz> <756 tgz> <757 tgz> --retry-walls <file> \\
        [--report <E>/src/data/eval/khop4/build_report.json --khop3-report ... --khop2-report ... --parent-report ...]

--retry-walls is required: one "<node> <hours>" line per failed execution of a smoke or pilot node (from the job event
log logs/train_x1khop4w_4b.log on the AP), an empty file when nothing retried. The gate refuses to run without it.
Common block (exactly 7 lines): C0 (data D1 / D2), C1 j975, C1 j985 (khop3's P1 and P2 for the S1 pair and the tarball
md5s of round G), C2 (a_975(ca, fwd, 3) >= 120 / 150), C3 (khop3 P11 on j985), C8 (smokes and pilot exits), C7c
(measured pilot total <= 6.0 GPU-h). Per arm a in {L, R} (CUR-L = j1025, K1X-L = j1026, CUR-R = j1027, K1X-R =
j1028), exactly 10 decision lines a-Q1 ... a-Q6, a-Q7b, a-Q9, a-Q10, a-Q11 and one INFO line a-Q7 (wall and GPU type,
never PASS / FAIL). Both arms are always evaluated and printed. Last line (card 0.5): common lines all PASS and R
decision lines all PASS -> "GATE PASS ARM R"; else common all PASS and L all PASS -> "GATE PASS ARM L"; else
"GATE FAIL: <failing lines>" ("; " separated). Run integrity (Amendment 3): when any of R-Q1, R-Q2, L-Q1, L-Q2 FAILs,
or an a-Q7b line FAILs on a missing wall ("a wall is missing"), that arm is not run as registered; the gate prints
"READING  not run as registered (arm a): ..." and the last line is "GATE FAIL: ..." whatever the other arm gives
(card 0.5: "not run as registered" -> GATE FAIL, no confirmation, rescue under the same id). Both arms are still
evaluated and printed. The 0.5 failure readings are printed as "READING" lines, and only for arms run as registered
(the both-arms reading only when both are). Walls are zone-aware (khop4_gpuh.wall: CDT = UTC-5, CST = UTC-6).
Every line fails closed on a missing, partial, non-object or non-finite input; every accuracy is an exact count
(Fraction(correct, items)) and every bar a rational. No gate line reads rc1..rc4 or any hint set of the 249 primary
items, and the gate prints none of them. It imports khop3_pilot_gate.py, khop3_table.py and khop4_table.py and edits
none of them.
"""
import argparse, hashlib, json, os, re, sys, tarfile, zlib
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402
import khop_pilot_gate as G  # noqa: E402  (exit_code, gsm_sets, ENV663)
import khop2_pilot_gate as G2  # noqa: E402  (PARENT_MD5)
import khop3_table as K3  # noqa: E402
import khop3_pilot_gate as GT3  # noqa: E402  (wall, md5)
import khop4_table as K4  # noqa: E402
import khop4_gpuh as GH  # noqa: E402  (zone-aware wall)

E = os.path.dirname(HERE)
F = Fraction
BAR = {"C2": F(120, 150), "C3_loss": F(1), "C3_k1": F("0.97"), "C7c": F(6), "Q3": F("0.85"), "Q4": F("0.80"),
       "Q5": F("0.32"), "Q6_cur": F(3, 100), "Q6_k1x": F(5, 100), "Q7b": F(23), "Q9_k2": F(128, 150),
       "Q9_k3": F(120, 150), "Q10": F(15, 100), "Q11": F("0.85")}
Q7_BAR = {"L": F("0.9"), "R": F("1.1")}               # INFO only
S1_MD5 = {"975": "453e235627b7455143b85b6460b84a5a", "985": "30a47caa409e67014527471726f56c69"}
S1_NAME = {"975": "975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp", "985": "985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp"}
S1_KIND = {"975": "CUR", "985": "K1X"}
PILOT_IDS = (1025, 1026, 1027, 1028)
JOBS = {("L", "CUR"): 1025, ("L", "K1X"): 1026, ("R", "CUR"): 1027, ("R", "K1X"): 1028}
OUT_MD5 = {"student_khop4_c27/train.jsonl": "92545dd42ca451c39e0e7c104aa78d1e",
           "student_khop4_c27/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a",
           "student_khop4_k1x/train.jsonl": "d7cc9716e5785461d16ec92156104261",
           "student_khop4_k1x/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a"}
IN_MD5 = {"student_khop3_c27/train.jsonl": "92545dd42ca451c39e0e7c104aa78d1e",
          "student_khop3_c27/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a",
          "student_khop3_k1x/train.jsonl": "d7cc9716e5785461d16ec92156104261",
          "student_khop3_k1x/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a",
          "eval/khop/train_kh.jsonl": "a16b1520f0fb8b24ffd8fa57ef696de0",
          "eval/khop/train_k1.jsonl": "4bb15f66c8d8843013eb0410c6a87e48"}
REPLAY_KEYS = ("train_kh_equals_rows_0_2999_of_student_khop3_c27", "train_k1_equals_rows_0_2999_of_student_khop3_k1x",
               "train_kh_equals_rows_0_2999_of_student_khop4_c27", "train_k1_equals_rows_0_2999_of_student_khop4_k1x")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def jload(p):
    """A JSON object from p, or None (missing, unparsable or not an object)."""
    try:
        v = json.load(open(p))
    except (OSError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def head3000(p):
    out = []
    with open(p, "rb") as f:
        for _ in range(3000):
            line = f.readline()
            if not line:
                break
            out.append(line)
    return b"".join(out)


def c0(a):
    """(ok, detail) of C0: D1 (the four student_khop4 files have the card md5s, equal to the khop4 report; each dir
    holds exactly train.jsonl and val.jsonl; the 6 inputs have the card md5s on disk, in the khop4 report and in the
    khop3 / khop / khop2 reports; the 12 parent khop files equal the khop and khop2 reports) and D2 (the khop4 report's
    replay lines all true, d_pass, no d_fail, inputs unchanged; and recomputed here: train_kh = rows 0..2,999 of
    student_khop4_c27/train.jsonl and train_k1 = rows 0..2,999 of student_khop4_k1x/train.jsonl, byte for byte)."""
    bad = []
    rep, rep3, rep2, repk = (jload(p) for p in (a.report, a.khop3_report, a.khop2_report, a.parent_report))
    if None in (rep, rep3, rep2, repk):
        return False, (f"report missing or not an object (khop4 {rep is not None}, khop3 {rep3 is not None}, khop2 "
                       f"{rep2 is not None}, khop {repk is not None})")
    rm = rep.get("md5") if isinstance(rep.get("md5"), dict) else {}
    if set(rm) != set(OUT_MD5):
        bad.append(f"khop4 report md5 keys {sorted(rm)}")
    for rel, want in OUT_MD5.items():
        p = os.path.join(a.data_root, rel)
        if not os.path.isfile(p) or md5(p) != want or rm.get(rel) != want:
            bad.append(f"{rel} md5 (card {want[:8]}, report {str(rm.get(rel))[:8]})")
    for d in ("student_khop4_c27", "student_khop4_k1x"):
        p = os.path.join(a.data_root, d)
        fs = sorted(os.listdir(p)) if os.path.isdir(p) else None
        if fs != ["train.jsonl", "val.jsonl"]:
            bad.append(f"{d} file set {fs}")
    ins = rep.get("inputs") if isinstance(rep.get("inputs"), dict) else {}
    m3 = rep3.get("md5") if isinstance(rep3.get("md5"), dict) else {}
    mk = repk.get("md5") if isinstance(repk.get("md5"), dict) else {}
    pm2 = rep2.get("parent_md5") if isinstance(rep2.get("parent_md5"), dict) else {}
    for rel, want in IN_MD5.items():
        p = os.path.join(a.data_root, rel)
        got = (ins.get(rel) or {}).get("md5") if isinstance(ins.get(rel), dict) else None
        par = m3.get(rel) if rel.startswith("student_") else mk.get(rel.split("/")[-1])
        if not os.path.isfile(p) or md5(p) != want or got != want or par != want:
            bad.append(f"input {rel} md5")
    for f, want in G2.PARENT_MD5.items():
        p = os.path.join(a.khop_dir, f + ".jsonl")
        if not os.path.isfile(p) or md5(p) != want or mk.get(f + ".jsonl") != want or pm2.get(f + ".jsonl") != want:
            bad.append(f"parent {f}")
    rp = rep.get("replay") if isinstance(rep.get("replay"), dict) else {}
    if sorted(rp) != sorted(REPLAY_KEYS) or not all(rp.get(k) is True for k in REPLAY_KEYS):
        bad.append(f"replay lines {rp}")
    if rep.get("d_pass") is not True or rep.get("d_fail") != [] or rep.get("inputs_unchanged") is not True:
        bad.append(f"d_pass {rep.get('d_pass')} d_fail {rep.get('d_fail')} inputs_unchanged {rep.get('inputs_unchanged')}")
    if rep.get("card") != "CARD_khop_v4":
        bad.append(f"card {rep.get('card')}")
    for rel, dst in (("eval/khop/train_kh.jsonl", "student_khop4_c27/train.jsonl"),
                     ("eval/khop/train_k1.jsonl", "student_khop4_k1x/train.jsonl")):
        p, q = os.path.join(a.data_root, rel), os.path.join(a.data_root, dst)
        if not (os.path.isfile(p) and os.path.isfile(q) and open(p, "rb").read() == head3000(q)):
            bad.append(f"D2 {rel} != rows 0..2,999 of {dst}")
    return not bad, ("D1 / D2 ok" if not bad else "; ".join(bad[:8]))


def read_retry(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"khop4_pilot_gate: --retry-walls <file> is required (empty when no node retried): {path!r}")
    return GT3.read_retry(path)          # [(node, Fraction hours)] or None for a malformed line


def read_log(path):
    """The logs/<name>.out text of a results tarball, read on its own (so the exit and wall lines of a landed job do not
    depend on its preds or metrics.json, which a-Q1 / a-Q2 check); None when the tarball or the log is missing or
    unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with tarfile.open(path) as t:
            ms = [m for m in t.getmembers() if m.isfile() and re.search(r"(?:^|/)logs/[^/]+\.out$", m.name)]
            if len(ms) != 1:
                return None
            return t.extractfile(ms[0]).read().decode("utf-8", "replace")
    except (OSError, tarfile.TarError, EOFError, zlib.error):    # EOFError: a truncated .tgz (fix round, F1 tests)
        return None


def smoke_ok(run, w):
    """(ok, detail): the smoke log ends with exit 0, has 'smoke ok A2' and exactly one weights import line of w."""
    if run is None:
        return False, "missing"
    log = run["log"]
    ex, ok_line, wl = G.exit_code(log), bool(re.search(r"^smoke ok A2$", log, re.M)), len(K3.weights_import(log, w))
    return ex == 0 and ok_line and wl == 1, f"exit {ex} smoke-ok {'yes' if ok_line else 'no'} weights {w[:4]} {wl}"


def gate(a):
    lines, fails, info, readings = [], [], [], []
    res = {}

    def line(name, ok, detail):
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        res[name] = ok
        if not ok:
            fails.append(name)
    retry = read_retry(a.retry_walls)
    khop = T.load_khop(a.khop_dir)
    khop_ok = all(len(khop.get(s, [])) == T.ROWS[s] for s in T.SETS)
    s2need = K3.s2_need(a.env663)
    gsm = G.gsm_sets(a.env663)
    env_ok = len(gsm) > len(G.GSM_BASE)

    def load(p, what, need):
        try:
            return K4.load(p, what, need), None
        except SystemExit as e:
            info.append(f"INFO {what} not loaded: {e}")
            return None, str(e)
    s1p = dict(zip(("975", "985"), a.s1 or []))
    s1 = {j: load(s1p.get(j), f"j{j}", K3.S1_NEED)[0] for j in ("975", "985")}
    pp = {1025: a.curl, 1026: a.k1xl, 1027: a.curr, 1028: a.k1xr}
    pil = {j: load(p, f"j{j}", s2need)[0] for j, p in pp.items()}
    rr = dict(zip(("980", "990", "756", "757"), a.recipe_ref or []))
    rref = {}
    for j in ("980", "990", "756", "757"):
        p = rr.get(j)
        rref[j] = load(p, f"recipe reference j{j}", s2need)[0] if p and K4.has_id(p, j) else None
        if p and not K4.has_id(p, j):
            info.append(f"INFO recipe reference {p} does not name j{j}")
    want_ids = {s: {x["item_id"] for x in khop.get(s, [])} for s in T.SETS}
    rseed = lambda r: tuple(r["metrics"]["args"].get(k) for k in ("seed", "order_seed"))
    pv = lambda x: "MISSING" if x is None else f"{100 * float(x):.1f}"
    ge = lambda x, b: x is not None and x >= b

    def cell(r, s, k):
        """Exact a(s, k) of run r or None (missing run / partial preds)."""
        if r is None or not khop_ok:
            return None
        ids = [x["item_id"] for x in khop[s] if x["k"] == k]
        p = r["preds"].get(s) or {}
        if not ids or any(i not in p for i in ids):
            return None
        return F(sum(int(p[i][0]) for i in ids), len(ids))

    def complete(r, s1_job):
        """(ok, detail) of P1 / a-Q1 completeness: the nine khop sets with the eval ids and the plain (S1) or GSM / hint
        (S2, preds_subq included) sets present."""
        if r is None:
            return False, "tarball missing or unreadable (see INFO)"
        miss = [s for s in T.SETS if not khop_ok or set(r["preds"].get(s, {})) != want_ids[s]]
        have = set(r.get("all_preds", []))
        need = list(K3.PLAIN) if s1_job else gsm
        gm = [g for g in need if g not in have]
        return (not miss and not gm and (env_ok or s1_job),
                f"khop sets {9 - len(miss)}/9 with the eval ids (off {miss[:3]}); {'plain' if s1_job else 'GSM / hint'} "
                f"sets {len(need) - len(gm)}/{len(need)} (missing {gm[:5]})" + ("" if env_ok else " env663 has no EXTRA_EVALS"))
    # --- common block
    ok, det = c0(a)
    line("C0", ok, det)
    for j in ("975", "985"):
        r, p = s1[j], s1p.get(j)
        if r is None:
            line(f"C1 j{j}", False, f"tarball {p} missing or unreadable")
            continue
        okc, detc = complete(r, True)
        try:
            ok2, det2 = K4.i4_s1(r, S1_KIND[j], 16)
        except SystemExit as e:
            ok2, det2 = False, str(e)
        hm = md5(p)
        okn = os.path.basename(p) == f"results_{S1_NAME[j]}.tgz"
        line(f"C1 j{j}", okc and ok2 and rseed(r) == (16, 16) and hm == S1_MD5[j] and okn,
             f"P1: {detc}; seed / order_seed {rseed(r)}; P2: {det2}; tarball md5 {hm[:8]} (round G {S1_MD5[j][:8]})"
             + ("" if okn else f"; name {os.path.basename(p)} is not results_{S1_NAME[j]}.tgz"))
    a975 = cell(s1["975"], "khop_ca_fwd", 3)
    c2 = ge(a975, BAR["C2"])
    line("C2", c2, f"a_975(ca, fwd, 3) {pv(a975)} (bar 120 / 150 = 80.0)")
    lm = None
    if s1["985"] is not None:
        try:
            lm = K3.i7_loss(s1["985"]["log"])
        except SystemExit as e:
            info.append(f"INFO C3 loss lines unreadable: {e}")
    k1 = {(t, o): cell(s1["985"], f"khop_{t}_{o}", 1) for t, o in T.CELLS6}
    line("C3", lm is not None and lm[0] <= BAR["C3_loss"] and all(ge(v, BAR["C3_k1"]) for v in k1.values()),
         f"j985 loss mean (step >= 760) {'MISSING' if lm is None else '%.3f (n %d)' % (float(lm[0]), lm[1])} (bar <= 1.0); "
         "k=1 cells " + " ".join(f"{t}/{o} {pv(v)}" for (t, o), v in k1.items()) + " (bar 97.0)")
    smp = (a.smokes or "").split(",") if a.smokes else []
    smr = {}
    for i, j in enumerate(PILOT_IDS):
        p = smp[i] if i < len(smp) else None
        smr[j] = None
        if p and re.match(rf"^results_smoke_{j}_", os.path.basename(p)) and os.path.exists(p):
            try:
                smr[j] = T.load_run(p)
            except Exception as e:
                info.append(f"INFO smoke_{j} not loaded: {e}")
    w_of = {1025: S1_NAME["975"], 1026: S1_NAME["985"], 1027: S1_NAME["975"], 1028: S1_NAME["985"]}
    sm = {j: smoke_ok(smr[j], w_of[j]) for j in PILOT_IDS}
    plog = {j: read_log(pp[j]) for j in PILOT_IDS}
    ex = {j: (G.exit_code(plog[j]) if plog[j] else None) for j in PILOT_IDS}
    line("C8", all(v[0] for v in sm.values()) and all(v == 0 for v in ex.values()),
         " ".join(f"smoke_{j} {sm[j][1]}" for j in PILOT_IDS) + "; " + " ".join(f"j{j} exit {ex[j]}" for j in PILOT_IDS))
    wh = {j: (GH.wall(plog[j]) if plog[j] else None) for j in PILOT_IDS}
    sw = {j: (GH.wall(smr[j]["log"]) if smr[j] else None) for j in PILOT_IDS}
    if retry is None:
        line("C7c", False, f"retry-walls file {a.retry_walls} has a malformed line (need '<node> <hours>')")
    else:
        parts = list(sw.values()) + list(wh.values())
        rs = sum((h for _, h in retry), F(0))
        tot = sum((x for x in parts if x is not None), F(0)) + rs
        line("C7c", None not in parts and tot <= BAR["C7c"],
             f"smokes {' '.join('-' if v is None else '%.2f' % v for v in sw.values())} + jobs "
             f"{' '.join('-' if v is None else '%.2f' % v for v in wh.values())} + retries {len(retry)} ({float(rs):.2f} h) "
             f"= {float(tot):.2f} GPU-h (cap 6.0" + ("" if None not in parts else "; a wall is MISSING") + ")")
    # --- the two arm blocks
    ref_ok, refs, rlab = False, [], None
    try:
        rlab, rp_ = a.ref.split("=", 1)
        rl = rp_.split(",")
    except (ValueError, AttributeError):
        rl = []
    ids_ok = (rlab == "s16" and len(rl) == 2 and len({os.path.realpath(p) for p in rl}) == 2
              and all(K4.has_id(p, j) for j, p in zip(("520", "663"), rl)))
    for p in rl:
        refs.append(load(p, "a-Q6 reference", K3.PLAIN)[0])
    ref_ok = ids_ok and len(refs) == 2 and all(r is not None and rseed(r) == (16, 16) for r in refs)
    canon = {s: ({i for i, _ in refs[0]["plain"].get(s, [])} if refs and refs[0] else set()) for s in K3.PLAIN}

    def xpl(r, s, what):
        try:
            return K3.xplain(r, s, canon[s], what) if r is not None and canon[s] else None
        except SystemExit:
            return None
    refv = {}
    for s in K3.PLAIN:
        rv = [xpl(r, s, "a-Q6 reference") for r in refs]
        refv[s] = (rv[0] + rv[1]) / 2 if ref_ok and len(rv) == 2 and None not in rv else None
    w975, w985 = (GH.wall(s1[j]["log"]) if s1[j] else None for j in ("975", "985"))
    wall_missing = {}
    cur_gsm_fail = {"R": False, "L": False}    # CUR-a below its a-Q6 bar with the reference readable (Codex round 3)
    for arm in ("R", "L"):
        cj, kj = JOBS[(arm, "CUR")], JOBS[(arm, "K1X")]
        cur, k1x = pil[cj], pil[kj]
        okc = [complete(cur, False), complete(k1x, False)]
        line(f"{arm}-Q1", okc[0][0] and okc[1][0], f"j{cj}: {okc[0][1]} | j{kj}: {okc[1][1]}")
        q2 = []
        for kind, j, r in (("CUR", cj, cur), ("K1X", kj, k1x)):
            if r is None:
                q2.append((False, f"j{j} no run"))
                continue
            try:
                o1, d1 = K4.i4_s2(r, arm, kind, 16, S1_NAME["975" if kind == "CUR" else "985"])
                ref_s2, ref_b = rref["980" if kind == "CUR" else "990"], rref["756" if kind == "CUR" else "757"]
                if ref_s2 is None or (arm == "R" and ref_b is None):
                    o2, d2 = False, "a recipe reference is missing or unreadable"
                else:
                    o2, d2 = K4.recipe_pilot(r, arm, kind, ref_s2, ref_b)
                okn = K4.run_name(r) == K4.s2_name(arm, kind, 16)
            except SystemExit as e:
                o1, d1, o2, d2, okn = False, str(e), False, "", False
            q2.append((o1 and o2 and okn, f"j{j}: {d1}; {d2}" + ("" if okn else f"; run name {K4.run_name(r)}")))
        line(f"{arm}-Q2", all(x[0] for x in q2), " | ".join(x[1] for x in q2))
        v3 = {(t, o): cell(cur, f"khop_{t}_{o}", 1) for t, o in T.CELLS6}
        line(f"{arm}-Q3", all(ge(v, BAR["Q3"]) for v in v3.values()),
             f"j{cj} " + " ".join(f"{t}/{o} {pv(v)}" for (t, o), v in v3.items()) + " (bar 85.0)")
        try:
            q4 = K3.K2.xqs_pair(cur, khop, f"j{cj} qs pair") if cur is not None and khop_ok else None
        except SystemExit:
            q4 = None
        line(f"{arm}-Q4", ge(q4, BAR["Q4"]), f"j{cj} query-swap pair k=1 {pv(q4)} (bar 80.0)")
        try:
            b5 = K3.K2.xacc(cur, "khop_blind", [x["item_id"] for x in khop["khop_blind"]], f"j{cj} blind") \
                if cur is not None and khop_ok else None
        except SystemExit:
            b5 = None
        line(f"{arm}-Q5", b5 is not None and b5 <= BAR["Q5"], f"j{cj} blind {pv(b5)} (bar <= 32.0)")
        ok6, det6 = ref_ok, [f"refs {rlab} {[os.path.basename(p) for p in rl]} (need j520 then j663, distinct, seeds "
                             f"(16, 16)): {'ok' if ref_ok else 'NO'}"]
        for s in K3.PLAIN:
            for j, r, b in ((cj, cur, BAR["Q6_cur"]), (kj, k1x, BAR["Q6_k1x"])):
                v = xpl(r, s, f"j{j}")
                if s in ("gsm8kx", "gsm8kv2"):
                    o = v is not None and refv[s] is not None and v >= refv[s] - b
                    ok6 &= o
                    if j == cj and not o:
                        cur_gsm_fail[arm] = True
                    det6.append(f"j{j} {s} {pv(v)} (reference mean(j520, j663) {pv(refv[s])}, bar "
                                f"{pv(None if refv[s] is None else refv[s] - b)})")
                else:
                    info.append(f"INFO {arm}-Q6 j{j} {s} {pv(v)} (reference {pv(refv[s])})")
        line(f"{arm}-Q6", ok6, " ".join(det6))
        gpu = {j: (re.findall(r"^gpu 0: ([^,]+),", plog[j], re.M) if plog[j] else []) for j in (cj, kj)}
        lines.append(f"INFO  {arm}-Q7: " + " ".join(
            f"j{j} {'MISSING' if wh[j] is None else '%.2f h' % wh[j]} ({gpu[j][0] if gpu[j] else 'gpu ?'})"
            + ("" if wh[j] is None or wh[j] <= Q7_BAR[arm] else " above") for j in (cj, kj))
            + f" (reference {float(Q7_BAR[arm])} h; INFO, a slow placement is hardware)")
        ws = (w975, w985, wh[cj], wh[kj])
        wall_missing[arm] = None in ws
        if None in ws:
            line(f"{arm}-Q7b", False, "a wall is missing (975, 985, CUR-a, K1X-a: "
                 + " ".join('-' if x is None else '%.2f' % x for x in ws) + ")")
        else:
            proj = 4 * sum(ws)
            line(f"{arm}-Q7b", proj <= BAR["Q7b"], f"projected confirmation 4 x ({' + '.join('%.2f' % x for x in ws)}) = "
                 f"{float(proj):.2f} GPU-h (cap 23.0)")
        a2, a3 = cell(cur, "khop_ca_fwd", 2), cell(cur, "khop_ca_fwd", 3)
        ok9 = ge(a2, BAR["Q9_k2"]) and ge(a3, BAR["Q9_k3"])
        line(f"{arm}-Q9", ok9, f"a_{cj}(ca, fwd, 2) {pv(a2)} (bar 128 / 150 = 85.3), a_{cj}(ca, fwd, 3) {pv(a3)} (bar "
             f"120 / 150 = 80.0)" + ("" if ok9 else f" (installed, then lost at s16, arm {arm})"))
        k3 = cell(k1x, "khop_ca_fwd", 3)
        d10 = None if None in (a3, k3) else a3 - k3
        line(f"{arm}-Q10", ge(d10, BAR["Q10"]), f"a_{cj}(ca, fwd, 3) - a_{kj}(ca, fwd, 3) {pv(d10)} (bar 15.0)")
        v11 = {(t, o): cell(k1x, f"khop_{t}_{o}", 1) for t, o in T.CELLS6}
        line(f"{arm}-Q11", all(ge(v, BAR["Q11"]) for v in v11.values()),
             f"j{kj} " + " ".join(f"{t}/{o} {pv(v)}" for (t, o), v in v11.items()) + " (bar 85.0)")
        # INFO curves (no rc / hint line)
        for j, r in ((cj, cur), (kj, k1x)):
            if r is None:
                continue
            for s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_vb_fwd", "khop_cag_fwd"):
                info.append(f"INFO j{j} {s:13s} " + " ".join(f"k{k} {pv(cell(r, s, k))}" for k in range(1, 9)))
            h = r["metrics"].get("history") or []
            tim = h and all(isinstance(x, dict) and K3.finite(x.get("s_per_step")) and K3.finite(x.get("peak_gib"))
                            for x in h)
            info.append(f"INFO j{j} " + (f"s_per_step {sum(x['s_per_step'] for x in h) / len(h):.2f}, peak_gib "
                                         f"{max(x['peak_gib'] for x in h):.1f}" if tim else
                                         "s_per_step / peak_gib UNREADABLE (a history row without a finite counter)"))
            lr_ok = h and all(isinstance(x, dict) and K3.finite(x.get("lr")) for x in h)   # Codex round 3 finding 2
            info.append(f"INFO j{j} max history lr {'%.3g' % max(x['lr'] for x in h) if lr_ok else 'UNREADABLE'} (args.lr "
                        f"{r['metrics'].get('args', {}).get('lr') if isinstance(r['metrics'].get('args'), dict) else '-'}; "
                        "INFO, the decision is in a-Q2)")
    # INFO (card section 3, Codex round 2 finding 1): plain sets incl. val for the four pilot jobs and the khop3 S2 pair
    # j980 / j990, and the j980 / j990 curves next to the pilot curves. No line reads them.
    def plain_txt(r, j):
        vals = [f"{s} {pv(xpl(r, s, f'j{j}'))}" for s in K3.PLAIN]
        try:
            rows = K3.read_plain(r["path"], ("val",)).get("val") or []
        except (SystemExit, OSError, ValueError, KeyError, tarfile.TarError, EOFError, zlib.error):
            rows = []
        ok = rows and len({i for i, _ in rows}) == len(rows)
        return " ".join(vals + [f"val {pv(F(sum(c for _, c in rows), len(rows)) if ok else None)}"
                                + (f" (n {len(rows)})" if ok else "")])
    for j, r in [(x, pil[x]) for x in PILOT_IDS] + [(int(x), rref[x]) for x in ("980", "990")]:
        if r is None:
            info.append(f"INFO j{j} plain MISSING (run not loaded)")
            continue
        info.append(f"INFO j{j} plain {plain_txt(r, j)}")
    for j in ("980", "990"):
        if rref[j] is None:
            continue
        for s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_vb_fwd", "khop_cag_fwd"):
            info.append(f"INFO j{j} (khop3 S2) {s:13s} " + " ".join(f"k{k} {pv(cell(rref[j], s, k))}" for k in range(1, 9)))
    for j in ("975", "985"):
        if s1[j] is not None:
            info.append(f"INFO j{j} (khop3 S1) khop_ca_fwd " + " ".join(f"k{k} {pv(cell(s1[j], 'khop_ca_fwd', k))}"
                                                                     for k in range(1, 9)))
    # --- run integrity (Amendment 3) and the pilot failure readings of card 0.5
    a_ = lambda arm, q: res.get(f"{arm}-{q}")
    misrun = {x: not (a_(x, "Q1") and a_(x, "Q2")) or wall_missing.get(x, True) for x in ("R", "L")}
    for arm in ("R", "L"):
        if misrun[arm]:
            why = [f"{arm}-{q}" for q in ("Q1", "Q2") if not a_(arm, q)] + (
                [f"{arm}-Q7b (a wall is missing)"] if wall_missing.get(arm, True) else [])
            readings.append(f"READING  not run as registered (arm {arm}): {', '.join(why)} FAIL; rescue the node under "
                            "the same id (card 0.5); nothing of this arm is read and no arm is chosen")
    if not misrun["R"] and not misrun["L"] and not a_("R", "Q9") and not a_("L", "Q9") and res.get("C2"):
        readings.append("READING  installed, then lost (both arms): at s16 neither a quarter of the LR nor a one-third "
                        "replay keeps 3-step composition through the GSM stage; follow-up (c) is closed at this dose")
    gsm_txt = {"L": "GSM not recovered at LR 7.5e-7 (the capacity and the GSM recovery compete for the LR mass)",
               "R": "the replay costs GSM after a chain stage although KH kept GSM from the base model"}
    lost_txt = {"L": "a quarter of the LR still overwrites the capacity (forgetting is fast early)",
                "R": "a one-third share does not maintain the capacity"}
    for arm in ("R", "L"):
        if misrun[arm]:
            continue
        if a_(arm, "Q9") and ref_ok and cur_gsm_fail[arm]:      # card 0.5: "a-Q6 fails for CUR-a", not for K1X-a
            readings.append(f"READING  kept, GSM damaged (arm {arm}): {gsm_txt[arm]}")
        if not a_(arm, "Q9") and a_(arm, "Q6"):
            readings.append(f"READING  lost, GSM intact (arm {arm}): {lost_txt[arm]}")
        if not a_(arm, "Q10") or not a_(arm, "Q11"):
            readings.append(f"READING  control invalid (arm {arm}): the twin reaches the same k = 3 accuracy or loses "
                            "single-hop reading; the arm cannot reach PASS")
    common_ok = all(res.get(c) for c in K4.COMMON)
    arm_ok = {x: all(res.get(n) for n in K4.arm_lines(x)) for x in ("R", "L")}
    if misrun["R"] or misrun["L"]:
        last = "GATE FAIL: " + "; ".join(fails)
    elif common_ok and arm_ok["R"]:
        last = "GATE PASS ARM R"
    elif common_ok and arm_ok["L"]:
        last = "GATE PASS ARM L"
    else:
        last = "GATE FAIL: " + "; ".join(fails)
    return lines, info, readings, last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s1", nargs=2, required=True, metavar=("J975", "J985"))
    ap.add_argument("--curl", required=True, help="j1025 tarball (CUR-L)")
    ap.add_argument("--k1xl", required=True, help="j1026 tarball (K1X-L)")
    ap.add_argument("--curr", required=True, help="j1027 tarball (CUR-R)")
    ap.add_argument("--k1xr", required=True, help="j1028 tarball (K1X-R)")
    ap.add_argument("--smokes", required=True, help="<smoke_1025 tgz>,<smoke_1026 tgz>,<smoke_1027 tgz>,<smoke_1028 tgz>")
    ap.add_argument("--ref", required=True, help="s16=<j520 tgz>,<j663 tgz>")
    ap.add_argument("--recipe-ref", nargs=4, required=True, metavar=("J980", "J990", "J756", "J757"))
    ap.add_argument("--retry-walls", required=True)
    ap.add_argument("--report", default=os.path.join(E, "src/data/eval/khop4/build_report.json"))
    ap.add_argument("--khop3-report", default=os.path.join(E, "src/data/eval/khop3/build_report.json"))
    ap.add_argument("--khop2-report", default=os.path.join(E, "src/data/eval/khop2/build_report.json"))
    ap.add_argument("--parent-report", default=os.path.join(E, "src/data/eval/khop/build_report.json"))
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--data-root", default=os.path.join(E, "src/data"))
    ap.add_argument("--env663", default=G.ENV663)
    a = ap.parse_args(argv)
    lines, info, readings, last = gate(a)
    for l in lines + info + readings:
        print(l)
    print(last)
    return 0 if last.startswith("GATE PASS ARM ") else 1


if __name__ == "__main__":
    sys.exit(main())
