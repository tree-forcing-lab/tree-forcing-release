#!/usr/bin/env python3
"""CARD_khop_v4 readout: does a stage-2 protection (arm a: L = the A2 GSM epoch at LR 7.5e-7, R = the A2 GSM rows with a
one-third chain replay) keep the k-hop capacity installed by the chain stage and GSM intact, and does the kept capacity
raise GSM rc2? CUR1 / K1X1 = the stage-1 models (s16: j975 / j985 of khop3; s17-20: j1029-1032 / j1033-1036), CUR /
K1X = the final models of the confirmed arm (s16: the arm's pilot pair, j1025 / j1026 for L, j1027 / j1028 for R;
s17-20: j1037-1040 / j1041-1044). Standard library, Python 3.9. Primitives of khop_table.py, khop2_table.py and
khop3_table.py by import; none of them is edited.

    python3 -I khop4_table.py --gate-output gate.txt --retry-walls walls.txt [--results-dir DIR] \\
        --run CUR1:s16=<975 tgz> ... --run CUR:s16=<1025 or 1027 tgz> ... --run K1X1:... --run K1X:... \\
        --run J980:s16=<980 tgz> --run J990:s16=<990 tgz> --run D27:s16=<875 tgz> ... \\
        [--run KH:s16=<756 tgz> ... --run K1:s16=<757 tgz> ...  (arm R)] \\
        --ref-orig s16=<520 tgz> ... --ref-retrain s16=<663 tgz> ... [--json out.json]

The arm is read from the gate output's last line ("GATE PASS ARM R" / "GATE PASS ARM L"); the gate output must be a
full, self-consistent khop4 gate output (gate_text_arm). Decision values are exact (card 2.7): every khop, plain and rc
cell is Fraction(correct, items) from the preds files; CIs and t compare in squared rational form (t = 2.78 at n = 5);
the bars are rationals. The registered population is fixed: the 20 jobs of the arm (each with its exact run name), the
10 reference jobs (j520 j525 j530 j540 j542 / j663 j664 j665 j540 j666, by id), and the context jobs of card 2.3 (D27
j875-879, j980 / j990 for both arms (recipe references), KH j756 j758-761 and K1 j757 j762-765 for R). A missing,
partial, NaN or unreadable preds file, log or metrics.json, a metrics.json that is not a JSON object, a missing wall, a
seed that differs from the label, two tarballs for one id (--results-dir) stop the readout (UNREADABLE); no seed, item
or set is dropped and no denominator shrinks. Pairing is by the seed recorded in metrics.json.
I4 (run as registered) checks every log line of card 2.5 and the all-args recipe check: metrics.json args of each job
equal those of its recipe reference over ALL keys except an explicit allow-list (recipe_diff; a key on one side only
fails). Verdict (card 0.7, in this order; the first that applies): MANIPULATION FAILED ("not run as registered",
"not installed", "installed, then lost in the GSM stage (arm a)", "control not a control", "GSM damaged" (L: "GSM
damaged or not recovered")), PASS, NEGATIVE, BETWEEN "gain from extra updates, not composition", BETWEEN "gain over
best A2, control invalid", BETWEEN "gain below the bar", NULL, BETWEEN "unresolved". The rc3 line uses readings 2-8 on
D_3 / C_3 and never changes the verdict; under MANIPULATION FAILED it prints the rc3 numbers as INFO only.
"""
import argparse, json, math, os, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import khop_table as T  # noqa: E402
import khop2_table as K2  # noqa: E402
import khop3_table as K3  # noqa: E402
import khop_pilot_gate as G  # noqa: E402  (exit_code, gsm_sets, ENV663)
import khop4_gpuh as GH  # noqa: E402  (zone-aware wall, Amendment 3)

E = os.path.dirname(HERE)
F = Fraction
SEEDS5 = (16, 17, 18, 19, 20)
S17 = (17, 18, 19, 20)
NEW = ("CUR1", "CUR", "K1X1", "K1X")
CONTEXT = {"R": ("J980", "J990", "D27", "KH", "K1"), "L": ("J980", "J990", "D27")}
PLAIN = K3.PLAIN
SCORED = [s for s in T.SETS if s != "khop_blind"]
XBAR = {"I1": F("0.90"), "I2": F("0.85"), "I3": F("0.30"), "I7_k1": F("0.98"), "I7_loss": F(1), "I7_k1_s2": F("0.90"),
        "N1": F(2, 100), "gain": F(3, 100)}
BONF_T = F("3.50")
LR = {"L": 7.5e-07, "R": 3e-06}
REPLAY = {"CUR": "data/eval/khop/train_kh.jsonl", "K1X": "data/eval/khop/train_k1.jsonl"}
GSM_TRAIN = "data/student_gsm8k_tf_vt/train.jsonl"
S2_COUNTS = {"L": (5923, 186), "R": (8923, 279)}
S1_ROWS, S1_STEPS = 27000, 844
PILOT = {("L", "CUR"): 1025, ("L", "K1X"): 1026, ("R", "CUR"): 1027, ("R", "K1X"): 1028}
S1_FIRST = {"CUR": 1029, "K1X": 1033}
S2_FIRST = {"CUR": 1037, "K1X": 1041}
S16_S1 = {"CUR": 975, "K1X": 985}
CTX_JOBS = {"J980": {16: 980}, "J990": {16: 990}, "D27": {s: 875 + s - 16 for s in SEEDS5},
            "KH": {16: 756, **{s: 758 + s - 17 for s in S17}}, "K1": {16: 757, **{s: 762 + s - 17 for s in S17}}}
ALLOW = {"pilot_L": {"lr", "out"}, "pilot_R": {"extra_train", "out"}, "pilot_R_base": {"model", "out", "save"},
         "s2_confirm": {"seed", "order_seed", "model", "out"},
         "s1_confirm": {"seed", "order_seed", "train", "out", "eval", "eval_subq"}}


def unreadable(msg):
    raise SystemExit(f"khop4_table: UNREADABLE: {msg} (the readout never reads a partial or broken input)")


K2.unreadable = unreadable                # the imported exact helpers stop with this family's message
K3.unreadable = unreadable


# --- names and identities ----------------------------------------------------------------------------------------------
def kind_of(arm):
    return "CUR" if arm.startswith("CUR") else "K1X"


def s1_id(kind, seed):
    return S16_S1[kind] if seed == 16 else S1_FIRST[kind] + seed - 17


def s1_name(kind, seed):
    d, t = ("kd27", "khopcur1-c1lr3ch") if kind == "CUR" else ("kd27k1", "khopk1x1-c1lr3ch")
    return f"{s1_id(kind, seed)}_4b_{d}_A2-{t}_s{seed}_fsdp"


def s2_id(a, kind, seed):
    return PILOT[(a, kind)] if seed == 16 else S2_FIRST[kind] + seed - 17


def s2_name(a, kind, seed):
    w = s1_id(kind, seed)
    tag = {("L", "CUR"): f"khop4curlw{w}-c1lr075ch", ("L", "K1X"): f"khop4k1xlw{w}-c1lr075ch",
           ("R", "CUR"): f"khop4currw{w}-c1lr3ch", ("R", "K1X"): f"khop4k1xrw{w}-c1lr3ch"}[(a, kind)]
    return f"{s2_id(a, kind, seed)}_4b_gsmvt_A2-{tag}_s{seed}_fsdp"


def expected_name(arm, a, seed):
    return s1_name(kind_of(arm), seed) if arm.endswith("1") else s2_name(a, kind_of(arm), seed)


def s1_train(kind, seed):
    fam = "khop3" if seed == 16 else "khop4"
    return f"data/student_{fam}_{'c27' if kind == 'CUR' else 'k1x'}/train.jsonl"


def has_id(path, j):
    """Reference / context identity: the basename names job j as a whole id ((^|_)<j>(_|.tgz$)), never a look-alike."""
    return re.search(rf"(^|_){j}(_|\.tgz$)", os.path.basename(str(path))) is not None


# --- loading -----------------------------------------------------------------------------------------------------------
def load(path, what, need=PLAIN):
    """khop3_table.load (every preds file checked, args.eval sets complete) plus: metrics.json must be a JSON object
    with an args object and a history list; any other exception of the loader is UNREADABLE, never a smaller set."""
    try:
        r = K3.load(path, what, need)
    except SystemExit:
        raise
    except Exception as e:
        unreadable(f"{what}: {path} does not load ({type(e).__name__}: {e})")
    m = r.get("metrics")
    if not isinstance(m, dict) or not isinstance(m.get("args"), dict) or not isinstance(m.get("history"), list):
        unreadable(f"{what}: {path} metrics.json is not an object with args and history")
    return r


def parse_specs(specs, what):
    """'ARM:s<seed>=<path>' (or 's<seed>=<path>' when what is a reference) -> {arm: {seed: path}}; a malformed or
    repeated entry stops."""
    out = {}
    for s in specs or []:
        m = re.match(r"^(?:([A-Z0-9]+):)?s(\d+)=(.+)$", s)
        if not m:
            raise SystemExit(f"khop4_table: bad {what} entry {s!r}")
        arm, seed, p = m.group(1) or "", int(m.group(2)), m.group(3)
        if seed in out.setdefault(arm, {}):
            raise SystemExit(f"khop4_table: {what} {arm} s{seed} given twice")
        out[arm][seed] = p
    return out


# --- log lines and the recipe check ------------------------------------------------------------------------------------
def recipe_diff(args, ref, allow):
    """The keys of two metrics.json args objects that differ, outside allow; a key present on one side only is listed
    even when allowed. Values compare as canonical JSON (so True and 1, 3e-06 and "3e-06" differ)."""
    if not isinstance(args, dict) or not isinstance(ref, dict):
        return ["<args not an object>"]
    j = lambda v: json.dumps(v, sort_keys=True)
    bad = []
    for k in sorted(set(args) | set(ref)):
        if k not in args or k not in ref:
            bad.append(f"{k} (one side only)")
        elif k not in allow and j(args[k]) != j(ref[k]):
            bad.append(k)
    return bad


def run_name(run):
    """The run name: from the tarball basename results_<name>.tgz (never from a directory part of the path), else from
    the log's first '== <name> on <host>' line, else None."""
    m = re.match(r"^results_(.+)\.tgz$", os.path.basename(str(run.get("path"))))
    if m:
        return m.group(1)
    n = re.findall(r"^== (\S+) on \S+ ", run.get("log") or "", re.M)
    return n[0] if n else None


def stage_lines(log):
    return re.findall(r"^stage (\S+): (\d+) examples per pass, (\d+) steps$", log, re.M)


def history_ok(h, steps):
    """Every metrics.json history row (not only the last one; Codex round 1 finding 1) is an object with an int step
    and a finite loss, the steps strictly increase, and the last step is the schedule's final step."""
    if not isinstance(h, list) or not h:
        return False
    if not all(isinstance(x, dict) and isinstance(x.get("step"), int) and not isinstance(x.get("step"), bool)
               and K3.finite(x.get("loss")) for x in h):
        return False
    return all(x["step"] < y["step"] for x, y in zip(h, h[1:])) and h[-1]["step"] == steps


def history_txt(h, ok):
    n = len(h) if isinstance(h, list) else "not a list"
    return f"{n} rows {'ok' if ok else 'BAD (a row without a finite loss / int step, or steps)'}"


def last_step(log):
    return [x["step"] for x in K3.step_rows(log)][-1:]


def i4_s1(run, kind, seed):
    """(ok, detail) of I4 for one S1 job: 27,000 rows / 844 steps, no extra-train line, final step 844 in the log and as
    the last metrics history row (history_ok: every row an object with an int step and a finite loss), the train path
    of the job's DATA (khop3 for 975 / 985, khop4 for 1029-1036), the base model, the label's seed, exactly one
    weights export line of its own name, exit 0."""
    log, m = run["log"], run["metrics"]
    a, h = m["args"], m["history"]
    name = run_name(run)
    st, ls, hok = stage_lines(log), last_step(log), history_ok(h, S1_STEPS)
    ok = (st == [("all", str(S1_ROWS), str(S1_STEPS))] and not re.search(r"^extra-train:", log, re.M)
          and ls == [S1_STEPS] and hok and a.get("train") == s1_train(kind, seed)
          and a.get("model") == "models/Qwen3.5-4B" and (a.get("seed"), a.get("order_seed")) == (seed, seed)
          and name == s1_name(kind, seed) and len(K3.weights_export(log, name)) == 1 and G.exit_code(log) == 0)
    return ok, (f"stage {st} last step {ls} history {history_txt(h, hok)} train {a.get('train')} seed "
                f"{a.get('seed')}/{a.get('order_seed')} "
                f"weights {K3.weights_export(log, name or '')} exit {G.exit_code(log)}")


def i4_s2(run, arm, kind, seed, w):
    """(ok, detail) of the I4 S2 log / value lines of one S2 job of arm (L / R), kind (CUR / K1X), from the S1 run w:
    one weights import line of w; L: 'stage all: 5923 examples per pass, 186 steps' and no extra-train line; R: the
    two extra-train lines (3,000 rows of the kind's replay file; total 8,923 with 5,923 from the gsmvt train file) and
    'stage all: 8923 examples per pass, 279 steps'; final step in the log and as the last history row (history_ok:
    every row an object with an int step and a finite loss); args.model = models/<w>/model, args.train the gsmvt
    file, args.lr exactly 7.5e-07 (L) or 3e-06 (R), args.extra_train exactly '' (L) or the replay path (R),
    metrics.json extra_train absent (L) or {<path>: 3000} (R); the label's seed; exit 0."""
    log, m = run["log"], run["metrics"]
    a, h = m["args"], m["history"]
    rows, steps = S2_COUNTS[arm]
    st, ls, hok = stage_lines(log), last_step(log), history_ok(h, steps)
    ex = re.findall(r"^extra-train: (\d+) rows from (\S+) ", log, re.M)
    tl = re.findall(r"^extra-train: total training rows (\d+) \((\d+) from (\S+)\)$", log, re.M)
    if arm == "L":
        ok_x = not ex and not tl and not re.search(r"^extra-train:", log, re.M) and a.get("extra_train") == "" \
            and "extra_train" not in m
    else:
        ok_x = (ex == [("3000", REPLAY[kind])] and tl == [("8923", "5923", GSM_TRAIN)]
                and len(re.findall(r"^extra-train:", log, re.M)) == 2 and a.get("extra_train") == REPLAY[kind]
                and m.get("extra_train") == {REPLAY[kind]: 3000})
    lr = a.get("lr")
    ok = (len(K3.weights_import(log, w)) == 1 and st == [("all", str(rows), str(steps))] and ok_x
          and ls == [steps] and hok and a.get("model") == f"models/{w}/model"
          and a.get("train") == GSM_TRAIN
          and isinstance(lr, float) and lr == LR[arm] and (a.get("seed"), a.get("order_seed")) == (seed, seed)
          and G.exit_code(log) == 0)
    return ok, (f"weights {K3.weights_import(log, w)} stage {st} extra-train {ex} total {tl} last step {ls} "
                f"history {history_txt(h, hok)} "
                f"lr {lr!r} extra_train {a.get('extra_train')!r} / {m.get('extra_train')!r} model {a.get('model')} seed "
                f"{a.get('seed')}/{a.get('order_seed')} exit {G.exit_code(log)}")


def recipe_pilot(run, arm, kind, ref_s2, ref_base):
    """(ok, detail) of the all-args recipe check of a pilot S2 job: against j980 (CUR) / j990 (K1X) except {lr, out}
    (L) or {extra_train, out} (R); for R also against the s16 KH (CUR) / K1 (K1X) job except {model, out, save}."""
    a = run["metrics"]["args"]
    bad = recipe_diff(a, ref_s2["metrics"]["args"], ALLOW[f"pilot_{arm}"])
    det = f"vs j{'980' if kind == 'CUR' else '990'} differ outside {sorted(ALLOW[f'pilot_{arm}'])}: {bad}"
    if arm == "R":
        b2 = recipe_diff(a, ref_base["metrics"]["args"], ALLOW["pilot_R_base"])
        bad = bad + b2
        det += f"; vs j{'756' if kind == 'CUR' else '757'} differ outside {sorted(ALLOW['pilot_R_base'])}: {b2}"
    return not bad, det


def recipe_s2_confirm(run, pilot_run):
    bad = recipe_diff(run["metrics"]["args"], pilot_run["metrics"]["args"], ALLOW["s2_confirm"])
    return not bad, f"vs the s16 pilot of the arm differ outside {sorted(ALLOW['s2_confirm'])}: {bad}"


def recipe_s1_confirm(run, kind, ref):
    """Against j975 (CUR1) / j985 (K1X1) except {seed, order_seed, train, out, eval, eval_subq}; train must be the
    khop4 path, eval the reference list with only the val entry's directory changed from student_khop3_* to
    student_khop4_*, eval_subq the reference value with the same change (Amendment 2)."""
    a, r = run["metrics"]["args"], ref["metrics"]["args"]
    bad = recipe_diff(a, r, ALLOW["s1_confirm"])
    sub = lambda v: v.replace("data/student_khop3_", "data/student_khop4_", 1) if isinstance(v, str) else v
    rev = r.get("eval") if isinstance(r.get("eval"), list) else []
    want_eval = [sub(x) if x.startswith("val=") else x for x in rev]
    if not any(x.startswith("val=data/student_khop3_") for x in rev):
        bad.append("reference eval has no val=data/student_khop3_ entry")
    if a.get("eval") != want_eval:
        bad.append("eval (not the reference list with the val dir changed)")
    if a.get("eval_subq") != sub(r.get("eval_subq")) or not str(r.get("eval_subq", "")).startswith("data/student_khop3_"):
        bad.append(f"eval_subq {a.get('eval_subq')!r}")
    if a.get("train") != s1_train(kind, 17):
        bad.append(f"train {a.get('train')!r}")
    return not bad, f"vs j{S16_S1[kind]} differ outside {sorted(ALLOW['s1_confirm'])} or wrong values: {bad}"


# --- statistics and text -----------------------------------------------------------------------------------------------
pts_ci, pts_full, acc_ci, fs = K3.pts_ci, K3.pts_full, K3.acc_ci, K3.fs


def pass_rule(xs):
    X = K2.XS(xs)
    return X.ok() and X.m >= XBAR["gain"] and X.lo_ge(0)


def bonferroni(xs):
    """INFO: t >= 3.50 (two arms, df 4), exact: m > 0 and m^2 n >= 3.5^2 var."""
    X = K2.XS(xs)
    return X.ok() and X.m > 0 and X.m * X.m * X.n >= BONF_T * BONF_T * X.var


ARM_TEXT = {"R": "before an A2 GSM epoch with a one-third replay of those chains",
            "L": "before an A2 GSM epoch at one quarter of the A2 LR"}


def rc_reading(j, arm, D, Dorig, Dre, C, V, vtxt, kstar):
    """Readings 2-8 of card 0.7 on rc_j for arm (L / R); the dicts are complete (verdict checks them)."""
    xd, xc = list(D.values()), list(C.values())
    XD, XC, XO, XR = K2.XS(xd), K2.XS(xc), K2.XS(Dorig.values()), K2.XS(Dre.values())
    d, c = pts_ci(xd), pts_ci(xc)
    st = fs(xd)
    if pass_rule(xd) and XC.lo_gt(0) and V:
        return "PASS", None, (f"in Qwen3.5-4B with plain GSM intact and k*(ca, fwd) = {kstar} kept, symbolic chain "
                              f"training (27,000 rows) {ARM_TEXT[arm]} raises one-pass GSM rc{j} on the 249 primary items "
                              f"by {d} points over the best A2 set and by {c} over the same schedule with single-hop "
                              "queries on the same chains")
    if XO.hi_lt(0) and XR.hi_lt(0):
        s = (f"with plain GSM intact and k* >= 3 kept, the chain curriculum (arm {arm}) lowers GSM rc{j} by {d} against "
             f"both A2 sets (originals {pts_ci(Dorig.values())}, re-trains {pts_ci(Dre.values())})")
        if XC.hi_lt(0) and V:
            s += "; the symbolic composition training interferes with GSM multi-step items"
        else:
            s += (f"; the K1X twin is not above CUR-{arm} (C_{j} {c}): the cost is chain training plus extra updates, "
                  "not composition")
        return "NEGATIVE", None, s + f"; V {vtxt}"
    if pass_rule(xd) and V and not XC.lo_gt(0):
        return "BETWEEN", "gain from extra updates, not composition", f"D_{j} {d}, C_{j} {c} (CI lower <= 0)"
    if pass_rule(xd) and not V:
        return "BETWEEN", "gain over best A2, control invalid", f"D_{j} {d}; V fails: {vtxt}; C_{j} {c} (INFO)"
    if XD.lo_gt(0) and XD.m < XBAR["gain"]:
        return "BETWEEN", "gain below the bar", (f"arm {arm} raises GSM rc{j} by {d} over the best A2 set, below the "
                                                 f"pre-registered +3 bar; C_{j} = {c}")
    if not XD.lo_gt(0) and XD.hi_lt(XBAR["gain"]):
        s = (f"the kept capacity (k*(ca, fwd) = {kstar}, plain intact) does not raise GSM rc{j}: D_{j} = {d}; the data "
             f"exclude a gain of more than {100 * st[3]:+.1f} points")
        if XD.hi_lt(0):
            drift = 100 * sum(Dorig[k] - Dre[k] for k in Dorig) / len(Dorig)
            s += (f" (below the original A2 set only (the A2 drift of {float(drift):+.2f} points on rc{j}, re-train - "
                  "original, is not excluded))")
        return "NULL", None, s + f"; C_{j} {c}; V {vtxt}"
    return "BETWEEN", "unresolved", f"D_{j} {d} (CI lower <= 0 and CI upper >= +3.0); C_{j} {c}"


def verdict(d):
    """d: {"arm": "L" / "R", "I": {"I1".."I7": bool}, "kstar_cur1", "kstar_cur" (int), "n1_cur", "n1_k1x" (bool),
    "n1_k1x_fail_sets", "rc": {j: {"D", "Dorig", "Dre", "C": {seed: value}}} for j = 2, 3}. Returns (reading,
    sub-label, lines, rc3 line)."""
    arm = d.get("arm")
    if arm not in ("L", "R"):
        unreadable(f"arm {arm!r}")
    I = d["I"]
    if sorted(I) != [f"I{i}" for i in range(1, 8)] or not all(isinstance(v, bool) for v in I.values()):
        unreadable(f"installation lines {I}")
    for k in ("n1_cur", "n1_k1x"):
        if not isinstance(d.get(k), bool):
            unreadable(f"{k} {d.get(k)!r}")
    if not all(isinstance(d.get(k), int) and not isinstance(d.get(k), bool) for k in ("kstar_cur1", "kstar_cur")):
        unreadable(f"k* {d.get('kstar_cur1')!r} / {d.get('kstar_cur')!r}")
    V = I["I7"] and d["n1_k1x"]
    sets = d.get("n1_k1x_fail_sets") or []
    vtxt = ("holds (I7 and N1 for K1X)" if V else
            "fails (" + ", ".join(x for x, ok in (("I7", I["I7"]), ("N1 for K1X" + (f" on {', '.join(sets)}" if sets
                                                                                  else ""), d["n1_k1x"])) if not ok) + ")")
    r = {j: d["rc"][j] for j in (2, 3)}
    for j in (2, 3):
        for n in ("D", "Dorig", "Dre", "C"):
            K2.need_seeds(r[j][n], f"{n}_{j}")
    lines = [f"arm {arm}; k*(CUR1, ca, fwd) = {d['kstar_cur1']}; k*(CUR-{arm}, ca, fwd) = {d['kstar_cur']}; V {vtxt}",
             f"D_2 {pts_full(r[2]['D'].values())} (D_2^orig {pts_ci(r[2]['Dorig'].values())}, D_2^re "
             f"{pts_ci(r[2]['Dre'].values())}); C_2 {pts_full(r[2]['C'].values())}",
             f"D_3 {pts_full(r[3]['D'].values())}; C_3 {pts_full(r[3]['C'].values())}",
             f"INFO Bonferroni for two arms (t >= 3.50, df 4) on D_2: {'cleared' if bonferroni(r[2]['D'].values()) else 'not cleared'}"]
    rc3 = rc_reading(3, arm, r[3]["D"], r[3]["Dorig"], r[3]["Dre"], r[3]["C"], V, vtxt, d["kstar_cur"])
    bad = [x for x in ("I1", "I2", "I3", "I4") if not I[x]]
    mf = None
    if bad:
        mf = ("not run as registered", f"failing: {', '.join(bad)}; fix and re-run under the same ids, nothing is read")
    elif d["kstar_cur1"] < 3:
        mf = ("not installed", f"k*(CUR1, ca, fwd) = {d['kstar_cur1']} < 3: stage 1 did not reproduce the parent's "
                               "capacity at five seeds")
    elif d["kstar_cur"] < 3:
        mf = (f"installed, then lost in the GSM stage (arm {arm})",
              f"k*(CUR1, ca, fwd) = {d['kstar_cur1']} >= 3 and k*(CUR-{arm}, ca, fwd) = {d['kstar_cur']} < 3: the "
              "protection that kept the capacity at s16 does not keep it at five seeds; both curves and the forgetting "
              "line are printed; no transfer sentence")
    elif not I["I6"]:
        mf = ("control not a control", f"I6: CUR-{arm} - K1X-{arm} at (ca, fwd, 3) has CI lower <= 0; C_2 isolates no "
                                       "composition")
    elif not d["n1_cur"]:
        mf = ("GSM damaged or not recovered" if arm == "L" else "GSM damaged",
              f"N1 fails for CUR-{arm}; D_2 {pts_ci(r[2]['D'].values())} is INFO only")
    if mf:
        rc3 = ("INFO", f"MANIPULATION FAILED: {mf[0]}",
               f"D_3 {pts_ci(r[3]['D'].values())} (D_3^orig {pts_ci(r[3]['Dorig'].values())}, D_3^re "
               f"{pts_ci(r[3]['Dre'].values())}); C_3 {pts_ci(r[3]['C'].values())}; no rc3 reading and no transfer "
               "sentence under MANIPULATION FAILED")
        return "MANIPULATION FAILED", mf[0], lines + [mf[1]], rc3
    reading, sub, claim = rc_reading(2, arm, r[2]["D"], r[2]["Dorig"], r[2]["Dre"], r[2]["C"], V, vtxt, d["kstar_cur"])
    if reading == "PASS":
        if rc3[0] == "PASS":
            claim += f", and rc3 by {pts_ci(r[3]['D'].values())}"
        else:
            claim += (f"; rc3 does not meet the full rc3 rule (secondary reading {rc3[0]}"
                      + (f" \"{rc3[1]}\"" if rc3[1] else "") + f"; D_3 {pts_ci(r[3]['D'].values())})")
        claim += ("; composition of given steps is part of the rc2 limit and training it on clean chains reaches GSM "
                  "(H of CARD_khop_v1, binding not depth, is weakened for rc2); no mechanism is named, the claim stays "
                  "within the symbolic format, small values and this schedule")
    elif reading == "NULL":
        claim += ("; the model composes 3+ given steps in one pass and does not use that on GSM items, so the rc2 limit "
                  "is not the serial composition of given operations (supports binding, not depth, for the transfer "
                  "route; a surface gap, symbolic wording and values 1-99, is not excluded: the cag follow-up)")
    return reading, sub, lines + [f"claim: {claim}"], rc3


# --- the gate output ---------------------------------------------------------------------------------------------------
# The lines khop4_pilot_gate.py prints (stage/prep_khop4_confirm.py holds the same lists; readout/test_khop4_pilot_gate
# checks both against the gate's real output).
COMMON = ("C0", "C1 j975", "C1 j985", "C2", "C3", "C8", "C7c")
ARM_Q = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7b", "Q9", "Q10", "Q11")
LINE_RE = re.compile(r"^(PASS|FAIL)  (C0|C1 j975|C1 j985|C2|C3|C8|C7c|[LR]-Q(?:1|2|3|4|5|6|7b|9|10|11)): ")
INFO_RE = re.compile(r"^INFO  ([LR])-Q7: ")


def arm_lines(a):
    return tuple(f"{a}-{q}" for q in ARM_Q)


def gate_text_arm(text):
    """(arm or None, reason) for the text of a khop4 gate output. Arm only when: the last non-empty line is exactly
    'GATE PASS ARM R' or 'GATE PASS ARM L' and no other line starts with 'GATE'; every line that starts with 'PASS' or
    'FAIL' is a khop4 line name, and each of the 7 common and 2 x 10 arm lines appears exactly once; exactly one
    'INFO  L-Q7: ' and one 'INFO  R-Q7: ' line; no common line FAILs; no arm is "not run as registered" (Amendment 3:
    no R-Q1 / R-Q2 / L-Q1 / L-Q2 FAILs, no a-Q7b FAILs on a missing wall, no 'READING  not run as registered' line);
    and the arm recomputed by card 0.5 (R when every R line passes, else L when every L line passes) equals the last
    line's arm."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None, "empty"
    m = re.match(r"^GATE PASS ARM ([LR])$", lines[-1])
    if not m:
        return None, f"does not end with GATE PASS ARM R / L ({lines[-1:]})"
    if any(l.startswith("GATE") for l in lines[:-1]):
        return None, "a second GATE line"
    got = {}
    for l in lines:
        if l.startswith("PASS") or l.startswith("FAIL"):
            mm = LINE_RE.match(l)
            if not mm:
                return None, f"a PASS / FAIL line that is not a khop4 gate line: {l[:60]!r}"
            if mm.group(2) in got:
                return None, f"line {mm.group(2)} twice"
            got[mm.group(2)] = mm.group(1)
    want = set(COMMON) | set(arm_lines("L")) | set(arm_lines("R"))
    if set(got) != want:
        return None, f"lines missing {sorted(want - set(got))[:5]}"
    info = [INFO_RE.match(l).group(1) for l in lines if INFO_RE.match(l)]
    if sorted(info) != ["L", "R"]:
        return None, f"INFO a-Q7 lines {info} (need one per arm)"
    if any(got[c] != "PASS" for c in COMMON):
        return None, "a common line FAILs"
    misrun = sorted({a for a in ("R", "L") if any(got[f"{a}-{q}"] != "PASS" for q in ("Q1", "Q2"))}
                    | {x.group(1) for x in (re.match(r"^FAIL  ([LR])-Q7b: a wall is missing", l) for l in lines) if x}
                    | {x.group(1) for x in (re.match(r"^READING  not run as registered \(arm ([LR])\)", l) for l in lines)
                       if x})
    if misrun:
        return None, (f"arm {' and '.join(misrun)} not run as registered (a-Q1 / a-Q2 FAIL or a-Q7b on a missing wall): "
                      "rescue under the same ids, no arm is chosen (Amendment 3)")
    rule = ("R" if all(got[x] == "PASS" for x in arm_lines("R")) else
            "L" if all(got[x] == "PASS" for x in arm_lines("L")) else None)
    if rule != m.group(1):
        return None, f"the last line names arm {m.group(1)}, the line results give {rule}"
    return m.group(1), "ok"


def gate_arm(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"khop4_table: no gate output {path!r}")
    arm, why = gate_text_arm(open(path).read())
    if arm is None:
        raise SystemExit(f"khop4_table: the gate output {path} is not a passing khop4 gate output: {why}")
    return arm


def read_retry_walls(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"khop4_table: the retry-walls file is required (empty when no node retried): {path!r}")
    out = []
    for i, l in enumerate(open(path).read().splitlines()):
        if not l.strip():
            continue
        p = l.split()
        h = F(p[1]) if len(p) == 2 and re.match(r"^[0-9]+(\.[0-9]+)?$", p[1]) else None
        if h is None or not re.match(r"^[A-Za-z0-9_.:-]+$", p[0]):
            raise SystemExit(f"khop4_table: retry-walls line {i + 1} is not '<node> <hours>': {l!r}")
        out.append((p[0], h))
    return out


def count_landed(results_dir, ids):
    """{id: number of results_<id>_*.tgz files} (smoke_ tarballs are not counted)."""
    names = os.listdir(results_dir)
    return {j: sum(1 for n in names if re.match(rf"^results_{j}_.*\.tgz$", n)) for j in ids}


# --- main --------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--khop-dir", default=os.path.join(E, "src/data/eval/khop"))
    ap.add_argument("--hint-dir", default=os.path.join(E, "src/data/eval/hint"))
    ap.add_argument("--run", action="append", default=[], help="ARM:s<seed>=<tgz or dir>")
    ap.add_argument("--ref-orig", action="append", default=[])
    ap.add_argument("--ref-retrain", action="append", default=[])
    ap.add_argument("--gate-output", required=True)
    ap.add_argument("--retry-walls", required=True)
    ap.add_argument("--results-dir", default="", help="checked for two landed tarballs of one id")
    ap.add_argument("--json", default="")
    ap.add_argument("--env663", default=G.ENV663)
    a = ap.parse_args(argv)
    arm_a = gate_arm(a.gate_output)
    retry = read_retry_walls(a.retry_walls)
    khop = T.load_khop(a.khop_dir)
    for s in T.SETS:
        if len(khop.get(s, [])) != T.ROWS[s]:
            unreadable(f"{s} has {len(khop.get(s, []))} rows in --khop-dir {a.khop_dir}, CARD_khop_v1 1.2 has {T.ROWS[s]}")
    if not os.path.isfile(os.path.join(a.hint_dir or "", "hint_plain.jsonl")):
        unreadable(f"no hint files in --hint-dir {a.hint_dir!r}: the rc cells need them")
    specs = parse_specs(a.run, "--run")
    ctx = CONTEXT[arm_a]
    unknown = sorted(set(specs) - set(NEW) - set(ctx))
    if unknown:
        raise SystemExit(f"khop4_table: arms {unknown} are not part of arm {arm_a}'s readout")
    for x in NEW:
        if sorted(specs.get(x, {})) != list(SEEDS5):
            unreadable(f"arm {x} has seeds {sorted(specs.get(x, {}))}, the card reads exactly {list(SEEDS5)}")
    for x in ctx:
        if sorted(specs.get(x, {})) != sorted(CTX_JOBS[x]):
            unreadable(f"context {x} has seeds {sorted(specs.get(x, {}))}, the card names {sorted(CTX_JOBS[x])}")
    refs_in = {"orig": parse_specs(a.ref_orig, "--ref-orig").get("", {}),
               "retrain": parse_specs(a.ref_retrain, "--ref-retrain").get("", {})}
    for n, d in refs_in.items():
        if sorted(d) != list(SEEDS5):
            unreadable(f"reference set {n} has seeds {sorted(d)}, the card reads exactly {list(SEEDS5)}")
    # identities before any load: exact run names for the 20 jobs, whole ids for the references and the context
    for x in NEW:
        for sd, p in specs[x].items():
            nm = expected_name(x, arm_a, sd)
            if not os.path.isdir(p) and os.path.basename(p) != f"results_{nm}.tgz":
                unreadable(f"{x} s{sd} is {p}, the card names {nm}")
    for x in ctx:
        for sd, p in specs[x].items():
            if not has_id(p, CTX_JOBS[x][sd]):
                unreadable(f"context {x} s{sd} is {p}, the card names j{CTX_JOBS[x][sd]}")
    for n, d in refs_in.items():
        for sd, p in d.items():
            if not has_id(p, K2.REF_JOBS[n][sd]):
                unreadable(f"reference {n} s{sd} is {p}, the card names j{K2.REF_JOBS[n][sd]}")
    ids = sorted({s1_id(k, s) for k in ("CUR", "K1X") for s in SEEDS5} |
                 {s2_id(arm_a, k, s) for k in ("CUR", "K1X") for s in SEEDS5} |
                 {j for x in ctx for j in CTX_JOBS[x].values()} |
                 {int(j) for n in K2.REF_JOBS for j in K2.REF_JOBS[n].values()})
    if a.results_dir:
        cnt = count_landed(a.results_dir, ids)
        two = sorted(j for j, c in cnt.items() if c > 1)
        if two:
            unreadable(f"two or more landed tarballs for ids {two} in {a.results_dir}")
    s2need = K3.s2_need(a.env663)
    need = {"CUR1": K3.S1_NEED, "K1X1": K3.S1_NEED, "CUR": s2need, "K1X": s2need, "J980": s2need, "J990": s2need,
            "KH": s2need, "K1": s2need}
    runs = {x: {s: load(p, f"{x} s{s}", need.get(x, PLAIN)) for s, p in sorted(d.items())} for x, d in specs.items()}
    # Codex round 4: the preds files that args.eval does not list must be present too (the gate's a-Q1 / P1 form):
    # every GSM / hint set of the 663 env, preds_subq included, once in each S2 run; preds_subq (eval_subq) once in
    # each S1 run.
    files_need = {"CUR": tuple(G.gsm_sets(a.env663)), "K1X": tuple(G.gsm_sets(a.env663)), "CUR1": ("subq",),
                  "K1X1": ("subq",)}
    for x, d in runs.items():
        for sd, r in d.items():
            if x in NEW and run_name(r) != expected_name(x, arm_a, sd):
                unreadable(f"{x} s{sd}: the run is {run_name(r)}, the card names {expected_name(x, arm_a, sd)}")
            fmiss = [g for g in files_need.get(x, ()) if r["all_preds"].count(g) != 1]
            if fmiss:
                unreadable(f"{x} s{sd}: preds files {fmiss[:5]} missing or not exactly once in {r.get('path')}")
            K2.check_seed(r, sd, f"arm {x} s{sd}")
            if x in NEW:    # Codex rounds 1 and 5: timing counters checked here, before any line is printed
                h = r["metrics"]["history"]
                bad_t = [i for i, y in enumerate(h) if not (isinstance(y, dict) and K3.finite(y.get("s_per_step"))
                                                             and K3.finite(y.get("peak_gib")))]
                if not h or bad_t:
                    unreadable(f"{x} s{sd}: metrics.json history has no rows or rows {bad_t[:5]} without a finite "
                               "s_per_step and peak_gib")
            K3.history_finite(r, f"{x} s{sd}")
            if x in NEW and GH.wall(r["log"]) is None:
                unreadable(f"timing {x} s{sd}: no wall in the log of {r.get('path')}")
    refs = {n: {s: load(p, f"{n} s{s}") for s, p in d.items()} for n, d in refs_in.items()}
    for n, d in refs.items():
        for sd, r in d.items():
            K2.check_seed(r, sd, f"reference {n} s{sd}")
            K3.history_finite(r, f"reference {n} s{sd}")
    out = {"lines": [], "I": {}, "arm": arm_a}
    pr = lambda *x: (print(*x), out["lines"].append(" ".join(str(y) for y in x)))
    pr(f"khop4 readout, arm {arm_a} (read from the gate output's last line)")
    byk = {s: T.by_k(khop[s]) for s in T.SETS}
    cellarms = NEW + tuple(c for c in ("J980", "J990", "KH", "K1", "D27") if c in runs)
    xc = {x: {s: {k: {sd: K2.xacc(r, s, ids_, f"{x} s{sd} {s} k{k}") for sd, r in runs[x].items()}
                  for k, ids_ in sorted(byk[s].items())} for s in SCORED} for x in cellarms if x in NEW or x != "D27"}
    cell = lambda arm, s, k: xc[arm][s][k]
    blind_ids = [r["item_id"] for r in khop["khop_blind"]]
    # I4: log lines, values and the all-args recipe check
    i4r = {}
    for x in NEW:
        kind = kind_of(x)
        for sd, r in runs[x].items():
            if x.endswith("1"):
                ok, det = i4_s1(r, kind, sd)
                if sd != 16:
                    ok2, det2 = recipe_s1_confirm(r, kind, runs[x][16])
                    ok, det = ok and ok2, det + "; " + det2
            else:
                ok, det = i4_s2(r, arm_a, kind, sd, s1_name(kind, sd))
                if sd == 16:
                    base = runs.get("KH" if kind == "CUR" else "K1", {}).get(16)
                    ok2, det2 = recipe_pilot(r, arm_a, kind, runs["J980" if kind == "CUR" else "J990"][16], base)
                else:
                    ok2, det2 = recipe_s2_confirm(r, runs[x][16])
                ok, det = ok and ok2, det + "; " + det2
            i4r[(x, sd)] = (ok, det)
    I = out["I"]
    I["I1"] = all(K2.XS(cell("CUR", f"khop_{t}_{o}", 1).values()).m >= XBAR["I1"] for t, o in T.CELLS6)
    qs_cur = [K2.xqs_pair(r, khop, f"CUR s{sd} qs pair") for sd, r in runs["CUR"].items()]
    I["I2"] = K2.XS(qs_cur).m >= XBAR["I2"]
    bl = {x: [K2.xacc(r, "khop_blind", blind_ids, f"{x} s{sd} blind") for sd, r in runs[x].items()] for x in NEW}
    I["I3"] = all(K2.XS(bl[x]).m <= XBAR["I3"] for x in ("CUR", "CUR1"))
    I["I4"] = all(v[0] for v in i4r.values())
    ks1 = K2.xkstar({k: cell("CUR1", "khop_ca_fwd", k) for k in range(1, 7)})
    ks = K2.xkstar({k: cell("CUR", "khop_ca_fwd", k) for k in range(1, 7)})
    I["I5"] = ks >= 3
    i6 = K2.pdiff(cell("CUR", "khop_ca_fwd", 3), cell("K1X", "khop_ca_fwd", 3))
    K2.need_seeds(i6, "I6")
    I["I6"] = K2.XS(i6.values()).lo_gt(0)
    i7l = {sd: K3.i7_loss(r["log"]) for sd, r in runs["K1X1"].items()}
    i7_k1 = {f"{t}/{o}": K2.XS(cell("K1X1", f"khop_{t}_{o}", 1).values()).m for t, o in T.CELLS6}
    i7_k1s2 = {f"{t}/{o}": K2.XS(cell("K1X", f"khop_{t}_{o}", 1).values()).m for t, o in T.CELLS6}
    I["I7"] = (all(v[0] <= XBAR["I7_loss"] for v in i7l.values()) and all(v >= XBAR["I7_k1"] for v in i7_k1.values())
               and all(v >= XBAR["I7_k1_s2"] for v in i7_k1s2.values()))
    pr("installation:", json.dumps(I))
    pr(f"I1 CUR-{arm_a} k=1 cells " + " ".join(f"{t}/{o} {acc_ci(cell('CUR', f'khop_{t}_{o}', 1).values())}"
                                             for t, o in T.CELLS6) + " (bar m >= 0.90)")
    pr(f"I2 CUR-{arm_a} qs pair k=1 {acc_ci(qs_cur)} (bar m >= 0.85) | I3 blind CUR-{arm_a} {acc_ci(bl['CUR'])} CUR1 "
       f"{acc_ci(bl['CUR1'])} (bar m <= 0.30)")
    for k_, v in sorted(i4r.items()):
        pr(f"I4 {k_[0]} s{k_[1]}: {'ok' if v[0] else 'FAIL'} ({v[1]})")
    pr(f"I5 k*(CUR-{arm_a}, ca, fwd) {ks} (k*(CUR1) {ks1}) | I6 CUR-{arm_a} - K1X-{arm_a} at (ca, fwd, 3) "
       f"{pts_ci(i6.values())} (CI lower > 0)")
    ks4 = K2.xkstar({k: {s: v for s, v in cell("CUR", "khop_ca_fwd", k).items() if s in S17} for k in range(1, 7)})
    ks14 = K2.xkstar({k: {s: v for s, v in cell("CUR1", "khop_ca_fwd", k).items() if s in S17} for k in range(1, 7)})
    pr(f"INFO I5 over s17-20 alone (n 4, t 3.18; s16 passed the gate's installation line): k*(CUR-{arm_a}) {ks4}, "
       f"k*(CUR1) {ks14}")
    pr("I7 K1X1 loss mean of the step lines >= 760: " + " ".join(f"s{sd} {float(v[0]):.3f} (n {v[1]})"
                                                               for sd, v in sorted(i7l.items()))
       + " (bar <= 1.0) | K1X1 k=1 cells " + " ".join(f"{c} {float(v):.3f}" for c, v in i7_k1.items())
       + f" (bar m >= 0.98) | K1X-{arm_a} k=1 cells " + " ".join(f"{c} {float(v):.3f}" for c, v in i7_k1s2.items())
       + " (bar m >= 0.90)")
    # plain sets (exact counts; best-A2 per set)
    pid = K3.plain_ids(refs["orig"][16], "orig s16")
    xp = {n: {sd: {p: K3.xplain(r, p, pid[p], f"{n} s{sd}") for p in PLAIN} for sd, r in refs[n].items()}
          for n in ("orig", "retrain")}
    for x in runs:
        xp[x] = {sd: {p: K3.xplain(r, p, pid[p], f"{x} s{sd}") for p in PLAIN} for sd, r in runs[x].items()}
    bestp = {p: K2.best_set({n: xp[n] for n in ("orig", "retrain")}, p) for p in PLAIN}
    get = lambda src, key: {s: v[key] for s, v in src.items()}
    n1, n1_4 = {}, {}
    for x in ("CUR", "K1X", "CUR1", "K1X1"):
        n1[x], n1_4[x] = {}, {}
        for p in PLAIN:
            xs_, rs_ = get(xp[x], p), get(xp[bestp[p]], p)
            K2.need_seeds(xs_, f"plain {x} {p}")
            xm, rm = K2.XS(xs_.values()).m, K2.XS(rs_.values()).m
            n1[x][p] = xm >= rm - XBAR["N1"]
            n1_4[x][p] = (K2.XS([xs_[s] for s in S17]).m >= K2.XS([rs_[s] for s in S17]).m - XBAR["N1"])
            pr(f"plain {x:4s} {p:8s} mean {100 * float(xm):.2f} | best-A2 ({bestp[p]}) {100 * float(rm):.2f} | X - best-A2 "
               f"{pts_ci(K2.pdiff(xs_, rs_).values())}" + ("" if x in ("CUR", "K1X") else " (stage-1 model, reported)"))
        if x in ("CUR", "K1X"):
            pr(f"N1 {x}-{arm_a}: {'holds' if all(n1[x].values()) else 'fails'} {json.dumps(n1[x])}")
            pr(f"INFO N1 {x}-{arm_a} over s17-20 alone (n 4; s16 passed a-Q6): "
               f"{'holds' if all(n1_4[x].values()) else 'fails'} {json.dumps(n1_4[x])}")
    n1_cur, n1_k1x = all(n1["CUR"].values()), all(n1["K1X"].values())
    # rc cells
    rc = {n: {sd: K2.xrc_cells(r["path"], a.hint_dir, f"{n} s{sd}") for sd, r in refs[n].items()} for n in refs}
    for x in ("CUR", "K1X") + tuple(c for c in ("KH", "K1", "J980", "J990", "D27") if c in runs):
        rc[x] = {sd: K2.xrc_cells(r["path"], a.hint_dir, f"{x} s{sd}") for sd, r in runs[x].items()}
    pr("rc populations (primary items, pfx_table.py): " + ", ".join(f"rc{k} {n}" for k, n in K2.RC_N.items()))
    for name, d_ in rc.items():
        for s, c in sorted(d_.items()):
            pr(f"rc cells {name:8s} s{s} " + " ".join(f"rc{k} {100 * float(x):.1f}" for k, x in c.items()))
    bestrc = {j: K2.best_set({n: rc[n] for n in ("orig", "retrain")}, j) for j in (1, 2, 3, 4)}
    rcd = {}
    for j in (1, 2, 3, 4):
        rcd[j] = {"D": K2.pdiff(get(rc["CUR"], j), get(rc[bestrc[j]], j)),
                  "Dorig": K2.pdiff(get(rc["CUR"], j), get(rc["orig"], j)),
                  "Dre": K2.pdiff(get(rc["CUR"], j), get(rc["retrain"], j)),
                  "C": K2.pdiff(get(rc["CUR"], j), get(rc["K1X"], j)),
                  "K1Xb": K2.pdiff(get(rc["K1X"], j), get(rc[bestrc[j]], j))}
    v, sub, lines, rc3 = verdict({"arm": arm_a, "I": I, "kstar_cur1": ks1, "kstar_cur": ks, "n1_cur": n1_cur,
                                  "n1_k1x": n1_k1x, "n1_k1x_fail_sets": [p for p in PLAIN if not n1["K1X"][p]],
                                  "rc": rcd})
    for l in lines:
        pr(l)
    out["verdict"], out["sublabel"] = v, sub
    pr(f"VERDICT (arm {arm_a}): {v}" + (f" ({sub})" if sub else ""))
    out["rc3"] = rc3
    pr(f"rc3 reading (secondary, never changes the verdict): {rc3[0]}" + (f" ({rc3[1]})" if rc3[1] else "")
       + f": {rc3[2]}")
    # 2.2 curves and forgetting
    for x in NEW:
        for sd in SEEDS5:
            pr(f"curve {x:4s} s{sd} (ca, fwd) " + " ".join(f"k{k} {100 * float(cell(x, 'khop_ca_fwd', k)[sd]):.1f}"
                                                        for k in range(1, 9)))
    for c in ("J980", "J990"):
        pr(f"curve {c} s16 (ca, fwd) (khop3 context) " + " ".join(f"k{k} {100 * float(cell(c, 'khop_ca_fwd', k)[16]):.1f}"
                                                                 for k in range(1, 9)))
    d27c = {sd: T.curve(r, khop, "khop_ca_fwd") for sd, r in runs["D27"].items()}
    pr("curve D27 (ca, fwd) (parent, context) " + " | ".join(f"k{k} {acc_ci([c.get(k) for c in d27c.values()])}"
                                                           for k in range(1, 9)))
    for k in (2, 3, 4):
        pr(f"forgetting a(CUR-{arm_a}, ca, fwd, {k}) - a(CUR1, ca, fwd, {k}) "
           f"{pts_ci(K2.pdiff(cell('CUR', 'khop_ca_fwd', k), cell('CUR1', 'khop_ca_fwd', k)).values())}"
           f" | K1X: {pts_ci(K2.pdiff(cell('K1X', 'khop_ca_fwd', k), cell('K1X1', 'khop_ca_fwd', k)).values())}")
    # 2.3 secondary
    for j in (1, 2, 3, 4):
        pr(f"rc{j} (best-A2 {bestrc[j]}): D {pts_ci(rcd[j]['D'].values())} | D^orig {pts_ci(rcd[j]['Dorig'].values())} | "
           f"D^re {pts_ci(rcd[j]['Dre'].values())} | C {pts_ci(rcd[j]['C'].values())} | K1X - best-A2 "
           f"{pts_ci(rcd[j]['K1Xb'].values())}")
    if arm_a == "R":
        for x, c in (("CUR", "KH"), ("K1X", "K1")):
            for j in (1, 2, 3, 4):
                pr(f"{x}-R - {c} rc{j} {pts_ci(K2.pdiff(get(rc[x], j), get(rc[c], j)).values())} (same stage-2 recipe "
                   "from the base model: the effect of stage 1)")
            for p in PLAIN:
                pr(f"{x}-R - {c} {p} {pts_ci(K2.pdiff(get(xp[x], p), get(xp[c], p)).values())}")
            for k in (2, 3, 4):
                pr(f"{x}-R - {c} (ca, fwd, {k}) "
                   f"{pts_ci(K2.pdiff(cell(x, 'khop_ca_fwd', k), cell(c, 'khop_ca_fwd', k)).values())}")
    else:
        for x, c in (("CUR", "J980"), ("K1X", "J990")):
            pr(f"INFO {x}-L - {c[1:]} at s16 (n 1, the LR alone): "
               + " ".join(f"rc{j} {100 * float(rc[x][16][j] - rc[c][16][j]):+.1f}" for j in (1, 2, 3, 4)) + " | "
               + " ".join(f"{p} {100 * float(xp[x][16][p] - xp[c][16][p]):+.1f}" for p in PLAIN) + " | "
               + " ".join(f"(ca, fwd, {k}) {100 * float(cell(x, 'khop_ca_fwd', k)[16] - cell(c, 'khop_ca_fwd', k)[16]):+.1f}"
                          for k in (2, 3, 4)))
    other = "L" if arm_a == "R" else "R"
    o_ids = {PILOT[(other, "CUR")], PILOT[(other, "K1X")]}
    for l in open(a.gate_output).read().splitlines():
        mj = re.match(r"^INFO j(\d+) ", l)               # Codex round 2 finding 2: the gate's single-space INFO lines too
        if (l.startswith((f"PASS  {other}-", f"FAIL  {other}-", f"INFO  {other}-", f"INFO {other}-"))
                or (mj and int(mj.group(1)) in o_ids)):
            pr(f"INFO unchosen arm {other} (s16 gate line, no decision): {l}")
    if n1_cur:
        for k in (2, 3):
            g = K2.pdiff(cell("CUR", "khop_ca_fwd", k), get(rc["CUR"], k))
            pr(f"G{k} = a(CUR-{arm_a}, ca, fwd, {k}) - rc{k}(CUR-{arm_a}) {pts_ci(g.values())}")
    else:
        pr("G2 / G3: not printed (N1 fails for CUR)")
    for x in ("CUR", "CUR1"):
        kd = {f"{t}/{o}": K2.xkstar({k: cell(x, f"khop_{t}_{o}", k) for k in range(1, 7)}) for t, o in T.CELLS6}
        kd["cag/fwd"] = K2.xkstar({k: cell(x, "khop_cag_fwd", k) for k in range(1, 7)})
        pr(f"k* {x}: {json.dumps(kd)}")
        for t, o in [("wp", "fwd"), ("wp", "rev"), ("vb", "fwd"), ("vb", "rev"), ("cag", "fwd"), ("ca", "rev")]:
            pr(f"a({x}, {t}, {o}, 2..4) " + " ".join(f"k{k} {acc_ci(cell(x, f'khop_{t}_{o}', k).values())}"
                                                   for k in (2, 3, 4)))
    add3 = T.alladd_ids(khop, "khop_ca_fwd").get(3, [])
    gold_i = {r["item_id"]: "ABCDE".index(r["gold_label"]) for r in khop["khop_ca_fwd"]}
    os3 = [r["item_id"] for r in khop["khop_ca_fwd"] if r["k"] == 3 and T.rank_probe(r, "rank_sum")[1] != [gold_i[r["item_id"]]]]
    for name, sel in (("all-+/-", add3), ("order-sensitive", os3)):
        vals = [K2.xacc(r, "khop_ca_fwd", sel, f"CUR s{sd} {name}") for sd, r in runs["CUR"].items()]
        pr(f"{name} items of (ca, fwd, 3) under CUR-{arm_a} (n {len(sel)}): {acc_ci(vals)}")
    # timing and compute
    tot = F(0)
    for x in NEW:
        for sd, r in sorted(runs[x].items()):
            h = r["metrics"]["history"]      # every row has a finite s_per_step and peak_gib (checked at load)
            sps = [y["s_per_step"] for y in h]
            w = GH.wall(r["log"])
            tot += w
            gpu = re.findall(r"^gpu 0: ([^,]+),", r["log"], re.M)
            sz = K3.weights_export(r["log"], run_name(r) or "") if x.endswith("1") else []
            pr(f"timing {x} s{sd}: wall {float(w):.2f} h ({gpu[0] if gpu else 'gpu ?'}), s_per_step "
               f"{sum(sps) / len(sps):.2f}, peak_gib "
               f"{max(y['peak_gib'] for y in h):.1f}" + (f", weights tar {sz[0]}" if sz else ""))
    rw = sum((h for _, h in retry), F(0))
    pr(f"compute: the 20 landed jobs {float(tot):.2f} GPU-h (logs; the s16 S1 pair 975 / 985 was counted in khop3) + "
       f"retry walls {float(rw):.2f} GPU-h over {len(retry)} failed executions = {float(tot + rw):.2f}; the smokes are in "
       "khop4_gpuh.py 1025 1049 <results dir> (run_khop4.sh prints it and stops on an n/a row)")
    out.update({"kstar_cur1": ks1, "kstar_cur": ks, "N1": {x: n1[x] for x in ("CUR", "K1X")},
                "rc": {j: {n: {s: float(x) for s, x in d_.items()} for n, d_ in rcd[j].items()} for j in rcd}})
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
