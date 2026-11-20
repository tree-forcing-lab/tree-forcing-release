"""Training directories of the k-hop design (family khop3, parent khop2): the curriculum stage-1 files for the arms CUR and
K1X, and data/eval/khop3/build_report.json. No chain item is drawn anew.

    python3 scripts/build_khop3.py --khop2 <E>/src/data/eval/khop2 --khop <E>/src/data/eval/khop \\
        --gsm-dir <E>/src/data/student_gsm8k_tf_vt --out-root <E>/src/data --report <E>/src/data/eval/khop3

This script imports scripts/build_khop.py and scripts/build_khop2.py and edits neither. Steps (the design 1.2):
1. Inputs. train_kd27.jsonl (--khop2) must have the md5 of the khop2 build_report.json and 92545dd4...; train_k1.jsonl
   (--khop) the khop2 report's parent_md5 and 4bb15f66...; the gsmvt val.jsonl (--gsm-dir) 9f55fec5... and 393 rows;
   the 12 the k-hop design files the md5s of the khop build_report.json (build_khop2.check_parent). Any other value is a
   SystemExit. --out-root, --report or an output directory equal to an input directory is a SystemExit.
2. student_khop3_c27/train.jsonl = train_kd27.jsonl byte for byte; student_khop3_c27/val.jsonl = the gsmvt val.jsonl
   byte for byte (train_arm.sbatch always evaluates $DATA/val.jsonl).
3. student_khop3_k1x/train.jsonl: line i = the K1 twin of line i of train_kd27.
   a. Replay (replay): build_khop.Item.row is wrapped to record the Item of every rendered item_id (the last call per
      item_id wins, so a redrawn row maps to the accepted item). build_khop.build_train(random.Random(<khop seeds.train>),
      ..., n_kh, n_kd_extra) and build_khop2.new_rows(<khop2 seed>, <new_per_task>, forbid_keys, forbid_hash, cycle0)
      run with the parameters the two parent reports record and the inputs build_khop2.build computes from the 12
      parent files. check_replay stops the build unless the replayed rows, in kd27 order, are dict-equal to the 27,000
      train_kd27 rows and the replayed K1 rows are dict-equal to the 3,000 train_k1 rows.
   b. Rows 0..2,999 (KH) get the train_k1 line byte for byte. Rows 3,000..26,999: it.row(item_id + "-k1", order,
      layout(i1, k, rng1), hop=1, group=<the kd27 row's group>), rng1 = random.Random(f"khop3-k1-{item_id}"), i1 = the
      running count of earlier K1X rows with the same (task, k), starting from the train_k1 counts; order from the kd27
      row's group.
   student_khop3_k1x/val.jsonl = the gsmvt val.jsonl byte for byte.
4. build_report.json in --report: md5s and row counts of the 4 files and the 3 inputs, the replay results and the K1X
   checks (gold letter shares, statement text equal to the kd27 twin's, query = the target chain's hop-1 variable and
   gold = its value by build_khop.parse_prompt, ca / wp option rule, vb options = the five roots, build_khop.row_rule_ok,
   eval prompt-hash overlap, kd27 prompt-hash overlap (reported; only k = 1 rows may share), distinct item_ids, one-hot
   teachers, the 32 gsmvt keys, token totals of K1X and kd27 with the tokenizer the khop2 report records (characters
   when none is recorded; a recorded tokenizer that does not load is a SystemExit), K1X within 1% of kd27).
   d_fail lists every failed line; d_pass is true only when it is empty. main exits 2 on a failed line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_khop as B  # noqa: E402
import build_khop2 as B2  # noqa: E402

KD27_MD5 = "92545dd42ca451c39e0e7c104aa78d1e"
K1_MD5 = "4bb15f66c8d8843013eb0410c6a87e48"
VAL_MD5 = "9f55fec541246e41401a7b5fa6c5510a"
N_KD27, N_K1, N_VAL = 27000, 3000, 393
SHARE = (0.18, 0.22)              # gold letter shares of K1X
TOK_TOL = 0.01                    # K1X token total within 1% of kd27
DIRS = {"c27": "student_khop3_c27", "k1x": "student_khop3_k1x"}
OUT_FILES = ("train.jsonl", "val.jsonl")


class ReplayError(SystemExit):
    pass


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5_file(p: Path) -> str:
    return md5_bytes(Path(p).read_bytes())


def lines_of(p: Path) -> list[bytes]:
    return Path(p).read_bytes().splitlines(keepends=True)


def stmt_text(prompt: str) -> str:
    """The prompt up to the query sentence (the statements, as in build_khop.parse_prompt)."""
    head = prompt.partition("\nOptions:\n")[0]
    m = B._Q.search(head)
    if not m:
        raise ValueError("no khop query")
    return head[:m.start()]


def check_inputs(khop2: Path, khop: Path, gsm: Path) -> dict:
    """Step 1: the parent reports and the md5s of every input; SystemExit on any mismatch."""
    rp2 = khop2 / "build_report.json"
    if not rp2.exists():
        raise SystemExit(f"build_khop3: no khop2 report {rp2}")
    rep2 = json.loads(rp2.read_text())
    kd27 = khop2 / "train_kd27.jsonl"
    if not kd27.exists():
        raise SystemExit(f"build_khop3: no {kd27}")
    got = md5_file(kd27)
    if got != (rep2.get("md5") or {}).get("train_kd27.jsonl") or got != KD27_MD5:
        raise SystemExit(f"build_khop3: md5 of {kd27} is {got}; the khop2 report says "
                         f"{(rep2.get('md5') or {}).get('train_kd27.jsonl')}, expected {KD27_MD5}")
    rep1 = B2.check_parent(khop)            # the 12 khop files = the khop report (SystemExit otherwise)
    k1 = khop / "train_k1.jsonl"
    got = md5_file(k1)
    if got != (rep2.get("parent_md5") or {}).get("train_k1.jsonl") or got != K1_MD5:
        raise SystemExit(f"build_khop3: md5 of {k1} is {got}; the khop2 report's parent_md5 says "
                         f"{(rep2.get('parent_md5') or {}).get('train_k1.jsonl')}, expected {K1_MD5}")
    for s in B2.PARENT_FILES:
        if (rep2.get("parent_md5") or {}).get(f"{s}.jsonl") != rep1["md5"][f"{s}.jsonl"]:
            raise SystemExit(f"build_khop3: the khop2 report's parent_md5 of {s}.jsonl differs from the khop report")
    val = gsm / "val.jsonl"
    if not val.exists():
        raise SystemExit(f"build_khop3: no {val}")
    got, n = md5_file(val), len(lines_of(val))
    if got != VAL_MD5 or n != N_VAL:
        raise SystemExit(f"build_khop3: {val} has md5 {got} and {n} rows; expected {VAL_MD5}, {N_VAL} rows")
    return {"khop": rep1, "khop2": rep2}


def stream_params(reps: dict) -> dict:
    """The parent generator parameters, read from the two reports (not assumed)."""
    r1, r2 = reps["khop"], reps["khop2"]
    tot = lambda s: (r1.get("counts") or {}).get(s, {}).get("total")
    n_kh, n_kd = tot("train_kh"), tot("train_kd")
    nt = len(B.TASKS)
    if not isinstance(n_kh, int) or not isinstance(n_kd, int) or n_kh % nt or n_kd % nt or n_kd <= n_kh:
        raise SystemExit(f"build_khop3: the khop report's train counts ({n_kh}, {n_kd}) do not give per-task sizes")
    p = {"train_seed": (r1.get("seeds") or {}).get("train"), "n_kh": n_kh // nt, "n_kd_extra": (n_kd - n_kh) // nt,
         "khop2_seed": r2.get("seed"), "new_per_task": r2.get("new_per_task")}
    if not all(isinstance(v, int) for v in p.values()):
        raise SystemExit(f"build_khop3: parent generator parameters missing from the reports: {p}")
    return p


def replay(khop: Path, params: dict):
    """Step 3a: run the two parent streams with Item.row wrapped. Returns (kd27 rows, K1 rows, {item_id: Item},
    {item_id: number of row() calls}; more than one call happens only on a prompt-hash redraw in build_khop2)."""
    rec, calls = {}, Counter()
    orig = B.Item.row

    def wrapped(self, item_id, *a, **kw):
        r = orig(self, item_id, *a, **kw)
        rec[item_id] = self                  # last call wins: a redrawn row maps to the accepted item
        calls[item_id] += 1
        return r

    rows = {s: [json.loads(l) for l in lines_of(khop / f"{s}.jsonl")] for s in B2.PARENT_FILES}
    forbid_keys, forbid_hash = set(), set()   # build_khop2.build's inputs, from the 12 parent files
    for s in B2.PARENT_FILES:
        for r in rows[s]:
            forbid_hash.add(B.phash(r["prompt"]))
            k = B2.row_key(r)
            if k is not None:
                forbid_keys.add(k)
    cycle0 = Counter((B.task_of(r), r["depth"]) for r in rows["train_kd"])
    B.Item.row = wrapped
    try:
        _, k1, kd, _, _ = B.build_train(random.Random(params["train_seed"]), defaultdict(Counter), params["n_kh"],
                                         params["n_kd_extra"])
        new, _, _, _ = B2.new_rows(params["khop2_seed"], params["new_per_task"], forbid_keys, forbid_hash, cycle0)
    finally:
        B.Item.row = orig
    return kd + new, k1, rec, calls


def check_replay(rep_kd27: list, rep_k1: list, kd27: list, k1: list) -> dict:
    """Step 3a: the replayed rows must be dict-equal to the files, row by row; else the build stops."""
    eq_kd = sum(a == b for a, b in zip(rep_kd27, kd27))
    eq_k1 = sum(a == b for a, b in zip(rep_k1, k1))
    res = {"kd27_rows_equal": eq_kd, "kd27_rows": len(kd27), "kd27_replayed": len(rep_kd27),
           "k1_rows_equal": eq_k1, "k1_rows": len(k1), "k1_replayed": len(rep_k1)}
    if not (len(rep_kd27) == len(kd27) == eq_kd == N_KD27 and len(rep_k1) == len(k1) == eq_k1 == N_K1):
        raise ReplayError(f"build_khop3: the replay does not reproduce the parent files {res}; no other generator is "
                          "chosen after the fact (a design change is needed)")
    return res


def k1x_lines(kd27: list, k1_lines: list, rec: dict) -> list[bytes]:
    """Step 3b: the K1X lines in kd27 order."""
    k1 = [json.loads(l) for l in k1_lines]
    out = []
    for i in range(N_K1):
        if k1[i]["item_id"] != kd27[i]["item_id"] + "-k1":
            raise SystemExit(f"build_khop3: train_k1 line {i} ({k1[i]['item_id']}) is not the twin of kd27 line {i}")
        out.append(k1_lines[i])
    cnt = Counter((B.task_of(r), r["depth"]) for r in k1)
    for r in kd27[N_K1:]:
        task, k, order = B.task_of(r), r["depth"], r["group"].split("/")[2]
        it = rec[r["item_id"]]
        rng1 = random.Random(f"khop3-k1-{r['item_id']}")
        i1 = cnt[(task, k)]
        cnt[(task, k)] += 1
        out.append(B2.dump(it.row(r["item_id"] + "-k1", order, B.layout(i1, k, rng1), hop=1, group=r["group"])))
    return out


def twin_checks(kd27: list, k1x: list) -> dict:
    """The K1X lines of step 4, from the rows alone (parse_prompt)."""
    c = Counter()
    shares = defaultdict(Counter)
    for a, t in zip(kd27, k1x):
        task, k = B.task_of(t), t["depth"]
        if stmt_text(t["prompt"]) != stmt_text(a["prompt"]):
            c["statement_mismatch"] += 1
        Pa, Pt = B.parse_prompt(a["prompt"]), B.parse_prompt(t["prompt"])
        root = Pa["chain_of"][Pa["query"]]
        hop1 = [s["name"] for s in Pa["stmts"] if s["parent"] == root]
        if len(hop1) != 1 or Pt["query"] != hop1[0]:
            c["query_not_hop1"] += 1
        gold_val = Pt["options"][B.LETTERS.index(t["gold_label"])]
        if Pt["query"] not in Pt["values"] or gold_val != Pt["values"][Pt["query"]]:
            c["gold_not_hop1_value"] += 1
        if task in B.RANGES:
            if not B.option_set_ok(Pt["options"], set(Pt["literals"])):
                c["option_rule_fail"] += 1
        else:
            roots = sorted(s["c"] for s in Pt["stmts"] if s["op"] == "root")
            if sorted(Pt["options"]) != roots:
                c["vb_options_not_roots"] += 1
        if B.row_rule_ok(t):
            c["row_rule_errors"] += 1
        if t["teacher"] != {L: (1.0 if L == t["gold_label"] else 0.0) for L in B.LETTERS}:
            c["teacher_not_onehot"] += 1
        if set(t) != set(B.GSMVT_KEYS) or set(a) != set(B.GSMVT_KEYS):
            c["keyset_differs"] += 1
        if (t["group"], t["depth"]) != (a["group"], a["depth"]) or t["item_id"] != a["item_id"] + "-k1":
            c["id_group_depth_differs"] += 1
        shares[f"{task}/k{k}"][t["gold_label"]] += 1
        shares["all"][t["gold_label"]] += 1
    out = {x: c[x] for x in ("statement_mismatch", "query_not_hop1", "gold_not_hop1_value", "option_rule_fail",
                             "vb_options_not_roots", "row_rule_errors", "teacher_not_onehot", "keyset_differs",
                             "id_group_depth_differs")}
    out["gold_shares"] = {cell: {L: round(cn[L] / sum(cn.values()), 4) for L in B.LETTERS}
                          for cell, cn in sorted(shares.items())}
    out["gold_counts"] = {cell: {L: cn[L] for L in B.LETTERS} for cell, cn in sorted(shares.items())}
    return out


def load_tok(rep2: dict, tok):
    """tok "auto": the tokenizer the khop2 report records (None -> characters); a recorded one that does not load stops
    the build. Tests pass tok=None (characters) or a tokenizer object."""
    if tok != "auto":
        return tok
    name = rep2.get("tokenizer")
    if not name:
        return None
    try:
        import transformers
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as e:                    # fail closed: never fall back to characters silently
        raise SystemExit(f"build_khop3: the khop2 report records the tokenizer {name}, which does not load ({e})")


def n_units(tok, prompt: str) -> int:
    return len(prompt) if tok is None else B.n_tokens(tok, prompt)


def build(khop2: Path, khop: Path, gsm: Path, out_root: Path, report_dir: Path, tok="auto") -> dict:
    t0 = time.time()
    inputs = {p.resolve() for p in (khop2, khop, gsm)}
    outs = {out_root.resolve(), report_dir.resolve()} | {(out_root / d).resolve() for d in DIRS.values()}
    if inputs & outs:
        raise SystemExit(f"build_khop3: --out-root / --report must not be an input directory ({sorted(map(str, inputs & outs))})")
    reps = check_inputs(khop2, khop, gsm)
    params = stream_params(reps)
    in_files = [khop2 / "train_kd27.jsonl", khop2 / "build_report.json", gsm / "val.jsonl"] + \
               [khop / f"{s}.jsonl" for s in B2.PARENT_FILES] + [khop / "build_report.json"]
    md5_before = {str(p): md5_file(p) for p in in_files}
    kd27_lines, k1_lines, val_bytes = lines_of(khop2 / "train_kd27.jsonl"), lines_of(khop / "train_k1.jsonl"), \
        (gsm / "val.jsonl").read_bytes()
    kd27, k1 = [json.loads(l) for l in kd27_lines], [json.loads(l) for l in k1_lines]
    rep_kd27, rep_k1, rec, calls = replay(khop, params)
    replay_res = check_replay(rep_kd27, rep_k1, kd27, k1)
    replay_res["ids_rendered_more_than_once"] = sum(v > 1 for v in calls.values())
    k1x = k1x_lines(kd27, k1_lines, rec)
    k1x_rows = [json.loads(l) for l in k1x]
    files = {("c27", "train.jsonl"): b"".join(kd27_lines), ("c27", "val.jsonl"): val_bytes,
             ("k1x", "train.jsonl"): b"".join(k1x), ("k1x", "val.jsonl"): val_bytes}
    for d in DIRS.values():
        p = out_root / d
        extra = sorted(x.name for x in p.iterdir() if x.name not in OUT_FILES) if p.exists() else []
        if extra:
            raise SystemExit(f"build_khop3: {p} holds other files {extra}; the bundle needs exactly {list(OUT_FILES)}")
    for (a, f), b in files.items():
        (out_root / DIRS[a]).mkdir(parents=True, exist_ok=True)
        (out_root / DIRS[a] / f).write_bytes(b)
    fails = []
    tw = twin_checks(kd27, k1x_rows)
    for x in ("statement_mismatch", "query_not_hop1", "gold_not_hop1_value", "option_rule_fail", "vb_options_not_roots",
              "row_rule_errors", "teacher_not_onehot", "keyset_differs", "id_group_depth_differs"):
        if tw[x]:
            fails.append(f"K1X {x}: {tw[x]}")
    bad_sh = {c: s for c, s in tw["gold_shares"].items() if not all(SHARE[0] <= v <= SHARE[1] for v in s.values())}
    if bad_sh:
        fails.append(f"K1X gold shares outside {SHARE}: {bad_sh}")
    if k1x[:N_K1] != k1_lines:
        fails.append("K1X rows 0..2,999 are not train_k1 byte for byte")
    ids = [r["item_id"] for r in k1x_rows]
    ids_ok = len(set(ids)) == len(ids) == N_KD27
    if not ids_ok:
        fails.append("K1X item_ids not distinct or not 27,000 rows")
    ev_hash = {B.phash(json.loads(l)["prompt"]) for s in B.EVAL_FILES for l in lines_of(khop / f"{s}.jsonl")}
    kd_hash = {B.phash(r["prompt"]) for r in kd27}
    k1x_hash = [B.phash(r["prompt"]) for r in k1x_rows]
    ev_overlap = len(set(k1x_hash) & ev_hash)
    kd_shared = [r for r, h in zip(k1x_rows, k1x_hash) if h in kd_hash]
    if ev_overlap:
        fails.append(f"K1X shares {ev_overlap} prompt hashes with the nine eval files")
    if any(r["depth"] != 1 for r in kd_shared):
        fails.append("K1X shares a prompt hash with train_kd27 on a row with k > 1")
    c27_ev = len(kd_hash & ev_hash)
    if c27_ev:
        fails.append(f"c27 shares {c27_ev} prompt hashes with the nine eval files")
    tk = load_tok(reps["khop2"], tok)
    tot = {"kd27": sum(n_units(tk, r["prompt"]) for r in kd27), "k1x": sum(n_units(tk, r["prompt"]) for r in k1x_rows)}
    rel = (tot["k1x"] - tot["kd27"]) / tot["kd27"]
    if abs(rel) > TOK_TOL:
        fails.append(f"K1X token total {tot['k1x']} is {100 * rel:+.2f}% of kd27's {tot['kd27']} (bar 1%)")
    md5_out = {f"{DIRS[a]}/{f}": md5_bytes(b) for (a, f), b in files.items()}
    if md5_out[f"{DIRS['c27']}/train.jsonl"] != KD27_MD5:
        fails.append("c27 train.jsonl md5 differs from train_kd27")
    if any(md5_out[f"{d}/val.jsonl"] != VAL_MD5 for d in DIRS.values()):
        fails.append("a val.jsonl md5 differs from the gsmvt val")
    md5_after = {str(p): md5_file(p) for p in in_files}
    if md5_after != md5_before:
        fails.append("an input file changed during the build")
    file_sets = {d: sorted(x.name for x in (out_root / d).iterdir()) for d in DIRS.values()}
    if any(v != sorted(OUT_FILES) for v in file_sets.values()):
        fails.append(f"output file sets {file_sets}")
    report = {
        "card": "CARD_khop_v3", "generator": "scripts/build_khop3.py",
        "stream_params": params,
        "inputs": {"train_kd27.jsonl": {"md5": md5_before[str(khop2 / "train_kd27.jsonl")], "rows": len(kd27)},
                   "train_k1.jsonl": {"md5": md5_before[str(khop / "train_k1.jsonl")], "rows": len(k1)},
                   "val.jsonl": {"md5": md5_before[str(gsm / "val.jsonl")], "rows": len(lines_of(gsm / "val.jsonl"))}},
        "parent_report_md5": {"khop": md5_before[str(khop / "build_report.json")],
                              "khop2": md5_before[str(khop2 / "build_report.json")]},
        "parent_md5": {f"{s}.jsonl": md5_before[str(khop / f'{s}.jsonl')] for s in B2.PARENT_FILES},
        "md5": md5_out,
        "rows": {f"{DIRS[a]}/{f}": len(b.splitlines()) for (a, f), b in files.items()},
        "file_sets": file_sets,
        "replay": replay_res,
        "k1x_rows_0_2999_equal_train_k1": k1x[:N_K1] == k1_lines,
        "k1x_checks": tw,
        "item_ids_distinct": ids_ok,
        "eval_hash_overlap": {"k1x": ev_overlap, "c27": c27_ev},
        "kd27_hash_overlap": {"k1x_rows": len(kd_shared), "by_k": dict(Counter(f"k{r['depth']}" for r in kd_shared))},
        "token_totals": {**tot, "rel_k1x_minus_kd27": round(rel, 6), "unit": "chars" if tk is None else "tokens",
                         "tokenizer": None if tk is None else getattr(tk, "name_or_path", None)},
        "updates": {"train_rows": N_KD27, "updates_mbs2_ga16": -(-N_KD27 // 32)},
        "bars": {"gold_share": list(SHARE), "token_rel": TOK_TOL},
        "inputs_unchanged": md5_after == md5_before,
        "d_fail": fails, "d_pass": not fails, "seconds": round(time.time() - t0, 1)}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--khop2", default="data/eval/khop2", help="train_kd27.jsonl and the khop2 build_report.json")
    ap.add_argument("--khop", default="data/eval/khop", help="the 12 khop files and their build_report.json")
    ap.add_argument("--gsm-dir", default="data/student_gsm8k_tf_vt", help="the gsmvt dir (val.jsonl)")
    ap.add_argument("--out-root", default="data", help="student_khop3_c27 / student_khop3_k1x are written here")
    ap.add_argument("--report", default="data/eval/khop3")
    a = ap.parse_args(argv)
    rep = build(Path(a.khop2), Path(a.khop), Path(a.gsm_dir), Path(a.out_root), Path(a.report))
    print(json.dumps({k: rep[k] for k in ("md5", "rows", "replay", "eval_hash_overlap", "kd27_hash_overlap",
                                          "token_totals", "d_pass", "d_fail", "seconds")}, indent=1))
    if not rep["d_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
