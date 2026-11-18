"""k-hop dose training files for CARD_khop_v2 (family khop2, parent khop): train_kd27, train_kd_x3, train_kh_x3 and
build_report.json, built on top of the CARD_khop_v1 files without changing them.

    python3 scripts/build_khop2.py --src <E>/src/data/eval/khop --out <E>/src/data/eval/khop2 \\
        [--tokenizer <Qwen3.5-4B tokenizer dir>] [--gsm-train <gsmvt train.jsonl>]

This script imports scripts/build_khop.py and does not edit it. It does not call build_khop.build(): build_eval() would
redraw the eval items against the new training keys (so the eval files would change), and the task-major loop of
build_train() would reorder the stream (so the result would not start with train_kd). Instead (card section 1.2):

1. Read the 12 parent files from --src. Stop (SystemExit) when the md5 of any file differs from the parent's
   build_report.json.
2. Forbidden sets: the value keys (row_key: task + sorted chain value tuples, parsed from the prompt; the rule of
   build_khop.build) and the prompt hashes (build_khop.phash) of every row of the nine eval files (blind rows: hash only)
   and of train_kh, train_k1, train_kd.
3. New rows: random.Random(--seed), task-major over ("ca", "wp", "vb"), j = 3000 .. 3000 + --new-per-task - 1 per task,
   k = 1 + j % 6, order fwd if (j // 6) % 2 == 0 else rev (the build_train rule), item_id "khop-tr-<task>-<j:05d>".
   it = build_khop.gen_item(task, k, rng, stats); the value key is checked first, then slots = build_khop.layout(i, k,
   rng) and the row are made and the prompt hash is checked; a forbidden or already used key or hash redraws the item
   (at most MAX_REDRAW redraws per row, else BuildError). i continues the per (task, k) cycle of the parent: it starts
   at the train_kd count of the (task, k) cell (500) and advances only when a row is kept.
4. Write train_kd27.jsonl (the train_kd.jsonl lines byte for byte, then the new rows in ca, wp, vb blocks),
   train_kd_x3.jsonl and train_kh_x3.jsonl (the parent file --repeat times: copy 1 byte for byte, copy c >= 2 with
   item_id + "-r<c>", problem_id and every other field unchanged) and build_report.json.

build_report.json uses the parent report's keys (md5, counts, gold_shares, d1_baselines, d4_tokens, d3_disjointness,
bars, d_fail, d_pass, train_tokens), with the stems train_kd27, train_kd_x3, train_kh_x3 and train_kd27_new (the new rows
alone, D1 / D2 / D4 only), so build_khop.d_lines and cell_of work on it unchanged. Extra keys: parent_md5, counts_order
(task / order / k), nested, copies_ok, new_distinct_keys, distinct_value_keys, redraws, resample_rate.
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

MAX_REDRAW = 2000                 # card 1.2: at most 2,000 redraws per row
J0 = 3000                         # first new j per task (train_kd ends at khop-tr-<task>-02999)
NEW_FILES = ("train_kd27", "train_kd_x3", "train_kh_x3")
D_STEMS = NEW_FILES + ("train_kd27_new",)
PARENT_FILES = B.EVAL_FILES + B.TRAIN_FILES


class CachedTok:
    """A tokenizer wrapper that tokenizes each distinct text once (the x3 files repeat every prompt three times)."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}
        self.name_or_path = getattr(tok, "name_or_path", None)

    def __call__(self, text, add_special_tokens=False):
        key = (text, add_special_tokens)
        if key not in self.cache:
            self.cache[key] = {"input_ids": list(self.tok(text, add_special_tokens=add_special_tokens)["input_ids"])}
        return self.cache[key]


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def read_lines(p: Path) -> list[bytes]:
    """The lines of a jsonl file as raw bytes, each with its trailing newline."""
    return p.read_bytes().splitlines(keepends=True)


def dump(row: dict) -> bytes:
    """One jsonl line in build_khop.write_jsonl's format."""
    return (json.dumps(row, ensure_ascii=False) + "\n").encode()


def row_key(r: dict):
    """The value key of a row from the prompt alone (build_khop.build's rule): task + sorted chain value tuples; None
    for a blind row."""
    P = B.parse_prompt(r["prompt"])
    if not P["stmts"]:
        return None
    roots = [s["name"] for s in P["stmts"] if s["op"] == "root"]
    kids = {s["parent"]: s["name"] for s in P["stmts"] if s["parent"]}
    chains = []
    for x in roots:
        seq = [x]
        while seq[-1] in kids:
            seq.append(kids[seq[-1]])
        chains.append(tuple(P["values"][n] for n in seq))
    return B.task_of(r) + ":" + json.dumps(sorted(chains))


def check_parent(src: Path) -> dict:
    """The parent report; SystemExit when a parent file is missing or its md5 differs from the report."""
    rp = src / "build_report.json"
    if not rp.exists():
        raise SystemExit(f"build_khop2: no parent report {rp}")
    rep = json.loads(rp.read_text())
    for s in PARENT_FILES:
        p = src / f"{s}.jsonl"
        if not p.exists():
            raise SystemExit(f"build_khop2: no parent file {p}")
        got, want = md5_bytes(p.read_bytes()), rep["md5"].get(f"{s}.jsonl")
        if got != want:
            raise SystemExit(f"build_khop2: md5 of {p} is {got}, the parent report says {want}")
    return rep


def copies(lines: list[bytes], repeat: int) -> list[bytes]:
    """The file `repeat` times: copy 1 byte for byte, copy c >= 2 with item_id + "-r<c>" (every other field unchanged)."""
    out = list(lines)
    for c in range(2, repeat + 1):
        for l in lines:
            r = json.loads(l)
            r["item_id"] = f"{r['item_id']}-r{c}"
            out.append(dump(r))
    return out


def copies_ok(lines: list[bytes], parent: list[bytes], repeat: int) -> bool:
    """D6 for a copy file: len = repeat x parent; copy 1 equals the parent byte for byte; copy c >= 2 differs from the
    parent only in item_id (= parent id + "-r<c>"); all item_ids distinct."""
    n = len(parent)
    if len(lines) != repeat * n or lines[:n] != parent:
        return False
    for c in range(2, repeat + 1):
        for a, b in zip(parent, lines[(c - 1) * n:c * n]):
            ra, rb = json.loads(a), json.loads(b)
            if rb.get("item_id") != f"{ra['item_id']}-r{c}" or {**rb, "item_id": ra["item_id"]} != ra:
                return False
            if dump({**rb, "item_id": ra["item_id"]}) != a:
                return False
    ids = [json.loads(l)["item_id"] for l in lines]
    return len(set(ids)) == len(ids)


def new_rows(seed: int, n_per_task: int, forbid_keys: set, forbid_hash: set, cycle0: Counter):
    """The new training rows (task-major, card 1.2 step 3). Returns (rows, keys, redraws per (task, k), stats)."""
    rng = random.Random(seed)
    stats = defaultdict(Counter)
    redraws = Counter()
    used_keys, used_hash = set(), set()
    cyc = Counter(cycle0)
    rows, keys = [], []
    for task in B.TASKS:
        for j in range(J0, J0 + n_per_task):
            k = 1 + j % 6
            order = "fwd" if (j // 6) % 2 == 0 else "rev"
            iid = f"khop-tr-{task}-{j:05d}"
            i = cyc[(task, k)]
            for _ in range(MAX_REDRAW + 1):
                it = B.gen_item(task, k, rng, stats)
                vk = B.value_key(task, it.ch)
                if vk in forbid_keys or vk in used_keys:
                    redraws[(task, k)] += 1
                    continue
                r = it.row(iid, order, B.layout(i, k, rng))
                h = B.phash(r["prompt"])
                if h in forbid_hash or h in used_hash:
                    redraws[(task, k)] += 1
                    continue
                break
            else:
                raise B.BuildError(f"{task} k={k} j={j}: no new value key / prompt in {MAX_REDRAW} redraws")
            cyc[(task, k)] += 1
            used_keys.add(vk)
            used_hash.add(h)
            rows.append(r)
            keys.append(vk)
    return rows, keys, redraws, stats


def build(src: Path, out: Path, tok=None, seed: int = 3, n_per_task: int = 6000, repeat: int = 3,
          gsm_train: Path | None = None) -> dict:
    t0 = time.time()
    if out.resolve() == src.resolve():
        raise SystemExit("build_khop2: --out must not be the parent directory")
    prep = check_parent(src)
    parent_md5_before = {s: md5_bytes((src / f"{s}.jsonl").read_bytes()) for s in PARENT_FILES}
    raw = {s: read_lines(src / f"{s}.jsonl") for s in PARENT_FILES}
    rows = {s: [json.loads(l) for l in raw[s]] for s in PARENT_FILES}
    ev_keys, ev_hash, forbid_keys, forbid_hash = set(), set(), set(), set()
    for s in PARENT_FILES:
        for r in rows[s]:
            h = B.phash(r["prompt"])
            forbid_hash.add(h)
            k = row_key(r)                                  # blind rows: hash only
            if k is not None:
                forbid_keys.add(k)
            if s in B.EVAL_FILES:
                ev_hash.add(h)
                if k is not None:
                    ev_keys.add(k)
    cycle0 = Counter((B.task_of(r), r["depth"]) for r in rows["train_kd"])
    new, new_keys, redraws, stats = new_rows(seed, n_per_task, forbid_keys, forbid_hash, cycle0)
    lines = {"train_kd27": raw["train_kd"] + [dump(r) for r in new],
             "train_kd_x3": copies(raw["train_kd"], repeat),
             "train_kh_x3": copies(raw["train_kh"], repeat)}
    out.mkdir(parents=True, exist_ok=True)
    for s, ls in lines.items():
        (out / f"{s}.jsonl").write_bytes(b"".join(ls))
    files = {s: [json.loads(l) for l in ls] for s, ls in lines.items()}
    files["train_kd27_new"] = new
    ctok = CachedTok(tok) if tok is not None else None
    rep = B.d_lines(files, ctok)
    fails = list(rep["fail"])
    # D3: each new file against the nine eval files; the new rows against every parent file and each other
    d3 = {}
    for s in NEW_FILES:
        fk = {row_key(r) for r in files[s]}
        fh = {B.phash(r["prompt"]) for r in files[s]}
        d3[s] = {"shared_prompt_hashes": len(fh & ev_hash), "shared_value_keys": len(fk & ev_keys)}
        if d3[s]["shared_prompt_hashes"] or d3[s]["shared_value_keys"]:
            fails.append(f"D3 {s} {d3[s]}")
    parsed_new = [row_key(r) for r in new]
    tr_keys = {row_key(r) for s in B.TRAIN_FILES for r in rows[s]}
    new_distinct = len(set(parsed_new))
    d3["train_kd27_new"] = {"shared_value_keys_parent_train": len(set(parsed_new) & tr_keys),
                            "shared_value_keys_eval": len(set(parsed_new) & ev_keys),
                            "shared_prompt_hashes_parent": len({B.phash(r["prompt"]) for r in new} & forbid_hash),
                            "parsed_key_equals_generator_key": parsed_new == new_keys}
    d3["shared_prompt_hashes"] = sum(d3[s]["shared_prompt_hashes"] for s in NEW_FILES)
    d3["shared_value_keys"] = sum(d3[s]["shared_value_keys"] for s in NEW_FILES)
    if new_distinct != len(new) or any(d3["train_kd27_new"][x] for x in ("shared_value_keys_parent_train",
                                                                          "shared_value_keys_eval",
                                                                          "shared_prompt_hashes_parent")) \
            or not d3["train_kd27_new"]["parsed_key_equals_generator_key"]:
        fails.append(f"D3 new rows: distinct keys {new_distinct} of {len(new)}, {d3['train_kd27_new']}")
    # D6
    n_kd = len(raw["train_kd"])
    nested = lines["train_kd27"][:n_kd] == raw["train_kd"] and len(lines["train_kd27"]) == n_kd + len(new)
    cok = {"train_kd_x3": copies_ok(lines["train_kd_x3"], raw["train_kd"], repeat),
           "train_kh_x3": copies_ok(lines["train_kh_x3"], raw["train_kh"], repeat)}
    ids = [r["item_id"] for r in files["train_kd27"]]
    ids_ok = len(set(ids)) == len(ids)
    keyset_ok = all(set(r) == set(B.GSMVT_KEYS) for r in new)
    rule_errs = [r["item_id"] for r in new if B.row_rule_ok(r)]
    if not nested:
        fails.append("D6 train_kd27 does not start with train_kd byte for byte")
    for s, ok in cok.items():
        if not ok:
            fails.append(f"D6 {s} is not {repeat} x its parent apart from the item_id suffixes")
    if not ids_ok:
        fails.append("D6 train_kd27 item_ids not distinct")
    if not keyset_ok:
        fails.append("new rows: key set differs from the 32 gsmvt keys")
    if rule_errs:
        fails.append(f"new rows: {len(rule_errs)} rows break a 1.1 rule, e.g. {rule_errs[0]}")
    parent_md5_after = {s: md5_bytes((src / f"{s}.jsonl").read_bytes()) for s in PARENT_FILES}
    if parent_md5_after != parent_md5_before:
        fails.append("parent files changed during the build")
    report = {
        "card": "CARD_khop_v2", "generator": "scripts/build_khop2.py", "seed": seed, "new_per_task": n_per_task,
        "repeat": repeat, "parent_report_md5": md5_bytes((src / "build_report.json").read_bytes()),
        "parent_md5": {f"{s}.jsonl": parent_md5_after[s] for s in PARENT_FILES},
        "md5": {f"{s}.jsonl": md5_bytes((out / f"{s}.jsonl").read_bytes()) for s in NEW_FILES},
        "counts": {s: dict(sorted(Counter(B.cell_of(r, s) for r in rs).items())) | {"total": len(rs)}
                   for s, rs in files.items()},
        "counts_order": {s: dict(sorted(Counter(f"{B.task_of(r)}/{r['group'].split('/')[2]}/k{r['depth']}"
                                                for r in rs).items())) for s, rs in files.items()},
        "gold_shares": rep["d2"], "d1_baselines": rep["d1"], "d4_tokens": rep["d4"],
        "d3_disjointness": d3, "nested": nested, "copies_ok": cok, "item_ids_distinct": ids_ok,
        "new_keyset_ok": keyset_ok, "new_rule_errors": len(rule_errs), "new_distinct_keys": new_distinct,
        "distinct_value_keys": {s: len({row_key(r) for r in files[s]}) for s in files},
        "redraws": {f"{t}/k{k}": redraws[(t, k)] for t in B.TASKS for k in range(1, 7)},
        "resample_rate": {f"{t}/k{k}": {"draws": c["draws"], "items": c["items"],
                                        "draws_per_item": round(c["draws"] / max(1, c["items"]), 2)}
                          for (t, k), c in sorted(stats.items())},
        "bars": B.BARS, "d_fail": fails, "d_pass": not fails,
        "tokenizer": getattr(tok, "name_or_path", None), "seconds": None}
    if ctok is not None:
        report["train_tokens"] = {s: sum(B.n_tokens(ctok, r["prompt"]) for r in files[s]) for s in files}
        report["train_tokens"]["parent_train_kd_report"] = (prep.get("train_tokens") or {}).get("train_kd")
        if gsm_train and gsm_train.exists():
            g = [json.loads(l) for l in open(gsm_train) if l.strip()]
            report["train_tokens"]["gsmvt_train_reference"] = sum(B.n_tokens(ctok, r["prompt"]) for r in g)
            report["train_tokens"]["gsmvt_train_rows"] = len(g)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="data/eval/khop", help="the CARD_khop_v1 files and their build_report.json")
    ap.add_argument("--out", default="data/eval/khop2")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--new-per-task", type=int, default=6000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tokenizer", default="", help="Qwen3.5-4B tokenizer dir for D4 and the token totals (optional)")
    ap.add_argument("--gsm-train", default="", help="gsmvt train.jsonl for the reference token total (optional)")
    a = ap.parse_args(argv)
    tok = None
    if a.tokenizer:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    try:
        rep = build(Path(a.src), Path(a.out), tok, a.seed, a.new_per_task, a.repeat,
                    Path(a.gsm_train) if a.gsm_train else None)
    except B.BuildError as e:
        raise SystemExit(f"build_khop2: {e}")
    print(json.dumps({"counts": {s: c["total"] for s, c in rep["counts"].items()}, "md5": rep["md5"],
                      "new_distinct_keys": rep["new_distinct_keys"], "redraws": rep["redraws"],
                      "d_pass": rep["d_pass"], "d_fail": rep["d_fail"], "seconds": rep["seconds"]}, indent=1))
    if not rep["d_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
