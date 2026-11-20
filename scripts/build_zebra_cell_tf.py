#!/usr/bin/env python3
"""Zebra cell questions with proof trees for Tree Forcing.

Hard puzzles of the generator (zebra.generate with zebra.HARD_WEIGHTS, up to 80 clues: no clue places
anyone), each asked as ONE cell (zebra.make_cell_item: "What is the <attribute> of the person who lives in the k-th
house?", the attribute's n values as options). Every train / val item carries a proof tree in the K&K subq format:
zebra.proof_tree (one-level case splits refuted by propagation, max_cases=60 so that every 3x3-4x5 puzzle is solved),
restricted to the ANCESTORS of the asked cell's placement node (the facts the answer depends on, closed under
depends_on; node ids and order kept). zebra.audit_tree must return no error on the kept nodes of every item.

 ~/.venvs/sev/bin/python scripts/build_zebra_cell_tf.py # needs `tokenizers` for the token table

Writes
 data/student_zebra_cell_tf/{train,val}.jsonl train 300 / val 75 per size, sizes 3x3 3x4 4x4 4x5 (1,200 / 300)
 data/eval/zebra_cell_in.jsonl in-range test, 150 per size of the train sizes (600)
 data/eval/zebra_cell_ex.jsonl extrapolation test, 150 per size 5x5 5x6 6x6 (450), no trees
 data/student_zebra_cell_tf/summary.json counts, node / token tables, md5s, seeds, dedup
Seeds: random.Random(base + 10000 * size_index + k), bases 3,000,000 (train), 4,000,000 (val), 5,000,000 (test); the
cell is drawn with random.Random(7 * base + 10000 * size_index + k), the tree with random.Random(f"tree-{item_id}").
Every puzzle is unique by zebra.canonical across all splits and against every earlier probe file.
CPU, deterministic.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill import zebra as Z  # noqa: E402

SIZES = [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]
TRAIN_SIZES = SIZES[:4]
EX_SIZES = SIZES[4:]
BASE = {"train": 3_000_000, "val": 4_000_000, "test": 5_000_000}
MAX_CASES = 60
TOKENIZER = ROOT / "results/weights/T1-Qwen3.5-2B-Base-student_v3-s2/model/tokenizer.json"
SUBQ_TEMPLATE = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) {a}\nB) {b}\nAnswer:"   # student.py
FINAL_TEMPLATE = "{problem}\n\nAnswer:"
PROBE_FILES = ["data/student_zebra_probe/train.jsonl", "data/student_zebra_probe/val.jsonl", "data/eval/zebra_probe.jsonl",
               "data/student_zebra_probe2/train.jsonl", "data/student_zebra_probe2/val.jsonl", "data/eval/zebra_probe2.jsonl",
               "data/student_zebra_probe3/train.jsonl", "data/student_zebra_probe3/val.jsonl", "data/eval/zebra_probe3.jsonl",
               "data/student_zebra_probe3c/train.jsonl", "data/student_zebra_probe3c/val.jsonl", "data/eval/zebra_probe3c.jsonl"]


def ancestors(nodes: list[dict], root: str) -> list[dict]:
    by = {x["id"]: x for x in nodes}
    keep, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i not in keep:
            keep.add(i)
            stack += by[i]["depends_on"]
    return [x for x in nodes if x["id"] in keep]


def cell_root(nodes: list[dict], p: dict, cell) -> str:
    c, h = cell
    v = p["solution"][c][h - 1]
    hits = [x["id"] for x in nodes if x["type"] == "derive" and x.get("hyp") is None and x.get("fact")
            and x["fact"][0] == "at" and x["fact"][1] == [c, v] and x["fact"][2] == h]
    if len(hits) != 1:
        raise ValueError(f"cell {cell}: {len(hits)} placement nodes")
    return hits[0]


def statement_block(prompt: str) -> str:
    head, sep, _ = prompt.partition("\nOptions:\n")
    body, cut, last = head.rpartition("\n\n")
    assert sep and cut and last.startswith("Question: What is the ") and "\n\nQuestion: " not in body, prompt[:200]
    return body


def make(split: str, si: int, n: int, m: int, k: int, seen: dict, with_tree: bool) -> dict | None:
    base = BASE[split]
    p = Z.generate(n, m, random.Random(base + 10000 * si + k), weights=Z.HARD_WEIGHTS, max_clues=80)
    if p is None:
        return None
    key = Z.canonical(p)
    if key in seen:
        return {"dup": seen[key]}
    iid = f"zebracell-{split}-{n}x{m}-{k:05d}"
    it = Z.make_cell_item(p, random.Random(7 * base + 10000 * si + k), iid)
    rec = {"item_id": iid, "source": "zebra_tf", "prompt": it["prompt"], "labels": it["labels"],
           "gold_label": it["gold_label"], "teacher": it["teacher"], "size": it["size"], "depth": n,
           "zebra": p, "cell": it["cell"], "option_scheme": "cell", "random_subqs": [], "rationale": ""}
    if with_tree:
        nodes = Z.proof_tree(p, random.Random(f"tree-{iid}"), max_cases=MAX_CASES)
        if nodes is None:
            return {"notree": iid}
        sub = ancestors(nodes, cell_root(nodes, p, it["cell"]))
        errs = Z.audit_tree(p, sub)
        if errs:
            raise SystemExit(f"{iid}: audit_tree: {errs[:3]}")
        rec.update(subqs=sub, has_tree=True, tree_source="zebra_proof_ancestors", tree_fix=True,
                   n_nodes_whole=len(nodes))
    seen[key] = iid
    return rec


def eval_row(r: dict) -> dict:
    out = {k: r[k] for k in ("item_id", "source", "prompt", "labels", "gold_label", "size", "zebra", "cell")}
    out.update(group=f"zebra_tf_cell/{r['size']}", type="choice", label_order=r["labels"],
               label_names={x: f"o{i}" for i, x in enumerate(r["labels"])}, gold_probs=None)
    return out


def build(root: Path, per: dict) -> dict:
    seen, fails = {}, collections.Counter()
    for f in PROBE_FILES:
        fp = root / f
        if not fp.exists():
            fails[f"probe file missing: {f}"] += 1
            continue
        for line in fp.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                seen.setdefault(Z.canonical(x["zebra"]), f"probe:{f}")
    n_probe = len(seen)
    out = {"train": [], "val": [], "test_in": [], "test_ex": []}
    plan = [("train", "train", TRAIN_SIZES, per["train"], True), ("val", "val", TRAIN_SIZES, per["val"], True),
            ("test_in", "test", TRAIN_SIZES, per["test"], False), ("test_ex", "test", EX_SIZES, per["test"], False)]
    for name, split, sizes, n_per, tree in plan:
        for n, m in sizes:
            si = SIZES.index((n, m))
            k, got = 0, 0
            while got < n_per:
                r = make(split, si, n, m, k, seen, tree)
                k += 1
                if r is None:
                    fails[f"{name} {n}x{m} generate"] += 1
                elif "dup" in r:
                    fails[f"{name} {n}x{m} duplicate ({r['dup'].split(':')[0]})"] += 1
                elif "notree" in r:
                    fails[f"{name} {n}x{m} no tree"] += 1
                else:
                    out[name].append(r)
                    got += 1
    return {"out": out, "fails": dict(fails), "n_probe_puzzles": n_probe}


def tokens_table(rows: list[dict]) -> dict:
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER))
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"no tokenizer: {e}"}
    L = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    by = collections.defaultdict(list)
    for r in rows:
        fin = L(FINAL_TEMPLATE.format(problem=r["prompt"]))
        block = statement_block(r["prompt"])
        node = [L(SUBQ_TEMPLATE.format(problem=block, question=x["question"], a="Yes", b="No")) for x in r.get("subqs", [])]
        by[r["size"]].append((fin, len(node), sum(node), max(node, default=0)))
    tab = {}
    for s, v in by.items():
        tab[s] = {"items": len(v), "final_tokens_median": st.median(x[0] for x in v), "final_tokens_max": max(x[0] for x in v),
                  "nodes_median": st.median(x[1] for x in v), "nodes_max": max(x[1] for x in v),
                  "gfps_tokens_per_item_mean": round(st.mean(x[0] + x[2] for x in v)),
                  "longest_example": max(max(x[0], x[3]) for x in v)}
    tab["all"] = {"gfps_tokens_per_epoch": sum(x[0] + x[2] for v in by.values() for x in v),
                  "final_tokens_per_epoch": sum(x[0] for v in by.values() for x in v)}
    return tab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--train-per-size", type=int, default=300)
    ap.add_argument("--val-per-size", type=int, default=75)
    ap.add_argument("--test-per-size", type=int, default=150)
    ap.add_argument("--out-dir", default="data/student_zebra_cell_tf")
    ap.add_argument("--eval-in", default="data/eval/zebra_cell_in.jsonl")
    ap.add_argument("--eval-ex", default="data/eval/zebra_cell_ex.jsonl")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = build(root, {"train": a.train_per_size, "val": a.val_per_size, "test": a.test_per_size})
    out = res["out"]
    d = root / a.out_dir
    d.mkdir(parents=True, exist_ok=True)
    files = {f"{a.out_dir}/train.jsonl": out["train"], f"{a.out_dir}/val.jsonl": out["val"],
             a.eval_in: [eval_row(r) for r in out["test_in"]], a.eval_ex: [eval_row(r) for r in out["test_ex"]]}
    md5 = {}
    for rel, rows in files.items():
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (root / rel).write_text(text)
        md5[rel] = hashlib.md5(text.encode()).hexdigest()
    gold = collections.Counter((r["size"], r["labels"].index(r["gold_label"])) for r in out["train"] + out["val"])
    summary = {
        "counts": {k: dict(collections.Counter(r["size"] for r in v)) for k, v in out.items()},
        "fails": res["fails"], "n_probe_puzzles_deduped_against": res["n_probe_puzzles"],
        "max_cases": MAX_CASES, "tree": "ancestors of the asked cell's placement node (zebra.proof_tree)",
        "nodes_kept_over_whole": {s: round(st.mean(len(r["subqs"]) / r["n_nodes_whole"] for r in out["train"] if r["size"] == s), 3)
                                  for s in sorted({r["size"] for r in out["train"]})},
        "node_types": dict(collections.Counter(x["type"] for r in out["train"] for x in r["subqs"])),
        "node_truth_share": round(st.mean(float(x["truth"]) for r in out["train"] for x in r["subqs"]), 3),
        "gold_index_counts": {f"{s}:{i}": c for (s, i), c in sorted(gold.items())},
        "tokens_train": tokens_table(out["train"]), "md5": md5,
        "seeds": {"bases": BASE, "puzzle": "Random(base + 10000 * size_index + k)",
                  "cell": "Random(7 * base + 10000 * size_index + k)", "tree": "Random('tree-' + item_id)"}}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("counts", "fails", "nodes_kept_over_whole", "node_types", "node_truth_share",
                                              "tokens_train", "md5")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
