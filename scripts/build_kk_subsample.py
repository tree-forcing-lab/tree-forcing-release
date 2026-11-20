#!/usr/bin/env python3
"""K&K Tree Forcing data with a FIXED subset of k tree nodes per training puzzle (the node-density
ablation): data/student_kk_tf_n<k>/{train,val}.jsonl from data/student_kk_tf.

Train: every puzzle keeps k of its subqs (all of them when it has at most k), drawn without replacement by a generator
of its own (random.Random of "kk-subsample|<seed>|<k>|<item_id>", seeded through a str key, so the draw depends on
nothing else), in their original order; every other field is byte-for-byte the source's (json round trip, the same
key order) plus "subsample": {"k", "n_full"}. Val: copied byte-identical (the node probes and the epoch selection see
the full trees, as in the full arm).

Why a pure subset cannot leak: GFPS asks every node in the statements view (the preamble and the statement lines)
without its parents' results, so a kept node's view is exactly its view in the full arm; dropping nodes removes
questions and never adds text to a prompt. The loss convention (--tree-node-weight 0: --aux-total, 1, split evenly
over a problem's views; treeplan / student full-tree groups) gives every puzzle the same total node weight as in the
full arm, spread over fewer questions.

 python3 scripts/build_kk_subsample.py --k 3 # -> data/student_kk_tf_n3
 python3 scripts/build_kk_subsample.py --k 8 # -> data/student_kk_tf_n8
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def subsample(item: dict, k: int, seed: int = 0) -> dict:
    nodes = item.get("subqs") or []
    out = dict(item)
    if len(nodes) > k:
        r = random.Random(f"kk-subsample|{seed}|{k}|{item['item_id']}")
        keep = sorted(r.sample(range(len(nodes)), k))
        out["subqs"] = [nodes[i] for i in keep]
    out["subsample"] = {"k": k, "n_full": len(nodes)}
    return out


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def node_stats(items: list[dict]) -> dict:
    n = [len(it.get("subqs") or []) for it in items]
    types = collections.Counter(q.get("type") for it in items for q in it.get("subqs") or [])
    yes = [q.get("p_cot") for it in items for q in it.get("subqs") or [] if q.get("p_cot") is not None]
    return {"items": len(items), "nodes": sum(n), "nodes_per_item": [round(st.mean(n), 2), min(n), max(n)],
            "types": dict(sorted(types.items(), key=str)), "yes_rate": round(sum(p >= 0.5 for p in yes) / len(yes), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="data/student_kk_tf")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default data/student_kk_tf_n<k>")
    a = ap.parse_args(argv)
    src, out = ROOT / a.src, ROOT / (a.out or f"data/student_kk_tf_n{a.k}")
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(src / "train.jsonl") if line.strip()]
    sub = [subsample(it, a.k, a.seed) for it in items]
    for it, s in zip(items, sub):                              # a pure subset, in the original order
        ids = [q["id"] for q in it["subqs"]]
        kept = [q["id"] for q in s["subqs"]]
        assert kept == [i for i in ids if i in set(kept)] and len(kept) == min(a.k, len(ids)), it["item_id"]
    with open(out / "train.jsonl", "w") as f:
        for s in sub:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    shutil.copyfile(src / "val.jsonl", out / "val.jsonl")
    summary = {"script": "scripts/build_kk_subsample.py", "args": vars(a), "source": a.src,
               "source_md5": {f"{a.src}/{s}.jsonl": md5(src / f"{s}.jsonl") for s in ("train", "val")},
               "md5": {f"{out.relative_to(ROOT)}/{s}.jsonl": md5(out / f"{s}.jsonl") for s in ("train", "val")},
               "train_before": node_stats(items), "train_after": node_stats(sub),
               "puzzles_with_at_most_k_nodes": sum(len(it["subqs"]) <= a.k for it in items),
               "val": "byte-identical copy of the source val (full trees)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("md5", "train_before", "train_after", "puzzles_with_at_most_k_nodes")}, indent=1))


if __name__ == "__main__":
    main()
