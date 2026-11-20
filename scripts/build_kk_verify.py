#!/usr/bin/env python3
"""K&K verification control (arm "vf"): the training puzzles of data/student_kk_tf with each puzzle's
tree nodes replaced by the SAME NUMBER of correct yes/no questions that do not come from the solver's tree, namely
the truth of one inhabitant's statement under a full role assignment:

 If Alice were a knave, Karen a knight, Henry a knave and Grace a knight, would the statement by Henry,
 "Karen is a knight and Alice is a knave.", be true?

(the wording of the K&K case nodes, with a full assignment instead of one hypothesis). The assignments are drawn
uniformly from those that are neither the puzzle's solution nor any of its options' assignments, so no question states
the answer or a candidate answer. Truth: the statement parsed by cotdistill.sources._kk_formula, evaluated by _kk_val.
Balance within each puzzle: half of the chosen (assignment, speaker) pairs have a true statement and half a false one,
and within each half the question asks "true" for half of the pairs and "false" for the other half, so the answer is
Yes for exactly half of the questions and P(Yes | wording) = 1/2 (up to one question when a count is odd, and except
where a puzzle has too few pairs of one kind: counted in summary.json). Every other field is the source's; val is
copied byte-identical (the node probes keep measuring the tree nodes).

Checks (exit 1 on failure): every statement of every training puzzle parses; brute force over all assignments finds
exactly one consistent assignment, which is the gold option's (or, for "None of the above" golds, none of the options);
every question's truth is re-derived from its own text.

 python3 scripts/build_kk_verify.py # -> data/student_kk_tf_vrf
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cotdistill.sources import _kk_formula, _kk_val  # noqa: E402

STMT = re.compile(r'^- ([A-Z][a-z]+) says: "(.*)"$')
OPT = re.compile(r"^([A-Z])\) (.*)$")
QUESTION = re.compile(r'^If (.+), would the statement by ([A-Z][a-z]+), "(.*)", be (true|false)\?$')


def parse_puzzle(item: dict) -> dict:
    head, _, tail = item["prompt"].partition("\n\nQuestion: ")
    _, _, body = head.partition("\n\n")
    speakers, stmts = [], {}
    for line in body.split("\n"):
        m = STMT.match(line)
        if not m:
            raise ValueError(f"{item['item_id']}: statement line {line!r}")
        speakers.append(m.group(1)); stmts[m.group(1)] = (m.group(2), line)
    people = set(speakers)
    forms = {}
    for p, (s, _) in stmts.items():
        f = _kk_formula(s, people)
        if f is None:
            raise ValueError(f"{item['item_id']}: does not parse: {s!r}")
        forms[p] = f
    options = {}
    for line in tail.split("\n"):
        m = OPT.match(line)
        if m and m.group(2) != "None of the above":
            roles = {}
            for part in m.group(2).split(", "):
                name, role = part.split(": ")
                roles[name] = role == "knight"
            if set(roles) != people:
                raise ValueError(f"{item['item_id']}: option {m.group(1)} names {sorted(roles)}")
            options[m.group(1)] = roles
        elif m:
            options[m.group(1)] = None
    consistent = []
    for bits in itertools.product([True, False], repeat=len(speakers)):
        r = dict(zip(speakers, bits))
        if all(r[p] == _kk_val(forms[p], r) for p in speakers):
            consistent.append(r)
    if len(consistent) != 1:
        raise ValueError(f"{item['item_id']}: {len(consistent)} consistent assignments")
    sol = consistent[0]
    gold = options.get(item["gold_label"], "missing")
    if gold == "missing" or (gold is not None and gold != sol) or (gold is None and sol in options.values()):
        raise ValueError(f"{item['item_id']}: gold option does not match the brute-force solution")
    return {"speakers": speakers, "stmts": stmts, "forms": forms, "solution": sol,
            "forbidden": [sol] + [r for r in options.values() if r is not None]}


def assignment_phrase(speakers: list[str], roles: dict) -> str:
    parts = [f"{speakers[0]} were a {'knight' if roles[speakers[0]] else 'knave'}"]
    parts += [f"{p} a {'knight' if roles[p] else 'knave'}" for p in speakers[1:]]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def question(speakers, roles, speaker, text, wording) -> str:
    return f'If {assignment_phrase(speakers, roles)}, would the statement by {speaker}, "{text}", be {wording}?'


def parse_question(q: str, speakers: list[str]) -> tuple[dict, str, str, str]:
    m = QUESTION.match(q)
    if not m:
        raise ValueError(q)
    roles = {}
    for part in re.split(r", | and ", m.group(1)):
        w = part.split()
        roles[w[0]] = w[-1] == "knight"
    return roles, m.group(2), m.group(3), m.group(4)


def verify_nodes(item: dict, pz: dict, seed: int = 0, stats: dict | None = None) -> list[dict]:
    n = len(item.get("subqs") or [])
    sp = pz["speakers"]
    r = random.Random(f"kk-verify|{seed}|{item['item_id']}")
    pool = {True: [], False: []}
    for bits in itertools.product([True, False], repeat=len(sp)):
        roles = dict(zip(sp, bits))
        if roles in pz["forbidden"]:
            continue
        for s in sp:
            pool[_kk_val(pz["forms"][s], roles)].append((bits, s))
    for v in (True, False):
        r.shuffle(pool[v])
    want_t = n // 2 + (r.random() < 0.5 if n % 2 else 0)
    take_t = min(want_t, len(pool[True]))
    take_f = min(n - take_t, len(pool[False]))
    take_t = min(n - take_f, len(pool[True]))                 # refill from the true side if the false side ran short
    if stats is not None and (take_t, take_f) != (want_t, n - want_t):
        stats["unbalanced_items"] = stats.get("unbalanced_items", 0) + 1
    if stats is not None and take_t + take_f < n:
        stats["short_items"] = stats.get("short_items", 0) + 1
    chosen = [(b, s, True) for b, s in pool[True][:take_t]] + [(b, s, False) for b, s in pool[False][:take_f]]
    out = []
    for v in (True, False):
        grp = [c for c in chosen if c[2] is v]
        words = ["true"] * (len(grp) // 2) + ["false"] * (len(grp) - len(grp) // 2)
        if len(grp) % 2 and r.random() < 0.5:
            words = ["false"] * (len(grp) // 2) + ["true"] * (len(grp) - len(grp) // 2)
        r.shuffle(words)
        for (bits, s, val), w in zip(grp, words):
            out.append((bits, s, val, w))
    r.shuffle(out)
    nodes = []
    for j, (bits, s, val, w) in enumerate(out):
        roles = dict(zip(sp, bits))
        text, line = pz["stmts"][s]
        yes = (w == "true") == val
        nodes.append({"question": question(sp, roles, s, text, w),
                      "opposite": question(sp, roles, s, text, "false" if w == "true" else "true"),
                      "p_cot": 1.0 if yes else 0.0, "p_fresh": None, "truth": yes, "type": "verify",
                      "depends_on": [], "id": f"v{j}", "initial": None, "status": "direct", "node_depth": 0,
                      "source_tree": "verify", "local_lines": [line]})
    return nodes


def check_nodes(item: dict, pz: dict, nodes: list[dict]) -> None:
    for q in nodes:
        roles, s, text, w = parse_question(q["question"], pz["speakers"])
        if set(roles) != set(pz["speakers"]) or roles in pz["forbidden"] or text != pz["stmts"][s][0]:
            raise ValueError(f"{item['item_id']}: bad question {q['question']!r}")
        f = _kk_formula(text, set(pz["speakers"]))
        if ((w == "true") == _kk_val(f, roles)) != q["truth"] or q["p_cot"] != float(q["truth"]):
            raise ValueError(f"{item['item_id']}: wrong truth {q['question']!r}")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="data/student_kk_tf")
    ap.add_argument("--out", default="data/student_kk_tf_vrf")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    src, out = ROOT / a.src, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(src / "train.jsonl") if line.strip()]
    stats, cells, fails = {}, collections.Counter(), []
    rows = []
    for it in items:
        try:
            pz = parse_puzzle(it)
        except ValueError as e:
            fails.append(str(e)); continue
        nodes = verify_nodes(it, pz, a.seed, stats)
        check_nodes(it, pz, nodes)
        for q in nodes:
            wording = "true" if q["question"].endswith("be true?") else "false"
            cells[(wording, q["truth"])] += 1
        row = dict(it); row["subqs"] = nodes; row["tree_source"] = "verify_kk"
        rows.append(row)
    if fails:
        sys.exit(f"build_kk_verify: {len(fails)} puzzles failed: " + "; ".join(fails[:5]))
    with open(out / "train.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copyfile(src / "val.jsonl", out / "val.jsonl")
    n_src = sum(len(it["subqs"]) for it in items); n_new = sum(len(r["subqs"]) for r in rows)
    summary = {"script": "scripts/build_kk_verify.py", "args": vars(a),
               "source_md5": {f"{a.src}/{s}.jsonl": md5(src / f"{s}.jsonl") for s in ("train", "val")},
               "md5": {f"{a.out}/{s}.jsonl": md5(out / f"{s}.jsonl") for s in ("train", "val")},
               "items": len(rows), "statements_parsed": sum(len(parse_puzzle(it)["speakers"]) for it in items),
               "parse_failures": 0, "nodes_source": n_src, "nodes": n_new,
               "wording_x_answer": {f"{w}/{'yes' if t else 'no'}": c for (w, t), c in sorted(cells.items(), key=str)},
               "p_yes_given_true_wording": round(cells[("true", True)] / (cells[("true", True)] + cells[("true", False)]), 4),
               "p_yes_given_false_wording": round(cells[("false", True)] / (cells[("false", True)] + cells[("false", False)]), 4),
               **{k: stats.get(k, 0) for k in ("unbalanced_items", "short_items")},
               "val": "byte-identical copy of the source val (full trees)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
