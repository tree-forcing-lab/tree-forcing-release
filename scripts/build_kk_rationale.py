#!/usr/bin/env python3
"""Rationale-LM baseline data for K&K (arm A7, Distilling Step-by-Step style, and the no-node control
"rn"): data/student_kk_tf with the `rationale` field filled from the item's OWN verified solver tree, every other byte of
every row unchanged. The rationale carries exactly the information of the tree nodes, as text to be generated, so A7 /
rn against the tree arm (GFPS) differ only in the form of the supervision (written proof vs. node questions).

Rendering rule (fixed before any result):
 - nodes in topological order over `depends_on` (parents first; ties keep the node list order);
 - every node once, in its TRUE form: the node `question` when `truth` is true, else its `opposite` (no negatives,
 no question / answer pairs);
 - each true question turned into a declarative sentence by a fixed set of patterns (below); a node that matches no
 pattern stops the build (no silent fallback);
 - sentences joined by newlines. build_examples appends "\\n\\nAnswer: <letter>" itself.
 Patterns (q = the true question):
 Is N a knight|knave[ exactly when "S" is true]? -> N is a knight|knave[ exactly when "S" is true].
 Is the statement by N, "S", true|false? -> The statement by N, "S", is true|false.
 Does N's role require the statement "S" to be true|false? -> N's role requires the statement "S" to be true|false.
 If C, would BODY? -> If C, BODY'. with BODY' one of
 N be a knight|knave -> N would be a knight|knave
 the statement by N, "S", be true|false -> the statement by N, "S", would be true|false
 N's role require the statement "S" to be true|false -> N's role would require the statement "S" to be...
 N's statement contradict the rule that... -> N's statement would contradict the rule that...
Outputs:
 data/student_kk_tf_rat/{train,val}.jsonl rationale filled (A7 and the tree arms read the same rows)
 data/student_kk_tf_rat_nonodes/{train,val}.jsonl the same rows with subqs = [] (arm rn: tree path, no nodes)
 data/student_kk_tf_rat/summary.json counts, pattern counts, md5s, length stats
Rows are written as json.dumps(row) of the parsed base row with only the changed field replaced; the base rows are
checked to re-serialise to the base file byte for byte first, so every unchanged field is byte-identical.

 python3 scripts/build_kk_rationale.py
 CUDA_VISIBLE_DEVICES= ~/.venvs/sev/bin/python scripts/build_kk_rationale.py --token-stats <tokenizer dir>
The second step reads the built files only and measures each LM example as cotdistill.student.lm_loss does
(prompt view tail-encoded to lm_max_len // 2, plus the continuation), reporting how many would be cut at 3072 / 4096 /
5120 tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/student_kk_tf"
OUT = ROOT / "data/student_kk_tf_rat"
OUT_NN = ROOT / "data/student_kk_tf_rat_nonodes"
N = r"[A-Z][A-Za-z'\-]+"
TV = r"true|false"
S = r'"[^"]*"'
PATTERNS = [
    ("is_role", re.compile(rf"^Is (?P<n>{N}) (?P<rest>a (?:knight|knave)(?: exactly when {S} is (?:{TV}))?)\?$"),
     lambda m: f"{m['n']} is {m['rest']}."),
    ("stmt_tv", re.compile(rf"^Is the statement by (?P<n>{N}), (?P<s>{S}), (?P<tv>{TV})\?$"),
     lambda m: f"The statement by {m['n']}, {m['s']}, is {m['tv']}."),
    ("role_req", re.compile(rf"^Does (?P<n>{N})'s role require (?P<rest>the statement {S} to be (?:{TV}))\?$"),
     lambda m: f"{m['n']}'s role requires {m['rest']}."),
]
BODY = [
    ("c_role", re.compile(rf"^(?P<n>{N}) be (?P<r>a (?:knight|knave))$"), lambda m: f"{m['n']} would be {m['r']}"),
    ("c_stmt_tv", re.compile(rf"^the statement by (?P<n>{N}), (?P<s>{S}), be (?P<tv>{TV})$"),
     lambda m: f"the statement by {m['n']}, {m['s']}, would be {m['tv']}"),
    ("c_role_req", re.compile(rf"^(?P<n>{N})'s role require (?P<rest>the statement {S} to be (?:{TV}))$"),
     lambda m: f"{m['n']}'s role would require {m['rest']}"),
    ("c_contra", re.compile(rf"^(?P<n>{N})'s statement contradict (?P<rest>the rule that .+)$"),
     lambda m: f"{m['n']}'s statement would contradict {m['rest']}"),
]
COND = re.compile(rf"^If (?P<c>{N} were a (?:knight|knave)(?: and {N} were a (?:knight|knave))*), would (?P<b>.+)\?$")


def declarative(q: str, counts: dict) -> str:
    for name, rx, fn in PATTERNS:
        m = rx.match(q)
        if m:
            counts[name] = counts.get(name, 0) + 1
            return fn(m)
    m = COND.match(q)
    if m:
        for name, rx, fn in BODY:
            mb = rx.match(m["b"])
            if mb:
                counts[name] = counts.get(name, 0) + 1
                return f"If {m['c']}, {fn(mb)}."
    raise ValueError(f"no pattern for node question: {q!r}")


def topo(nodes: list[dict]) -> list[dict]:
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    deps = {n["id"]: [d for d in (n.get("depends_on") or []) if d in idx] for n in nodes}
    missing = [(n["id"], d) for n in nodes for d in (n.get("depends_on") or []) if d not in idx]
    if missing:
        raise ValueError(f"depends_on names unknown nodes: {missing[:3]}")
    done, out = set(), []
    while len(out) < len(nodes):
        ready = [n for n in nodes if n["id"] not in done and all(d in done for d in deps[n["id"]])]
        if not ready:
            raise ValueError("cycle in depends_on")
        n = min(ready, key=lambda x: idx[x["id"]])
        out.append(n)
        done.add(n["id"])
    return out


def rationale_of(row: dict, counts: dict) -> str:
    nodes = row.get("subqs") or []
    sent = []
    for n in topo(nodes):
        q = n["question"] if n["truth"] else n.get("opposite")
        if not q:
            raise ValueError(f"{row['item_id']}: node {n['id']} is false and has no opposite")
        sent.append(declarative(q, counts))
    return "\n".join(sent)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_NN.mkdir(parents=True, exist_ok=True)
    summ = {"base": {}, "out": {}, "patterns": {}, "rows": {}}
    for split in ("train", "val"):
        src = BASE / f"{split}.jsonl"
        lines = [l for l in src.read_text().splitlines() if l.strip()]
        rat, nn, chars, nodes, empty = [], [], [], [], 0
        for line in lines:
            row = json.loads(line)
            if json.dumps(row) != line:
                raise ValueError(f"{row['item_id']}: base row does not re-serialise byte for byte")
            if row.get("rationale"):
                raise ValueError(f"{row['item_id']}: base rationale is not empty")
            r = rationale_of(row, summ["patterns"]) if row.get("subqs") else ""
            empty += r == ""
            row["rationale"] = r
            rat.append(json.dumps(row))
            row2 = dict(row, subqs=[])
            nn.append(json.dumps(row2))
            if r:
                chars.append(len(r))
                nodes.append(len(row["subqs"]))
        (OUT / f"{split}.jsonl").write_text("\n".join(rat) + "\n")
        (OUT_NN / f"{split}.jsonl").write_text("\n".join(nn) + "\n")
        summ["base"][split] = md5(src)
        summ["out"][f"rat/{split}"] = md5(OUT / f"{split}.jsonl")
        summ["out"][f"nonodes/{split}"] = md5(OUT_NN / f"{split}.jsonl")
        summ["rows"][split] = {"rows": len(lines), "no_tree_no_rationale": empty,
                               "nodes_median": statistics.median(nodes), "nodes_max": max(nodes),
                               "chars_median": statistics.median(chars), "chars_p95": sorted(chars)[int(.95 * len(chars))],
                               "chars_max": max(chars)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def token_stats(tok_dir: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from transformers import AutoTokenizer
    from cotdistill.student import RATIONALE_PREFIX
    tok = AutoTokenizer.from_pretrained(tok_dir)
    res = {}
    for split in ("train", "val"):
        L = []
        for line in (OUT / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if not row["rationale"]:
                continue
            ans = max(row["teacher"], key=row["teacher"].get)
            pre = tok.encode(RATIONALE_PREFIX.format(problem=row["prompt"]), add_special_tokens=False)
            cont = tok.encode(f" {row['rationale'].strip()}\n\nAnswer: {ans}", add_special_tokens=False)
            L.append((len(pre), len(cont)))
        tot = sorted(a + b for a, b in L)
        res[split] = {"n": len(L), "total_median": statistics.median(tot), "total_p95": tot[int(.95 * len(tot))],
                      "total_max": tot[-1], "cont_median": statistics.median(b for _, b in L),
                      **{f"cut_at_{m}": sum(1 for a, b in L if min(a, m // 2) + b > m) for m in (3072, 4096, 5120)}}
    s = json.loads((OUT / "summary.json").read_text())
    s["tokens"] = dict(res, tokenizer=tok_dir)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--token-stats", default="", help="tokenizer dir: add LM token lengths to summary.json")
    a = ap.parse_args(argv)
    print(json.dumps(token_stats(a.token_stats) if a.token_stats else build(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
