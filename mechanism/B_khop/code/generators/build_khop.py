"""k-hop symbolic chains for CARD_khop_v1 (family khop, idea #4): eval sets, training files and build_report.json.

    python3 scripts/build_khop.py --out <E>/src/data/eval/khop [--tokenizer <Qwen3.5-4B tokenizer dir>]

Every item is a 5-option multiple-choice prompt in the GSM eval layout ("Question: <statements> <query>\\nOptions:\\nA) v
..."; the trainer and eval_student.py add "\\n\\nAnswer:"). Five chains per item (target T = chain 0, distractors D1..D4),
each of k hops. Tasks:
  ca   chained arithmetic, "bo = 7." then "ki = bo + 5." / "ki = bo − 3." / "ki = bo × 2." / "ki = bo × 3."; values in
       [1, 99], roots in [2, 60], addends in [2, 9]
  cag  ca with GSM-sized numbers (values [10, 999], roots [10, 400], addends [11, 99]); fwd only, reported only
  wp   the ca structure in words ("Mia has 7 marbles." / "Leo has 5 more marbles than Mia." / "... 3 fewer ..." /
       "... twice as many marbles as Mia." / "... three times as many ..."), one noun per item
  vb   pointer chasing, "bo = 47." (distinct roots in [10, 99]) then copies "ki = bo."; options = the five roots
ca / wp / cag (Amendment 2, build critic finding 1): the five chains share one skeleton, one root value r and one
op-kind sequence; only the constants differ. Hop 1 is + or - (five distinct constants, drawn without replacement), hops
2..k are uniform over the kinds that keep every chain in range, each chain draws its own constant. So the queried
chain's root and kinds are the same for all five options, and a policy that chases to the root and applies the kinds
without reading the constants is at chance by symmetry. The whole item is resampled when, for the final-value set or
for the hop-1 set, two options are equal, an option equals an integer written in the statements (roots, constants and
the multipliers 2 / 3 of "× 2" / "× 3"), or one option is 2x or 3x another. The additive part of the Amendment 1 rule
(|g - h| in [lo, hi]) is dropped: options that differ only by the constants cannot meet it. 10,000 resamples of one
item stop the build (about 2 draws per item are needed).
Premise order: fwd = a uniformly random interleaving of the five chains (a uniform linear extension: every statement after
the one it reads); rev = the exact reverse of the same item's fwd order.

Files (out dir): khop_{ca,wp,vb}_{fwd,rev}.jsonl (k = 1..6 x 150, k = 7, 8 x 100), khop_cag_fwd.jsonl (k = 1..6 x 150),
khop_ca_fwd_qs.jsonl (query-swap twin of khop_ca_fwd k = 1..6: query on D1_k, gold D1_k, id "<id>-qs"), khop_blind.jsonl
(the first 18 items of every (task, fwd, k <= 6) cell of ca / wp / vb, statements removed), train_kh.jsonl (3,000 rows:
1,000 per task, k = 1 + j % 6, order fwd / rev alternating per 6 rows), train_k1.jsonl (the same 3,000 statements and
orders, query on the target chain's hop-1 variable; ca / wp options = the hop-1 set, vb keeps the five roots),
train_kd.jsonl (train_kh's 3,000 rows + 6,000 more from the same generator stream), build_report.json.
Seeds: training stream 1 (KH rows first, then the KD rows), eval stream 2. Eval items whose prompt hash or value key
(task, sorted chain value tuples) occurs in any training row are redrawn (counted in the report).

Gold letter: in every (set, k) cell the i-th item puts T at position (i + 3 (k - 1)) mod 5 (the k offset balances the
18-row blind cells pooled per task), D1 at (p + 1 + (i // 5) mod 4) mod 5 (so the -qs twin, whose gold is D1, is
balanced too), the other three distractors in a seeded shuffle. Training rows cycle the same way per (file, task, k).

build_report.json (section 2.4 D-lines): counts per cell, md5 per file, gold-letter shares, the D1 lines per (set, k)
cell and pooled over k = 1..6 (d1_baselines): BASELINES (pick-max, pick-min, pick-median, closest-literal,
first-literal, last-literal, unwritten, query-op pair, query-op consistent, one-op parent, chase-mult, chase-sign; barred
on every set but khop_blind), EVAL_BARRED (query-op avoid; barred on the eval sets, reported on the training files) and
REPORTED (chase-last1, chase-drop1, chase-droplast, drop-one-best, hard share; partial-composition lines, no bar); the
resample rate per (task, k), prompt token lengths per (task, k) and token totals per training file (with --tokenizer),
disjointness counts and the pass / fail of every D-line. Every line is computed from the rows alone (parse_prompt) as an
expected accuracy (a rule picks uniformly among its candidates; no candidate: uniformly among the five options), so
tests recompute it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E"]
KINDS = ["+", "-", "*2", "*3"]
GLYPH = {"+": "+", "-": "−", "*2": "×", "*3": "×"}
RANGES = {"ca": (1, 99, 2, 60, 2, 9), "wp": (1, 99, 2, 60, 2, 9), "cag": (10, 999, 10, 400, 11, 99)}
VB_LO, VB_HI = 10, 99
MAX_RESAMPLE = 10_000             # card 1.1 (Amendment 2: the shared-skeleton rule needs about 2 draws per item)
CONS = "bcdfghjklmnprstvwyz"                     # 19 consonants
EXCLUDE = {"be", "by", "do", "fa", "go", "ha", "he", "hi", "ho", "la", "lo", "ma", "me", "mu", "no", "nu", "pa", "pi",
           "re", "so", "ta", "ti", "to", "we", "ye", "yo", "ya", "wo", "is", "us", "ok", "oh"}
SYLLABLES = sorted(c + v for c in CONS for v in "aeiou" if c + v not in EXCLUDE)
PEOPLE = ["Mia", "Leo", "Ava", "Ben", "Zoe", "Max", "Ivy", "Sam", "Eli", "Amy", "Kai", "Nora", "Owen", "Ruby", "Finn",
          "Lily", "Jack", "Ella", "Noah", "Iris", "Theo", "Maya", "Luke", "Rosa", "Hugo", "Cleo", "Adam", "Tara", "Omar",
          "Gwen", "Paul", "Vera", "Ezra", "Dina", "Carl", "Lena", "Ivan", "Jade", "Kurt", "Faye", "Seth", "Hana", "Dev",
          "Yara", "Neil", "Beth", "Raul", "Gina", "Troy", "Wren", "Abel", "Uma", "Joel", "Kira", "the second author", "Pia", "Reid",
          "Sara", "Vito", "Zara"]
NOUNS = ["marbles", "stickers", "apples", "coins", "cards", "shells", "stamps", "pencils", "books", "beads", "buttons",
         "cookies", "crayons", "pebbles", "balloons", "candles", "ribbons", "tickets", "magnets", "acorns"]
TASKS = ("ca", "wp", "vb")
EVAL_K = {k: (150 if k <= 6 else 100) for k in range(1, 9)}
BLIND_N = 18
BARS = {"cell": 0.33, "pooled": 0.25, "gold_lo": 0.17, "gold_hi": 0.23, "max_tokens": 1536}
BASELINES = ("max", "min", "median", "closest_lit", "first_lit", "last_lit", "unwritten", "qop_pair", "qop_consistent",
             "one_op_parent", "chase_mult", "chase_sign")         # barred on every set but khop_blind (Amendment 2)
EVAL_BARRED = ("qop_avoid",)                    # barred on the eval sets, reported on the training files (Amendment 2)
REPORTED = ("chase_last1", "chase_drop1", "chase_droplast", "drop_one_best", "hard")   # no bar: partial-composition lines
ALL_LINES = BASELINES + EVAL_BARRED + REPORTED
GSMVT_KEYS = ("item_id", "source", "group", "type", "prompt", "labels", "gold_label", "teacher", "options",
              "option_kinds", "recipe", "depth", "dep_level", "subqs", "random_subqs", "rationale", "augment", "has_tree",
              "tree_fix", "row_kind", "problem_id", "node_status", "licence_class", "licence", "label_basis",
              "node_basis", "pool_rank", "n_steps", "tree_source", "value_nodes", "worked_steps", "verify_nodes")
EVAL_EXTRA = ("task", "order", "k", "label_order", "gold_probs")
assert len(SYLLABLES) >= 45 and len(PEOPLE) == 60 and len(NOUNS) == 20
assert not any(a != b and b.startswith(a) for a in PEOPLE for b in PEOPLE), "a name is a prefix of another"


class BuildError(RuntimeError):
    pass


# --- chains (critique/sim_khop_a1.py, ported) ---------------------------------------------------------------------
def _feasible(v: int, o: str, R) -> bool:
    vlo, vhi, _, _, clo, _ = R
    return {"+": v + clo <= vhi, "-": v - clo >= vlo, "*2": 2 * v <= vhi, "*3": 3 * v <= vhi}[o]


def apply_op(v: int, o: str, c: int | None) -> int:
    return v + c if o == "+" else v - c if o == "-" else 2 * v if o == "*2" else 3 * v


def _const(v: int, o: str, R, rng) -> int | None:
    vlo, vhi, _, _, clo, chi = R
    if o == "+":
        return rng.randint(clo, min(chi, vhi - v))
    if o == "-":
        return rng.randint(clo, min(chi, v - vlo))
    return None


def one_op(h: int, g: int, clo: int, chi: int) -> bool:
    return clo <= abs(g - h) <= chi or g == 2 * h or g == 3 * h


def mult_pair(h: int, g: int) -> bool:
    return g == 2 * h or g == 3 * h


def option_set_ok(opts, lits) -> bool:
    """Amendment 2 option rule: distinct, no option equals a written number, no option 2x or 3x another. The additive
    part of the Amendment 1 rule (|g - h| in [lo, hi]) is dropped: on a shared skeleton the five options differ only by
    the constants, so it cannot be met (BUILD_REPORT, fix round)."""
    if len(set(opts)) < len(opts) or any(o in lits for o in opts):
        return False
    return not any(h != g and mult_pair(h, g) for h in opts for g in opts)


def chain_literals(ops) -> set:
    """Integers the chain's statements write: the addends and, for x2 / x3, the multiplier (ca / cag glyph form)."""
    out = set()
    for o, c in ops:
        out.add(c if c is not None else (2 if o == "*2" else 3))
    return out


def draw_arith_item(task: str, k: int, rng, stats: Counter):
    """Five chains of an arithmetic task (ca / wp / cag) on one shared skeleton (Amendment 2): one root value r and one
    op-kind sequence for all five chains; only the constants differ between chains. Hop 1 is + or - (uniform over the
    kinds whose constant range holds 5 distinct values), with 5 distinct hop-1 constants drawn without replacement; hops
    2..k are uniform over the kinds that keep every chain's value in range, each chain's constant drawn on its own. The
    whole item is resampled when, for the final-value set or for the hop-1 set, two options are equal, an option equals
    a number written in the statements, or one option is 2x or 3x another (mult_pair). Returns [(values, ops)] * 5."""
    R = RANGES[task]
    vlo, vhi, rlo, rhi, clo, chi = R
    for n in range(MAX_RESAMPLE):
        stats["draws"] += 1
        r = rng.randint(rlo, rhi)
        vals, ops = [[r] for _ in range(5)], [[] for _ in range(5)]
        good = True
        for j in range(1, k + 1):
            cur = [v[-1] for v in vals]
            if j == 1:
                span = {"+": (clo, min(chi, vhi - r)), "-": (clo, min(chi, r - vlo))}
                feas = [o for o in ("+", "-") if span[o][1] - span[o][0] + 1 >= 5]
            else:
                feas = [o for o in KINDS if all(_feasible(v, o, R) for v in cur)]
            if not feas:
                good = False
                break
            o = rng.choice(feas)
            cs = rng.sample(range(span[o][0], span[o][1] + 1), 5) if j == 1 else [_const(v, o, R, rng) for v in cur]
            for c in range(5):
                ops[c].append((o, cs[c]))
                vals[c].append(apply_op(cur[c], o, cs[c]))
        if not good:
            continue
        ch = list(zip(vals, ops))
        lits = {r}
        for _, op in ch:
            lits |= chain_literals(op) if task != "wp" else {c for _, c in op if c is not None}
        fin = [v[k] for v in vals]
        h1 = [v[1] for v in vals]
        if option_set_ok(fin, lits) and option_set_ok(h1, lits):
            stats["items"] += 1
            return ch
    raise BuildError(f"{task} k={k}: no item passes the option rule after {MAX_RESAMPLE} resamples")


def draw_vb_item(k: int, rng, stats: Counter):
    roots = rng.sample(range(VB_LO, VB_HI + 1), 5)
    stats["draws"] += 1
    stats["items"] += 1
    return [([r] * (k + 1), [("copy", None)] * k) for r in roots]


# --- rendering ------------------------------------------------------------------------------------------------------
def interleave(k: int, rng) -> list[tuple[int, int]]:
    """A uniformly random interleaving of five chains of k + 1 statements: [(chain, position in chain)]."""
    seq = [c for c in range(5) for _ in range(k + 1)]
    rng.shuffle(seq)
    nxt = [0] * 5
    out = []
    for c in seq:
        out.append((c, nxt[c]))
        nxt[c] += 1
    return out


def statement(task: str, names, noun, ch, c: int, j: int) -> str:
    vals, ops = ch[c]
    me = names[c][j]
    if task == "wp":
        if j == 0:
            return f"{me} has {vals[0]} {noun}."
        o, cst = ops[j - 1]
        par = names[c][j - 1]
        return {"+": f"{me} has {cst} more {noun} than {par}.", "-": f"{me} has {cst} fewer {noun} than {par}.",
                "*2": f"{me} has twice as many {noun} as {par}.",
                "*3": f"{me} has three times as many {noun} as {par}."}[o]
    if j == 0:
        return f"{me} = {vals[0]}."
    o, cst = ops[j - 1]
    par = names[c][j - 1]
    if o == "copy":
        return f"{me} = {par}."
    return f"{me} = {par} {GLYPH[o]} {cst if cst is not None else (2 if o == '*2' else 3)}."


def query_text(task: str, name: str, noun) -> str:
    return f"How many {noun} does {name} have?" if task == "wp" else f"What is the value of {name}?"


def render(body: str, opts) -> str:
    return f"Question: {body}\nOptions:\n" + "\n".join(f"{L}) {v}" for L, v in zip(LETTERS, opts))


def layout(i: int, k: int, rng) -> list[int]:
    """slot -> chain index: T (chain 0) at p = (i + 3 (k - 1)) mod 5, D1 (chain 1) at (p + 1 + (i // 5) mod 4) mod 5, the
    other three chains shuffled into the rest."""
    p = (i + 3 * (k - 1)) % 5
    q = (p + 1 + (i // 5) % 4) % 5
    rest = [2, 3, 4]
    rng.shuffle(rest)
    slots = [None] * 5
    slots[p], slots[q] = 0, 1
    for s in range(5):
        if slots[s] is None:
            slots[s] = rest.pop()
    return slots


def value_key(task: str, ch) -> str:
    return task + ":" + json.dumps(sorted(tuple(v) for v, _ in ch))


def phash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def base_row(item_id, task, k, prompt, opts, gold_slot, kinds, group) -> dict:
    gold = LETTERS[gold_slot]
    return {"item_id": item_id, "source": "khop", "group": group, "type": "choice", "prompt": prompt,
            "labels": list(LETTERS), "gold_label": gold, "teacher": {L: (1.0 if L == gold else 0.0) for L in LETTERS},
            "options": [str(v) for v in opts], "option_kinds": kinds, "recipe": "khop", "depth": k, "dep_level": k,
            "subqs": [], "random_subqs": [], "rationale": "", "augment": None, "has_tree": False, "tree_fix": False,
            "row_kind": "final", "problem_id": item_id, "node_status": "ok", "licence_class": "PERM",
            "licence": "synthetic (build_khop.py)", "label_basis": "gold", "node_basis": "program", "pool_rank": 0,
            "n_steps": k, "tree_source": "khop", "value_nodes": [], "worked_steps": [], "verify_nodes": []}


class Item:
    """One generated item: chains, names, noun, fwd order; renders rows for every query / order / option variant."""

    def __init__(self, task, k, ch, rng):
        self.task, self.k, self.ch = task, k, ch
        n = 5 * (k + 1)
        pool = PEOPLE if task == "wp" else SYLLABLES
        flat = rng.sample(pool, n)
        self.names = [flat[c * (k + 1):(c + 1) * (k + 1)] for c in range(5)]
        self.noun = rng.choice(NOUNS) if task == "wp" else None
        self.fwd = interleave(k, rng)

    def statements(self, order: str) -> list[str]:
        seq = self.fwd if order == "fwd" else list(reversed(self.fwd))
        return [statement(self.task, self.names, self.noun, self.ch, c, j) for c, j in seq]

    def options(self, hop: int) -> list[int]:
        """The option value of every chain: vb the roots; ca / wp / cag the value after `hop` hops."""
        return [v[0] if self.task == "vb" else v[hop] for v, _ in self.ch]

    def row(self, item_id, order, slots, *, query_chain=0, hop=None, blind=False, group=None) -> dict:
        """hop: the queried hop (default k); query_chain: whose variable is asked (gold = that chain's option)."""
        hop = self.k if hop is None else hop
        vals = self.options(hop)
        opts = [vals[c] for c in slots]
        q = query_text(self.task, self.names[query_chain][hop], self.noun)
        body = q if blind else " ".join(self.statements(order)) + " " + q
        gold_slot = slots.index(query_chain)
        kinds = ["gold" if c == query_chain else "cross" for c in slots]
        return base_row(item_id, self.task, self.k, render(body, opts), opts, gold_slot, kinds,
                        group or f"khop/{self.task}/{order}/k{self.k}")


# --- parsing (the rows alone; used by the D-lines, tests and readout/khop_table.py) ---------------------------------
_CA = re.compile(r"^([a-z]{2}) = ([a-z]{2}|\d+)(?: ([+−×]) (\d+))?\.$")
_WP = [(re.compile(r"^(\w+) has (\d+) (\w+)\.$"), "root"),
       (re.compile(r"^(\w+) has (\d+) more (\w+) than (\w+)\.$"), "+"),
       (re.compile(r"^(\w+) has (\d+) fewer (\w+) than (\w+)\.$"), "-"),
       (re.compile(r"^(\w+) has twice as many (\w+) as (\w+)\.$"), "*2"),
       (re.compile(r"^(\w+) has three times as many (\w+) as (\w+)\.$"), "*3")]
_Q = re.compile(r"(What is the value of ([a-z]{2})\?|How many (\w+) does (\w+) have\?)$")


def parse_prompt(prompt: str) -> dict:
    """{"stmts": [{"name", "parent", "op", "c", "lits"}], "query": name, "options": [int], "literals": [int in text
    order], "values": {name: value}, "chain_of": {name: root name}, "pos": {name: statement index}}. Blind prompts have
    no statements. Raises ValueError on a prompt that is not a khop prompt."""
    head, _, tail = prompt.partition("\nOptions:\n")
    if not head.startswith("Question: ") or not tail:
        raise ValueError("not a khop prompt")
    body = head[len("Question: "):]
    m = _Q.search(body)
    if not m:
        raise ValueError("no khop query")
    query = m.group(2) or m.group(4)
    sents = [s for s in re.split(r"(?<=\.) ", body[:m.start()].strip()) if s]
    opts = [int(l.split(") ", 1)[1]) for l in tail.split("\n")]
    stmts = []
    for s in sents:
        mm = _CA.match(s)
        if mm:
            name, src, glyph, c = mm.groups()
            if src.isdigit():
                stmts.append({"name": name, "parent": None, "op": "root", "c": int(src)})
            elif glyph is None:
                stmts.append({"name": name, "parent": src, "op": "copy", "c": None})
            else:
                op = {"+": "+", "−": "-"}.get(glyph) or ("*2" if c == "2" else "*3")
                stmts.append({"name": name, "parent": src, "op": op, "c": int(c)})
            continue
        for rx, kind in _WP:
            mm = rx.match(s)
            if mm:
                g = mm.groups()
                if kind == "root":
                    stmts.append({"name": g[0], "parent": None, "op": "root", "c": int(g[1])})
                elif kind in "+-":
                    stmts.append({"name": g[0], "parent": g[3], "op": kind, "c": int(g[1])})
                else:
                    stmts.append({"name": g[0], "parent": g[2], "op": kind, "c": None})
                break
        else:
            raise ValueError(f"unparsed statement {s!r}")
    by = {s["name"]: s for s in stmts}
    values, chain_of = {}, {}

    def val(n):
        if n not in values:
            s = by[n]
            if s["op"] == "root":
                values[n], chain_of[n] = s["c"], n
            else:
                pv = val(s["parent"])
                values[n] = pv if s["op"] == "copy" else apply_op(pv, s["op"], s["c"])
                chain_of[n] = chain_of[s["parent"]]
        return values[n]

    for s in stmts:
        val(s["name"])
    lits = [int(x) for s in sents for x in re.findall(r"\d+", s)]
    return {"stmts": stmts, "query": query, "options": opts, "literals": lits, "values": values, "chain_of": chain_of,
            "pos": {s["name"]: i for i, s in enumerate(stmts)}, "by": by}


def task_of(row: dict) -> str:
    return row["group"].split("/")[1]


def baselines(row: dict) -> dict:
    """Expected accuracy of every D1 rule on one row (uniform among a rule's candidates; none: uniform over five)."""
    P = parse_prompt(row["prompt"])
    opts = P["options"]
    gold = LETTERS.index(row["gold_label"])
    task = task_of(row)
    clo, chi = (RANGES[task][4], RANGES[task][5]) if task in RANGES else (2, 9)
    lits = P["literals"]
    q = P["by"].get(P["query"])                   # the queried statement (None in blind rows)

    def pick(cands):
        cands = [i for i in range(5) if i in cands]
        if not cands:
            return 0.2
        return (1.0 / len(cands)) if gold in cands else 0.0

    def closest(targets):
        if not targets:
            return pick([])
        d = [min(abs(o - t) for t in targets) for o in opts]
        return pick([i for i in range(5) if d[i] == min(d)])

    r = {"max": pick([opts.index(max(opts))]), "min": pick([opts.index(min(opts))]),
         "median": pick([opts.index(sorted(opts)[2])]), "closest_lit": closest(lits),
         "first_lit": closest(lits[:1]), "last_lit": closest(lits[-1:]),
         "unwritten": pick([i for i in range(5) if opts[i] not in set(lits)])}
    if q is not None and q["op"] in KINDS:
        o, c = q["op"], q["c"]
        r["qop_pair"] = pick([i for i in range(5) if any(j != i and apply_op(opts[j], o, c) == opts[i] for j in range(5))])
        cons = lambda g: (o == "*2" and g % 2 == 0) or (o == "*3" and g % 3 == 0) or (o == "+" and g > c) or o == "-"
        r["qop_consistent"] = pick([i for i in range(5) if cons(opts[i])])
        qp = [i for i in range(5) if any(j != i and apply_op(opts[j], o, c) == opts[i] for j in range(5))]
        r["qop_avoid"] = pick([i for i in range(5) if i not in qp]) if qp else pick([])
    else:
        r["qop_pair"] = r["qop_consistent"] = r["qop_avoid"] = pick([])
    r["one_op_parent"] = pick([i for i in range(5) if any(j != i and one_op(opts[j], opts[i], clo, chi) for j in range(5))])
    # Amendment 2 (build critic finding 1): magnitude policies that chase the queried chain to its root and read its op
    # kinds but not (all of) its constants; a constant not read is replaced by the mean constant (lo + hi) / 2.
    # Barred: chase_mult (x2 / x3 only), chase_sign (every constant at the mean). Reported (partial composition, no bar):
    # chase_last1 (only the queried statement's constant exact), chase_drop1 (all exact but hop 1), chase_droplast (all
    # exact but the last hop), drop_one_best (per row the best of "all exact but additive hop j" over j), hard (1.0 when
    # no drop-one estimate has the gold among its nearest options: every constant of the target is needed).
    # vb (copies only) and blind rows: no estimate, uniform; hard is None.
    hops = query_hops(P)
    if hops is not None and task in RANGES:
        root, qops = hops
        n = len(qops)
        mean_c = (clo + chi) / 2
        x_mult = float(root)
        for o, _ in qops:
            x_mult *= {"*2": 2, "*3": 3}.get(o, 1)
        r["chase_mult"] = closest([x_mult])
        r["chase_sign"] = closest([chase_est(root, qops, set(), mean_c)])
        r["chase_last1"] = closest([chase_est(root, qops, {n - 1}, mean_c)])
        r["chase_drop1"] = closest([chase_est(root, qops, set(range(1, n)), mean_c)])
        r["chase_droplast"] = closest([chase_est(root, qops, set(range(n - 1)), mean_c)])
        drops = [closest([chase_est(root, qops, set(range(n)) - {j}, mean_c)]) for j, (o, _) in enumerate(qops)
                 if o in ("+", "-")]
        r["drop_one_best"] = max(drops) if drops else pick([])     # no additive hop: never in generated rows
        r["hard"] = float(all(d == 0 for d in drops)) if drops else None
    else:
        r["chase_mult"] = r["chase_sign"] = r["chase_last1"] = r["chase_drop1"] = r["chase_droplast"] = \
            r["drop_one_best"] = pick([])
        r["hard"] = None
    return r


def chase_est(root: int, qops, exact: set, mean_c: float) -> float:
    """The queried chain's value with x2 / x3 applied and every +c / -c whose hop index (0-based) is not in `exact`
    replaced by +/- mean_c."""
    x = float(root)
    for j, (o, c) in enumerate(qops):
        if o in ("*2", "*3"):
            x *= 2 if o == "*2" else 3
        else:
            x += (c if j in exact else mean_c) * (1 if o == "+" else -1)
    return x


def hard_item(row: dict):
    """True when every constant of the queried chain is needed (no drop-one estimate puts the gold among its nearest
    options); None for vb and blind rows. Used by readout/khop_table.py for the hard-subset accuracy (Amendment 2)."""
    h = baselines(row)["hard"]
    return None if h is None else bool(h)


def query_hops(P: dict):
    """(root value, [(kind, const)] from hop 1 to the queried hop) of the queried chain, or None (blind / vb copies)."""
    q = P["by"].get(P["query"])
    if q is None:
        return None
    ops = []
    while q["op"] != "root":
        if q["op"] == "copy":
            return None
        ops.append((q["op"], q["c"]))
        q = P["by"][q["parent"]]
    return q["c"], ops[::-1]


def row_rule_ok(row: dict) -> list[str]:
    """The 1.1 rules checked on one row from the prompt alone: gold in options, options distinct, every option is one
    chain's value at the queried hop, the query's chain is the gold; ca / wp / cag: the shared skeleton (one root value
    and one op-kind sequence for the five chains, hop 1 + or - with five distinct constants) and, for the shown option
    set and for the hop-1 set, no option equals a written number and no option is 2x or 3x another. [] = ok."""
    errs = []
    P = parse_prompt(row["prompt"])
    opts = P["options"]
    if row["gold_label"] not in LETTERS or len(set(opts)) != 5:
        errs.append("options not distinct / gold missing")
    if [str(o) for o in opts] != row["options"]:
        errs.append("options field differs from the prompt")
    if not P["stmts"]:
        return errs
    task = task_of(row)
    roots = [s["name"] for s in P["stmts"] if s["op"] == "root"]
    if len(roots) != 5:
        errs.append(f"{len(roots)} chains")
        return errs
    kids = {s["parent"]: s["name"] for s in P["stmts"] if s["parent"]}
    chains = []
    for r in roots:
        seq = [r]
        while seq[-1] in kids:
            seq.append(kids[seq[-1]])
        chains.append(seq)
    if len({len(c) for c in chains}) != 1 or len(chains[0]) - 1 != row["depth"]:
        errs.append("chain lengths differ from k")
    hop = next((j for c in chains for j, n in enumerate(c) if n == P["query"]), None)
    qchain = next((ci for ci, c in enumerate(chains) if P["query"] in c), None)
    if hop is None:
        errs.append("query is not a chain variable")
        return errs
    vals_at = (lambda h: [P["values"][c[h]] for c in chains])
    shown = vals_at(hop)
    if sorted(shown) != sorted(opts):
        errs.append("options are not the chains' values at the queried hop")
    if opts[LETTERS.index(row["gold_label"])] != shown[qchain]:
        errs.append("gold is not the queried chain's value")
    if task in RANGES:
        clo, chi = RANGES[task][4], RANGES[task][5]
        lits = set(P["literals"])
        for name, s in (("shown", shown), ("hop-1", vals_at(1))):
            if not option_set_ok(s, lits):
                errs.append(f"{name} set breaks the option rule")
        by = P["by"]
        if len({by[c[0]]["c"] for c in chains}) != 1:
            errs.append("roots differ (skeleton)")
        if len({tuple(by[n]["op"] for n in c[1:]) for c in chains}) != 1:
            errs.append("op kinds differ (skeleton)")
        h1 = [by[c[1]] for c in chains]
        if h1[0]["op"] not in ("+", "-") or len({x["c"] for x in h1}) != 5:
            errs.append("hop 1 is not + / - with five distinct constants (skeleton)")
        vlo, vhi = RANGES[task][0], RANGES[task][1]
        if any(not vlo <= v <= vhi for v in P["values"].values()):
            errs.append("a value leaves the range")
    return errs


def fwd_is_topological(prompt: str) -> bool:
    P = parse_prompt(prompt)
    return all(s["parent"] is None or P["pos"][s["parent"]] < P["pos"][s["name"]] for s in P["stmts"])


# --- build ----------------------------------------------------------------------------------------------------------
def gen_item(task, k, rng, stats):
    ch = draw_vb_item(k, rng, stats[(task, k)]) if task == "vb" else draw_arith_item(task, k, rng, stats[(task, k)])
    return Item(task, k, ch, rng)


def build_train(rng, stats, n_kh: int = 1000, n_kd_extra: int = 2000):
    """KH rows (n_kh per task), K1 rows (the same items, hop-1 query) and the KD extra rows (n_kd_extra per task), with
    the value keys and prompt hashes of every training row."""
    kh, k1, kd_extra, keys, hashes = [], [], [], set(), set()
    cyc = Counter()
    for part, n0, n1, out in (("kh", 0, n_kh, kh), ("kd", n_kh, n_kh + n_kd_extra, kd_extra)):
        for task in TASKS:
            for j in range(n0, n1):
                k = 1 + j % 6
                order = "fwd" if (j // 6) % 2 == 0 else "rev"
                it = gen_item(task, k, rng, stats)
                keys.add(value_key(task, it.ch))
                iid = f"khop-tr-{task}-{j:05d}"
                i = cyc[("kh", task, k)]
                cyc[("kh", task, k)] += 1
                r = it.row(iid, order, layout(i, k, rng))
                out.append(r)
                hashes.add(phash(r["prompt"]))
                if part == "kh":
                    i1 = cyc[("k1", task, k)]
                    cyc[("k1", task, k)] += 1
                    r1 = it.row(iid + "-k1", order, layout(i1, k, rng), hop=1,
                                group=f"khop/{task}/{order}/k{k}")
                    k1.append(r1)
                    hashes.add(phash(r1["prompt"]))
    return kh, k1, kh + kd_extra, keys, hashes


def build_eval(rng, stats, train_keys, train_hashes, counts=None):
    """{file stem: rows}; eval items colliding with a training row (value key or prompt hash) are redrawn."""
    counts = counts or {t: dict(EVAL_K) for t in TASKS} | {"cag": {k: 150 for k in range(1, 7)}}
    sets = defaultdict(list)
    redrawn = Counter()
    seen = set()
    for task in ("ca", "wp", "vb", "cag"):
        for k, n in counts[task].items():
            for i in range(n):
                while True:
                    it = gen_item(task, k, rng, stats)
                    vk = value_key(task, it.ch)
                    if vk in train_keys or vk in seen:
                        redrawn[f"{task}/k{k}/value_key"] += 1
                        continue
                    break
                seen.add(vk)
                slots = layout(i, k, rng)
                iid = f"khop-{task}-k{k}-{i:04d}"
                orders = ("fwd",) if task == "cag" else ("fwd", "rev")
                for o in orders:
                    r = it.row(iid, o, slots)
                    r.update({"task": task, "order": o, "k": k, "label_order": list(LETTERS), "gold_probs": None})
                    sets[f"khop_{task}_{o}"].append(r)
                if task == "ca" and k <= 6:
                    r = it.row(iid + "-qs", "fwd", slots, query_chain=1)
                    r.update({"task": task, "order": "fwd", "k": k, "label_order": list(LETTERS), "gold_probs": None})
                    sets["khop_ca_fwd_qs"].append(r)
                if task in TASKS and k <= 6 and i < BLIND_N:
                    r = it.row(iid, "fwd", slots, blind=True, group=f"khop/{task}/blind/k{k}")
                    r.update({"task": task, "order": "blind", "k": k, "label_order": list(LETTERS), "gold_probs": None})
                    sets["khop_blind"].append(r)
    for stem, rows in sets.items():
        clash = [r["item_id"] for r in rows if phash(r["prompt"]) in train_hashes]
        if clash:
            raise BuildError(f"{stem}: {len(clash)} prompts equal a training prompt, e.g. {clash[0]}")
    return dict(sets), dict(redrawn)


EVAL_FILES = ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev", "khop_cag_fwd",
              "khop_ca_fwd_qs", "khop_blind")
TRAIN_FILES = ("train_kh", "train_k1", "train_kd")


def cell_of(row: dict, stem: str) -> str:
    """The D-line cell of a row: eval sets by k (blind by task and k), training files by task and k."""
    t, _, k = row["group"].split("/")[1:]
    if stem.startswith("train"):
        return f"{t}/{k}"
    return f"{t}/{k}" if stem == "khop_blind" else k


def line_means(bs: list) -> dict:
    """Cell means of every D1 line (ALL_LINES); a line with no value in the cell (hard on vb / blind rows) is None."""
    out = {}
    for b in ALL_LINES:
        xs = [x[b] for x in bs if x[b] is not None]
        out[b] = round(sum(xs) / len(xs), 4) if xs else None
    return out


def barred(stem: str) -> tuple:
    """The D1 lines with a bar on this file: BASELINES everywhere, EVAL_BARRED on the eval sets only."""
    return BASELINES + (() if stem.startswith("train") else EVAL_BARRED)


def d_lines(files: dict, tok=None) -> dict:
    """D1 (every line of ALL_LINES per cell and pooled over k = 1..6; bars (barred()) on cells of 100-150 rows and on
    the pooled cells of every set but khop_blind; REPORTED lines have no bar), D2 (gold share per cell; khop_blind pooled per task), D4 (token lengths, with a tokenizer).
    Training files: cells (task, k), pooled per task."""
    rep = {"d1": {}, "d2": {}, "d4": {}, "fail": []}
    for stem, rows in files.items():
        cells = defaultdict(list)
        for r in rows:
            cells[cell_of(r, stem)].append(r)
        b1, b2 = {}, {}
        pooled = defaultdict(list)
        for c, rs in sorted(cells.items()):
            bs = [baselines(r) for r in rs]
            b1[c] = {"n": len(rs), **line_means(bs)}
            kk = int(c.split("k")[-1])
            pkey = (c.split("/")[0] if "/" in c else "all")
            if kk <= 6:
                pooled[pkey].extend(bs)
            g = Counter(r["gold_label"] for r in rs)
            b2[c] = {L: round(g[L] / len(rs), 4) for L in LETTERS}
            if stem != "khop_blind" and 100 <= len(rs):
                for b in barred(stem):
                    if b1[c][b] > BARS["cell"]:
                        rep["fail"].append(f"D1 {stem} {c} {b} {b1[c][b]} > {BARS['cell']}")
                if not all(BARS["gold_lo"] <= v <= BARS["gold_hi"] for v in b2[c].values()):
                    rep["fail"].append(f"D2 {stem} {c} gold shares {b2[c]}")
        for pk, bs in pooled.items():
            b1[f"pooled_k1-6/{pk}"] = {"n": len(bs), **line_means(bs)}
            if stem != "khop_blind":
                for b in barred(stem):
                    if b1[f"pooled_k1-6/{pk}"][b] > BARS["pooled"]:
                        rep["fail"].append(f"D1 {stem} pooled {pk} {b} {b1[f'pooled_k1-6/{pk}'][b]} > {BARS['pooled']}")
        if stem == "khop_blind":
            for t in TASKS:
                rs = [r for r in rows if task_of(r) == t]
                g = Counter(r["gold_label"] for r in rs)
                b2[f"pooled/{t}"] = {L: round(g[L] / len(rs), 4) for L in LETTERS}
                if not all(BARS["gold_lo"] <= v <= BARS["gold_hi"] for v in b2[f"pooled/{t}"].values()):
                    rep["fail"].append(f"D2 khop_blind pooled {t} gold shares {b2[f'pooled/{t}']}")
        rep["d1"][stem], rep["d2"][stem] = b1, b2
    if tok is not None:
        for stem, rows in files.items():
            lens = defaultdict(list)
            for r in rows:
                lens[f"{task_of(r)}/k{r['depth']}"].append(n_tokens(tok, r["prompt"]))
            rep["d4"][stem] = {c: {"mean": round(sum(v) / len(v), 1), "max": max(v)} for c, v in sorted(lens.items())}
            mx = max(max(v) for v in lens.values())
            if mx >= BARS["max_tokens"]:
                rep["fail"].append(f"D4 {stem} longest prompt {mx} tokens >= {BARS['max_tokens']}")
    return rep


def n_tokens(tok, prompt: str) -> int:
    """Tokens of the final question text the trainer and eval_student.py build (prompt + "\\n\\nAnswer:")."""
    return len(tok(prompt + "\n\nAnswer:", add_special_tokens=False)["input_ids"])


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def write_jsonl(p: Path, rows) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(out: Path, tok=None, train_seed: int = 1, eval_seed: int = 2, n_kh: int = 1000, n_kd_extra: int = 2000,
          eval_counts=None, gsm_train: Path | None = None) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(Counter)
    kh, k1, kd, keys, hashes = build_train(random.Random(train_seed), stats, n_kh, n_kd_extra)
    ev, redrawn = build_eval(random.Random(eval_seed), stats, keys, hashes, eval_counts)
    files = {**{s: ev[s] for s in EVAL_FILES if s in ev}, "train_kh": kh, "train_k1": k1, "train_kd": kd}
    for stem, rows in files.items():
        write_jsonl(out / f"{stem}.jsonl", rows)
    rep = d_lines(files, tok)
    ev_keys = set()
    ev_hash = set()
    for s in EVAL_FILES:
        for r in files.get(s, []):
            ev_hash.add(phash(r["prompt"]))
    # value keys of the eval items from the rows alone (parse), against the training rows' parsed keys
    def row_key(r):
        P = parse_prompt(r["prompt"])
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
        return task_of(r) + ":" + json.dumps(sorted(chains))
    for s in EVAL_FILES:
        for r in files.get(s, []):
            kk = row_key(r)
            if kk:
                ev_keys.add(kk)
    tr_keys = {row_key(r) for s in TRAIN_FILES for r in files[s]}
    tr_hash = {phash(r["prompt"]) for s in TRAIN_FILES for r in files[s]}
    d3 = {"shared_prompt_hashes": len(ev_hash & tr_hash), "shared_value_keys": len(ev_keys & tr_keys),
          "eval_redrawn": redrawn}
    if d3["shared_prompt_hashes"] or d3["shared_value_keys"]:
        rep["fail"].append(f"D3 {d3}")
    report = {
        "card": "CARD_khop_v1", "generator": "scripts/build_khop.py", "seeds": {"train": train_seed, "eval": eval_seed},
        "counts": {s: dict(sorted(Counter(cell_of(r, s) for r in rows).items())) | {"total": len(rows)}
                   for s, rows in files.items()},
        "md5": {f"{s}.jsonl": md5_file(out / f"{s}.jsonl") for s in files},
        "gold_shares": rep["d2"], "d1_baselines": rep["d1"], "d4_tokens": rep["d4"],
        "resample_rate": {f"{t}/k{k}": {"draws": c["draws"], "items": c["items"],
                                        "draws_per_item": round(c["draws"] / max(1, c["items"]), 2)}
                          for (t, k), c in sorted(stats.items())},
        "d3_disjointness": d3, "bars": BARS, "d_fail": rep["fail"], "d_pass": not rep["fail"],
        "tokenizer": getattr(tok, "name_or_path", None), "seconds": round(time.time() - t0, 1)}
    if tok is not None:
        report["train_tokens"] = {s: sum(n_tokens(tok, r["prompt"]) for r in files[s]) for s in TRAIN_FILES}
        if gsm_train and gsm_train.exists():
            g = [json.loads(l) for l in open(gsm_train) if l.strip()]
            report["train_tokens"]["gsmvt_train_reference"] = sum(n_tokens(tok, r["prompt"]) for r in g)
            report["train_tokens"]["gsmvt_train_rows"] = len(g)
    (out / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="", help="Qwen3.5-4B tokenizer dir for D4 and the token totals (optional)")
    ap.add_argument("--gsm-train", default="", help="gsmvt train.jsonl for the reference token total (optional)")
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    try:
        rep = build(Path(a.out), tok, gsm_train=Path(a.gsm_train) if a.gsm_train else None)
    except BuildError as e:
        raise SystemExit(f"build_khop: {e}")
    print(json.dumps({"counts": {s: c["total"] for s, c in rep["counts"].items()}, "d_pass": rep["d_pass"],
                      "d_fail": rep["d_fail"], "seconds": rep["seconds"]}, indent=1))
    if not rep["d_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
