#!/usr/bin/env python3
"""FOLIO as a Tree Forcing source with prover-verified nodes (CHTC batch 27, cloud/chtc/README_27.md).

Input: data/raw/folio/folio_v2_{train,validation}.jsonl (FOLIO v2, CC BY-SA 4.0): natural-language premises and
conclusion, their first-order-logic annotations (premises-FOL, conclusion-FOL) and the label True / False / Uncertain.

1. Parse every FOL string (connectives ¬ ∧ ∨ ⊕ → ↔, quantifiers ∀ ∃ (scope: Parser), = and ≠, predicates over
   constants and bound variables) into z3 over one uninterpreted sort. A string that does not parse is retried once
   after counted mechanical repairs (repair(): a trailing comma, unbalanced parentheses); one that still does not
   parse completely drops its item (reported).
2. Prover label: premises |= conclusion -> True; premises |= not conclusion -> False; else Uncertain (z3 unknown or a
   timeout drops the item; unsatisfiable premises drop it). Only items whose prover label equals the gold label are
   kept (FOLIO's FOL annotations contain errors; the agreement rate and every drop reason go to summary.json).
3. Nodes (the tree): ground atoms P(c1, ..) obtained by instantiating each predicate occurrence of the item's
   formulas with the item's constants (bound variables -> every constant; arity 1 or 2). A node is kept when the
   premises entail the atom (truth True) or its negation (truth False); undetermined atoms are not asked. An atom
   logically equivalent to the conclusion or to its negation is dropped (it would state the answer). Items with an
   unbound variable letter u-z (an annotation slip; it would be read as a constant "x") are dropped. Node fields
   as the K&K subqs the trainer reads (question, opposite, truth, p_cot, type, id, depends_on [], node_depth 0,
   source_tree); type "parse" when the atom or its negation is a conjunct of a premise, else "derive". Question
   'Is it true that <c> has the property "<p>"?' / 'Is it true that <a> stands in the relation "<p>" to <b>?' (a
   frame well-formed for any predicate name), or "Is it false that ..." (truth follows the wording). Which atoms are
   asked in which wording: balance_nodes, so that the wording does not predict the answer (over each split
   P(Yes | true wording) = P(Yes | false wording) = P(Yes) = 1/2); a question-only logistic regression
   (shortcut_baseline) is reported in summary.json.
4. Items, K&K style: a preamble paragraph, the premises as "- <statement>" lines, then one paragraph "Question: Based
   on the statements, is the following conclusion true, false, or uncertain? Conclusion: <conclusion>", fixed options
   A) True B) False C) Uncertain, one-hot teacher, source "folio_tf" (treeplan.FINAL_QUESTION["folio_tf"] on branch
   folio cuts the statement block: preamble + statements, without the conclusion).
   Splits: FOLIO v2 train grouped by story_id, 15% of the stories to val (seed 0, no story in both); test = FOLIO
   v2 validation (no story shared with train), the items whose prover label equals the gold label; also the whole
   validation set in the same format, FOLIO gold, no nodes (data/eval/folio_tf_test_all.jsonl, report only).
Writes data/student_folio_tf/{train,val}.jsonl, data/eval/folio_tf_test{,_all}.jsonl, data/student_folio_tf/summary.json.
A rebuild gives the same bytes.

  ~/.venvs/sev/bin/python scripts/build_folio_tf.py                     # needs z3-solver
  ~/.venvs/sev/bin/python scripts/build_folio_tf.py --token-stats <tokenizer dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

import z3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/folio"
OUT = ROOT / "data/student_folio_tf"
TEST = ROOT / "data/eval/folio_tf_test.jsonl"
TEST_ALL = ROOT / "data/eval/folio_tf_test_all.jsonl"
TIMEOUT_MS = 60000        # a safety net only: every check is bounded by RLIMIT (deterministic, machine-independent)
RLIMIT = 20_000_000       # z3 resource units per check; a check that hits it counts as unknown (summary.json)
LABELS = ["A", "B", "C"]
FOLIO_OPTIONS = {"True": "True", "False": "False", "Uncertain": "Uncertain"}
PREAMBLE = ("Each statement below is true. Judge the conclusion using only these statements and valid logical "
            "reasoning: it is true if the statements imply it, false if they imply its negation, and uncertain "
            "otherwise.")


# ------------------------------------------------------------------------------------------------- parsing
TOKEN = re.compile(r"\s*(?:(?P<q>[∀∃])\s*(?P<var>[a-z][a-z0-9]*)?|(?P<op>¬|∧|∨|⊕|→|↔|⟷|—>|->|=|≠|\(|\)|,)"
                   r"|(?P<id>[^\W][\w.'\-]*))")


class ParseError(ValueError):
    pass


def tokenize(s: str) -> list[tuple[str, str]]:
    s = s.strip()
    out, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = TOKEN.match(s, i)
        if not m or m.end() == i:
            raise ParseError(f"bad character {s[i]!r} at {i}")
        if m.group("q"):
            out.append(("Q", m.group("q") + (m.group("var") or "x")))   # "∃(" with no variable: x (5 strings)
        elif m.group("op"):
            op = {"⟷": "↔", "—>": "→", "->": "→"}.get(m.group("op"), m.group("op"))
            out.append(("OP", op))
        else:
            ident = m.group("id")
            if ident.endswith("-") or ident.endswith("."):     # "—" alone or a trailing period: not an identifier
                raise ParseError(f"bad identifier {ident!r}")
            out.append(("ID", ident))
        i = m.end()
    return out


class Parser:
    """Recursive descent: iff < implies (right) < or / xor (left) < and < unary (¬, quantifier, atom, parens).
    Quantifier scope: narrow=True, a quantifier followed by "(" (or by another quantifier) binds that parenthesised
    group only, as FOLIO writes "∀x (∃y (...) → A(x))"; otherwise, and always with narrow=False, the scope extends as
    far right as possible (an iff-level formula)."""

    def __init__(self, toks, narrow=True):
        self.t, self.i, self.narrow = toks, 0, narrow

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def eat(self, kind, val=None):
        tok = self.peek()
        if tok[0] != kind or (val is not None and tok[1] != val):
            raise ParseError(f"expected {val or kind} at token {self.i}, got {tok}")
        self.i += 1
        return tok[1]

    def parse(self):
        f = self.iff(frozenset())
        if self.i != len(self.t):
            raise ParseError(f"trailing tokens from {self.i}: {self.t[self.i:self.i + 4]}")
        return f

    def iff(self, bound):
        a = self.implies(bound)
        while self.peek() == ("OP", "↔"):
            self.i += 1
            a = ("iff", a, self.implies(bound))
        return a

    def implies(self, bound):
        a = self.disj(bound)
        if self.peek() == ("OP", "→"):
            self.i += 1
            return ("imp", a, self.implies(bound))
        return a

    def disj(self, bound):
        a = self.conj(bound)
        while self.peek() in (("OP", "∨"), ("OP", "⊕")):
            op = "or" if self.peek()[1] == "∨" else "xor"
            self.i += 1
            a = (op, a, self.conj(bound))
        return a

    def conj(self, bound):
        a = self.unary(bound)
        while self.peek() == ("OP", "∧"):
            self.i += 1
            a = ("and", a, self.unary(bound))
        return a

    def unary(self, bound):
        kind, val = self.peek()
        if (kind, val) == ("OP", "¬"):
            self.i += 1
            return ("not", self.unary(bound))
        if kind == "Q":
            self.i += 1
            q, var = val[0], val[1:]
            nxt = self.peek()
            body = (self.unary(bound | {var}) if self.narrow and (nxt == ("OP", "(") or nxt[0] == "Q")
                    else self.iff(bound | {var}))
            return ("forall" if q == "∀" else "exists", var, body)
        if (kind, val) == ("OP", "("):
            self.i += 1
            f = self.iff(bound)
            self.eat("OP", ")")
            return f
        if kind == "ID":
            if self.peek(1) == ("OP", "("):
                name = self.eat("ID")
                self.eat("OP", "(")
                args = [self.term(bound)]
                while self.peek() == ("OP", ","):
                    self.i += 1
                    args.append(self.term(bound))
                self.eat("OP", ")")
                return ("pred", name, tuple(args))
            if self.peek(1) in (("OP", "="), ("OP", "≠")):
                a = self.term(bound)
                op = self.eat("OP")
                b = self.term(bound)
                return ("eq", a, b) if op == "=" else ("not", ("eq", a, b))
            name = self.eat("ID")
            return ("pred", name, ())
        raise ParseError(f"unexpected token {self.peek()} at {self.i}")

    def term(self, bound):
        name = self.eat("ID")
        if self.peek() == ("OP", "("):
            raise ParseError(f"function term {name}(...) not supported")
        return ("var", name) if name in bound else ("const", name)


def repair(s: str) -> tuple[str, list[str]]:
    """Mechanical fixes of FOLIO annotation typos, each counted: a trailing comma before ")"; unbalanced parentheses
    (missing ")" appended at the end, a surplus of ")" removed from the end or, if inside, matched by "(" prepended).
    A repaired string still has to parse completely, and its item still has to pass the prover-label check."""
    fixes = []
    t = re.sub(r",\s*\)", ")", s)
    if t != s:
        fixes.append("trailing_comma")
    o, c = t.count("("), t.count(")")
    if o > c:
        t += ")" * (o - c)
        fixes.append("missing_close")
    elif c > o:
        k = c - o
        u = t.rstrip()
        while k and u.endswith(")"):
            trial = u[:-1]
            if trial.count("(") >= trial.count(")") - (k - 1):
                u, k = trial.rstrip(), k - 1
            else:
                break
        t = "(" * k + u if k else u
        fixes.append("surplus_close")
    return t, fixes


def parse_scoped(s: str) -> tuple:
    """Narrow scope; if that leaves a variable letter unbound ("∀x (A(x)) → B(x)", the annotator meant the whole
    line), the maximal-scope reading. Returns (formula, "scope_maximal" or None)."""
    f = Parser(tokenize(s), narrow=True).parse()
    if free_vars(f):
        g = Parser(tokenize(s), narrow=False).parse()
        if not free_vars(g):
            return g, "scope_maximal"
    return f, None


def parse(s: str, fixes: list | None = None):
    try:
        r, sc = parse_scoped(s)
        f = []
    except ParseError:
        t, f = repair(s)
        if not f:
            raise
        r, sc = parse_scoped(t)
    if fixes is not None:
        fixes.extend(f + ([sc] if sc else []))
    return r


# ------------------------------------------------------------------------------------------------- z3
class World:
    """One item's z3 vocabulary: sort Obj, constants, predicates keyed by (name, arity)."""

    def __init__(self, ctx=None):
        """A fresh z3 context per item (default), so no item's answers depend on what was solved before it."""
        self.ctx = ctx or z3.Context()
        self.Obj = z3.DeclareSort("Obj", self.ctx)
        self.consts, self.preds = {}, {}

    def const(self, name):
        if name not in self.consts:
            self.consts[name] = z3.Const("c_" + name, self.Obj)
        return self.consts[name]

    def pred(self, name, k):
        key = (name, k)
        if key not in self.preds:
            self.preds[key] = z3.Function(f"p_{name}_{k}", *([self.Obj] * k), z3.BoolSort(self.ctx))
        return self.preds[key]

    def term(self, t, env):
        return env[t[1]] if t[0] == "var" else self.const(t[1])

    def to_z3(self, f, env=None):
        env = env or {}
        op = f[0]
        if op == "pred":
            p = self.pred(f[1], len(f[2]))
            return p(*[self.term(a, env) for a in f[2]])
        if op == "eq":
            return self.term(f[1], env) == self.term(f[2], env)
        if op == "not":
            return z3.Not(self.to_z3(f[1], env))
        if op in ("and", "or"):
            return (z3.And if op == "and" else z3.Or)(self.to_z3(f[1], env), self.to_z3(f[2], env))
        if op == "xor":
            return z3.Xor(self.to_z3(f[1], env), self.to_z3(f[2], env))
        if op == "imp":
            return z3.Implies(self.to_z3(f[1], env), self.to_z3(f[2], env))
        if op == "iff":
            return self.to_z3(f[1], env) == self.to_z3(f[2], env)
        if op in ("forall", "exists"):
            v = z3.Const(f"v_{f[1]}_{len(env)}", self.Obj)
            body = self.to_z3(f[2], {**env, f[1]: v})
            return (z3.ForAll if op == "forall" else z3.Exists)([v], body)
        raise ValueError(op)


def walk(f):
    yield f
    for x in f[1:]:
        if isinstance(x, tuple) and x and isinstance(x[0], str) and x[0] in (
                "pred", "eq", "not", "and", "or", "xor", "imp", "iff", "forall", "exists"):
            yield from walk(x)


def constants_of(f, out):
    for g in walk(f):
        if g[0] == "pred":
            out.update(a[1] for a in g[2] if a[0] == "const")
        elif g[0] == "eq":
            out.update(a[1] for a in (g[1], g[2]) if a[0] == "const")
    return out


def solver(ctx):
    s = z3.Solver(ctx=ctx)
    s.set("timeout", TIMEOUT_MS)
    s.set("rlimit", RLIMIT)
    return s


UNKNOWN = {"n": 0}       # node / leak checks that ended unknown (resource limit), per build


def entails(solver, goal) -> str:
    """'yes' if the asserted formulas entail goal, 'no' if not (counter-model), 'unknown' on timeout / unknown."""
    solver.push()
    solver.add(z3.Not(goal))
    r = solver.check()
    solver.pop()
    if r == z3.unknown:
        UNKNOWN["n"] += 1
    return "yes" if r == z3.unsat else "no" if r == z3.sat else "unknown"


# ------------------------------------------------------------------------------------------------- rendering
def words(name: str) -> list[str]:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", " ", name)
    return [w for w in re.split(r"[\s_\-]+", s) if w]


def const_text(name: str, text: str) -> str:
    """A constant as the story writes it: its words (CamelCase split) found case-insensitively in the premises or
    conclusion text, with that casing; else the words, lower-cased when there are several."""
    w = words(name)
    m = re.search(r"\b" + r"[\s\-]*".join(map(re.escape, w)) + r"\b", text, flags=re.I) if w else None
    return m.group(0) if m else " ".join(w).lower() if len(w) > 1 else name


def claim(pname: str, args: list[str]) -> str:
    """A frame that is well-formed for any predicate name (noun, adjective or verb phrase)."""
    ph = " ".join(w.lower() for w in words(pname))
    if len(args) == 1:
        return f'{args[0]} has the property "{ph}"'
    return f'{args[0]} stands in the relation "{ph}" to {args[1]}'


# ------------------------------------------------------------------------------------------------- items
FINAL_Q = "Question: Based on the statements, is the following conclusion true, false, or uncertain? Conclusion: "
OPTIONS_TXT = "Options:\nA) True\nB) False\nC) Uncertain"
LABEL_OF = {"True": "A", "False": "B", "Uncertain": "C"}


def clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def prompt_of(row: dict) -> str:
    """K&K-style: preamble paragraph, one statement per line, the final question paragraph naming the conclusion
    (treeplan.FINAL_QUESTION["folio_tf"] on branch folio), fixed options."""
    lines = [clean(x) for x in row["premises"].split("\n") if clean(x)]
    concl = clean(row["conclusion"])
    assert lines and concl and not any(x.startswith("Question:") for x in lines), row["example_id"]
    return (f"{PREAMBLE}\n\n" + "\n".join(f"- {x}" for x in lines) + f"\n\n{FINAL_Q}{concl}\n{OPTIONS_TXT}")


def conjuncts(f):
    if f[0] == "and":
        return conjuncts(f[1]) + conjuncts(f[2])
    return [f]


def free_vars(f, bound=frozenset()):
    """Variable letters (u v w x y z) used as terms outside any quantifier binding them (FOLIO annotation slips such as
    'MeansToDestination(x, drive) ∨ Student(james)'): the parser reads them as constants, so the item is dropped.
    Other single letters are real constants in FOLIO (the grades a, b, c of GradeIn(a, englishClass))."""
    out = set()
    if f[0] in ("forall", "exists"):
        return free_vars(f[2], bound | {f[1]})
    if f[0] == "pred":
        return {a[1] for a in f[2] if a[0] == "const" and re.fullmatch(r"[u-z]", a[1])}
    if f[0] == "eq":
        return {a[1] for a in (f[1], f[2]) if a[0] == "const" and re.fullmatch(r"[u-z]", a[1])}
    for x in f[1:]:
        if isinstance(x, tuple):
            out |= free_vars(x, bound)
    return out


def valid(goal) -> bool:
    s = solver(goal.ctx)
    s.add(z3.Not(goal))
    r = s.check()
    if r == z3.unknown:
        UNKNOWN["n"] += 1
    return r == z3.unsat


def analyse(row: dict) -> dict:
    """Prover label and verified nodes of one FOLIO row; {'drop': reason} when it cannot be used."""
    fixes = []
    try:
        prem = [parse(x, fixes) for x in row["premises-FOL"].split("\n") if x.strip()]
        concl = parse(row["conclusion-FOL"], fixes)
    except ParseError as e:
        return {"drop": "parse", "why": str(e)}
    if any(free_vars(f) for f in prem + [concl]):
        return {"drop": "free_variable", "fixes": fixes}
    W = World()
    zp = [W.to_z3(f) for f in prem]
    zc = W.to_z3(concl)
    s = solver(W.ctx)
    s.add(*zp)
    sat = s.check()
    if sat == z3.unsat:
        return {"drop": "premises_unsat", "fixes": fixes}
    if sat != z3.sat:
        return {"drop": "premises_unknown", "fixes": fixes}
    t, f = entails(s, zc), entails(s, z3.Not(zc))
    if t == "yes" and f == "yes":
        return {"drop": "premises_unsat", "fixes": fixes}
    if "unknown" in (t, f) and not (t == "yes" or f == "yes"):
        return {"drop": "prover_unknown", "fixes": fixes}
    label = "True" if t == "yes" else "False" if f == "yes" else "Uncertain"
    consts = sorted(constants_of(concl, set()).union(*[constants_of(x, set()) for x in prem]))
    cand = set()
    for g in [x for f_ in prem + [concl] for x in walk(f_) if x[0] == "pred"]:
        if not 1 <= len(g[2]) <= 2:
            continue
        slots = [[a[1]] if a[0] == "const" else consts for a in g[2]]
        for a0 in slots[0]:
            for a1 in (slots[1] if len(slots) > 1 else [None]):
                cand.add((g[1], (a0,) if a1 is None else (a0, a1)))
    prem_lits = set()                                   # ground literals stated by a premise (one of its conjuncts)
    for f_ in prem:
        for c in conjuncts(f_):
            g = c[1] if c[0] == "not" else c
            if g[0] == "pred" and g[2] and all(a[0] == "const" for a in g[2]):
                prem_lits.add((g[1], tuple(a[1] for a in g[2])))
    nodes, leak = [], 0
    for name, args in sorted(cand):
        atom = W.pred(name, len(args))(*[W.const(a) for a in args])
        yes, no = entails(s, atom), entails(s, z3.Not(atom))
        if yes == "yes" and no == "yes" or "yes" not in (yes, no):
            continue                                    # undetermined (or unknown): not asked
        if valid(atom == zc) or valid(atom == z3.Not(zc)):
            leak += 1                                   # logically the conclusion or its negation: states the answer
            continue
        nodes.append((name, args, yes == "yes", "parse" if (name, args) in prem_lits else "derive"))
    nodes.sort(key=lambda n: (n[3] != "parse", n[0], n[1]))
    return {"label": label, "nodes": nodes, "fixes": fixes, "leak_dropped": leak}


def _key(example_id, name, args) -> float:
    return random.Random(f"folio_tf|{example_id}|{name}|{'|'.join(args)}").random()


def recurring_names(rows: list) -> set:
    """Lower-cased constant names that occur in the FOL of two or more FOLIO stories (train and validation together,
    so a name shared by a train story and a test story counts)."""
    st = {}
    for r in rows:
        for x in r["premises-FOL"].split("\n") + [r["conclusion-FOL"]]:
            if not x.strip():
                continue
            try:
                f = parse(x)
            except ParseError:
                continue
            for c in constants_of(f, set()):
                st.setdefault(c.lower(), set()).add(r["story_id"])
    return {c for c, v in st.items() if len(v) > 1}


def balance_nodes(kept: list, recurring: set) -> dict:
    """_balance on each stratum separately, so that no surface feature that recurs across stories predicts the
    answer together with the wording: the frame (arity 1 "has the property", arity 2 "stands in the relation"; asked
    relations held 60-72% of the time and properties 39-44% before this) and the first argument when its name occurs
    in two or more FOLIO stories (recurring_names; FOLIO reuses names: atoms about Mary held 1 / 33 in train, 0 / 24
    in val); names of one story only cannot carry over to another story and are pooled.
    P(Yes | wording, stratum) = 1/2."""
    strat = lambda n: (len(n[1]), n[1][0].lower() if n[1][0].lower() in recurring else "*")
    out = {r["example_id"]: [] for r, _ in kept}
    for key in sorted({strat(n) for _, res in kept for n in res["nodes"]}):
        sub = [(r, dict(res, nodes=[n for n in res["nodes"] if strat(n) == key])) for r, res in kept]
        for e, ns in _balance(sub).items():
            out[e] += ns
    for e in out:
        out[e].sort(key=lambda n: (n[3] != "parse", n[0], n[1], n[4]))
    return out


def _balance(kept: list) -> dict:
    """Which atoms are asked, and in which wording, so that the wording does not predict the answer: over the split,
    P(Yes | "Is it true that") = P(Yes | "Is it false that") = 1/2 and P(Yes) = 1/2.
    The answer is Yes iff (wording true) == (atom holds). With H holding and N not-holding atoms (H > N: about 80% of
    the atoms the premises decide hold), both conditionals are 1/2 only if the asked holding and not-holding nodes are
    equal in number within each wording: a = b true-worded, c = d false-worded. So: every not-holding atom is asked in
    both wordings (b = d = N) and 2N holding atoms are asked, N true-worded and N false-worded (if H < 2N: all holding
    atoms, and as many not-holding wordings). The holding atoms are spread over the items in proportion to their
    count (largest remainder, at least one per item that has one), chosen and worded by a coin keyed by item and atom.
    kept: [(row, res)]; returns {example_id: [(name, args, holds, type, flipped)]}."""
    H = [(r["example_id"], n) for r, res in kept for n in res["nodes"] if n[2]]
    N = [(r["example_id"], n) for r, res in kept for n in res["nodes"] if not n[2]]
    target = min(len(H), 2 * len(N))
    a, c = (target + 1) // 2, target // 2
    nh = {}
    for eid, _ in H:
        nh[eid] = nh.get(eid, 0) + 1
    order = [r["example_id"] for r, _ in kept if nh.get(r["example_id"])]
    quota = {e: 0 for e in order}
    if target:
        base = target >= len(order)
        for e in order:
            quota[e] = 1 if base else 0
        left = target - sum(quota.values())
        share = {e: (target - (len(order) if base else 0)) * (nh[e] - quota[e]) / max(1, len(H) - sum(quota.values()))
                 for e in order}
        for e in order:
            add = min(nh[e] - quota[e], int(share[e]))
            quota[e] += add
            left -= add
        rest = sorted(order, key=lambda e: (-(share[e] - int(share[e])), _key(e, "quota", [])))
        while left > 0:
            for e in rest:
                if left and quota[e] < nh[e]:
                    quota[e] += 1
                    left -= 1
    out = {r["example_id"]: [] for r, _ in kept}
    k = 0
    for r, res in kept:
        e = r["example_id"]
        hold = sorted((n for n in res["nodes"] if n[2]), key=lambda n: _key(e, n[0], n[1]))[:quota.get(e, 0)]
        for n in hold:                                   # alternate over the split: a true-worded, c false-worded
            out[e].append((n[0], n[1], True, n[3], k % 2 == 1))
            k += 1
    assert k == target
    nots = sorted(N, key=lambda x: _key(x[0], x[1][0], x[1][1]))
    for i, (e, n) in enumerate(nots):
        if i < a:
            out[e].append((n[0], n[1], False, n[3], False))
        if len(nots) - 1 - i < c:
            out[e].append((n[0], n[1], False, n[3], True))
    return out


def node_fields(row: dict, nodes) -> list[dict]:
    """nodes: (name, args, holds, type, flipped) from balance_nodes; plain wording "Is it true that ...", flipped
    "Is it false that ..."; truth = the answer to the question asked."""
    text = row["premises"] + "\n" + row["conclusion"]
    out = []
    for k, (name, args, holds, ntype, flip) in enumerate(nodes, 1):
        c = claim(name, [const_text(a, text) for a in args])
        q, opp = f"Is it true that {c}?", f"Is it false that {c}?"
        if flip:
            q, opp = opp, q
        truth = holds != flip
        out.append({"question": q, "opposite": opp, "truth": truth, "p_cot": 1.0 if truth else 0.0, "p_fresh": None,
                    "type": ntype, "depends_on": [], "id": f"n{k}", "node_depth": 0, "initial": None,
                    "status": "direct", "flipped": flip, "atom_holds": holds, "source_tree": "z3",
                    "atom": [name, list(args)]})
    return out


def build_item(row: dict, res: dict | None, split: str, nodes=None) -> dict:
    """res None: an unverified item (the unfiltered test file), FOLIO's gold label, no nodes. nodes: balance_nodes'."""
    gold = LABEL_OF[row["label"]]
    subqs = node_fields(row, nodes) if res else []
    return {"item_id": f"foliotf-{split}-{row['example_id']}", "source": "folio_tf", "prompt": prompt_of(row),
            "type": "choice", "group": f"folio_tf/{row['label']}", "labels": LABELS, "label_order": list(LABEL_OF),
            "label_names": {v: k for k, v in LABEL_OF.items()}, "gold_label": gold, "gold_probs": None,
            "teacher": {L: float(L == gold) for L in LABELS}, "subqs": subqs, "random_subqs": [], "rationale": "",
            "has_tree": bool(subqs), "tree_source": "folio_z3", "tree_fix": True,
            "depth": len([x for x in row["premises"].splitlines() if x.strip()]), "story_id": row["story_id"],
            "example_id": row["example_id"], "prover_label": res["label"] if res else None,
            "leak_dropped": res["leak_dropped"] if res else 0}


NEG = re.compile(r"\b(?:not|no|non|never|without|un\w+|dis\w+|in(?:capable|active|valid))\b")


def shortcut_baseline() -> dict:
    """Question-only baselines (no premises): logistic regression on the node question text, trained on the train
    nodes, scored on val / test nodes. Features: lower-cased word unigrams and bigrams, and in the second model also
    every unigram crossed with the wording (true / false), which can learn "this predicate or constant tends to hold".
    Full-batch gradient descent on numpy, L2 in (1e-1, 1e-2, 1e-3), the best on val reported (optimistic for the
    shortcut). Also the yes rate of nodes whose predicate or constant text has a negation-like word."""
    import numpy as np
    items = {k: [json.loads(line) for line in open(p)]
             for k, p in (("train", OUT / "train.jsonl"), ("val", OUT / "val.jsonl"), ("test", TEST))}
    sets = {k: [q for x in v for q in x["subqs"]] for k, v in items.items()}

    def feats(q, cross):
        w = re.findall(r"[a-z0-9]+", q["question"].lower())
        f = set(w) | {a + "_" + b for a, b in zip(w, w[1:])}
        if cross:
            f |= {("F:" if q["flipped"] else "T:") + t for t in set(w)}
        return f

    def fit(train, cross, l2):
        vocab = {f: i for i, f in enumerate(sorted({f for q in train for f in feats(q, cross)}))}

        def mat(qs):
            X = np.zeros((len(qs), len(vocab) + 1))
            X[:, -1] = 1.0
            for r, q in enumerate(qs):
                for f in feats(q, cross):
                    if f in vocab:
                        X[r, vocab[f]] = 1.0
            return X, np.array([float(q["truth"]) for q in qs])
        X, y = mat(train)
        w = np.zeros(X.shape[1])
        for _ in range(800):
            pr = 1 / (1 + np.exp(-X @ w))
            w -= 0.5 * (X.T @ (pr - y) / len(y) + l2 * np.r_[w[:-1], 0.0])
        return lambda qs: int((((mat(qs)[0] @ w) > 0) == mat(qs)[1]).sum()), len(vocab)

    out = {}
    for cross in (False, True):
        res = {}
        for l2 in (1e-1, 1e-2, 1e-3):
            score, nf = fit(sets["train"], cross, l2)
            res[l2] = {k: round(score(sets[k]) / len(sets[k]), 4) for k in ("train", "val", "test")}
        best = max(res, key=lambda l2: res[l2]["val"])
        name = "words+wording" if cross else "words"
        out[name] = dict(res[best], l2=best, n_features=nf)
        # steadier: 5 folds of all nodes grouped by story (train + val + test), l2 1e-2 fixed
        allq = [(x["story_id"], q) for v in items.values() for x in v for q in x["subqs"]]
        st = sorted({sid for sid, _ in allq}, key=str)
        random.Random(0).shuffle(st)
        fold = {sid: i % 5 for i, sid in enumerate(st)}
        cor = 0
        for k in range(5):
            score, _ = fit([q for sid, q in allq if fold[sid] != k], cross, 1e-2)
            cor += score([q for sid, q in allq if fold[sid] == k])
        acc = cor / len(allq)
        out[name]["story_grouped_5fold"] = {"acc": round(acc, 4), "n": len(allq),
                                            "se": round((acc * (1 - acc) / len(allq)) ** 0.5, 4)}
    out["majority_yes_share"] = {k: round(sum(q["truth"] for q in v) / max(1, len(v)), 4) for k, v in sets.items()}
    out["n_nodes"] = {k: len(v) for k, v in sets.items()}
    neg = {}
    for k, v in sets.items():
        g = [q for q in v if NEG.search(q["question"].split(" that ", 1)[1].lower())]
        neg[k] = [sum(q["truth"] for q in g), len(g)]
    out["negation_word_nodes_yes"] = neg
    return out


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build(seed: int = 0) -> dict:
    train_rows = [json.loads(l) for l in open(RAW / "folio_v2_train.jsonl") if l.strip()]
    val_rows = [json.loads(l) for l in open(RAW / "folio_v2_validation.jsonl") if l.strip()]
    summ = {"z3": z3.get_version_string(), "rlimit": RLIMIT, "timeout_ms": TIMEOUT_MS, "splits": {}, "drops": {},
            "agreement": {}}
    stories = sorted({r["story_id"] for r in train_rows}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    rng = random.Random(seed)
    rng.shuffle(stories)
    val_st = set(stories[: round(0.15 * len(stories))])
    summ["story_overlap_train_test"] = len({r["story_id"] for r in train_rows} & {r["story_id"] for r in val_rows})
    groups = {"train": [r for r in train_rows if r["story_id"] not in val_st],
              "val": [r for r in train_rows if r["story_id"] in val_st], "test": val_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    recurring = recurring_names(train_rows + val_rows)
    summ["recurring_names"] = len(recurring)
    for split, rows in groups.items():
        kept, drops, agree, n_det, fixes_n = [], {}, 0, 0, {}
        UNKNOWN["n"] = 0
        for r in rows:
            res = analyse(r)
            for f in res.get("fixes", []):
                fixes_n[f] = fixes_n.get(f, 0) + 1
            if "drop" in res:
                drops[res["drop"]] = drops.get(res["drop"], 0) + 1
                continue
            n_det += 1
            if res["label"] != r["label"]:
                key = f"label_{r['label']}_prover_{res['label']}"
                drops[key] = drops.get(key, 0) + 1
                continue
            agree += 1
            kept.append((r, res))
        chosen = balance_nodes(kept, recurring)
        n_atoms = {"holding": sum(1 for _, res in kept for n in res["nodes"] if n[2]),
                   "not_holding": sum(1 for _, res in kept for n in res["nodes"] if not n[2])}
        kept = [build_item(r, res, split, chosen[r["example_id"]]) for r, res in kept]
        path = TEST if split == "test" else OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
        nn = [len(x.get("subqs") or []) for x in kept]
        summ["splits"][split] = {"rows": len(rows), "kept": len(kept), "md5": md5(path), "path": str(path.relative_to(ROOT)),
                                 "labels": {L: sum(1 for x in kept if x["prover_label"] == L) for L in FOLIO_OPTIONS},
                                 "nodes_mean": round(statistics.mean(nn), 2) if nn else 0, "nodes_median": statistics.median(nn) if nn else 0,
                                 "nodes_max": max(nn) if nn else 0, "items_without_nodes": sum(1 for n in nn if n == 0),
                                 "node_truth_true": sum(1 for x in kept for q in x.get("subqs") or [] if q["truth"]),
                                 "node_truth_false": sum(1 for x in kept for q in x.get("subqs") or [] if not q["truth"]),
                                 "node_types": {t: sum(1 for x in kept for q in x.get("subqs") or [] if q["type"] == t) for t in ("parse", "derive")}}
        summ["drops"][split] = drops
        summ.setdefault("repairs", {})[split] = fixes_n
        summ["agreement"][split] = {"prover_decided": n_det, "agree": agree, "rate": round(agree / max(1, n_det), 4)}
        summ["splits"][split]["leak_nodes_dropped"] = sum(x["leak_dropped"] for x in kept)
        summ["splits"][split]["solver_unknown_checks"] = UNKNOWN["n"]
        summ["splits"][split]["node_yes_share"] = round(summ["splits"][split]["node_truth_true"] / max(1, sum(nn)), 4)
        qs = [q for x in kept for q in x["subqs"]]
        for w, flip in (("true", False), ("false", True)):
            g = [q["truth"] for q in qs if q["flipped"] == flip]
            summ["splits"][split][f"yes_given_{w}_wording"] = [sum(g), len(g), round(sum(g) / max(1, len(g)), 4)]
        summ["splits"][split]["atoms_decided"] = n_atoms
        summ["splits"][split]["atoms_asked"] = {"holding": len({(x["item_id"], tuple(q["atom"][:1]) + tuple(q["atom"][1])) for x in kept for q in x["subqs"] if q["atom_holds"]}),
                                                "not_holding": len({(x["item_id"], tuple(q["atom"][:1]) + tuple(q["atom"][1])) for x in kept for q in x["subqs"] if not q["atom_holds"]})}
    allp = TEST_ALL
    allp.write_text("".join(json.dumps(build_item(r, None, "test"), ensure_ascii=False) + "\n" for r in val_rows))
    summ["splits"]["test_all"] = {"rows": len(val_rows), "kept": len(val_rows), "md5": md5(allp),
                                  "path": str(allp.relative_to(ROOT)), "labels_gold": {L: sum(1 for r in val_rows if r["label"] == L) for L in FOLIO_OPTIONS},
                                  "note": "FOLIO v2 validation, all items, FOLIO gold labels, unverified, no nodes"}
    summ["question_only_baseline"] = shortcut_baseline()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    return summ


def token_stats(tok_dir: str, src: str) -> dict:
    """Prompt tokens and the GFP-S node sequences (student.SUBQ_TEMPLATE on the statement block), with treeplan from
    src (the folio branch's src/, which defines the folio_tf cut)."""
    sys.path.insert(0, src)
    from transformers import AutoTokenizer
    from cotdistill.treeplan import statement_block
    tmpl = "{problem}\n\nIntermediate question: {question}\nOptions:\nA) Yes\nB) No\nAnswer: A"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    res = {}
    for split, path in (("train", OUT / "train.jsonl"), ("val", OUT / "val.jsonl"), ("test", TEST), ("test_all", TEST_ALL)):
        P, T, M = [], [], []
        for line in open(path):
            x = json.loads(line)
            P.append(len(tok.encode(x["prompt"], add_special_tokens=False)))
            sb = statement_block(x)
            qs = [len(tok.encode(tmpl.format(problem=sb, question=q["question"]), add_special_tokens=False))
                  for q in x.get("subqs") or []]
            T.append(P[-1] + sum(qs))
            M.append(max([P[-1]] + qs))
        res[split] = {"prompt_median": statistics.median(P), "prompt_max": max(P),
                      "tree_tokens_per_item_mean": round(statistics.mean(T)), "tree_tokens_per_item_max": max(T),
                      "longest_single_sequence": max(M), "budget": 4096}
    s = json.loads((OUT / "summary.json").read_text())
    s["tokens"] = dict(res, tokenizer=tok_dir, note="tree tokens = prompt + every node sequence (one item's GFP-S load)")
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--token-stats", default="", help="tokenizer dir")
    ap.add_argument("--src", default=str(ROOT / "src"), help="src/ with the folio_tf cut (the folio branch worktree)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(token_stats(a.token_stats, a.src) if a.token_stats else build(a.seed), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
