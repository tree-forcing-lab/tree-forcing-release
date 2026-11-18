"""Zebra (logic grid) puzzles for Tree Forcing: a generator with an exact solver, our own clue wording, and
full-arrangement multiple-choice items (the K&K format: the options are complete answers, the distractors are wrong
arrangements that break the fewest clues).

A puzzle has N houses in a row (1..N, left to right) and M categories (name, house color, pet, ...), each with N
values; the solution assigns every value of every category to one house, a permutation per category. An entity is a
(category, value) pair. Clues (all true of the solution, together they fix it uniquely):
  found(X, k)      X lives in house k                     notfound(X, k)   X does not live in house k
  same(X, Y)       X is Y (one person; X, Y of different categories)        notsame(X, Y)    X is not Y
  left(X, Y)       X lives somewhere to the left of Y     right(X, Y)      X lives somewhere to the right of Y
  dleft(X, Y)      X lives directly to the left of Y      next(X, Y)       X and Y live next to each other
  gap1(X, Y)       exactly one house lies between X and Y
The solver (count_solutions) propagates the clues and the one-value-per-house rule on bitmask domains and branches;
it is exact (checked against brute force in tests/test_zebra.py)."""
from __future__ import annotations

import itertools
import random
from collections import Counter

CATEGORIES = {
    # key: (header line, values pool, phrase of an entity)
    "name": ("Each person has a different first name", ["Mara", "Tobias", "Ines", "Rafael", "Greta", "Omar", "Lena",
                                                         "Victor"], "{v}"),
    "color": ("Each house is painted a different color", ["red", "blue", "green", "yellow", "white", "gray"],
              "the person in the {v} house"),
    "pet": ("Each person keeps a different pet", ["cat", "dog", "parrot", "rabbit", "turtle", "hamster"],
            "the person who keeps the {v}"),
    "drink": ("Each person prefers a different drink", ["tea", "coffee", "milk", "juice", "water", "cocoa"],
              "the person who drinks {v}"),
    "sport": ("Each person plays a different sport", ["tennis", "soccer", "golf", "rugby", "hockey", "cricket"],
              "the person who plays {v}"),
    "job": ("Each person has a different job", ["baker", "nurse", "pilot", "teacher", "lawyer", "farmer"], "the {v}"),
    "instrument": ("Each person plays a different instrument", ["piano", "violin", "guitar", "drums", "flute", "cello"],
                   "the person who plays the {v}"),
    "fruit": ("Each person has a different favorite fruit", ["apple", "banana", "cherry", "mango", "grape", "pear"],
              "the person whose favorite fruit is the {v}"),
}
LABEL = {"name": "name", "color": "house color", "pet": "pet", "drink": "drink", "sport": "sport", "job": "job",
         "instrument": "instrument", "fruit": "favorite fruit"}
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
UNARY = ("found", "notfound")
BINARY = ("same", "notsame", "left", "right", "dleft", "next", "gap1")
CLUE_WEIGHTS = {"found": 0.13, "notfound": 0.15, "same": 0.24, "notsame": 0.06, "left": 0.09, "right": 0.07,
                "dleft": 0.1, "next": 0.1, "gap1": 0.06}
FINAL_QUESTION = "Question: Which option gives the correct arrangement?"
# v3 (11-08, "as hard as possible"): no position clue that places someone, few "not in house k" clues, relations only
HARD_WEIGHTS = {"found": 0.0, "notfound": 0.04, "same": 0.26, "notsame": 0.08, "left": 0.12, "right": 0.1,
                "dleft": 0.14, "next": 0.14, "gap1": 0.12}
CELL_QUESTION = "Question: What is the {label} of the person who lives in the {ordinal} house?"


def holds(kind: str, px: int, py_or_k: int) -> bool:
    """Does clue `kind` hold for X in house px and Y in house py (unary kinds: house k)?"""
    a, b = px, py_or_k
    return {"found": a == b, "notfound": a != b, "same": a == b, "notsame": a != b, "left": a < b, "right": a > b,
            "dleft": a + 1 == b, "next": abs(a - b) == 1, "gap1": abs(a - b) == 2}[kind]


def make_solution(n: int, cats: list[str], rng: random.Random) -> dict:
    """{category: [value of house 1, ..., value of house n]}: values sampled from the pool, then shuffled."""
    return {c: rng.sample(CATEGORIES[c][1], n) for c in cats}


def positions(sol: dict) -> dict:
    """{(category, value): house (1-based)}."""
    return {(c, v): i + 1 for c, vals in sol.items() for i, v in enumerate(vals)}


def clue_true(clue: tuple, pos: dict) -> bool:
    kind = clue[0]
    if kind in UNARY:
        return holds(kind, pos[clue[1]], clue[2])
    return holds(kind, pos[clue[1]], pos[clue[2]])


# ------------------------------------------------------------------------------------------------------------ solver
def _bits(n: int) -> int:
    return (1 << n) - 1


def _propagate(dom: dict, clues: list, groups: dict, n: int) -> bool:
    """Shrink the domains (bit h-1 = house h possible) to a fixpoint; False on a wipe-out. In place."""
    changed = True
    while changed:
        changed = False
        for cl in clues:
            kind, x, y = cl[0], cl[1], cl[2]
            if kind in UNARY:
                continue
            for a, b, rel in ((x, y, kind), (y, x, None)):
                da, db = dom[a], dom[b]
                keep = 0
                for ha in range(1, n + 1):
                    if not da >> (ha - 1) & 1:
                        continue
                    for hb in range(1, n + 1):
                        if db >> (hb - 1) & 1 and (holds(kind, ha, hb) if rel else holds(kind, hb, ha)):
                            keep |= 1 << (ha - 1)
                            break
                if keep != da:
                    if not keep:
                        return False
                    dom[a] = keep
                    changed = True
        for ents in groups.values():                     # one value per house within a category
            for e in ents:
                d = dom[e]
                if d & (d - 1) == 0:                      # placed: remove its house from the others
                    for f in ents:
                        if f != e and dom[f] & d:
                            dom[f] &= ~d
                            if not dom[f]:
                                return False
                            changed = True
            for h in range(n):                            # a house only one value of the category can take
                cand = [e for e in ents if dom[e] >> h & 1]
                if not cand:
                    return False
                if len(cand) == 1 and dom[cand[0]] != 1 << h:
                    dom[cand[0]] = 1 << h
                    changed = True
    return True


def initial_domains(n: int, cats: list[str], vals: dict, clues: list) -> tuple[dict, dict] | None:
    groups = {c: [(c, v) for v in vals[c]] for c in cats}
    dom = {e: _bits(n) for ents in groups.values() for e in ents}
    for cl in clues:
        if cl[0] == "found":
            dom[cl[1]] &= 1 << (cl[2] - 1)
        elif cl[0] == "notfound":
            dom[cl[1]] &= ~(1 << (cl[2] - 1))
        if not dom[cl[1]]:
            return None
    return dom, groups


def count_solutions(n: int, cats: list[str], vals: dict, clues: list, limit: int = 2) -> int:
    """Number of arrangements satisfying every clue, counted up to `limit`."""
    init = initial_domains(n, cats, vals, clues)
    if init is None:
        return 0
    dom, groups = init
    found = [0]

    def rec(d: dict) -> None:
        if found[0] >= limit or not _propagate(d, clues, groups, n):
            return
        open_ = [e for e in d if d[e] & (d[e] - 1)]
        if not open_:
            found[0] += 1
            return
        e = min(open_, key=lambda x: bin(d[x]).count("1"))
        for h in range(n):
            if d[e] >> h & 1:
                d2 = dict(d)
                d2[e] = 1 << h
                rec(d2)
                if found[0] >= limit:
                    return

    rec(dict(dom))
    return found[0]


def solves_by_propagation(n: int, cats: list[str], vals: dict, clues: list) -> bool:
    """True when propagation alone (no branching) fixes every entity."""
    init = initial_domains(n, cats, vals, clues)
    if init is None:
        return False
    dom, groups = init
    return _propagate(dom, clues, groups, n) and all(d & (d - 1) == 0 for d in dom.values())


# --------------------------------------------------------------------------------------------------------- generator
def random_clue(sol: dict, pos: dict, rng: random.Random, weights: dict | None = None) -> tuple:
    """A random clue that is true of the solution (kinds drawn with `weights`, default CLUE_WEIGHTS)."""
    n = len(next(iter(sol.values())))
    ents = list(pos)
    kinds, w = zip(*(weights or CLUE_WEIGHTS).items())
    while True:
        kind = rng.choices(kinds, w)[0]
        x = rng.choice(ents)
        if kind == "found":
            return ("found", x, pos[x])
        if kind == "notfound":
            k = rng.choice([h for h in range(1, n + 1) if h != pos[x]])
            return ("notfound", x, k)
        y = rng.choice([e for e in ents if e[0] != x[0]])
        if kind == "same":
            y = next(e for e in ents if e[0] == y[0] and pos[e] == pos[x])
        if holds(kind, pos[x], pos[y]):
            return (kind, x, y)


def generate(n: int, m: int, rng: random.Random, max_clues: int = 60, max_tries: int = 20,
             weights: dict | None = None) -> dict | None:
    """One puzzle with a unique solution and a clue set from which no clue can be dropped: categories (name always
    first), the solution, the clues (in a random order). None if no unique set within max_clues (rare)."""
    for _ in range(max_tries):
        cats = ["name"] + rng.sample([c for c in CATEGORIES if c != "name"], m - 1)
        sol = make_solution(n, cats, rng)
        pos = positions(sol)
        vals = {c: sorted(sol[c]) for c in cats}
        clues, seen = [], set()
        while len(clues) < max_clues:
            cl = random_clue(sol, pos, rng, weights)
            key = canon_clue(cl)
            if key in seen:
                continue
            seen.add(key)
            clues.append(cl)
            if count_solutions(n, cats, vals, clues) == 1:
                break
        else:
            continue
        for cl in list(rng.sample(clues, len(clues))):    # drop clues the solution does not need
            rest = [c for c in clues if c is not cl]
            if count_solutions(n, cats, vals, rest) == 1:
                clues = rest
        rng.shuffle(clues)
        assert all(clue_true(c, pos) for c in clues) and count_solutions(n, cats, vals, clues) == 1
        return {"n": n, "cats": cats, "solution": sol, "clues": [list(_jsonable(c)) for c in clues]}
    return None


def canon_clue(cl: tuple) -> tuple:
    kind = cl[0]
    if kind in ("same", "notsame", "next", "gap1"):
        return (kind,) + tuple(sorted([cl[1], cl[2]]))
    if kind == "right":
        return ("left", cl[2], cl[1])
    return tuple(cl)


def canonical(p: dict) -> tuple:
    """Puzzle identity: the solution as a set of house rows plus the clue set (order and left/right wording ignored)."""
    rows = tuple(sorted(tuple(sorted((c, p["solution"][c][h]) for c in p["cats"])) for h in range(p["n"])))
    return rows, tuple(sorted(canon_clue(clue_tuple(c)) for c in p["clues"]))


def _jsonable(cl: tuple) -> tuple:
    return tuple(list(x) if isinstance(x, tuple) else x for x in cl)


def clue_tuple(c) -> tuple:
    return tuple(tuple(x) if isinstance(x, list) else x for x in c)


# --------------------------------------------------------------------------------------------------------- rendering
def phrase(e: tuple) -> str:
    return CATEGORIES[e[0]][2].format(v=e[1])


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def render_clue(cl: tuple) -> str:
    kind, x = cl[0], phrase(cl[1])
    if kind == "found":
        return _cap(f"{x} lives in the {ORDINAL[cl[2]]} house.")
    if kind == "notfound":
        return _cap(f"{x} does not live in the {ORDINAL[cl[2]]} house.")
    y = phrase(cl[2])
    return _cap({"same": f"{x} is {y}.", "notsame": f"{x} is not {y}.",
                 "left": f"{x} lives somewhere to the left of {y}.", "right": f"{x} lives somewhere to the right of {y}.",
                 "dleft": f"{x} lives directly to the left of {y}.", "next": f"{x} and {y} live next to each other.",
                 "gap1": f"There is exactly one house between {x} and {y}."}[kind])


def render_grid(sol: dict, cats: list[str]) -> str:
    n = len(sol[cats[0]])
    return "; ".join(f"House {h + 1}: " + ", ".join(sol[c][h] for c in cats) for h in range(n))


def render_problem(p: dict, options_note: bool = True) -> str:
    """The puzzle text before the final question: preamble, one line per category, the numbered clues."""
    n, cats = p["n"], p["cats"]
    head = (f"There are {n} houses in a row, numbered 1 to {n} from left to right. Each house is home to exactly one "
            f"person, and no two people share any of the following attributes.")
    lines = [f"- {CATEGORIES[c][0]}: " + ", ".join(sorted(p["solution"][c])) for c in cats]
    clues = [f"{i + 1}. {render_clue(clue_tuple(c))}" for i, c in enumerate(p["clues"])]
    order = ", ".join(LABEL[c] for c in cats)
    note = (f"\n\nEach option lists the houses from left to right, giving for each house the person's {order}, in "
            f"that order." if options_note else "")
    return head + "\n" + "\n".join(lines) + note + "\n\nClues:\n" + "\n".join(clues)


# -------------------------------------------------------------------------------------------------------- distractors
def violations(p: dict, sol: dict) -> int:
    pos = positions(sol)
    return sum(not clue_true(clue_tuple(c), pos) for c in p["clues"])


def _swap(sol: dict, rng: random.Random) -> dict:
    s = {c: list(v) for c, v in sol.items()}
    c = rng.choice(list(s))
    i, j = rng.sample(range(len(s[c])), 2)
    s[c][i], s[c][j] = s[c][j], s[c][i]
    return s


def _key(sol: dict, cats: list[str]) -> tuple:
    return tuple(tuple(sol[c]) for c in cats)


def distractor_pool(p: dict, rng: random.Random, size: int = 1500) -> list[tuple[int, dict]]:
    """Wrong arrangements with their number of broken clues: random walks of 1-6 swaps from the solution, and local
    minima of the broken-clue count reached by greedy swap descent from random arrangements (these are not near the
    solution in the grid sense, so the options do not cluster around the answer)."""
    cats, gold = p["cats"], {c: list(v) for c, v in p["solution"].items()}
    gk = _key(gold, cats)
    out = {}
    for _ in range(size // 2):
        s = gold
        for _ in range(rng.randint(1, 6)):
            s = _swap(s, rng)
        k = _key(s, cats)
        if k != gk:
            out[k] = (violations(p, s), s)
    for _ in range(size // 30):
        s = {c: rng.sample(v, len(v)) for c, v in gold.items()}
        v = violations(p, s)
        for _ in range(200):
            t = _swap(s, rng)
            vt = violations(p, t)
            if vt <= v:
                s, v = t, vt
            k = _key(s, cats)
            if k != gk:
                out[k] = (v, s)
    return sorted(out.values(), key=lambda x: x[0])


def hamming(a: dict, b: dict, cats: list[str]) -> int:
    return sum(x != y for c in cats for x, y in zip(a[c], b[c]))


def choose_options(p: dict, rng: random.Random, k: int = 5, slack: int = 1) -> list[dict] | None:
    """k distractors from the pool's near misses (broken clues <= the pool minimum + slack, widened if too few), drawn
    so that the answer is not the Hamming medoid of the options more often than chance: candidates are added one by
    one, each time preferring (at random among the ties) one that keeps the answer off the medoid."""
    pool = distractor_pool(p, rng)
    if not pool:
        return None
    cats, gold = p["cats"], p["solution"]
    lo = pool[0][0]
    while True:
        near = [s for v, s in pool if v <= lo + slack]
        if len(near) >= 3 * k or slack > 6:
            break
        slack += 1
    if len(near) < k:
        return None
    rng.shuffle(near)
    chosen = []
    for s in near:
        if len(chosen) == k:
            break
        trial = chosen + [s]
        opts = [gold] + trial
        dist = {i: sum(hamming(a, b, cats) for b in opts) for i, a in enumerate(opts)}
        if len(trial) == k and min(dist, key=dist.get) == 0 and rng.random() > 1 / (k + 1):
            continue                                     # the answer would be the medoid: skip, unless chance allows
        chosen.append(s)
    return chosen if len(chosen) == k else None


def _apply(sol: dict, sw: tuple) -> dict:
    c, i, j = sw
    s = {k: list(v) for k, v in sol.items()}
    s[c][i], s[c][j] = s[c][j], s[c][i]
    return s


def symmetric_options(p: dict, rng: random.Random, k: int = 5, tries: int = 300) -> list[dict] | None:
    """k distractors with no option-only cue: a centre C = the answer with one swap s* (two houses' values of one
    category exchanged); every option is C with one more swap, the swaps pairwise disjoint (no shared cell), and the
    answer is the option whose swap undoes s*. So every option is 2 cells from C and 4 cells from every other option:
    the Hamming medoid and the per-cell majority (= C) single out no option. Among `tries` random draws the one whose
    distractors break the fewest clues (sum) is kept. None when the grid has fewer than k + 1 disjoint swaps."""
    cats, gold, n = p["cats"], p["solution"], p["n"]
    if len(cats) * (n // 2) < k + 1:
        return None
    swaps = [(c, i, j) for c in cats for i in range(n) for j in range(i + 1, n)]
    best = None
    for _ in range(tries):
        s0 = rng.choice(swaps)
        used = {(s0[0], s0[1]), (s0[0], s0[2])}
        centre = _apply(gold, s0)
        chosen = []
        for sw in rng.sample(swaps, len(swaps)):
            cells = {(sw[0], sw[1]), (sw[0], sw[2])}
            if cells & used:
                continue
            chosen.append(sw)
            used |= cells
            if len(chosen) == k:
                break
        if len(chosen) < k:
            continue
        ds = [_apply(centre, sw) for sw in chosen]
        score = sum(violations(p, d) for d in ds)
        if best is None or score < best[0]:
            best = (score, ds)
    return best[1] if best else None


def local_minima(p: dict, rng: random.Random, restarts: int = 120, patience: int = 60) -> list[tuple[int, dict]]:
    """Wrong arrangements that break few clues and owe nothing to the answer: greedy swap descent (sideways moves
    allowed) from random arrangements; each run's end point is kept unless it is the answer."""
    cats, gold = p["cats"], p["solution"]
    gk = _key(gold, cats)
    out = {}
    for _ in range(restarts):
        s = {c: rng.sample(v, len(v)) for c, v in gold.items()}
        v, stall = violations(p, s), 0
        while stall < patience and v > 0:
            t = _swap(s, rng)
            vt = violations(p, t)
            if vt < v:
                s, v, stall = t, vt, 0
            else:
                if vt == v and rng.random() < 0.5:
                    s = t
                stall += 1
        k = _key(s, cats)
        if v > 0 and k != gk:
            out[k] = (v, s)
    return sorted(out.values(), key=lambda x: x[0])


def far_options(p: dict, rng: random.Random, k: int = 5, frac: float = 0.5, min_frac: float = 0.3,
                restarts: int = 120, tries: int = 800, top: int = 40,
                one_broken: bool = False) -> tuple[list[dict], float] | None:
    """k distractors that break few clues and each differ from the answer in at least `frac` of the cells (relaxed in
    steps of 0.1 down to min_frac when too few exist), so the answer cannot be told from a few cells: every clue has to
    be checked against the options, as in K&K. Low-violation arrangements cluster around the answer, which would make
    it the most central option; so the set is drawn to give the answer a centrality rank fixed in advance at random
    (uniform over the k + 1 ranks; rank by total Hamming distance to the other options, ties broken at random), and
    the answer is the unique option closest to the per-cell majority arrangement with probability 1/(k+1). Candidates: the `top` arrangements with the
    fewest broken clues among those far enough from the answer. Returns (distractors, the fraction reached) or None."""
    cats, gold = p["cats"], p["solution"]
    n, cells = p["n"], p["n"] * len(cats)
    pool = local_minima(p, rng, restarts=restarts, patience=max(60, 4 * cells))
    target = rng.randrange(k + 1)                         # the answer's centrality rank (0 = most central)
    allow_major = rng.random() < 1 / (k + 1)
    f = frac
    while f >= min_frac - 1e-9:
        need = -(-int(round(f * cells * 100)) // 100)
        far = [(v, sol) for v, sol in pool if hamming(sol, gold, cats) >= need]
        if one_broken:                                    # v3: exactly one broken clue, a relation (not a position)
            one = [(v, sol) for v, sol in far if v == 1 and not broken_unary(p, sol)]
            far = one if len(one) >= k else far
        rng.shuffle(far)
        far.sort(key=lambda x: x[0])
        cand = [sol for _, sol in far[:top]]
        if len(cand) >= k:
            for t in range(tries):
                ds = rng.sample(cand, k)
                opts = [gold] + ds
                tot = [sum(hamming(a, b, cats) for b in opts) for a in opts]
                order = sorted(range(k + 1), key=lambda i: (tot[i], rng.random()))
                if order.index(0) != target:
                    continue
                maj = {c: [Counter(o[c][h] for o in opts).most_common(1)[0][0] for h in range(n)] for c in cats}
                dm = [hamming(o, maj, cats) for o in opts]
                top_major = dm[0] < min(dm[1:])
                if t < tries // 2 and top_major != allow_major:   # the second half of the draws drops this condition
                    continue
                if t >= tries // 2 and top_major and not allow_major:
                    continue
                return ds, f
        f = round(f - 0.1, 2)
    return None


def broken_unary(p: dict, sol: dict) -> bool:
    pos = positions(sol)
    return any(c[0] in UNARY and not clue_true(clue_tuple(c), pos) for c in p["clues"])


def option_cues(gold: dict, ds: list[dict], cats: list[str]) -> list[str]:
    """Option-only cues that pick the answer: "medoid" (least total Hamming distance to the other options, ties
    count) and "majority" (the option closest to the per-cell majority arrangement, ties count)."""
    opts = [gold] + ds
    n = len(gold[cats[0]])
    dist = [sum(hamming(a, b, cats) for b in opts) for a in opts]
    maj = {c: [Counter(o[c][h] for o in opts).most_common(1)[0][0] for h in range(n)] for c in cats}
    dm = [hamming(o, maj, cats) for o in opts]
    out = []
    if dist[0] == min(dist):
        out.append("medoid")
    if dm[0] == min(dm):
        out.append("majority")
    return out


def make_item(p: dict, rng: random.Random, item_id: str, source: str = "zebra_tf", scheme: str = "symmetric") -> dict | None:
    """The multiple-choice item: problem text, FINAL_QUESTION, 6 options (answer at a random letter)."""
    far = None
    if scheme in ("far", "far1"):                         # v2 (11-07): options far apart; far1 (v3): one broken clue each
        r = far_options(p, rng, restarts=240 if scheme == "far1" else 120, one_broken=scheme == "far1")
        if r is None:
            return None
        ds, far = r
    else:
        ds = symmetric_options(p, rng)
        if ds is None:                                   # small grids: near misses, answer off the medoid
            ds, scheme = choose_options(p, rng), "near"
    if ds is None:
        return None
    opts = [p["solution"]] + ds
    order = list(range(len(opts)))
    rng.shuffle(order)
    labels = [chr(65 + i) for i in range(len(opts))]
    gold = labels[order.index(0)]
    text = (render_problem(p) + "\n\n" + FINAL_QUESTION + "\nOptions:\n"
            + "\n".join(f"{L}) {render_grid(opts[j], p['cats'])}" for L, j in zip(labels, order)))
    return {"item_id": item_id, "source": source, "prompt": text, "labels": labels, "gold_label": gold,
            "teacher": {L: float(L == gold) for L in labels}, "zebra": p,
            "option_violations": {L: violations(p, opts[j]) for L, j in zip(labels, order)},
            "size": f"{p['n']}x{len(p['cats'])}", "option_scheme": scheme, **({"far_frac": far} if far else {})}


def make_cell_item(p: dict, rng: random.Random, item_id: str, source: str = "zebra_tf") -> dict:
    """The puzzle asked as one cell (ZebraLogic's multiple-choice mode): "What is the <attribute> of the person who
    lives in the <k-th> house?", the attribute's n values as options in random order. The cell is drawn at random
    among those no clue states outright (v3 puzzles have no position-placing clue, so any cell)."""
    pos = positions(p["solution"])
    stated = {(c[1][0], c[2]) for c in map(clue_tuple, p["clues"]) if c[0] == "found"}
    cells = [(c, h) for c in p["cats"] for h in range(1, p["n"] + 1) if (c, h) not in stated]
    c, h = rng.choice(cells)
    vals = list(p["solution"][c])
    rng.shuffle(vals)
    labels = [chr(65 + i) for i in range(len(vals))]
    gold = labels[vals.index(p["solution"][c][h - 1])]
    q = CELL_QUESTION.format(label=LABEL[c], ordinal=ORDINAL[h])
    text = (render_problem(p, options_note=False) + "\n\n" + q + "\nOptions:\n"
            + "\n".join(f"{L}) {v}" for L, v in zip(labels, vals)))
    return {"item_id": item_id, "source": source, "prompt": text, "labels": labels, "gold_label": gold,
            "teacher": {L: float(L == gold) for L in labels}, "zebra": p, "cell": [c, h],
            "size": f"{p['n']}x{len(p['cats'])}", "option_scheme": "cell"}


# ------------------------------------------------------------------------------------------------------- proof trees
# A deduction engine with justifications. Facts: ("at", e, h) = entity e lives in house h; ("not", e, h) = it does not.
# Rules (the fact's "rule"; "prem" = the facts it follows from, "clue" = the clue it reads):
#   found / notfound   a position clue read off the text (no premise)
#   placed             e is placed in house h, so e is in no other house       (folded into "at" in the tree)
#   taken              e' is placed in house h, so no other value of e's category is in house h   (folded too)
#   clue               a relation clue rules a house out for X: no house left for Y satisfies the relation with it
#                      (prem: the facts that ruled out Y's compatible houses)
#   only_house         every other house is ruled out for e
#   only_person        house h is ruled out for every other value of e's category
#   case               assuming e in house h leads to a contradiction (prem: the hypothetical chain)
#   hyp                the assumption itself (inside a case)
# Contradictions: ("none", e) no house left for e; ("empty", category, h) no value of the category left for house h.

class Deducer:
    def __init__(self, p: dict):
        self.n, self.cats = p["n"], p["cats"]
        self.clues = [clue_tuple(c) for c in p["clues"]]
        self.groups = {c: [(c, v) for v in sorted(p["solution"][c])] for c in self.cats}
        self.dom = {e: _bits(self.n) for g in self.groups.values() for e in g}
        self.facts: dict = {}
        self.order: list = []
        self.contra = None

    def copy(self) -> "Deducer":
        d = Deducer.__new__(Deducer)
        d.n, d.cats, d.clues, d.groups = self.n, self.cats, self.clues, self.groups
        d.dom, d.facts, d.order, d.contra = dict(self.dom), dict(self.facts), list(self.order), None
        return d

    def _add(self, key, rule, prem, clue=None) -> None:
        work = [(key, rule, prem, clue)]
        while work and self.contra is None:
            key, rule, prem, clue = work.pop(0)
            if key in self.facts:
                continue
            kind, e, h = key
            bit = 1 << (h - 1)
            if kind == "at":
                if not self.dom[e] & bit:                 # e was ruled out of h: contradiction at e
                    self.facts[key] = {"rule": rule, "prem": prem, "clue": clue}
                    self.order.append(key)
                    self.contra = (("none", e), [("not", e, h), key])
                    return
                self.facts[key] = {"rule": rule, "prem": prem, "clue": clue}
                self.order.append(key)
                for h2 in range(1, self.n + 1):
                    if h2 != h and self.dom[e] >> (h2 - 1) & 1:
                        work.append((("not", e, h2), "placed", [key], None))
                for e2 in self.groups[e[0]]:
                    if e2 != e and self.dom[e2] & bit:
                        work.append((("not", e2, h), "taken", [key], None))
                continue
            if not self.dom[e] & bit:
                continue
            self.facts[key] = {"rule": rule, "prem": prem, "clue": clue}
            self.order.append(key)
            self.dom[e] &= ~bit
            if not self.dom[e]:
                self.contra = (("none", e), [("not", e, x) for x in range(1, self.n + 1)])
                return
            if self.dom[e] & (self.dom[e] - 1) == 0:
                h0 = self.dom[e].bit_length()
                if ("at", e, h0) not in self.facts:
                    work.append((("at", e, h0), "only_house", [("not", e, x) for x in range(1, self.n + 1) if x != h0], None))
            cand = [e2 for e2 in self.groups[e[0]] if self.dom[e2] & bit]
            if not cand:
                self.contra = (("empty", e[0], h), [("not", e2, h) for e2 in self.groups[e[0]]])
                return
            if len(cand) == 1 and ("at", cand[0], h) not in self.facts:
                work.append((("at", cand[0], h), "only_person", [("not", e2, h) for e2 in self.groups[e[0]] if e2 != cand[0]],
                             None))

    def start(self) -> None:
        for i, cl in enumerate(self.clues):
            if cl[0] == "found":
                self._add(("at", cl[1], cl[2]), "found", [], i)
            elif cl[0] == "notfound":
                self._add(("not", cl[1], cl[2]), "notfound", [], i)

    def propagate(self) -> None:
        changed = True
        while changed and self.contra is None:
            changed = False
            for i, cl in enumerate(self.clues):
                kind = cl[0]
                if kind in UNARY:
                    continue
                for a, b, fwd in ((cl[1], cl[2], True), (cl[2], cl[1], False)):
                    for h in range(1, self.n + 1):
                        if not self.dom[a] >> (h - 1) & 1:
                            continue
                        compat = [h2 for h2 in range(1, self.n + 1) if (holds(kind, h, h2) if fwd else holds(kind, h2, h))]
                        if any(self.dom[b] >> (h2 - 1) & 1 for h2 in compat):
                            continue
                        self._add(("not", a, h), "clue", [("not", b, h2) for h2 in compat], i)
                        changed = True
                        if self.contra is not None:
                            return

    def solved(self) -> bool:
        return all(d & (d - 1) == 0 for d in self.dom.values())

    def case_split(self) -> bool:
        """One assumption ("e lives in house h") that propagation refutes; the shortest refutation is kept and its
        conclusion ("not", e, h) added with rule case. False when no single assumption is refuted."""
        best = None
        for e, d in self.dom.items():
            if d & (d - 1) == 0:
                continue
            for h in range(1, self.n + 1):
                if not d >> (h - 1) & 1:
                    continue
                t = self.copy()
                t._add(("at", e, h), "hyp", [], None)
                if t.contra is None:
                    t.propagate()
                if t.contra is None:
                    continue
                chain = t.closure(t.contra[1], stop=set(self.facts))
                if best is None or len(chain) < len(best[2]):
                    best = (e, h, chain, t)
        if best is None:
            return False
        e, h, chain, t = best
        self.facts.setdefault(("case", e, h), {"rule": "case_world", "facts": {k: t.facts[k] for k in chain},
                                                "contra": t.contra, "hyp": ("at", e, h)})
        self._add(("not", e, h), "case", [("case", e, h)], None)
        return True

    def closure(self, keys, stop: set) -> list:
        """The facts the keys rest on, premises first (stopping at the facts in `stop`)."""
        seen, out = set(), []

        def visit(k):
            if k in seen or k in stop or k not in self.facts:
                return
            seen.add(k)
            for q in self.facts[k].get("prem", []):
                visit(q)
            out.append(k)
        for k in keys:
            visit(k)
        return out

    def run(self, max_cases: int = 6) -> bool:
        self.start()
        self.propagate()
        cases = 0
        while self.contra is None and not self.solved():
            if cases >= max_cases or not self.case_split():
                return False
            cases += 1
            self.propagate()
        return self.contra is None and self.solved()


def _ent(e) -> str:
    return phrase(e)


def _fact_q(key, hyp=None) -> tuple[str, str, bool]:
    """(question, opposite wording, answer of the question) for a fact, inside an assumption when hyp is given."""
    kind, e, h = key
    x, H = _ent(e), ORDINAL[h]
    if hyp is None:
        if kind == "at":
            return _cap(f"does {x} live in the {H} house?"), _cap(f"does {x} live somewhere other than the {H} house?"), True
        return _cap(f"could {x} live in the {H} house?"), f"Is the {H} house ruled out for {x}?", False
    pre = f"If {_ent(hyp[1])} lived in the {ORDINAL[hyp[2]]} house, "
    if kind == "at":
        return f"{pre}would {x} live in the {H} house?", f"{pre}would {x} live somewhere other than the {H} house?", True
    return f"{pre}could {x} live in the {H} house?", f"{pre}would the {H} house be ruled out for {x}?", False


def _contra_q(contra, hyp) -> tuple[str, str, bool]:
    pre = f"If {_ent(hyp[1])} lived in the {ORDINAL[hyp[2]]} house, "
    if contra[0][0] == "none":
        x = _ent(contra[0][1])
        return f"{pre}would any house be left for {x}?", f"{pre}would every house be ruled out for {x}?", False
    _, c, h = contra[0]
    return (f"{pre}could any {LABEL[c]} fit the {ORDINAL[h]} house?",
            f"{pre}would every {LABEL[c]} be ruled out for the {ORDINAL[h]} house?", False)


def _ej(e):
    return [e[0], e[1]]


def proof_tree(p: dict, rng: random.Random, max_cases: int = 6) -> list[dict] | None:
    """The tree nodes (K&K subq format) of a puzzle, or None when one-level case splits do not solve it.
    Node types: parse (one concrete instance of each relation clue the proof reads: the clue's two people at their
    true houses, answer Yes, or a placement the proof rules out with it, answer No), derive (facts of the main line:
    position clues, clue eliminations, only-house / only-person placements, case conclusions), case (facts inside a
    refuted assumption, ending in its contradiction). Placed / taken facts are folded into the placement they follow
    from; the assumption itself is not asked (every case node states it). About half of the yes/no nodes are asked in
    their opposite wording (answer flipped). Each node keeps what audit_tree needs: fact, hyp, rule, clue, base."""
    d = Deducer(p)
    if not d.run(max_cases=max_cases):
        return None
    pos = positions(p["solution"])
    nodes, idmap, parse_of = [], {}, {}

    def new(rec):
        rec["id"] = f"n{len(nodes) + 1}"
        nodes.append(rec)
        return rec["id"]

    def fold(k, facts):
        while k in facts and facts[k]["rule"] in ("placed", "taken"):
            k = facts[k]["prem"][0]
        return k

    def parse_node(ci, fact_key):
        if ci in parse_of:
            return parse_of[ci]
        kind, x, y = d.clues[ci]
        _, a, h = fact_key
        if rng.random() < 0.5 or fact_key[0] != "not":
            h1, h2 = pos[x], pos[y]                       # the clue's two people where they live: Yes
        else:                                             # the ruled-out placement against the other's true house: No
            h1, h2 = (h, pos[y]) if a == x else (pos[x], h)
        t = holds(kind, h1, h2)
        q = (f"According to clue {ci + 1}, could {_ent(x)} live in the {ORDINAL[h1]} house if {_ent(y)} lived in the "
             f"{ORDINAL[h2]} house?")
        parse_of[ci] = new({"question": q, "opposite": None, "truth": t, "base": t, "type": "parse", "depends_on": [],
                            "clue": ci + 1, "instance": [h1, h2]})
        return parse_of[ci]

    def emit(k, facts, hyp=None, world=None):
        """Node id of fact k: k lives in `facts` (the main line, or a case world when world is set)."""
        k = fold(k, facts)
        if world is not None and k not in facts:          # a main-line fact read inside a case
            return emit(k, d.facts)
        mk = (world, k)
        if mk in idmap:
            return idmap[mk]
        f = facts[k]
        if f["rule"] == "hyp":
            return None
        deps = []
        if f["rule"] == "case":
            ck = f["prem"][0]
            w = d.facts[ck]
            inner, hk = w["facts"], w["hyp"]
            cdeps = [emit(q, inner, hyp=hk, world=ck) for q in w["contra"][1]]
            cq, co, ct = _contra_q(w["contra"], hk)
            deps = [new({"question": cq, "opposite": co, "truth": ct, "base": ct, "type": "case",
                         "depends_on": sorted({x for x in cdeps if x}), "rule": "contradiction",
                         "contra": [w["contra"][0][0]] + [_ej(x) if isinstance(x, tuple) else x for x in w["contra"][0][1:]],
                         "hyp": ["at", _ej(hk[1]), hk[2]],
                         "parent_facts": [[q[0], _ej(q[1]), q[2]] for q in w["contra"][1]]})]
        else:
            for q in f.get("prem", []):
                deps.append(emit(q, facts, hyp=hyp, world=world))
            if f["rule"] == "clue":
                deps.append(parse_node(f["clue"], k))
        qq, oo, tt = _fact_q(k, hyp)
        rec = {"question": qq, "opposite": oo, "truth": tt, "base": tt, "type": "case" if world else "derive",
               "depends_on": sorted({x for x in deps if x}), "rule": f["rule"], "fact": [k[0], _ej(k[1]), k[2]],
               "hyp": ["at", _ej(hyp[1]), hyp[2]] if hyp else None,
               "prem_facts": [[q[0], _ej(q[1]), q[2]] for q in f.get("prem", []) if isinstance(q[1], tuple)]}
        if f.get("clue") is not None:
            rec["clue"] = f["clue"] + 1
        idmap[mk] = new(rec)
        return idmap[mk]

    for e in pos:
        emit(("at", e, pos[e]), d.facts)
    depth = {}
    for x in nodes:
        depth[x["id"]] = 1 + max((depth[y] for y in x["depends_on"]), default=-1)
    for x in nodes:
        x["node_depth"] = depth[x["id"]]
        x["flipped"] = x["opposite"] is not None and rng.random() < 0.5
        if x["flipped"]:
            x["question"], x["opposite"], x["truth"] = x["opposite"], x["question"], not x["truth"]
        x["p_cot"] = 1.0 if x["truth"] else 0.0
        x.update({"p_fresh": None, "initial": None, "status": "direct", "source_tree": "solver"})
    return nodes


def _consistent(p: dict, facts: list, clue=None) -> list[dict]:
    """All placements of the entities the facts (and the clue) mention, one value per house within a category, that
    satisfy the facts and the clue (a solution-free local check: nothing else about the puzzle is used)."""
    n = p["n"]
    ents = []
    for f in facts:
        e = tuple(f[1])
        if e not in ents:
            ents.append(e)
    if clue is not None:
        for e in clue[1:3]:
            if isinstance(e, tuple) and e not in ents:
                ents.append(e)
    out = []

    def ok(asg):
        for kind, e, h in facts:
            e = tuple(e)
            if e in asg and (asg[e] == h) != (kind == "at"):
                return False
        return True

    def rec(i, asg):
        if not ok(asg):
            return
        if i == len(ents):
            if clue is None or holds(clue[0], asg[clue[1]], asg[clue[2]]):
                out.append(dict(asg))
            return
        e = ents[i]
        for h in range(1, n + 1):
            if any(f[0] == e[0] and asg[f] == h for f in asg):
                continue
            asg[e] = h
            rec(i + 1, asg)
            del asg[e]
    rec(0, {})
    return out


def audit_tree(p: dict, nodes: list[dict]) -> list[str]:
    """Checks of a tree, as messages (empty = clean): every main-line fact agrees with the solution; every derived
    fact follows from its premise facts (and its clue, and the assumption inside a case) alone, by enumerating the
    placements of the entities they mention; every contradiction node's premises admit no placement; every parse node
    evaluates its clue correctly; asked answers = base answers flipped when the wording is; dependencies point back."""
    pos = positions(p["solution"])
    clues = [clue_tuple(c) for c in p["clues"]]
    ids = {x["id"]: i for i, x in enumerate(nodes)}
    bad = []
    for i, x in enumerate(nodes):
        if any(ids.get(y, i) >= i for y in x["depends_on"]):
            bad.append(f"{x['id']}: dependency not earlier")
        if x["truth"] != (x["base"] != x.get("flipped", False)):
            bad.append(f"{x['id']}: asked answer vs base")
        if x["type"] == "parse":
            kind = clues[x["clue"] - 1][0]
            if holds(kind, *x["instance"]) != x["base"]:
                bad.append(f"{x['id']}: parse instance")
            continue
        hyp = x.get("hyp")
        pf = [nodes[ids[y]]["fact"] for y in x["depends_on"] if nodes[ids[y]].get("fact")]
        if x.get("rule") == "contradiction":
            fs = pf + [hyp]
            if _consistent(p, fs):
                bad.append(f"{x['id']}: contradiction premises are satisfiable")
            continue
        kind, e, h = x["fact"]
        e = tuple(e)
        if hyp is None and (pos[e] == h) != (kind == "at"):
            bad.append(f"{x['id']}: main-line fact false in the solution")
        if x["rule"] in ("found", "notfound"):
            cl = clues[x["clue"] - 1]
            if (cl[0], cl[1], cl[2]) != ({"at": "found", "not": "notfound"}[kind], e, h):
                bad.append(f"{x['id']}: position clue mismatch")
            continue
        if x["rule"] == "case":                           # refuted assumption: its contradiction node assumes it
            cs = [nodes[ids[y]] for y in x["depends_on"]]
            if kind != "not" or [c.get("hyp") for c in cs] != [["at", list(e), h]]:
                bad.append(f"{x['id']}: case conclusion vs its assumption")
            continue
        prem = pf + ([hyp] if hyp else [])
        cl = clues[x["clue"] - 1] if x["rule"] == "clue" else None
        sols = _consistent(p, prem + [[ "at" if kind == "not" else "not", list(e), h]], cl)
        if sols:
            bad.append(f"{x['id']}: {x['rule']} does not follow from its premises")
    return bad
