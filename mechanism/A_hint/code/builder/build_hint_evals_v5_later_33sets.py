#!/usr/bin/env python3
"""Build the OPC (oracle-prefix capacity) eval sets: cue-free GSM8K items (gsm8k_mc_v2_xprob) with the first k worked
steps written into the prompt as the repo's "Worked steps:" block (student.chain_block + insert_chain), plus controls.

    PYTHONPATH=<repo>/src python build_hint_evals.py --steps <repo>/data/eval/gsm8k_mc_v2_steps.jsonl \
        --x <repo>/data/eval/gsm8k_mc_v2_xprob.jsonl --out data/eval/hint [--seed 20261021]

Items: x items whose steps item (same test index) has n >= 2 worked steps. Sets written (one jsonl each):
  hint_plain            the item unchanged                                   (n >= 2 items)
  hint_true_r<r>        first k = n - r true steps shown, r = 1 .. 6 remaining (r = 1: n >= 2 items, ...)
  hint_full             all n steps shown (the answer step included): sanity ceiling
  hint_other_r<r>       k lines from another item's chain (student.teacher_partners), r in 1..5
  hint_placebo_r<r>     k sentences of the item's own question text (student._pl_sentences), r in 1..5
  hint_plan_r<r>        the first k steps as a PLAN: results hidden ('?'), linked operands '[Step i]', r in 1..5
  hint_wrongres_r<r>    a CONSISTENT wrong chain: every result wrong and at least one tier away from the truth (an
                        operator slip when it gives a clean number, else +-max(2, 10%)), and every operand that comes
                        from an earlier step (a_from / b_from) rewritten to that step's wrong value, r in {1, 2}.
                        No true result of steps 1..k appears in the block. The M3 control.
  hint_wrongres_near_r<r> the old near-miss control: true operands kept, results off by +-1 .. +-2 (secondary)
Every row keeps item_id, labels, gold_label, gold_probs, label_order, option_kinds, depth of the x item and adds
n, r, k, kind, shown (the lines), shown_equals_option (a shown value equals an option text), partner (other).
Prints a manifest (counts, md5) and writes it to <out>/MANIFEST.json.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys

import random

try:                                   # the repo helpers (need torch at import); else byte-equal local copies below
    from cotdistill import student as st
    HAVE_REPO = True
except ImportError:
    HAVE_REPO = False


class _Local:
    """Copies of the cotdistill.student helpers the builder uses (pin 994ff50), for a Python without torch. Kept
    equal to the repo functions; test_build_hint_evals.py asserts the equality whenever the repo is importable."""
    CHAIN_HEADER = "Worked steps:"
    _PL_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

    @staticmethod
    def worked_steps(item):
        if "worked_steps" in item:
            return item["worked_steps"] or None
        return None

    @staticmethod
    def slot_values(item):
        out = []
        for w in _Local.worked_steps(item) or []:
            v = w.get("value")
            v = str(v).strip() if v is not None else str(w.get("text", "")).rpartition("=")[2].strip()
            if not v:
                raise ValueError(f"{item.get('item_id')}: worked step {w.get('id')} has no value")
            out.append(v)
        return out

    @staticmethod
    def chain_block(texts, first=1):
        if not texts:
            return ""
        return f"\n\n{_Local.CHAIN_HEADER}\n" + "\n".join(f"Step {first + i}: {t}" for i, t in enumerate(texts))

    @staticmethod
    def insert_chain(text, block):
        if not block:
            return text
        head_, sep, tail = text.rpartition("\nOptions:\n")
        if not sep:
            raise ValueError("no options block to write the worked steps before")
        return head_ + block + sep + tail

    @staticmethod
    def _pl_sentences(text):
        out = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
            out += [t for t in _Local._PL_SENT.split(line) if t.strip()]
        return out

    @staticmethod
    def teacher_partners(items, seed, need=None):
        chains = {it["item_id"]: tuple(w["text"] for w in _Local.worked_steps(it) or []) for it in items}
        ids = sorted((i for i, c in chains.items() if c), key=str)
        by_min, out = {}, {}
        for iid in ids:
            m = max(1, int((need or {}).get(iid, 1)))
            if m not in by_min:
                by_min[m] = [j for j in ids if len(chains[j]) >= m]
            pool = by_min[m]
            rng = random.Random(f"teacher-partner|{seed}|{iid}")
            pick = None
            for _ in range(64):
                j = pool[rng.randrange(len(pool))] if pool else None
                if j is not None and j != iid and chains[j] != chains[iid]:
                    pick = j
                    break
            if pick is None:
                ok = [j for j in pool if j != iid and chains[j] != chains[iid]]
                if not ok:
                    raise ValueError(f"teacher partners: no other item with >= {m} worked steps for {iid}")
                pick = ok[rng.randrange(len(ok))]
            out[iid] = pick
        return out


if not HAVE_REPO:
    st = _Local

R_TRUE = (1, 2, 3, 4, 5, 6)
R_CTRL = (1, 2, 3, 4, 5)     # other-item and wrong-result controls (v2: to r = 5, for the far cells rc = 3)
R_PLACEBO = (1, 2, 3, 4, 5)  # length / position control at every r of the capacity curve
R_PLAN = (1, 2, 3, 4, 5)     # plan rendering: operations shown, results hidden, linked operands as [Step i]


def read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def index_of(item_id):
    m = re.search(r"-test-(\d+)$", item_id)
    if not m:
        raise ValueError(item_id)
    return int(m.group(1))


def head(prompt):
    return prompt.rpartition("\nOptions:\n")[0]


def option_texts(prompt):
    tail = prompt.rpartition("\nOptions:\n")[2]
    return [ln.split(") ", 1)[1].strip() for ln in tail.split("\n") if ") " in ln]


def option_map(prompt):
    tail = prompt.rpartition("\nOptions:\n")[2]
    return {ln.split(") ", 1)[0].strip(): ln.split(") ", 1)[1].strip() for ln in tail.split("\n") if ") " in ln}


def fmt_like(v, ref):
    """Format number v like the string ref (integer if ref has no '.', else the same number of decimals)."""
    if "." in ref:
        d = len(ref.split(".")[1])
        return f"{v:.{d}f}"
    return str(int(round(v)))


def wrong_value(true, forbidden):
    """A wrong result for a step: the first of true+1, true-1, true+2, true-2, 2*true, true/2, true+10 that is a
    number of the same format, not equal to the true value or to any forbidden text. None if none works."""
    try:
        t = float(true.replace(",", ""))
    except ValueError:
        return None
    for v in (t + 1, t - 1, t + 2, t - 2, 2 * t, t / 2, t + 10):
        s = fmt_like(v, true)
        if s != true and s not in forbidden and not s.startswith("-"):
            return s
    return None


def wrong_line(text, true, wrong):
    """The step text with its result replaced: 'a op b = true' -> 'a op b = wrong'."""
    lhs, sep, rhs = text.rpartition("=")
    if not sep or rhs.strip() != true:
        raise ValueError(f"step text {text!r} does not end in = {true}")
    return f"{lhs}= {wrong}"


def plan_line(step):
    """PLAN rendering of a step: result hidden ("?"), operands that come from earlier steps written "[Step i]"
    (a_from / b_from are 0-based). Must equal cotdistill.student.plan_line (tested by equality once the repo change
    lands; until then this copy is the builder's source)."""
    a = f"[Step {int(step['a_from']) + 1}]" if step.get("a_from") is not None else str(step["a"])
    b = f"[Step {int(step['b_from']) + 1}]" if step.get("b_from") is not None else str(step["b"])
    return f"{a}% of {b} = ?" if step["op"] == "% of" else f"{a} {step['op']} {b} = ?"


def line_text(a, op, b, value):
    """The repo's step text format: '60% of 5 = 3' for the pct kind, 'a op b = value' otherwise."""
    return f"{a}% of {b} = {value}" if op == "% of" else f"{a} {op} {b} = {value}"


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def far_wrong_value(a, op, b, true, forbidden):
    """A wrong result at least one tier from the truth: the operator slip (x <-> +, - -> +, / -> x, % of -> +) when it
    gives a clean number of the same format, else true +- max(2, 10% of |true|) (rounded to the format), then x2, /2.
    Never the true value, never a forbidden text, never within +-1 of the truth. None if nothing works."""
    t, fa, fb = num(true), num(a), num(b)
    if t is None:
        return None
    cands = []
    if fa is not None and fb is not None:
        slip = {"×": fa + fb, "+": fa * fb, "−": fa + fb, "÷": fa * fb, "% of": fa + fb}.get(op)
        if slip is not None:
            cands.append(slip)
    d = max(2.0, abs(t) * 0.1)
    cands += [t + d, t - d, 2 * t, t / 2]
    for v in cands:
        sv = fmt_like(v, true)
        if num(sv) is None or sv == true or sv in forbidden or sv.startswith("-") or abs(num(sv) - t) < 1.5:
            continue
        return sv
    return None


def placebo_lines(x, k):
    """k lines of the item's own text (its question sentences, in order, cycled), as student.placebo_lines does for
    sources with a statement block (gsm8k_v2 has none, so the sentences are split here with the same splitter)."""
    q = head(x["prompt"])
    q = q[len("Question: "):] if q.startswith("Question: ") else q
    sents = [t.strip() for t in st._pl_sentences(q) if t.strip()]
    if not sents:
        raise ValueError(f"{x['item_id']}: no sentence")
    return [sents[i % len(sents)] for i in range(k)]


def remaining_path(steps, k):
    """Longest dependency chain among the steps NOT shown (indices >= k), in steps: the serial work left after the first
    k results are written. Links come from a_from / b_from (0-based step indices). k = 0 gives the chain's full
    critical path; k = n gives 0."""
    memo = {}
    def rp(j):
        if j in memo:
            return memo[j]
        best = 0
        for key in ("a_from", "b_from"):
            src = steps[j].get(key)
            if src is not None and k <= int(src) < j:
                best = max(best, rp(int(src)))
        memo[j] = best + 1
        return memo[j]
    return max((rp(j) for j in range(k, len(steps))), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    steps = {index_of(it["item_id"]): it for it in read(a.steps)}
    xs = read(a.x)
    os.makedirs(a.out, exist_ok=True)

    items = []          # (x item, steps item, texts, values)
    mism = 0
    for x in xs:
        s = steps.get(index_of(x["item_id"]))
        if s is None:
            raise SystemExit(f"{x['item_id']}: no steps item")
        if head(x["prompt"]) != head(s["prompt"]):
            mism += 1
            continue
        ws = st.worked_steps(s) or []
        if len(ws) < 2:
            continue
        items.append((x, s, [w["text"] for w in ws], st.slot_values(s)))
    if mism:
        raise SystemExit(f"{mism} items whose question text differs between the x file and the steps file")

    # partners: another item with at least n - 1 steps and a different chain (fixed by seed and item)
    need = {s["item_id"]: len(t) - 1 for _, s, t, _ in items}
    partners = st.teacher_partners([s for _, s, _, _ in items], a.seed, need)
    by_id = {s["item_id"]: (x, s, t, v) for x, s, t, v in items}

    sets: dict[str, list] = {"hint_plain": [], "hint_full": []}
    for r in R_TRUE:
        sets[f"hint_true_r{r}"] = []
    for r in R_CTRL:
        for kind in ("other", "wrongres", "wrongres_near"):
            sets[f"hint_{kind}_r{r}"] = []
    for r in R_PLACEBO:
        sets[f"hint_placebo_r{r}"] = []
    for r in R_PLAN:
        sets[f"hint_plan_r{r}"] = []
    skipped_wrong = 0

    def row(x, kind, n, r, k, lines, extra=None):
        opts = option_texts(x["prompt"])
        d = {kk: x[kk] for kk in ("item_id", "source", "labels", "label_order", "gold_label", "gold_probs", "option_kinds", "depth") if kk in x}
        d.update({"type": x.get("type", "choice"), "group": f"hint/{kind}/n{min(n, 7)}/r{r}", "n": n, "r": r, "k": k, "kind": kind,
                  "prompt": st.insert_chain(x["prompt"], st.chain_block(lines)) if lines else x["prompt"], "shown": lines,
                  "r_path": remaining_path(cur_steps, k if kind != "plain" else 0), "path": remaining_path(cur_steps, 0)})
        shown_vals = [ln.rpartition("=")[2].strip() for ln in lines]
        d["shown_equals_option"] = any(v in opts for v in shown_vals)
        gold_text = option_map(x["prompt"]).get(x["gold_label"])
        d["shown_equals_gold"] = gold_text is not None and gold_text in shown_vals
        if extra:
            d.update(extra)
        return d

    cur_steps = None
    for x, s, texts, values in items:
        n = len(texts)
        steps = st.worked_steps(s)
        cur_steps = steps
        sets["hint_plain"].append(row(x, "plain", n, n, 0, []))
        sets["hint_full"].append(row(x, "full", n, 0, n, texts))
        for r in R_TRUE:
            k = n - r
            if k >= 1:
                sets[f"hint_true_r{r}"].append(row(x, "true", n, r, k, texts[:k]))
        p = partners.get(s["item_id"])
        px = by_id.get(p) if p else None
        opts = set(option_texts(x["prompt"]))
        for r in R_PLACEBO:
            k = n - r
            if k >= 1:
                sets[f"hint_placebo_r{r}"].append(row(x, "placebo", n, r, k, placebo_lines(x, k)))
        for r in R_PLAN:
            k = n - r
            if k >= 1:
                sets[f"hint_plan_r{r}"].append(row(x, "plan", n, r, k, [plan_line(w) for w in steps[:k]]))
        for r in R_CTRL:
            k = n - r
            if k < 1:
                continue
            if px is not None and len(px[2]) >= k:
                sets[f"hint_other_r{r}"].append(row(x, "other", n, r, k, px[2][:k], {"partner": p}))
            # near-miss control (secondary): true operands, result off by a small amount
            lines, ok = [], True
            for text, true in zip(texts[:k], values[:k]):
                w = wrong_value(true, opts | set(values))
                if w is None:
                    ok = False
                    break
                lines.append(wrong_line(text, true, w))
            if ok:
                sets[f"hint_wrongres_near_r{r}"].append(row(x, "wrongres_near", n, r, k, lines))
            else:
                skipped_wrong += 1
            # consistent far-wrong chain (the M3 control): wrong results propagate into linked operands
            lines, wrongs, ok = [], [], True
            for i, w in enumerate(steps[:k]):
                a_, b_ = str(w["a"]), str(w["b"])
                if w.get("a_from") is not None and 0 <= int(w["a_from"]) < len(wrongs):
                    a_ = wrongs[int(w["a_from"])]
                if w.get("b_from") is not None and 0 <= int(w["b_from"]) < len(wrongs):
                    b_ = wrongs[int(w["b_from"])]
                wv = far_wrong_value(a_, w["op"], b_, str(w["value"]).strip(), opts | set(values) | set(wrongs))
                if wv is None:
                    ok = False
                    break
                wrongs.append(wv)
                lines.append(line_text(a_, w["op"], b_, wv))
            if ok:
                sets[f"hint_wrongres_r{r}"].append(row(x, "wrongres", n, r, k, lines, {"wrong_values": wrongs}))
            else:
                skipped_wrong += 1

    manifest = {"seed": a.seed, "items_n_ge_2": len(items), "skipped_wrongres": skipped_wrong, "sets": {}}
    for name, rows in sets.items():
        path = os.path.join(a.out, name + ".jsonl")
        with open(path, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        manifest["sets"][name] = {"n": len(rows), "md5": md5,
                                  "shown_equals_option": sum(1 for d in rows if d["shown_equals_option"])}
        print(f"{name:20s} n {len(rows):5d}  md5 {md5[:8]}  shown=option {manifest['sets'][name]['shown_equals_option']}")
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"items with n >= 2: {len(items)}; wrong-result rows skipped: {skipped_wrong}")


if __name__ == "__main__":
    main()
