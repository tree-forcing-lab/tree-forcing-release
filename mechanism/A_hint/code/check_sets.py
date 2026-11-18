#!/usr/bin/env python3
"""Independent check of the 18 hint sets that jobs 500-513 evaluated (standard library only, no torch).

    python3 code/check_sets.py            # from the A_hint folder; prints a report, exit 1 on any failure

It re-derives every set rule from the two source files and never imports the builder:
  data/source/gsm8k_mc_v2_xprob.jsonl   cue-free GSM8K test items (options are other problems' answers)
  data/source/gsm8k_mc_v2_steps.jsonl   the same test items with their worked steps (atomic binary operations)
Checks:
  1. Item choice: the 1,129 items are exactly the x items whose steps row has >= 2 worked steps, in file order.
  2. r_k: for every row, n = len(worked_steps), k = n - r, len(shown) = k; for true and full rows
     shown == [step text for the first k steps]. plain has k = 0 (r = n), full has k = n (r = 0).
  3. Prompt: every prompt equals the x prompt with "\\n\\nWorked steps:\\nStep 1: ..." inserted before the last
     "\\nOptions:\\n" (student.chain_block + insert_chain at 994ff50); options and gold label unchanged.
  4. Controls: placebo lines are sentences of the item's own question; other lines are the first k lines of the
     partner's chain (partner != item); far-wrong (wrongres) blocks hold no true result of steps 1..k; near-miss
     (wrongres_near) lines keep the true left side and change the result to the first of +1, -1, +2, -2, x2, /2,
     +10 that is no option text and no true step value (so mostly +-1 or +-2, a few larger).
  5. The depth field is not n (reported, since the first author's reading may use depth).
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "source")
HINT = os.path.join(ROOT, "data", "hint")
SETS = ["hint_plain", "hint_full"] + [f"hint_true_r{r}" for r in range(1, 7)] + \
       [f"hint_{k}_r{r}" for k in ("other", "wrongres", "wrongres_near") for r in (1, 2)] + \
       [f"hint_placebo_r{r}" for r in range(1, 5)]
FAIL = []


def read(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
    return cond


def block(lines):
    return "" if not lines else "\n\nWorked steps:\n" + "\n".join(f"Step {i + 1}: {t}" for i, t in enumerate(lines))


def insert(prompt, blk):
    if not blk:
        return prompt
    head, sep, tail = prompt.rpartition("\nOptions:\n")
    return head + blk + sep + tail


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def main():
    xs = read(os.path.join(SRC, "gsm8k_mc_v2_xprob.jsonl"))
    steps = {d["item_id"]: d for d in read(os.path.join(SRC, "gsm8k_mc_v2_steps.jsonl"))}
    x_by = {d["item_id"]: d for d in xs}
    ncount = collections.Counter()
    chosen = []
    for x in xs:
        ws = steps[x["item_id"]].get("worked_steps") or []
        check(x["prompt"].rpartition("\nOptions:\n")[0] == steps[x["item_id"]]["prompt"].rpartition("\nOptions:\n")[0],
              f"{x['item_id']}: question text differs between x and steps files")
        ncount["null" if not ws else min(len(ws), 7)] += 1
        if len(ws) >= 2:
            chosen.append(x["item_id"])
    print(f"x items: {len(xs)}; worked-steps length over them: " +
          ", ".join(f"{k if k != 7 else '7+'}: {ncount[k]}" for k in ["null", 1, 2, 3, 4, 5, 6, 7]))
    print(f"items with n >= 2 (the hint item set): {len(chosen)}")

    texts = {i: [w["text"] for w in steps[i]["worked_steps"]] for i in chosen}
    rows = {s: read(os.path.join(HINT, s + ".jsonl")) for s in SETS}
    check([d["item_id"] for d in rows["hint_plain"]] == chosen, "plain item list != x items with n >= 2 in file order")
    near_tags = collections.Counter()
    depth_ne_n = sum(1 for d in rows["hint_plain"] if d["depth"] != d["n"])
    for s, rs in rows.items():
        kind = s.split("_", 1)[1].rsplit("_r", 1)[0] if "_r" in s[5:] else s[5:]
        bad = collections.Counter()
        for d in rs:
            i = d["item_id"]; x = x_by[i]; n = len(texts[i])
            bad["n"] += d["n"] != n
            bad["k=n-r"] += d["k"] != n - d["r"]
            bad["len(shown)=k"] += len(d["shown"]) != d["k"]
            bad["prompt"] += d["prompt"] != insert(x["prompt"], block(d["shown"]))
            bad["gold"] += d["gold_label"] != x["gold_label"]
            if kind in ("plain", "full", "true"):
                bad["shown=first k steps"] += d["shown"] != texts[i][:d["k"]]
            if kind == "plain":
                bad["plain k=0"] += d["k"] != 0
            if kind == "full":
                bad["full k=n"] += d["k"] != n
            if kind == "placebo":
                q = x["prompt"].rpartition("\nOptions:\n")[0]
                bad["placebo in question"] += any(ln not in q for ln in d["shown"])
            if kind == "other":
                p = d["partner"]
                bad["partner"] += p == i or d["shown"] != texts[p][:d["k"]]
            if kind == "wrongres":
                true_vals = {w["value"] for w in steps[i]["worked_steps"][:d["k"]]}
                bad["far: no true result"] += any(ln.rpartition("=")[2].strip() in true_vals for ln in d["shown"])
            if kind == "wrongres_near":
                for ln, t in zip(d["shown"], texts[i][:d["k"]]):
                    lt, _, vt = t.rpartition("=")
                    lw, _, vw = ln.rpartition("=")
                    t, w = num(vt), num(vw)
                    cands = [("+1", t + 1), ("-1", t - 1), ("+2", t + 2), ("-2", t - 2), ("x2", 2 * t), ("/2", t / 2), ("+10", t + 10)]
                    tag = next((nm for nm, c in cands if w is not None and abs(c - w) < 1e-9), None)
                    bad["near: same lhs, result from the builder's list"] += lt != lw or tag is None
                    near_tags[tag] += 1
        for key, v in bad.items():
            check(v == 0, f"{s}: {v} rows fail '{key}'")
        r_vals = sorted({d["r"] for d in rs})
        print(f"{s:22s} rows {len(rs):5d}  r {r_vals if len(r_vals) < 4 else str(r_vals[0]) + '..' + str(r_vals[-1])}  "
              f"k {min(d['k'] for d in rs)}..{max(d['k'] for d in rs)}  checks {'ok' if not any(bad.values()) else dict(bad)}")
    print("near-miss line changes (first of +1, -1, +2, -2, x2, /2, +10 that is no option and no true value): " +
          ", ".join(f"{k} {v}" for k, v in near_tags.most_common()))
    print(f"plain rows whose depth field != n: {depth_ne_n} of {len(rows['hint_plain'])} (depth is the socratic step count, "
          f"n counts atomic operations)")
    ex = rows["hint_true_r2"][0]
    print(f"example: {ex['item_id']} n {ex['n']} depth {ex['depth']} -> hint_true_r2 shows k = {ex['k']} step(s): {ex['shown']}")
    if FAIL:
        print("FAILED:\n  " + "\n  ".join(FAIL[:40]))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
