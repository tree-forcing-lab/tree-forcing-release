#!/usr/bin/env python3
"""Write data/example_prompts.md: the real prompt of every one of the 18 sets for one item.

 python3 code/make_examples.py # from the A_hint folder

The item is the shortest-question item with n >= 7 worked steps that appears in all 18 sets and whose shown values
never equal an option (so every r = 1..6 exists for it). The prompts are copied byte for byte from data/hint.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "data", "hint")
SETS = ["hint_plain"] + [f"hint_true_r{r}" for r in range(1, 7)] + ["hint_full"] + \
       [f"hint_placebo_r{r}" for r in range(1, 5)] + [f"hint_{k}_r{r}" for k in ("other", "wrongres", "wrongres_near") for r in (1, 2)]
rows = {s: {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(H, s + ".jsonl"))} for s in SETS}
cands = [i for i, d in rows["hint_plain"].items() if d["n"] >= 7 and all(i in rows[s] for s in SETS)
         and not any(rows[s][i]["shown_equals_option"] for s in SETS if s.startswith("hint_true"))]
iid = min(cands, key=lambda i: (len(rows["hint_plain"][i]["prompt"]), i))
d0 = rows["hint_plain"][iid]
out = [f"# One item in all 18 conditions: {iid}", "",
       f"n = {d0['n']} worked steps (atomic operations), depth field = {d0['depth']}, gold = {d0['gold_label']}.",
       "In every set, k = number of lines in the block and r = n - k. For true_r<r>, the block holds the FIRST k = n - r",
       "true steps, so r is the number of steps the model must still compute. Prompts are copied byte for byte.", ""]
for s in SETS:
    d = rows[s][iid]
    extra = "".join(f", {k} = {d[k]}" for k in ("partner", "wrong_values") if k in d)
    out += [f"## {s} (n {d['n']}, k {d['k']}, r {d['r']}{extra})", "", "```", d["prompt"], "```", ""]
open(os.path.join(ROOT, "data", "example_prompts.md"), "w").write("\n".join(out))
print(iid, len(cands), "candidates")
