#!/usr/bin/env python3
"""Print the short key tables of README.md from contrasts.csv and curves.csv (run compute_tables.py first).

    python3 results/readme_tables.py > results/readme_tables.md              # stage 1 (pilot)
    python3 results/readme_tables.py 2 > results/stage2_readme_tables.md     # stage 2 (confirmation)
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = sys.argv[1:] == ["2"]
P = "stage2_" if STAGE2 else ""
MODELS = ["A2-c1", "A2-c2", "GFPQ-c2", "GFPQR-c2"] if STAGE2 else ["base", "A2-c4", "A2-c12", "GFPQ-c4", "GFPQM-c4", "GFPQMR-c4"]
C = {(r["model"], r["contrast"]): r for r in csv.DictReader(open(os.path.join(HERE, P + "contrasts.csv")))}
A = {}
for r in csv.DictReader(open(os.path.join(HERE, P + "acc_per_seed.csv"))):
    A.setdefault((r["model"], r["set"]), []).append(float(r["acc_pts"]))


def f(x, sign=True):
    return "-" if x in ("", None) else (f"{float(x):+.1f}" if sign else f"{float(x):.1f}")


def cell(model, name, full=False):
    r = C[(model, name)]
    s = f(r["mean"])
    per = r["d_per_seed"].split(";")
    if len(per) > 1:
        s += " (" + " / ".join(f(x) for x in per) + ")"
    if full:
        tl = f"[{f(r['two_level_lo'])}, {f(r['two_level_hi'])}]"
        st = f"[{f(r['seed_t_lo'])}, {f(r['seed_t_hi'])}]" if r["seed_t_lo"] else "-"
        return s, st, tl
    return s


print("Table A. Headline contrasts per model (points; mean, then per seed, " + ("s16 / s17 / s18" if STAGE2 else "s12 / s13 / s14") + ").\n")
print("| model | plain acc | true_r1 - plain, all 1,129 | seed-t 95% | two-level 95% | true_r1 - plain, clean 918 | H (clean, n>=4, 440) |")
print("|---|---|---|---|---|---|---|")
for m in MODELS:
    pl = A[(m, "hint_plain")]
    s, st, tl = cell(m, "true_r1 - plain", True)
    print(f"| {m} | {sum(pl) / len(pl):.1f} | {s} | {st} | {tl} | {cell(m, 'true_r1 - plain (clean)')} | {cell(m, 'H: true_r1 - plain (clean, n>=4)')} |")

print("\nTable B. true_r - plain for r = 1..6 (each on the items of that r set; seed mean, two-level 95% interval).\n")
print("| model | " + " | ".join(f"r={r} ({n})" for r, n in zip(range(1, 7), (1129, 877, 576, 332, 158, 75))) + " |")
print("|---|" + "---|" * 6)
for m in MODELS:
    cells = []
    for r in range(1, 7):
        x = C[(m, f"true_r{r} - plain")]
        cells.append(f"{f(x['mean'])} [{f(x['two_level_lo'])}, {f(x['two_level_hi'])}]")
    print(f"| {m} | " + " | ".join(cells) + " |")

print("\nTable C. Fixed-item curve, 332 items with n >= 5 (accuracy, seed mean).\n")
cur = {}
for r in csv.DictReader(open(os.path.join(HERE, P + "curves.csv"))):
    if r["cohort"] == "n>=5" and r["seed"] == "mean":
        cur[(r["model"], r["condition"], r["r"])] = float(r["acc_pts"])
print("| model | plain | r=1 | r=2 | r=3 | r=4 | full |")
print("|---|---|---|---|---|---|---|")
for m in MODELS:
    print(f"| {m} | {cur[(m, 'plain', '')]:.1f} | " + " | ".join(f"{cur[(m, 'true', str(r))]:.1f}" for r in range(1, 5)) +
          f" | {cur[(m, 'full', '0')]:.1f} |")

print("\nTable C2. The same 332 fixed items: true_r - plain and D = true_r1 - true_r4 (seed mean, seed-t 95% for 3-seed arms,")
print("two-level 95%).\n")
print("| model | r=1 - plain | r=2 - plain | r=3 - plain | r=4 - plain | D = r1 - r4 |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    cells = []
    for nm in [f"[n>=5] true_r{r} - plain" for r in range(1, 5)] + ["[n>=5] true_r1 - true_r4 (D)"]:
        x = C[(m, nm)]
        st = f" t[{f(x['seed_t_lo'])}, {f(x['seed_t_hi'])}]" if x["seed_t_lo"] else ""
        cells.append(f"{f(x['mean'])}{st} b[{f(x['two_level_lo'])}, {f(x['two_level_hi'])}]")
    print(f"| {m} | " + " | ".join(cells) + " |")

print("\nTable D. Controls minus plain (each on the items of that control set; seed mean, two-level 95% interval).\n")
names = ["other_r1 - plain", "other_r2 - plain", "wrongres_r1 - plain", "wrongres_r2 - plain", "wrongres_near_r1 - plain",
         "wrongres_near_r2 - plain", "placebo_r1 - plain", "placebo_r2 - plain", "placebo_r3 - plain", "placebo_r4 - plain"]
print("| contrast | " + " | ".join(MODELS) + " |")
print("|---|" + "---|" * len(MODELS))
for nm in names:
    print(f"| {nm} | " + " | ".join(f"{f(C[(m, nm)]['mean'])} [{f(C[(m, nm)]['two_level_lo'])}, {f(C[(m, nm)]['two_level_hi'])}]" for m in MODELS) + " |")

print("\nTable E. True chain minus each control at r = 1 (shared items; seed mean, two-level 95% interval).\n")
print("| contrast | " + " | ".join(MODELS) + " |")
print("|---|" + "---|" * len(MODELS))
for nm in ["true_r1 - other_r1", "true_r1 - wrongres_r1", "true_r1 - wrongres_near_r1", "true_r1 - placebo_r1"]:
    print(f"| {nm} | " + " | ".join(f"{f(C[(m, nm)]['mean'])} [{f(C[(m, nm)]['two_level_lo'])}, {f(C[(m, nm)]['two_level_hi'])}]" for m in MODELS) + " |")
