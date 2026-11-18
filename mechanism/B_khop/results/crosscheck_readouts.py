#!/usr/bin/env python3
"""Cross-check: the per-seed cells in cells_per_seed.csv / contrasts.csv against the October readout JSON files.

    python3 -I crosscheck_readouts.py <dir with khop_readout_20261026.json, khop2_readout_20261027.json,
                                       khop4_readout_20261028.json>

The October readouts (exp/readout/khop_table.py, khop2_table.py, khop4_table.py) were written by other code from the
same tarballs. This script compares every per-seed curve value they stored with the value make_tables.py computed,
and the khop4 per-seed rc differences. Exit 1 on any difference above the CSV rounding (1e-6 per value). Not needed to rebuild the tables.
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1]
mine = {}
for r in csv.DictReader(open(os.path.join(HERE, "cells_per_seed.csv"))):
    if r["subset"] == "all":
        mine[(r["arm"], int(r["seed"]), r["set"], int(r["k"]))] = float(r["acc"]) / 100
bad = n = 0
for f in ("khop_readout_20261026.json", "khop2_readout_20261027.json"):
    j = json.load(open(os.path.join(d, f)))
    for arm, sets in j["curves"].items():
        for s, seeds in sets.items():
            for seed, ks in seeds.items():
                for k, v in ks.items():
                    if v is None:
                        continue
                    m = mine.get((arm, int(seed), s, int(k)))
                    n += 1
                    if m is None or abs(m - v) > 1e-6:      # cells_per_seed.csv rounds to 1e-4 percent
                        bad += 1
                        print("DIFF", f, arm, seed, s, k, v, m)
print(f"curve cells compared: {n}, differences: {bad}")
# khop4 per-seed rc contrasts
j = json.load(open(os.path.join(d, "khop4_readout_20261028.json")))
g = {}
for r in csv.DictReader(open(os.path.join(HERE, "gsm_per_seed.csv"))):
    g[(r["arm"], int(r["seed"]))] = r
m4 = 0
for rc, block in j["rc"].items():
    for name, (a, b) in (("Dorig", ("CUR-R", "A2orig")), ("Dre", ("CUR-R", "A2re")), ("C", ("CUR-R", "K1X-R"))):
        for seed, v in block[name].items():
            x = (float(g[(a, int(seed))][f"rc{rc}"]) - float(g[(b, int(seed))][f"rc{rc}"])) / 100
            m4 += 1
            if abs(x - v) > 1e-5:          # two values each rounded to 1e-6
                bad += 1
                print("DIFF khop4", rc, name, seed, v, x)
print(f"khop4 rc per-seed differences compared: {m4}; total differences: {bad}")
sys.exit(1 if bad else 0)
