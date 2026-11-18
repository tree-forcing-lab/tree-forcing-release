"""Readouts of batch 26 (README_26 §2: K&K DAIS-style arm vs the Tree / A2 control pool) and batch 27 (README_27 §4:
FOLIO, report-only), both registered before any of their results. Selection: per arm, best seed-mean val epoch, ties
to fewer. Statistic: ledger.py (per-seed items for K&K mid-epoch rows, as the registered b21 / b23 readouts;
paired_stats common items otherwise).  python3 analysis/b26_b27_readout.py"""
import collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import ledger as L
M = "Qwen3.5-4B"
L.ARMS["DAIS-style"] = L.kk("GFPS", "c6lr3chds", range(13, 19))
L.ARMS["DAIS-style s16-18"] = L.kk("GFPS", "c6lr3chds", (16, 17, 18))
print("# Batch 26: DAIS-style (GFPS + --tree-scaffold 1e9)")
for a in ("Tree", "DAIS-style", "A2"):
    e, _ = L.pick(L.ARMS[a]); print(f"  {a}: selected epoch {e}")
print("\n## PRIMARY (registered): Tree − DAIS-style, kkdeep, 6 seeds; gate mean >= +1.5, >= 5/6 > 0, 2L lower > 0\n" + L.HEAD)
L.compare("Tree", "DAIS-style", ["kkdeep"], L.ARMS)
print("\n## robustness (report)\n" + L.HEAD)
L.compare("Tree", "DAIS-style", ["kkdeep"], L.ARMS, items="common")
L.compare("Tree", "DAIS-style", ["kkdeep"], L.ARMS, select=False, label=" (end, 2,000)", items="common")
print("\n## also reported\n" + L.HEAD)
L.compare("DAIS-style", "A2", ["kkdeep"], L.ARMS)
L.compare("DAIS-style", "A2", ["kkdeep"], L.ARMS, select=False, label=" (end, 2,000)", items="common")
for x in ("kk",):
    L.compare("Tree", "DAIS-style", [x], L.ARMS); L.compare("DAIS-style", "A2", [x], L.ARMS)
L.compare("Tree s16-18", "DAIS-style s16-18", ["xkk"], L.ARMS)
L.compare("DAIS-style s16-18", "A2 s16-18", ["xkk"], L.ARMS)
for x in ("bbh", "gsm8k", "folio", "musr"):
    L.compare("Tree", "DAIS-style", [x], L.ARMS, select=False, label=" (end)", items="common")
    L.compare("DAIS-style", "A2", [x], L.ARMS, select=False, label=" (end)", items="common")
e, E = L.pick(L.ARMS["DAIS-style"])
print(f"\nDAIS-style ECE kkdeep (selected) {np.mean([L.ece(n, 'kkdeep', e, E[s]) for s, n in L.ARMS['DAIS-style'].items()]):.1f}; "
      f"bbh (end) {np.mean([L.ece(n, 'bbh') for n in L.ARMS['DAIS-style'].values()]):.1f}")

print("\n\n# Batch 27: FOLIO (report-only)")
F = {"FOLIO Tree": {s: f"GFPS-c6lr3chfo-{M}-student_folio_tf-s{s}" for s in range(13, 19)},
     "FOLIO A2": {s: f"A2-c12lr3chfo-{M}-student_folio_tf-s{s}" for s in range(13, 19)}}
for a in F:
    e, _ = L.pick(F[a]); print(f"  {a}: selected epoch {e}")
print("\n## primary report: ftest (132), plus ftall / folio / kk / kkdeep / xkk at the selected epochs\n" + L.HEAD)
for x in ("ftest", "ftall", "folio", "kk", "kkdeep", "xkk"):
    L.compare("FOLIO Tree", "FOLIO A2", [x], F, items="common")
for x in ("bbh",):
    L.compare("FOLIO Tree", "FOLIO A2", [x], F, select=False, label=" (end)", items="common")
print("\n## ftest accuracy by gold label (selected epochs; mean over seeds, run SD)")
for a in F:
    e, E = L.pick(F[a]); by = collections.defaultdict(list); unc = []
    for s, n in F[a].items():
        rr = L.rows(n, "ftest", e, E[s]); c = collections.defaultdict(list)
        for r in rr:
            pred = r["labels"][int(np.argmax(r["probs"]))]; c[r["gold_label"]].append(pred == r["gold_label"])
        for k, v in c.items(): by[k].append(100 * np.mean(v))
        unc.append(100 * np.mean([r["labels"][int(np.argmax(r["probs"]))] == "C" for r in rr]))
    print(f"  {a}: " + ", ".join(f"{k} {np.mean(v):.1f} ({np.std(v, ddof=1):.1f}, n={len(c[k])})" for k, v in sorted(by.items()))
          + f"; share predicted C (Uncertain) {np.mean(unc):.1f}")
print("\n## Tree arm node probes (preds_subq, end of training): accuracy by wording and atom truth")
for s, n in F["FOLIO Tree"].items():
    p = L.R / n / "preds_subq.jsonl"
    if not p.exists(): print(f"  s{s}: no preds_subq"); continue
    g = collections.defaultdict(list)
    for line in open(p):
        r = json.loads(line)
        if r.get("correct") is None: continue
        q = r.get("question", ""); wording = "false" if "Is it false that" in q else "true" if "Is it true that" in q else "other"
        g[(wording, r.get("target"))].append(r["correct"])
    print(f"  s{s}: overall {100 * np.mean([c for v in g.values() for c in v]):.1f} | " + ", ".join(f"{k[0]}/{k[1]} {100 * np.mean(v):.0f} (n={len(v)})" for k, v in sorted(g.items(), key=str)))
