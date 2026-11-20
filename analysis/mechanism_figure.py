"""Figure for the mechanism section: (a) one-pass accuracy on synthetic arithmetic chains by hop count; (b) GSM8K
accuracy on the 332 test problems with at least five steps as true intermediate steps are written into the prompt
(the hint test, seeds 16-18). python3 analysis/mechanism_figure.py SCRATCH"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCR = Path(sys.argv[1])
OUT = Path(__import__("os").environ.get("FIG_OUT", ROOT / "paper/figures/mechanism.pdf"))
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def read(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def corr(rows):
    return {r["item_id"]: int(r["labels"][int(np.argmax(r["probs"]))] == r["gold_label"]) for r in rows}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

# (a) serial budget
curve = json.load(open(SCR / "khop_curve.json"))
ks = list(range(1, 9))
style = {"untrained (750)": ("Untrained", OK["grey"], ":", "x"),
         "answer-only A2Z (751-755)": ("Answer-only", OK["blue"], "-", "o"),
         "27k chains D27 (875-879)": ("+27k chain examples", OK["verm"], "--", "s")}
for key, (lab, col, ls, mk) in style.items():
    ax1.plot(ks, curve[key], ls, color=col, marker=mk, ms=3.5, lw=1.3, label=lab)
ax1.axhline(20, color=OK["black"], lw=0.7, ls=(0, (2, 2)))
ax1.text(1.05, 21.5, "chance", fontsize=7)
ax1.axvspan(6.5, 8.4, color="#eeeeee", zorder=0)
ax1.text(6.6, 94, "not trained", fontsize=7)
ax1.set_xlim(0.7, 8.4); ax1.set_ylim(0, 103); ax1.set_xticks(ks)
ax1.set_xlabel("Serial steps (hops) in the chain"); ax1.set_ylabel("One-pass accuracy (%)")
ax1.legend(frameon=False, fontsize=7, loc="center right")
ax1.text(-0.2, 1.02, "(a)", transform=ax1.transAxes, fontsize=9)

# (b) hint test, fixed 332 items with n >= 5
meta = {s: {r["item_id"]: r for r in read(SCR / "hintA" / f"hint_{s}.jsonl")} for s in ("plain", "true_r4")}
items = sorted(i for i in meta["true_r4"] if meta["plain"][i]["n"] >= 5)
ARMS = {"Answer-only, 1 epoch": ([520, 525, 530], OK["blue"], "-", "o"),
        "Answer-only, 2 epochs": ([521, 526, 531], OK["sky"], "-.", "^"),
        "Tree Forcing, 2 epochs": ([522, 527, 532], OK["verm"], "--", "s"),
        "Permuted nodes, 2 epochs": ([523, 528, 533], OK["purple"], ":", "D")}
xs = ["plain", "true_r4", "true_r3", "true_r2", "true_r1", "full"]
xlab = ["none", "4", "3", "2", "1", "0"]
for lab, (jobs, col, ls, mk) in ARMS.items():
    ys = [100 * st.mean(np.mean([corr(read(SCR / "s2" / str(j) / f"preds_hint_{s}.jsonl"))[i] for i in items]) for j in jobs)
          for s in xs]
    ax2.plot(range(len(xs)), ys, ls, color=col, marker=mk, ms=3.5, lw=1.3, label=lab)
ax2.set_xticks(range(len(xs))); ax2.set_xticklabels(xlab)
ax2.set_xlabel("Steps left to compute (true earlier steps shown)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(30, 103)
ax2.legend(frameon=False, fontsize=6.5, loc="upper left")
ax2.text(-0.2, 1.02, "(b)", transform=ax2.transAxes, fontsize=9)
fig.tight_layout(w_pad=2.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(OUT)
