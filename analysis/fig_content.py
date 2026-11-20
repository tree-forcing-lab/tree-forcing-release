"""Figure 2: K&K-Deep accuracy of each training variant minus answer-only training, paired by seed at each method's
selected epoch (bars = mean, dots = seeds), grouped by what the variant changes; * = significant (both intervals
exclude zero). Reads $OUT/fig_content.json (default analysis/), writes $OUT/content.pdf (default paper/figures/).
 OUT=reproduced python3 analysis/fig_content.py"""
import json
import os
from pathlib import Path

import matplotlib; matplotlib.use("pdf"); import matplotlib.pyplot as plt; import numpy as np  # noqa: E401,E402
ROOT = Path(__file__).resolve().parents[1]
OUT = os.environ.get("OUT")
src = Path(OUT) / "fig_content.json" if OUT else ROOT / "analysis/fig_content.json"
dst = Path(OUT) / "content.pdf" if OUT else ROOT / "paper/figures/content.pdf"
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8})
data = json.load(open(src))
SIG = {"Tree Forcing"}
groups = [("", ["Tree Forcing"]),
          ("Content", ["Permuted nodes", "Verification control", "Read nodes only"]),
          ("Density", ["Three nodes per puzzle", "Eight nodes per puzzle"]),
          ("Format", ["Proof generation", "Proof gen., tree batching", "DAIS-style"])]
rows = [(g, l) for g, labs in groups for l in labs]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
y = len(rows) + len(groups) - 1; yt, yl = [], []; prev = None
for g, lab in rows:
    if prev is not None and g != prev: y -= 1
    prev = g
    v = data[lab]; m = float(np.mean(v))
    ax.barh(y, m, color="#0072B2" if lab == "Tree Forcing" else "#E69F00", height=0.62)
    ax.scatter(v, [y] * len(v), s=7, color="black", zorder=3, linewidths=0)
    ax.text(max(max(v), 0) + 0.3, y, f"{m:+.1f}" + ("*" if lab in SIG else ""), va="center", fontsize=7)
    yt.append(y); yl.append(lab + (" (3 seeds)" if lab == "Read nodes only" else "")); y -= 1
for g, labs in groups:
    if not g: continue
    ys = [yt[[r[1] for r in rows].index(l)] for l in labs]
    ax.text(8.4, (max(ys) + min(ys)) / 2, g, va="center", ha="right", fontsize=7.5, color="#555555", style="italic")
ax.axvline(0, color="#333333", lw=0.7)
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlim(-4.2, 8.5)
ax.set_xlabel("Gain over answer-only training on K&K-Deep (points)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(dst)
