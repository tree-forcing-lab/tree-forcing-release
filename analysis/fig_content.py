"""Figure: K&K-Deep accuracy of each training variant minus answer-only training, paired by seed at each method's
selected epoch (bars = mean, dots = seeds). Data: analysis/fig_content.json.  ~/.venvs/hf/bin/python analysis/fig_content.py"""
import json
import matplotlib; matplotlib.use("pdf"); import matplotlib.pyplot as plt; import numpy as np
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8})
data = json.load(open("analysis/fig_content.json"))
order = sorted(data, key=lambda k: np.mean(data[k]))
fig, ax = plt.subplots(figsize=(5.4, 2.8))
for y, lab in enumerate(order):
    v = data[lab]; m = np.mean(v)
    ax.barh(y, m, color="#0072B2" if lab == "Tree Forcing" else "#E69F00", height=0.6)
    ax.scatter(v, [y] * len(v), s=7, color="black", zorder=3, linewidths=0)
    ax.text(max(max(v), 0) + 0.3, y, f"{m:+.1f}", va="center", fontsize=7)
ax.axvline(0, color="#333333", lw=0.7)
ax.set_yticks(range(len(order))); ax.set_yticklabels([l + (" (3 seeds)" if l == "Read nodes only" else "") for l in order])
ax.set_xlabel("Gain over answer-only training on K&K-Deep (points)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("paper/figures/content.pdf")
