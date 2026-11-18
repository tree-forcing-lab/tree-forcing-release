# Mechanism experiments (Section 5 and the appendix "Mechanism Experiments")

Materials of the two experiment groups the paper uses in Section 5, as prepared by the second author: code (patches
against the base commit 994ff50, which is in this repository's history of `src/` and `scripts/`), generated
evaluation data with md5 lists, design notes and result tables recomputed from the prediction files.

| Folder | Experiment | Jobs |
|---|---|---|
| `A_hint/` | Hint test: true intermediate values of GSM8K solutions written into the prompt | 500-513 (pilot), 520-533 (the test reported in the paper) |
| `B_khop/` | Serial steps in one forward pass: synthetic arithmetic chains (k hops) and the GSM8K cost of chain training | k-hop rounds, see `B_khop/README.md` |

- In `A_hint/README.md`, `r` in `hint_true_r<r>` is the number of steps LEFT to compute: the file shows the first
  `n - r` true steps of an `n`-step solution.
- Large `.jsonl` files are gzip-compressed (`*.jsonl.gz`); `analysis/prepare_mechanism_inputs.sh` unpacks what the
  paper's scripts read (`analysis/mechanism_check.py`, `analysis/mechanism_figure.py`).
- A third group of October experiments (editing values in the prompt) is not used by the paper and not included.
- Release status: the second author's consent to publish these materials is pending; until then this folder is part
  of a private repository.
