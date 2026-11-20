# Hint test

**Question.** Can a model that answers in one forward pass use intermediate results when they are written into its
prompt?

**Items.** The GSM8K test problems whose worked solution has at least two steps (1,129 problems), where a step is one
binary operation from the calculator notes of the Socratic solution. Options are the cue-free ones (the four wrong
options are answers of other problems).

**Conditions** (`data/hint/hint_<condition>.jsonl`; a "Worked steps:" block sits between the question and the options):

| Condition | Block |
|---|---|
| `plain` | none |
| `true_r<r>` | the first n − r true steps, so r steps are left to compute (r = 1 … 6) |
| `full` | every step, including the answer |
| `placebo_r<r>` | as many of the problem's own sentences as `true_r<r>` has lines |
| `other_r<r>` | the first steps of another problem's solution |
| `wrongres_r<r>` | a consistently wrong chain (each result off by at least 1.5, later operands follow) |
| `wrongres_near_r<r>` | the true operands with results changed by a small amount, without propagation |

**Models.** Three runs per method trained on the GSM8K training set of the paper (answer-only for one and two
epochs, Tree Forcing and the permuted-nodes control for two epochs); the training arguments of each run are in
`configs/`. Their predictions on every hint set are in `data/stage2_preds.tgz` (one folder per run).

**Files.**
- `code/builder/build_hint_evals.py` builds the hint sets from `data/source/` (`rebuild.sh`, test in `test_build_hint_evals.py`); `code/check_sets.py` checks them independently; `code/make_examples.py` writes `data/example_prompts.md` (one problem in every condition).
- `data/md5.txt` lists the md5 of every data file.
