# Serial steps in one forward pass (k-hop chains)

**Question.** How many dependent arithmetic steps can a model compute in one forward pass, and does training on many
such chains carry over to word problems?

**Items.** Each question lists five interleaved chains of variable assignments that share a root value and a sequence
of operation types and differ only in their constants; it asks for the variable k hops from the root, and the five
options are the five chains' values at that depth (chance 20%). The test set has 150 items for each k from 1 to 6 and
100 for k = 7, 8 (`data/eval/khop/khop_ca_fwd.jsonl`; the other files are variants of wording and direction).
Training rows cover k = 1 … 6, so 7 and 8 hops are extrapolation.

**Models.** Answer-only GSM8K training evaluated on the chain sets, and the same training with 27,000 distinct chain
examples added (`data/eval/khop2/train_kd27.jsonl.gz`), five seeds each, plus
smaller doses and curricula of the same files (one to five seeds each); the arguments of each run are in `configs/`.
`data/eval/hint/` holds the GSM8K hint sets with serial-depth annotations used for the transfer test (two serial steps
left to compute).

**Files.**
- `code/generators/build_khop.py` (evaluation sets and the base training files), `build_khop2.py` (the 27,000-chain
  file), `build_khop3.py` / `build_khop4.py` (the curriculum training directories, `data/student_khop3_k1x/` and
  `data/eval/khop3`, `khop4`), with tests in `code/tests/`. The run tags are listed in the repository README (Run names).
- `code/trainer/` is the version of `train_student.py` / `train_arm.sbatch` used for chain training: it adds
  `--extra-train`, which mixes additional multiple-choice files into the training set.
- `data/md5.txt` lists the md5 of every data file.
- In the `build_report.json` files the local tokenizer path is replaced by `<path>`; `data/md5.txt` gives the md5 of the
  files as shipped.
