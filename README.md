# Tree Forcing: Training Single-Pass Decision Models on the Intermediate Conclusions of Verified Reasoning Trees

Code, data, run configurations and analysis scripts for the paper. Single-pass decision models read their answer from
the probability of each option after one forward pass. Tree Forcing turns each intermediate conclusion of a verified
reasoning tree into an extra yes/no question about the problem, answered with the same single-pass readout as the
final decision; at test time the model sees only the final question. The paper evaluates it on Knights-and-Knaves
puzzles, GSM8K, logic grid puzzles and FOLIO with Qwen3.5-4B (and Qwen3.5-9B on Knights-and-Knaves).

Paper: arXiv link to be added. Run outputs (predictions and metrics of every cited run): Zenodo DOI to be added.

## Layout

| Path | Content |
|---|---|
| `src/cotdistill/` | Training and evaluation library (single-pass readout, Tree Forcing batches, FSDP, metrics, puzzle generators, Zebra solver) |
| `scripts/` | `train_student.py` (training and evaluation), `eval_student.py`, every data builder the paper uses, `paired_stats.py`, `unpack_data.py` |
| `teacher_cache/solver_kk_v3f/` | The deterministic Knights-and-Knaves solver whose proofs are the reasoning trees |
| `hpc/` | `train_arm.sbatch` (one job = one training run; maps an arm name to trainer flags) and `single_gpu.patch` (applied to every 4B job, see below) |
| `configs/env/` | The job env file of each cited run (243 of the 259; the other 16 were submitted from the second author's job sets and are described by `configs/run_args/`) |
| `configs/run_args/` | The exact trainer arguments of every cited run (259 files, from each run's `metrics.json`) |
| `configs/chtc/` | HTCondor submit and DAG templates, the job wrapper and the container definitions |
| `data/` | The paper's training and evaluation sets that may be redistributed, gzip-compressed, with `MANIFEST.json` (md5 of each uncompressed file, line count, license) |
| `analysis/` | Scripts that recompute every number, table and figure of the paper from the run outputs; `run_manifest.csv` maps each paper element to its script and runs |
| `mechanism/` | Materials of the mechanism experiments of Section 5 (hint test, k-hop chains) |
| `paper/` | LaTeX source of the paper |

## Environment

All training ran in one container image built from `configs/chtc/cotd.def` and `configs/chtc/cotd_4b.def`
(Python 3.11.17, torch 2.11.0+cu128, CUDA 12.8, transformers 5.17.0, flash-linear-attention 0.5.2, fla-core 0.5.2,
triton 3.6.0, accelerate 1.15.0; full list in `configs/chtc/pip_freeze_cotd_4b.txt`). GPU drivers supporting CUDA 12.8
or later. Data builders and analysis scripts need only the Python standard library and numpy, except where noted
(`z3-solver` for FOLIO, `pandas` / `pyarrow` for parquet sources, `tokenizers` for token statistics, `matplotlib` for
figures).

Models: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Hugging Face).

## Data

```bash
python3 scripts/unpack_data.py        # data/**/*.jsonl.gz -> data/**/*.jsonl, md5-checked against data/MANIFEST.json
```

Rebuilding a dataset (run from the repository root with `src/` importable, e.g. `PYTHONPATH=src`):

| Dataset | Command | md5 of `train.jsonl` / `val.jsonl` (or the file) |
|---|---|---|
| Knights-and-Knaves trees (`data/student_kk_tf`) | `python3 scripts/build_kk_tf.py --no-tokens` | `e53d7f6d8e36351da60d34c3288c132b` / `8d352af42e529c20af16e145165bfcc5` |
| K&K with written proofs (`student_kk_tf_rat`, `_rat_nonodes`) | `python3 scripts/build_kk_rationale.py` | see `data/MANIFEST.json` |
| Three / eight nodes per puzzle (`student_kk_tf_n3`, `_n8`) | `python3 scripts/build_kk_subsample.py --k 3` (and `--k 8`) | see `data/MANIFEST.json` |
| Verification control (`student_kk_tf_vrf`) | `python3 scripts/build_kk_verify.py` | see `data/MANIFEST.json` |
| K&K-Deep / K&K diagnostic | `python3 scripts/build_kk_deep.py`, `python3 scripts/build_diag.py` | see `data/MANIFEST.json` |
| K&K-External (`data/eval/xkk.jsonl`) | `python3 scripts/build_eval_xkk.py` (downloads `K-and-K/knights-and-knaves`) | see `data/MANIFEST.json` |
| GSM8K trees (`student_gsm8k_tf_vt`) | `build_gsm8k_socratic.py`, then `build_gsm8k_tf.py`, `build_gsm8k_ops.py`, `build_gsm8k_si.py`, `build_gsm8k_vt.py` (GSM8K parquet under `data/raw/gsm8k`) | see `data/MANIFEST.json` |
| GSM8K with rationales (`student_gsm8k_tf_vt_rat`) | `python3 scripts/build_gsm8k_rationale.py` | see `data/MANIFEST.json` |
| GSM8K / GSM1K test sets | `python3 scripts/build_math_evals.py`, `python3 scripts/build_mc_xprob.py`, `python3 scripts/build_val_xprob.py` | GSM1K files (not shipped): `df2f83837642b96fd6db105dc8ebee22`, `f3d35073fcf474911892d1b2210f3c75` |
| Logic grid (Zebra) cells (`student_zebra_cell_tf`, `zebra_cell_in/ex`) | `python3 scripts/build_zebra_cell_tf.py` | see `data/MANIFEST.json` |
| FOLIO (`student_folio_tf`, `folio_tf_test*`) | `python3 scripts/build_folio_tf.py` (needs `data/raw/folio` from `yale-nlp/FOLIO` and `z3-solver`) | `2aa4a729a7bf05731863220fbb8a226a` / `016deb53467a435ba5a056e3edd3c4eb`; test `c64ec9c358fc6bfaaf9a9112003a4e85`, all `1e7c797c6188cce4a4f0a3fd07a9c7d4` |
| Held-out sets (BIG-Bench Hard, BIG-Bench Extra Hard, FOLIO, MuSR, ShARC, ZebraLogic) | `python3 scripts/build_items.py`, `python3 scripts/build_breadth.py`, `python3 scripts/build_eval_zebra.py` | not shipped: BBH `b56e0e7b47dc2ded9410fd6d58fa0718`, FOLIO `7968092b6f169828ab9ee06c2795b693`, ShARC `f8bae820a8f995c10ab4067ffcff9908`, ZebraLogic `f6c2fa2381a75e4cdb52544c6d9068d8` |

Rebuilding `student_kk_tf` exactly also needs its deduplication inputs, which are shipped:
`data/bench/train_kk.jsonl` and the Knights-and-Knaves records of an earlier mixture (`data/student_v3sl/*.jsonl`).

## Training

One run, locally on one GPU (the 4B recipe; `ARMS=GFPS` is Tree Forcing, `ARMS=A2` answer-only training):

```bash
patch -p1 < hpc/single_gpu.patch                  # the one-GPU FSDP path all 4B runs used
set -a; . configs/env/GFPS-c6lr3chv-Qwen3.5-4B-student_kk_tf-s16.env; set +a
export REPO=. NGPU=1 MODEL=/path/to/Qwen3.5-4B   # the cluster jobs set NGPU=1 at submission; REPO = this checkout
bash hpc/train_arm.sbatch                         # writes runs/<run name>/ (metrics.json, preds_*.jsonl)
```

`DRY_RUN=1 bash hpc/train_arm.sbatch` prints the trainer command line, which should match the run's
`configs/run_args/<run name>.json`. On an HTCondor cluster, `configs/chtc/train28_4b.sub` (4B, one GPU) and
`configs/chtc/train25_9b.sub` (9B, two GPUs) are the submit templates and `configs/chtc/make_dags_28.sh` builds the
DAG from a job list. A 4B Tree Forcing run on Knights-and-Knaves takes about 12 GPU hours on an NVIDIA L40S.

## Reproducing the paper's numbers

Download and unpack the run outputs so that `runs_archive/runs/` and `runs_archive/restored_runs/` exist (or point
`RUNS` and `RESTORED` at them), then:

```bash
python3 scripts/unpack_data.py
PYTHONPATH=src python3 analysis/ledger.py > ledger.md          # main comparisons, transfer, GSM8K, per-seed table
python3 analysis/b28_readout.py                                 # node ablations, permuted control, GSM8K schedule
python3 analysis/b26_b27_readout.py                             # DAIS-style control, FOLIO
python3 analysis/b23_readout.py                                 # proof-generation controls
python3 analysis/b25_readout.py                                 # 9B
python3 analysis/analysis_1117.py                               # K&K-Deep splits, BIG-Bench Hard by task, temperature scaling
python3 analysis/checks_1115.py; python3 analysis/checks_1117.py  # matched-length transfer, coupling by size, held-out sets
python3 analysis/fig_content_data.py && python3 analysis/fig_content.py         # Figure 2
bash analysis/prepare_mechanism_inputs.sh /tmp/mech
python3 analysis/mechanism_check.py /tmp/mech && python3 analysis/mechanism_figure.py /tmp/mech   # Section 5, Figure 3
```

`analysis/run_manifest.csv` lists, for each table, figure and in-text number, the script and the runs. Comparisons
pair runs by seed; each method's epoch is the one with the best validation accuracy averaged over its seeds. Four
answer-only GSM8K run folders (seeds 15 to 18) were overwritten by later re-runs of the same configuration; the
paper reports the first completed runs, which the archive holds under `restored_runs/` and the scripts read in place
of `runs/`.

## Licenses

- Code (`src/`, `scripts/`, `hpc/`, `analysis/`, `configs/`, `teacher_cache/`, `tests/`): Apache License 2.0
  (`LICENSE`).
- Data we generated (Knights-and-Knaves, logic grid puzzles, their trees and derived sets): CC BY 4.0.
- Data derived from third-party datasets keeps the source license: GSM8K (MIT), K&K-External from
  `K-and-K/knights-and-knaves` (CC BY-NC-SA 4.0), BIG-Bench Extra Hard (Apache 2.0), MuSR (CC BY 4.0). The license of
  each shipped file is in `data/MANIFEST.json`. Sets derived from GSM1K, BIG-Bench Hard, FOLIO, ShARC and ZebraLogic
  are not shipped; their builders and expected md5s are listed above.
- The LaTeX style files in `paper/` come from the TMLR template (Apache 2.0, `paper/LICENSE.tmlr-style`).

## Citation

To be added with the arXiv version.
