# Section A: hint experiments (jobs 500-513 and 520-533)

This folder holds the code, data and results of the October "hint" experiments (card `CARD_opc_sev6e_v4`).
Every number here was computed again from the preds files. No number was copied from a log.
All paths are relative to this folder.

**Two stages. Read this before you cite a number.**

- **Jobs 500-513 are the PILOT (Stage 1).** Saved 4-epoch weights (A2-c12: 12 epochs), seeds 12-14, eval only. The
  card uses the pilot for go / no-go only and makes no claim from it.
- **Jobs 520-533 are the registered CONFIRMATION (Stage 2).** Fresh training runs at seeds 16-18 (A2 at 1 and 2
  epochs, GFPQ-c2, GFPQR-c2), with the same 18 hint files in `EXTRA_EVALS`. The card's verdict comes from Stage 2.
  See the section "Stage 2: the registered confirmation". Cite confirmed numbers from there.

## The answer first: what r_k means

**r is the number of steps LEFT for the model to compute.** The block shows the first k = n - r true steps.

- n is the number of worked steps of the item. A "step" is one atomic binary operation (for example `13 - 4 = 9`).
  It is not a Socratic sub-question, and it is not the `depth` field. On 331 of the 1,129 items, `depth` and n differ.
- `hint_true_r1` shows steps 1 .. n-1. The model must compute the last step only.
- `hint_true_r4` shows steps 1 .. n-4. The model must compute 4 steps.
- `hint_plain` shows no step (k = 0, r = n). `hint_full` shows all n steps (k = n, r = 0).

Evidence:

1. Builder code, `code/builder/build_hint_evals.py`: `k = n - r`, then `row(..., texts[:k])` for every r in 1..6, and only
   `if k >= 1`. That is why the sets shrink with r (1,129 / 877 / 576 / 332 / 158 / 75 items).
2. Data check, `code/check_sets.py` (standard library, no builder import). For all 14,303 rows: `k = n - r`,
   `len(shown) = k`, and for true and full rows, `shown` = the first k steps of `data/source/gsm8k_mc_v2_steps.jsonl`.
   It also rebuilds every prompt from the source item. Output: `results/check_sets.txt`, "ALL CHECKS PASSED".
3. Example (`gsm8k_v2-test-0`, Janet's ducks, n = 3, depth 2). `hint_true_r2` shows one step, `16 − 3 = 13`. The model
   must still compute `13 − 4 = 9` and `9 × 2 = 18`. `data/example_prompts.md` shows the real prompt of one item
   (n = 9) in all 18 conditions.
4. The block format is the repo's own: `chain_block` and `insert_chain` in
   `code/repo_994ff50/src/cotdistill/student.py` (lines 1557-1572). The block goes between the question and "Options:".

## Design (one page; the 18 sets and items are the same in both stages)

**Items.** The 1,129 items are all GSM8K test items in `gsm8k_mc_v2_xprob.jsonl` (1,317 items) whose worked-step chain
has n >= 2 steps. Dropped: 165 items with no chain in the steps file (`worked_steps` null) and 23 items with one step. By n: 2: 252, 3: 301, 4: 244, 5: 174, 6: 83, 7+: 75. Chains come from `scripts/build_gsm8k_test_steps.py`
(the calculator notes of the GSM8K Socratic test solution, split into binary operations). The options are the
cue-free "x" options of `scripts/build_mc_xprob.py`: the 4 wrong options are final answers of OTHER test problems.
They are not the gsm8kv2 options (those are built from the problem's own numbers). No other filter was used.

**Conditions** (18 sets in `data/hint/`; one real prompt each in `data/example_prompts.md`):

| set | what the "Worked steps:" block holds | items |
|---|---|---|
| plain | no block; the x item unchanged | 1,129 |
| full | all n true steps, answer step included (ceiling check) | 1,129 |
| true_r1 .. true_r6 | the first k = n - r true steps | 1,129 / 877 / 576 / 332 / 158 / 75 |
| placebo_r1 .. r4 | k sentences of the item's own question, in order, cycled if too few (length control) | 1,129 / 877 / 576 / 332 |
| other_r1, r2 | the first k steps of another item's chain (partner seeded by item, 20261021) | 1,129 / 877 |
| wrongres_r1, r2 | a consistent far-wrong chain: each result wrong by at least 1.5 (operator slip, else +-max(2, 10%), x2, /2); operands linked to earlier steps take the wrong values; no true result appears | 1,111 / 861 |
| wrongres_near_r1, r2 | true operands, results changed to the first of +1, -1, +2, -2, x2, /2, +10 that is no option and no true value (+-1 in 97% of lines); no propagation | 1,129 / 877 |

All controls use the same k = n - r as true_r at the same r. Options and gold labels are byte-identical to the x item.
`shown_equals_option` marks rows where a shown result equals an option text; "clean" means that flag is false.

**Models** (saved weights, evaluated with no further training):

| label | eval jobs | weights (training job) | training |
|---|---|---|---|
| base | 500 | Qwen3.5-4B, post-trained, no fine-tuning | none |
| A2-c4 | 501-503 | 83, 84, 85 (seeds 12, 13, 14) | answer only, 4 epochs |
| A2-c12 | 504 | 41 (seed 12) | answer only, 12 epochs |
| GFPQ-c4 | 505-507 | 42, 70, 73 | answer + reasoning-tree node questions (yes/no), 4 epochs |
| GFPQM-c4 | 508-510 | 164, 172, 173 | GFPQ with `--node-format mc` (multiple-choice nodes), 4 epochs |
| GFPQMR-c4 | 511-513 | 165, 174, 186 | GFPQM with node targets shuffled within each problem (`--tree-target shuffled`) |

Shared training settings: data `data/student_gsm8k_tf/train.jsonl` (5,923 GSM8K train items), lr 3e-6, warmup 0.05,
32 items per update (micro batch 2 x 16), max length 1,536, label smoothing 0.1, seed = order seed, 1 GPU (FSDP,
CPU AdamW). Tree arms add `--tree-sources gsm8k_tf`. Exact arguments: `code/train_args/<job>.json`. Env files, code
pins, bundle md5s and diffs: `code/train_code/` (see `BUNDLES.txt`). Weights: `/staging/<netid>/weights/<job>.tar`
on CHTC (all 13 present on 2026-11-17).

**Evaluation.** `run_eval_iv.sh` runs `scripts/eval_student.py --max-len 1536 --bs 8` once per model on all 18 sets
(no chat template). The prediction is the option letter with the highest probability (`top1` in
`results/paired_stats.py`). Code: the bundle `cotd_code_data_hint_4b.tgz` (md5 2517b5eb), pin
`994ff5026e54cffe260d88709ce33afd4533d326`, printed in all 14 job logs; all 14 logs end with `== exit 0`. DAG 6906464,
2026-10-21. The bundle is not in this folder: it is at `~/cotd_chtc/cotd_code_data_hint_4b.tgz` on CHTC and at
`exp/logs/cueread/b6906464_src/` on the second author's Mac; its code is `code/repo_994ff50_full.tgz` and its hint data is
`data/hint/`.

## Code and statistics

**Code base and diff.** Jobs 500-513 ran 994ff50 with no patch (empty diff). Evidence: the eval bundle's repo tree
plus `code/opc_fsdp1.patch` gives, file for file, the repo tree of the Stage 2 bundle (pin `994ff50-fsdp1`). The only
changed file is `hpc/train_arm.sbatch`, which eval jobs do not use. The full tree is `code/repo_994ff50_full.tgz`.
The git repo itself was not reachable, so the match to the commit is not verified against git.
The eval `metrics.json` files hold only `args` and `eval`; `eval_student.py` writes no commit field. The training logs
print `code: repo @ unknown` for the same reason (the bundle has no .git). The pins above replace that field.

**Statistics.** `results/compute_tables.py` imports `paired_stats()` and `top1()` from a byte-identical copy of
`scripts/paired_stats.py`. Every contrast is paired on the items in both sets. Per seed: d = acc(A) - acc(B). Seed
level: Student t, df 2. Two-level bootstrap: resample seeds, then items with shared indices, 10,000 draws. Base and
A2-c12 have one run: no seed-t interval, and the two-level interval is an item bootstrap.
The `significant` column in the CSVs is `paired_stats()`'s own flag: both intervals exclude 0.

## Stage 1 results (pilot, jobs 500-513)

Full tables: `results/tables.md`. CSV for plots: `results/acc_per_seed.csv`, `contrasts.csv`, `curves.csv`,
`grid_n_r.csv`, `between_models.csv`, `path_slice.csv`. Points = accuracy x 100. The tables below come from
`results/readme_tables.py`.

Table A. Headline contrasts per model (mean, then per seed s12 / s13 / s14).

| model | plain acc | true_r1 - plain, all 1,129 | seed-t 95% | two-level 95% | true_r1 - plain, clean 918 | H (clean, n>=4, 440) |
|---|---|---|---|---|---|---|
| base | 33.4 | +12.5 | - | [+9.2, +15.8] | +14.6 | +16.4 |
| A2-c4 | 55.4 | +22.5 (+19.8 / +25.3 / +22.4) | [+15.6, +29.4] | [+19.0, +26.1] | +22.4 (+18.5 / +24.9 / +23.7) | +29.9 (+23.9 / +34.3 / +31.6) |
| A2-c12 | 54.7 | +18.4 | - | [+15.6, +21.3] | +18.5 | +23.4 |
| GFPQ-c4 | 58.9 | +24.0 (+24.5 / +23.0 / +24.4) | [+21.9, +26.1] | [+21.5, +26.7] | +23.1 (+23.7 / +22.8 / +22.9) | +32.3 (+35.5 / +30.9 / +30.5) |
| GFPQM-c4 | 60.7 | +20.6 (+22.1 / +20.0 / +19.8) | [+17.5, +23.7] | [+17.4, +23.9] | +19.6 (+20.7 / +18.6 / +19.4) | +35.3 (+35.7 / +34.1 / +36.1) |
| GFPQMR-c4 | 56.4 | +18.1 (+15.0 / +16.7 / +22.8) | [+7.9, +28.3] | [+14.0, +22.7] | +18.6 (+15.6 / +17.2 / +22.9) | +24.3 (+20.7 / +22.5 / +29.8) |

Table B. true_r - plain, each on the items of that r set (seed mean, two-level 95% interval).

| model | r=1 (1129) | r=2 (877) | r=3 (576) | r=4 (332) | r=5 (158) | r=6 (75) |
|---|---|---|---|---|---|---|
| base | +12.5 [+9.2, +15.8] | +3.6 [+0.0, +7.4] | +2.4 [-1.9, +6.9] | +0.0 [-5.7, +5.7] | -2.5 [-10.1, +5.1] | +6.7 [-5.3, +18.7] |
| A2-c4 | +22.5 [+19.0, +26.1] | +11.4 [+8.6, +14.3] | +5.7 [+2.4, +9.2] | +1.9 [-2.4, +6.2] | +1.9 [-4.4, +8.4] | -0.9 [-9.8, +8.4] |
| A2-c12 | +18.4 [+15.6, +21.3] | +8.8 [+5.8, +11.7] | +3.0 [-0.9, +6.9] | -1.2 [-6.0, +3.6] | +3.2 [-2.5, +8.9] | +4.0 [-6.7, +14.7] |
| GFPQ-c4 | +24.0 [+21.5, +26.7] | +12.8 [+9.7, +15.9] | +3.5 [+0.3, +6.8] | +1.1 [-3.9, +6.0] | +1.5 [-4.6, +8.0] | +2.7 [-7.1, +12.4] |
| GFPQM-c4 | +20.6 [+17.4, +23.9] | +0.1 [-4.5, +4.8] | -6.5 [-11.0, -1.9] | -8.2 [-14.1, -2.6] | -9.7 [-17.9, -1.5] | -7.1 [-20.0, +6.2] |
| GFPQMR-c4 | +18.1 [+14.0, +22.7] | +11.1 [+7.8, +14.6] | +3.3 [-0.3, +7.1] | +1.9 [-2.2, +6.1] | +3.8 [-2.3, +9.7] | +4.0 [-5.8, +13.8] |

Table B mixes r with item length, because each r set holds different items. Table C holds the items fixed.

Table C. Curve on one fixed set of 332 items with n >= 5 (accuracy, seed mean). For n >= 7 (75 items, r = 1..6), see
`results/tables.md` section 4.

| model | plain | r=1 | r=2 | r=3 | r=4 | full |
|---|---|---|---|---|---|---|
| base | 28.3 | 43.4 | 34.3 | 32.5 | 28.3 | 99.4 |
| A2-c4 | 39.1 | 69.5 | 53.1 | 44.6 | 41.0 | 97.2 |
| A2-c12 | 41.9 | 62.3 | 51.8 | 44.3 | 40.7 | 92.2 |
| GFPQ-c4 | 43.1 | 75.2 | 58.7 | 48.7 | 44.2 | 95.8 |
| GFPQM-c4 | 44.1 | 81.2 | 58.4 | 44.2 | 35.8 | 97.8 |
| GFPQMR-c4 | 40.2 | 65.0 | 53.7 | 45.2 | 42.1 | 96.1 |

Table C2. The same 332 fixed items with intervals: true_r - plain and D = true_r1 - true_r4 (seed mean; t[ ] =
seed-t 95% for 3-seed arms; b[ ] = two-level 95%). The n >= 7 rows are in `results/tables.md` section 2.

| model | r=1 - plain | r=2 - plain | r=3 - plain | r=4 - plain | D = r1 - r4 |
|---|---|---|---|---|---|
| base | +15.1 b[+8.7, +21.4] | +6.0 b[+0.0, +12.0] | +4.2 b[-1.5, +9.9] | +0.0 b[-5.7, +5.7] | +15.1 b[+8.7, +21.4] |
| A2-c4 | +30.4 t[+19.1, +41.7] b[+23.8, +37.0] | +14.1 t[+6.8, +21.3] b[+8.7, +19.3] | +5.5 t[+0.5, +10.5] b[+1.2, +10.0] | +1.9 t[-1.8, +5.6] b[-2.4, +6.2] | +28.5 t[+20.2, +36.8] b[+22.6, +34.2] |
| A2-c12 | +20.5 b[+14.8, +26.2] | +9.9 b[+4.5, +15.4] | +2.4 b[-3.0, +7.8] | -1.2 b[-6.0, +3.6] | +21.7 b[+16.3, +27.1] |
| GFPQ-c4 | +32.1 t[+26.0, +38.2] b[+26.5, +37.7] | +15.7 t[+8.2, +23.1] b[+10.1, +21.1] | +5.6 t[+0.3, +10.9] b[+0.9, +10.2] | +1.1 t[-4.5, +6.7] b[-3.9, +6.0] | +31.0 t[+25.8, +36.3] b[+25.3, +36.6] |
| GFPQM-c4 | +37.1 t[+35.0, +39.3] b[+31.8, +42.4] | +14.4 t[+8.0, +20.7] b[+8.4, +20.3] | +0.1 t[-3.4, +3.6] b[-5.0, +5.4] | -8.2 t[-9.4, -7.1] b[-14.1, -2.6] | +45.4 t[+42.3, +48.5] b[+39.7, +50.9] |
| GFPQMR-c4 | +24.8 t[+15.0, +34.6] b[+18.8, +31.1] | +13.6 t[+6.2, +20.9] b[+8.1, +19.1] | +5.0 t[-0.8, +10.8] b[+0.4, +9.8] | +1.9 t[-1.2, +5.0] b[-2.2, +6.1] | +22.9 t[+11.3, +34.5] b[+16.4, +29.3] |

Table D. Controls minus plain, each on the items of that control set (seed mean, two-level 95% interval).

| contrast | base | A2-c4 | A2-c12 | GFPQ-c4 | GFPQM-c4 | GFPQMR-c4 |
|---|---|---|---|---|---|---|
| other_r1 - plain | -7.4 [-10.4, -4.3] | -6.6 [-10.2, -3.1] | -7.2 [-9.8, -4.4] | -7.1 [-9.6, -4.5] | -16.3 [-20.7, -11.2] | -9.4 [-13.8, -4.6] |
| other_r2 - plain | -5.0 [-8.7, -1.4] | -5.2 [-8.8, -1.6] | -5.5 [-8.7, -2.3] | -7.4 [-10.4, -4.5] | -16.4 [-20.9, -11.8] | -7.3 [-11.0, -3.8] |
| wrongres_r1 - plain | -5.9 [-9.0, -2.8] | -3.8 [-6.9, -0.9] | -4.9 [-7.7, -2.1] | -6.3 [-9.3, -3.3] | -14.3 [-17.6, -11.2] | -4.8 [-7.6, -2.0] |
| wrongres_r2 - plain | -4.2 [-7.7, -0.6] | -3.0 [-6.4, +0.5] | -2.8 [-5.9, +0.5] | -2.9 [-6.0, +0.2] | -12.7 [-16.4, -9.1] | -2.5 [-5.6, +0.7] |
| wrongres_near_r1 - plain | -2.8 [-6.0, +0.4] | +2.0 [-0.9, +5.0] | +2.9 [-0.1, +5.8] | +1.2 [-1.4, +3.8] | -4.9 [-8.4, -1.5] | +0.5 [-2.3, +3.4] |
| wrongres_near_r2 - plain | -4.1 [-7.6, -0.6] | +0.5 [-2.8, +3.9] | +1.1 [-2.2, +4.4] | -0.4 [-4.0, +3.0] | -9.3 [-12.8, -5.8] | -1.5 [-4.6, +1.6] |
| placebo_r1 - plain | -1.9 [-4.6, +1.0] | +1.0 [-1.4, +3.4] | +0.4 [-1.7, +2.6] | -1.3 [-3.3, +0.6] | -9.4 [-11.9, -7.0] | -0.2 [-2.1, +1.7] |
| placebo_r2 - plain | +0.9 [-2.4, +4.2] | +0.2 [-2.4, +2.8] | -1.1 [-3.6, +1.3] | -2.0 [-4.4, +0.3] | -10.1 [-13.2, -6.9] | +1.1 [-1.7, +3.8] |
| placebo_r3 - plain | -0.5 [-4.7, +3.6] | +0.2 [-2.6, +3.1] | -1.0 [-4.2, +2.1] | -2.2 [-5.3, +1.0] | -8.4 [-12.2, -4.5] | +0.0 [-3.5, +3.3] |
| placebo_r4 - plain | -2.7 [-8.1, +3.0] | -1.7 [-5.6, +2.1] | -0.9 [-4.8, +2.7] | -2.4 [-7.0, +2.2] | -6.4 [-11.8, -1.0] | -0.5 [-4.4, +3.3] |

Table E. True chain minus each control at r = 1 (shared items; seed mean, two-level 95% interval). The r = 2 rows
are in `results/tables.md`.

| contrast | base | A2-c4 | A2-c12 | GFPQ-c4 | GFPQM-c4 | GFPQMR-c4 |
|---|---|---|---|---|---|---|
| true_r1 - other_r1 | +19.8 [+16.4, +23.1] | +29.1 [+26.2, +31.8] | +25.6 [+22.5, +28.9] | +31.1 [+28.0, +34.1] | +36.9 [+31.2, +42.1] | +27.6 [+21.0, +34.6] |
| true_r1 - wrongres_r1 | +18.4 [+15.1, +21.5] | +26.4 [+23.4, +29.4] | +23.6 [+20.3, +26.8] | +30.3 [+27.4, +33.2] | +34.8 [+31.1, +38.4] | +23.0 [+18.1, +28.8] |
| true_r1 - wrongres_near_r1 | +15.3 [+12.5, +18.2] | +20.5 [+17.9, +23.1] | +15.5 [+12.8, +18.3] | +22.8 [+20.3, +25.3] | +25.5 [+22.0, +29.3] | +17.6 [+14.0, +21.4] |
| true_r1 - placebo_r1 | +14.3 [+11.2, +17.4] | +21.5 [+18.8, +24.3] | +18.0 [+15.0, +20.9] | +25.3 [+22.6, +28.1] | +30.0 [+26.9, +33.1] | +18.3 [+13.9, +23.4] |

Short reading:

- With one step left, every model gains a lot (A2-c4 +22.5 on all items, +29.9 on clean items with n >= 4).
- The gain falls fast with r. On the fixed n >= 5 items, A2-c4 goes 69.5 / 53.1 / 44.6 / 41.0 for r = 1..4,
  against 39.1 plain (D = +28.5, seed-t [+20.2, +36.8]). With 4 steps left, the gain is +1.9 [-2.4, +6.2].
- The gain needs the true values. Another item's chain and a far-wrong chain both fall below plain. A near-miss chain
  stays within 3 points of plain, except for base and GFPQM-c4. The true chain beats every control by 14 to 37 points
  at r = 1.
- Placebo text costs A2-c4, A2-c12, GFPQ-c4 and GFPQMR-c4 at most 2.4 points (GFPQ-c4: -2.0 and -2.2 at r = 2, 3,
  seed-t intervals below 0). GFPQM-c4 is the exception: any block costs it 6 to 10 points, and its true_r3..r5 fall
  below plain. The card's gate fails for GFPQM-c4.
- Base: plain accuracy is 33.4, below the card's validity bar of 40, so by the card its row decides nothing.
- `results/tables.md` section 6: with one serial step left (remaining critical path 1), A2-c4 reaches 68.7, near
  native one-path items (61.7, only 20 items). With two left it reaches 50.8, against 72.4 for native two-path items.
- Between models (`results/tables.md` section 3, seed-paired against A2-c4): GFPQ-c4 and GFPQMR-c4 show no
  difference that passes both intervals (GFPQ-c4 gain at r = 1: +1.5, seed-t [-7.4, +10.4]). GFPQM-c4 differs on
  three rows: plain +5.3, true_r2 -6.7, gain at r = 2 -11.3 (both intervals exclude 0). These are pilot models.

Self-tests (`results/selftest.txt`): for all 252 (run, set) pairs, the preds item ids and gold labels equal the hint
file, and the recomputed accuracy equals the "acc" line of the job log. The October readout script
(`code/readout_oct/hint_table.py`) was run again on the same tarballs (`results/oct_readout_rerun.txt`). Its H, M2,
M3, M3b and M4 values equal the values above.

## Stage 2: the registered confirmation (jobs 520-533)

**What ran.** 12 training jobs, DAG 6913320, 2026-10-21, all exit 0. Code pin `994ff50-fsdp1` (= 994ff50 plus
`code/opc_fsdp1.patch`) in all 12 logs. Data `data/student_gsm8k_tf_vt/train.jsonl`, `--eval-value-render vt`,
lr 3e-6, 32 items per update, 1 GPU. The 18 hint files ran at the end of training on the final weights. No weights
were saved. Arguments: `code/train_args_stage2/<job>.json`.

| label | jobs (s16 / s17 / s18) | training |
|---|---|---|
| A2-c1 | 520 / 525 / 530 | answer only, 1 epoch |
| A2-c2 | 521 / 526 / 531 | answer only, 2 epochs |
| GFPQ-c2 | 522 / 527 / 532 | answer + yes/no tree nodes (`--tree-sources gsm8k_tf`), 2 epochs |
| GFPQR-c2 | 523 / 528 / 533 | GFPQ-c2 with node targets shuffled within each problem, 2 epochs |

The card amendment of 2026-10-21 20:45 (before any Stage 2 job) replaced the planned GFPQM-c2 / GFPQMR-c2 pair by
GFPQ-c2 / GFPQR-c2, because GFPQM-c4 failed the placebo check in the pilot.

**Data source.** The hint preds, valx preds, metrics and logs of the 12 jobs are in `data/stage2_preds.tgz`, cut
from the job result tarballs (md5 list `data/stage2_results_tgz.md5`; the tarballs on CHTC and the second author's Mac match it).
Warning: in the results repo, `runs/A2-c1lr3ch-Qwen3.5-4B-student_gsm8k_tf_vt-s16..s18` now hold jobs 663-665
(same run name, 2026-10-25). The Stage 2 A2-c1 preds are only in the repo's git history (for example commit
f3a31508 for job 520) and in `data/stage2_preds.tgz`. Self-test (`results/stage2_selftest.txt`): for all 216
(run, set) pairs, item ids and gold labels equal the hint files, and the recomputed accuracy equals `metrics.json`.
The recomputed valx equals `metrics.json` too.

**Card verdict, recomputed** (`results/stage2_verdict.md`, rules in `verdict()` of `results/compute_tables.py`):

Epoch rule (valx, 393 items): A2-c1 62.60 (63.61 / 61.32 / 62.85), A2-c2 61.41 (61.32 / 62.60 / 60.31) -> the P2 student is A2-c1.

### Checks (seed means) and primary numbers per arm

| arm | M1 full | M2 placebo-plain pooled (r1/r2/r3/r4) | M3 true-far | M3b true-near | M4 true-other | read (n=2) | gate | H clean n>=4 (per seed) | H seed-t 95% | H two-level 95% | D n>=5 (per seed) | D seed-t 95% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2-c1 | 95.6 | -0.9 (+0.1 / -1.4 / -1.0 / -2.7) | +35.0 | +23.9 | +32.7 | +4.5 | PASS | +36.4 (+38.2 / +34.1 / +37.0) | [+31.2, +41.7] | [+31.7, +41.4] | +35.9 (+34.6 / +38.6 / +34.6) | [+30.3, +41.6] |
| A2-c2 | 97.6 | -0.8 (-0.3 / -1.1 / -0.6 / -1.9) | +30.3 | +23.0 | +31.4 | +4.9 | PASS | +34.2 (+35.0 / +30.5 / +37.0) | [+25.8, +42.5] | [+28.6, +39.5] | +31.1 (+30.7 / +28.9 / +33.7) | [+25.1, +37.2] |
| GFPQ-c2 | 96.1 | -3.2 (-2.4 / -4.1 / -2.9 / -3.8) | +37.7 | +26.9 | +37.6 | +6.0 | FAIL | +38.6 (+42.0 / +35.5 / +38.2) | [+30.3, +46.8] | [+33.3, +43.8] | +38.3 (+40.4 / +34.6 / +39.8) | [+30.4, +46.1] |
| GFPQR-c2 | 96.9 | -1.0 (-0.7 / -0.9 / -1.5 / -1.2) | +32.4 | +24.0 | +34.9 | +4.8 | PASS | +36.2 (+36.6 / +31.1 / +40.9) | [+24.0, +48.4] | [+29.8, +42.6] | +34.9 (+35.2 / +31.0 / +38.6) | [+25.6, +44.3] |

### Decisions

- Gate on the P2 student A2-c1 (seed means): PASS. P2 is read only if the gate passes.
- P2 (primary): H = +36.4 on 440 clean items with n >= 4, seeds +38.2 / +34.1 / +37.0, seed t 29.89 (crit 4.30) -> **PASS**.
- P1: D = acc(r1) - acc(r4) on 332 items with n >= 5 = +35.9 -> capacity-limited.
- P3 (secondary) GFPQ-c2 - GFPQR-c2 on 2006 r in {1, 2} true rows: +1.7 / +5.0 / +1.6, mean +2.8, seed t 2.46, seed-t 95% [-2.1, +7.6], two-level 95% [+0.7, +5.1] -> between.
- P3 (secondary) GFPQ-c2 - GFPQR-c2 on 1672 r in {1, 2} true rows (clean only): +1.3 / +4.2 / +1.3, mean +2.3, seed t 2.30, seed-t 95% [-2.0, +6.5], two-level 95% [+0.2, +4.5] -> between.

Reading: the primary passes clearly. With every earlier result written, one forward pass of A2-c1 gains 36 points on
clean items with 4+ steps, and the gain falls to 0 with 4 steps left (D = +35.9). The content comparison P3 is
"between": positive on all 3 seeds but not significant at the seed level. GFPQ-c2 fails the placebo check by 0.2
points (pooled -3.2, bar 3). These match the October ledger entry.

**Stage 2 tables** (from `results/readme_tables.py 2`; full tables `results/stage2_tables.md`, CSVs
`results/stage2_*.csv`).

Table S2-A. Headline contrasts per model (points; mean, then per seed, s16 / s17 / s18).

| model | plain acc | true_r1 - plain, all 1,129 | seed-t 95% | two-level 95% | true_r1 - plain, clean 918 | H (clean, n>=4, 440) |
|---|---|---|---|---|---|---|
| A2-c1 | 60.6 | +25.3 (+26.7 / +23.3 / +25.9) | [+20.8, +29.8] | [+22.2, +28.3] | +25.3 (+26.3 / +23.2 / +26.4) | +36.4 (+38.2 / +34.1 / +37.0) |
| A2-c2 | 59.0 | +24.4 (+25.1 / +21.0 / +27.0) | [+16.7, +32.0] | [+20.5, +28.0] | +24.9 (+25.8 / +21.6 / +27.2) | +34.2 (+35.0 / +30.5 / +37.0) |
| GFPQ-c2 | 60.8 | +27.1 (+29.9 / +25.0 / +26.5) | [+20.8, +33.5] | [+23.7, +30.7] | +27.6 (+30.6 / +25.3 / +26.9) | +38.6 (+42.0 / +35.5 / +38.2) |
| GFPQR-c2 | 58.6 | +25.2 (+25.2 / +22.9 / +27.5) | [+19.5, +31.0] | [+22.0, +28.5] | +25.7 (+26.4 / +22.9 / +28.0) | +36.2 (+36.6 / +31.1 / +40.9) |

Table S2-B. true_r - plain for r = 1..6 (each on the items of that r set; seed mean, two-level 95% interval).

| model | r=1 (1129) | r=2 (877) | r=3 (576) | r=4 (332) | r=5 (158) | r=6 (75) |
|---|---|---|---|---|---|---|
| A2-c1 | +25.3 [+22.2, +28.3] | +13.3 [+9.6, +17.1] | +5.5 [+1.2, +9.9] | +0.1 [-5.0, +5.3] | +0.6 [-6.3, +7.6] | +7.1 [-3.6, +17.3] |
| A2-c2 | +24.4 [+20.5, +28.0] | +14.4 [+11.4, +17.6] | +4.8 [+0.8, +9.0] | +1.1 [-3.5, +5.4] | +1.7 [-6.3, +9.5] | +5.8 [-5.3, +16.4] |
| GFPQ-c2 | +27.1 [+23.7, +30.7] | +12.3 [+9.2, +15.2] | +4.1 [-0.3, +8.6] | +0.0 [-4.8, +4.7] | +0.8 [-6.8, +8.2] | +0.4 [-10.7, +12.0] |
| GFPQR-c2 | +25.2 [+22.0, +28.5] | +13.5 [+10.1, +17.0] | +5.5 [+1.6, +9.7] | +1.1 [-3.3, +5.4] | +2.5 [-3.4, +8.4] | +2.2 [-6.7, +11.1] |

Table S2-C. Fixed-item curve, 332 items with n >= 5 (accuracy, seed mean).

| model | plain | r=1 | r=2 | r=3 | r=4 | full |
|---|---|---|---|---|---|---|
| A2-c1 | 45.7 | 81.7 | 62.9 | 54.2 | 45.8 | 95.7 |
| A2-c2 | 44.7 | 76.9 | 61.2 | 49.1 | 45.8 | 97.3 |
| GFPQ-c2 | 43.8 | 82.0 | 60.4 | 50.4 | 43.8 | 96.1 |
| GFPQR-c2 | 41.7 | 77.7 | 58.3 | 49.1 | 42.8 | 97.8 |

Table S2-C2. The same 332 fixed items: true_r - plain and D = true_r1 - true_r4 (seed mean, seed-t 95% for 3-seed arms,
two-level 95%).

| model | r=1 - plain | r=2 - plain | r=3 - plain | r=4 - plain | D = r1 - r4 |
|---|---|---|---|---|---|
| A2-c1 | +36.0 t[+31.2, +40.9] b[+30.6, +41.5] | +17.2 t[+11.3, +23.0] b[+11.6, +22.7] | +8.5 t[+2.9, +14.2] b[+3.5, +13.7] | +0.1 t[-7.2, +7.4] b[-5.0, +5.3] | +35.9 t[+30.3, +41.6] b[+30.3, +41.6] |
| A2-c2 | +32.2 t[+21.0, +43.5] b[+25.3, +39.0] | +16.6 t[+9.1, +24.0] b[+10.7, +22.3] | +4.4 t[-0.9, +9.7] b[-0.6, +9.2] | +1.1 t[-4.2, +6.4] b[-3.5, +5.4] | +31.1 t[+25.1, +37.2] b[+25.3, +36.7] |
| GFPQ-c2 | +38.3 t[+34.1, +42.4] b[+32.8, +43.7] | +16.7 t[+11.8, +21.5] b[+11.2, +22.0] | +6.6 t[+1.2, +12.0] b[+1.6, +11.4] | +0.0 t[-5.6, +5.6] b[-4.8, +4.7] | +38.3 t[+30.4, +46.1] b[+32.1, +44.3] |
| GFPQR-c2 | +36.0 t[+25.1, +46.9] b[+29.4, +42.9] | +16.7 t[+9.8, +23.6] b[+11.1, +22.7] | +7.4 t[+2.6, +12.3] b[+2.7, +12.3] | +1.1 t[-2.7, +4.9] b[-3.3, +5.4] | +34.9 t[+25.6, +44.3] b[+28.5, +41.3] |

Table S2-D. Controls minus plain (each on the items of that control set; seed mean, two-level 95% interval).

| contrast | A2-c1 | A2-c2 | GFPQ-c2 | GFPQR-c2 |
|---|---|---|---|---|
| other_r1 - plain | -7.4 [-9.7, -5.2] | -7.0 [-9.9, -4.2] | -10.5 [-12.9, -8.1] | -9.7 [-12.3, -7.1] |
| other_r2 - plain | -7.3 [-10.4, -4.3] | -5.1 [-7.8, -2.6] | -10.0 [-12.9, -7.0] | -7.6 [-11.0, -4.3] |
| wrongres_r1 - plain | -9.6 [-13.1, -6.4] | -5.9 [-9.3, -2.5] | -10.5 [-13.4, -7.4] | -7.1 [-10.4, -3.7] |
| wrongres_r2 - plain | -5.9 [-9.9, -1.6] | -3.9 [-7.2, -0.8] | -7.3 [-10.3, -4.2] | -4.3 [-7.7, -0.7] |
| wrongres_near_r1 - plain | +1.4 [-1.5, +4.3] | +1.4 [-2.5, +4.9] | +0.2 [-2.5, +3.1] | +1.2 [-1.4, +3.8] |
| wrongres_near_r2 - plain | -0.7 [-4.1, +2.7] | -0.8 [-3.8, +2.1] | -1.6 [-4.9, +1.8] | -0.2 [-3.6, +3.3] |
| placebo_r1 - plain | +0.1 [-1.7, +1.9] | -0.3 [-2.0, +1.4] | -2.4 [-4.6, -0.2] | -0.7 [-2.5, +1.2] |
| placebo_r2 - plain | -1.4 [-3.9, +1.0] | -1.1 [-3.6, +1.3] | -4.1 [-6.7, -1.5] | -0.9 [-4.0, +2.1] |
| placebo_r3 - plain | -1.0 [-3.9, +2.0] | -0.6 [-3.1, +2.0] | -2.9 [-6.5, +0.6] | -1.5 [-5.9, +2.5] |
| placebo_r4 - plain | -2.7 [-6.6, +1.2] | -1.9 [-5.7, +1.8] | -3.8 [-7.8, +0.3] | -1.2 [-5.2, +2.7] |

Table S2-E. True chain minus each control at r = 1 (shared items; seed mean, two-level 95% interval).

| contrast | A2-c1 | A2-c2 | GFPQ-c2 | GFPQR-c2 |
|---|---|---|---|---|
| true_r1 - other_r1 | +32.7 [+29.5, +35.9] | +31.4 [+26.8, +36.2] | +37.6 [+34.3, +41.0] | +34.9 [+30.6, +39.1] |
| true_r1 - wrongres_r1 | +35.0 [+30.9, +39.1] | +30.3 [+27.1, +33.5] | +37.7 [+33.9, +41.5] | +32.4 [+27.4, +37.4] |
| true_r1 - wrongres_near_r1 | +23.9 [+21.5, +26.4] | +23.0 [+20.6, +25.4] | +26.9 [+23.9, +30.1] | +24.0 [+20.4, +27.4] |
| true_r1 - placebo_r1 | +25.2 [+22.2, +28.2] | +24.6 [+20.8, +28.2] | +29.5 [+25.3, +33.6] | +25.9 [+22.2, +29.4] |

Between arms (seed-paired, `results/stage2_tables.md` section 3): GFPQ-c2 - A2-c1 on the gain true_r1 - plain is
+1.8 (seed-t [-1.4, +5.0]); GFPQ-c2 - GFPQR-c2 on the same gain is +1.9 (seed-t [-5.3, +9.0]). Neither passes.

## Conditions run

All of these ran in jobs 500-513 (14 models) and in jobs 520-533 (12 runs), with no failure:

- plain, full, true_r1 .. true_r6, placebo_r1 .. r4, other_r1, r2, wrongres_r1, r2, wrongres_near_r1, r2 (18 sets).

Changes, drops and related runs:

- Card v1 had 16 sets. v3 added the consistent far-wrong chain as `wrongres` and kept the old near-miss set as
  `wrongres_near`. 18 sets ran.
- r = 7 and above was not built (44 prompts on the 29 items with n >= 8).
- wrongres drops 18 items at r = 1 and 16 at r = 2: no far-wrong value passed the rules for some step.
- Builder: `code/builder/build_hint_evals.py` is the version that built the evaluated files. It is recovered from
  session file history (the original scratch folder was deleted by the macOS /tmp cleanup). It reproduces all 18 files
  byte for byte (`code/builder/rebuild.sh`), and its own 11 tests pass on the bundle data.
- Later the same day the builder added the fields `path`, `r_path` (remaining critical path) and later
  `shown_equals_gold`, and grew to 33 sets (other / wrongres / wrongres_near r3-r5, placebo_r5, plan_r1-r5; version in
  `code/builder/build_hint_evals_v5_later_33sets.py`). Prompts of the 18 shared sets did not change. The 15 extra
  sets were not run in jobs 500-513. They ran in Stage 2b (jobs 540-553).
- The card names the Stage 1 bundle md5 8487cf14 and DAG ids 340-353. The bundle that ran is 2517b5eb, ids 500-513.
- Weight training jobs 70 and 73 failed their first smoke stage on host gpu2008. The retries finished with exit 0.
- The same 18 files (same md5) ran inside the Stage 2 training jobs 520-533 (seeds 16-18).
- Planned but not run: the SIB-c2 reader jobs (524 / 529 / 534, `jobs_opc2_sib_4b.dag`). The card ran them only if
  the pilot checks failed on A2; they passed.

## Differences from the first author's reading

1. r_k is "k = steps left", not "first k steps written". The block holds the first n - r steps.
2. A "step" is one atomic binary operation, not a Socratic sub-question. One item can have depth 2 and n = 9
   (`data/example_prompts.md`).
3. The 1,129 items are GSM8K test items, but the options are the cue-free x options (other problems' answers), not
   the gsm8kv2 options. Plain accuracy here is not gsm8kv2 accuracy.
4. Jobs 500-513 are the card's PILOT (Stage 1). The card says no claim comes from it: the models are 4-epoch
   checkpoints (past A2's best epoch, 1-2), seed 12 was the selection seed, and only base and A2-c12 have one run each.
   The pre-registered confirmation is Stage 2 (jobs 520-533: A2-c1 / c2, GFPQ-c2, GFPQR-c2, seeds 16-18), on the same 18
   files. Its section in this README holds the confirmed numbers and the card verdict.
5. The card's primary number is H on clean items with n >= 4 (440 items), not true_r1 - plain on all 1,129. Both are
   in Table A.
6. "GFPQ-MC shuffled" (GFPQMR) shuffles the node targets within each problem. It is a control for node content, not a
   shuffled hint.
7. The paper_inputs README says jobs 500-513 print no code pin. They do print `994ff50...` in every log.

## Not recovered or not verified

- The git commits 994ff50 and 5a0460b were not reachable from this account. The code is taken from the job bundles.
- The card file is the rebuilt copy from `paper_inputs/2026-11-02/cards` (the original was lost in the /tmp cleanup).
- The base row cannot be judged under the card (plain accuracy 33.4 < 40).

## How to run

```bash
python3 -I code/check_sets.py                 # set rules and r_k, standard library
python3 results/compute_tables.py --stage 1 --runs <results repo>/runs   # pilot tables (default --runs ../../../runs)
python3 results/compute_tables.py --stage 2   # confirmation tables + stage2_verdict.md, from data/stage2_preds.tgz
python3 -I results/readme_tables.py           # the short pilot tables above
python3 -I results/readme_tables.py 2         # the short Stage 2 tables
python3 -I code/make_examples.py              # data/example_prompts.md
bash code/builder/rebuild.sh <python with torch>   # rebuild the 18 sets and compare md5
```

## Files

- `code/builder/`: the recovered builder, its tests, the later 33-set version, `rebuild.sh`.
- `code/repo_994ff50/`: the eval-time copies of `student.py`, `eval_student.py`, `build_mc_xprob.py`,
  `build_gsm8k_test_steps.py`, `train_arm.sbatch`. `code/repo_994ff50_full.tgz`: the full repo tree of the bundle.
- `code/cluster/`: `run_eval_iv.sh`, `train_hint_4b.sub`, `jobs_hint_4b.dag`, the 14 env files, `CODE_PIN`.
- `code/train_args/`, `code/train_code/`: training arguments, env files, pins, bundle md5s, diffs.
- `code/readout_oct/hint_table.py`: the October readout. `code/card/`: the card. `code/opc_fsdp1.patch`: Stage 2 diff.
- `data/hint/`: the 18 evaluated sets and `MANIFEST.json`. `data/source/`: x items, steps, gsm8kv2 items.
- Stage 1 preds: not copied. They are `runs/5xx_4b_hint_*` in the results repo; their hint preds equal the job
  tarballs listed in `data/results_tgz.md5` (checked). `data/job_logs/`, `data/weight_train_logs/`: logs.
- `data/stage2_preds.tgz`: Stage 2 hint + valx preds, metrics and logs; `data/stage2_results_tgz.md5`: source tarballs.
- `code/train_args_stage2/`: Stage 2 training arguments.
- `data/md5.txt`, `results/md5.txt`, `code/md5.txt`: md5 of every file.
