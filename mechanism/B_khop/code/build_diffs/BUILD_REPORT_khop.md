# Build report: CARD_khop_v1 (family khop, ids 750-774), 2026-10-25 23:55 CDT

E = <tmp>
Nothing was bundled, copied to CHTC or submitted. No cluster command was run.

## 1. What changed

Code (E/src_khop/repo; diff against repo.base in E/src_khop/build.diff, made with `diff -ruN -x __pycache__ -x data`,
because repo/data is a dangling symlink in both trees):

| File | Change |
|---|---|
| scripts/train_student.py | New flag `--extra-train PATH[,PATH]` (default ""). The rows of each file are appended to `train_items` directly after `train_items = read_jsonl(args.train)`, before any filter, shuffle or order seed. A missing file or a duplicate item_id (against --train or inside the extra rows) is a SystemExit. Log lines: `extra-train: <n> rows from <path> (sources {...})` per file and `extra-train: total training rows <N> (<M> from <train>)`. metrics.json gets `"extra_train": {path: n}` only when the flag is set. |
| hpc/train_arm.sbatch | Smoke stage only: when XARGS holds `--extra-train`, the smoke trains on the first 16 rows of each extra file (one job-local file) instead of the full file. Without the flag every command is identical to repo.base (tested). |
| scripts/build_khop.py | New generator: all eval and training files, build_report.json, the row parser and the D-lines. |
| tests/test_build_khop.py, tests/test_extra_train.py, tests/test_khop_eval_paths.py | New tests (section 2). |

Why the sbatch change (a card deviation, flagged): the smoke stage runs `--epochs 0.5 --micro-bs 2 --grad-accum 2` on
16 gsmvt rows plus XARGS. With the full extra file that is about 1,508 rows = 377 updates for KH / K1 and about 4,508
rows = 1,127 updates for KD. j663 shows about 3.5 s of cpu-adamw per update (upd_s), so the smoke stage alone would
take about 22 min (KH / K1) and about 66 min (KD), longer than KD's real training. The smoke now runs 32 rows
(8 updates). The real arm still reads the full file.

Readout and stage (new files only):

| File | Content |
|---|---|
| E/readout/khop_table.py | Curves, I1-I4, N1, rc cells (pfx_table.py), G2 / G3, verdict() in the card's reading order, secondary contrasts, nearest-chain share. |
| E/readout/khop_pilot_gate.py | Pilot gate P0-P9; exit 0 = GATE PASS. |
| E/readout/run_khop.sh | Five-seed readout: all 25 tarballs (KH, K1, KD, A2Z s16-20, BASE0, A2Z96, both A2 sets). |
| E/readout/test_khop_table.py, test_khop_pilot_gate.py | Tests on synthetic tarballs. |
| E/stage/prep_khop.py | Writes all 22 envs (750-771), both subs, the smoke DAG and the pilot DAG. |
| E/stage/prep_khop_confirm.py | Writes confirm_x1khop_4b.dag only after a gate output whose last line is `GATE PASS`; `--dry-run` prints the table. |
| E/stage/test_prep_khop.py | Env / sub / DAG / confirm-generator tests. |
| E/stage/train_x1khop_4b.sub, eval_x1khop_4b.sub | train_x1b3v2_4b.sub / eval_x1b2_4b.sub with CODE_TGZ=cotd_code_data_x1khop_4b.tgz and logs/train_x1khop_4b.log / logs/eval_x1khop_4b.log (only these lines and the header comment differ; tested). |
| E/stage/smoke_x1khop_4b.dag, pilot_x1khop_4b.dag | Section 3 DAGs. |
| E/stage/condor/jobs_x1/750-771 envs | 22 envs (table in section 4). |

No eval_x1khop_4b.dag: the card puts every eval-only node into the smoke DAG (j750), the pilot DAG (j751) or the
confirmation DAG (j752-755, j771). A separate eval DAG would submit these jobs twice.

The card file was not edited.

## 2. Tests

Runner: E/src_khop/run_tests.sh (venv_pfx python, `python -m unittest`, COTD_REF_REPO = repo.base so that
test_frontier_train's A2 identity test runs against the snapshot). Output: E/src_khop/test_all.log.

| Suite | Result | What it checks |
|---|---|---|
| repo tests (test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train) | OK, 64 tests, 1 skip (test_frontier.TestRealData needs data/, known) |
| readout (test_khop_table, test_khop_pilot_gate) | OK, 16 tests | |
| stage (test_prep_khop) | OK, 4 tests | |

- Regression comparison for the shared sbatch (the only shared file edited besides train_student.py):
  tests.test_tree_target + tests.test_probe_slots run once in repo and once in repo.base (same venv, logs
  E/src_khop/logs/cmp_repo.log, cmp_repo.base.log): 23 tests each, the same 5 failures in both trees
  (test_tree_target: test_gfpsr_gfpqr_are_gfps_gfpq_with_shuffled_targets and the two data/-dependent tests;
  test_probe_slots.TestSbatch: test_knobs_and_probe, test_probe_only, the known macOS bash 3.2 xtrace failures).
  No new failure. The rest of the src_khop suite was not compared (not touched by the diff beyond these files).
- test_build_khop: determinism by seed; option rule; interleave is a linear extension with each chain first about 1/5;
  parser and the expected-accuracy convention on hand rows; on the real files: md5 and counts per cell (1.2), the key
  set (training = the 32 gsmvt keys; eval = 32 + task, order, k, label_order, gold_probs), every 1.1 rule recomputed
  from the prompt for every row, fwd topological and rev its exact reverse with the same id and options, K1 shares the
  KH statements and queries the target's hop-1 variable with the hop-1 set (vb: the roots), -qs twin, blind rows,
  train_kd starts with train_kh, disjointness, all D1 / D2 numbers recomputed from rows equal the report, D4
  recomputed with the Qwen3.5-4B tokenizer.
- test_extra_train: flag off byte-identical to repo.base (losses, eval metrics, preds; no extra_train key); flag on
  equals the reference trainer on one concatenated file (proves append before shuffle); log lines; metrics key; two
  files; refusals (duplicate vs --train, duplicate inside the file, missing file); 1-rank FSDP (torchrun
  --local-addr 127.0.0.1, cpu-adamw) on mixed GSM + khop rows with --eval-value-render vt and --eval-subq; the sbatch
  with a stub python: identical calls without the flag (normal, SMOKE_ONLY, a look-alike flag), the smoke subset for
  one and two files, the arm call keeps the full path, a missing file stops the smoke.
- test_khop_eval_paths (Amendment 1 finding 8): train_student in-job eval (1-rank FSDP, eval_max_len 6144, the
  sbatch passes no --eval-max-len) vs eval_student.py on the saved weights (--max-len 1536 --bs 8, run_eval_b2.sh
  form), 20 khop rows of every task / order / qs / blind: same item set, max |log p1 - log p2| <= 1e-5 (passes), the
  same token ids at 6144 and 1536 (GPT-2), and every row of the real khop eval files fits 1,536 Qwen tokens with the
  same ids at both lengths. The fallback replay j772 is therefore not needed.
- test_khop_pilot_gate: a good pilot passes; each of P0, P1 (khop set and GSM set), P2 (row total, NaN loss), P3, P4,
  P5, P6, P7, P8, P9 fails alone; P9 needs both conditions; the wall-time and exit parsers on the real j663 log (0.51 h)
  and the 663 env's 55 preds names.
- test_khop_table: every reading of section 0 reached (MANIPULATION FAILED first, PASS, PASS blocked by m(G2) < 0.15,
  NEGATIVE with and without the k* <= 1 line, NEGATIVE blocked by the dose (both BETWEEN reasons), NULL, BETWEEN on an
  unresolved k = 3 cell); SOLVED / FAILED / k*; t CI; I4; loaders; nearest-chain share; end to end through main() on
  five synthetic seeds per arm against the real A2-6k reference tarballs (prints VERDICT: PASS).

## 3. Data

Generator: E/src_khop/repo/scripts/build_khop.py (in the repo, so it travels with the bundle). Command run:

    venv_pfx/bin/python scripts/build_khop.py --out E/src/data/eval/khop \
      --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
      --gsm-train E/src/data/student_gsm8k_tf_vt/train.jsonl

(1,435 s; log E/src_khop/logs/build_khop.log; tokenizer files fetched from the HF hub into the local cache, no model.)

All 12 files plus build_report.json are in E/src/data/eval/khop/:

| File | Rows | md5 |
|---|---|---|
| khop_ca_fwd.jsonl | 1,100 | 46165e7174d125f6acfabada8610bc22 |
| khop_ca_rev.jsonl | 1,100 | b36b1abef4ce10c05f7721cb2f0bb7e3 |
| khop_wp_fwd.jsonl | 1,100 | 24a05544239c09bc50fa6d2c8fdc660f |
| khop_wp_rev.jsonl | 1,100 | 54b4f1a60f65d5890bbac6417eb28bd6 |
| khop_vb_fwd.jsonl | 1,100 | 7a023cda7092144421ed91fa18174327 |
| khop_vb_rev.jsonl | 1,100 | 156a2fdb5ed817c4b77a3c8569974fff |
| khop_cag_fwd.jsonl | 900 | 279e9f524ec094c30f50f3c4925328d8 |
| khop_ca_fwd_qs.jsonl | 900 | f10bf128e74155f9f8bc6fd165cd5f04 |
| khop_blind.jsonl | 324 | 155db4c5f7b86fc7b3937a85619cf54c |
| train_kh.jsonl | 3,000 | 06369ba8a7c927a2b4524055f540a81e |
| train_k1.jsonl | 3,000 | 2ff1e906fbbb9a51a5a42406a53332e9 |
| train_kd.jsonl | 9,000 | 5a4c8f05d2483fff1018add3fc409229 |
| build_report.json | - | 7f141a639b5c7ce64e418dda5ab7c04e |

Where the training data must go: the card's place, E/src/data/eval/khop/ (passed by path in XARGS). The task text
asked for E/src/data/student_khop_<name>/, but jobs/build_bundles.sh copies only E/src/data/eval,
student_gsm8k_tf_vt and student_gsm8k_tf_si, so a student_khop_* directory would not reach the bundle. data/eval/khop
is copied whole. Pre-submit check for sev-e0: `tar tzf cotd_code_data_x1khop_4b.tgz | grep data/eval/khop/` lists the
12 files (+ build_report.json), and `tar tzf ... | grep condor/jobs_x1/7[5-7]` lists the 22 envs.

D-lines (build_report.json, all pass, d_fail = []):
- D1: ten baselines (the card says "all nine"; it lists 7 + 3 = 10). Highest cell value 0.32 (khop_ca_fwd / rev k = 7,
  pick-min, 100 rows; bar 0.33); highest pooled value 0.233 (khop_cag_fwd median; bar 0.25). Every rule is an
  expected accuracy (uniform among its candidates; no candidate: 0.2), computed from the prompt alone.
- D2: gold share 0.198-0.205 in every non-blind cell; blind pooled per task 0.194-0.204.
- D3: 0 shared prompt hashes, 0 shared value keys (3 eval items redrawn for a value-key clash: ca k1, ca k2, wp k1).
- D4 (Qwen3.5-4B tokens of prompt + "\n\nAnswer:"): longest eval prompt 550 (wp k = 8), longest training prompt 441;
  far below 1,536.
- Resample rate (draws per kept item): ca 30 / 312 / 837 / 1,351 / 1,946 / 2,311 / 2,704 / 2,713 at k = 1..8; wp
  similar (k = 8: 3,330); cag 21-654; vb 1.

Generator readings the card left open, or where I deviated (all in the script docstring):
1. Resample cap 1,000,000, not 10,000. The card's rule needs about 2,300-3,300 draws per item on average at k = 6-8
   (author's own sim gives 3,255 at k = 6, max 7,514 of 12; the tail passes 10,000 often). At 10,000 the build
   stopped on ca k = 6. The cap is a safety stop only; the sampled distribution is unchanged.
2. "A number written in the text" = every integer written in the statements, so the multipliers 2 and 3 of "× 2" /
   "× 3" count (stricter than sim_khop_a1.py, which lists roots and addends only).
3. first-literal / last-literal baselines = the option closest to the first / last written number (an exact match is
   distance 0), not "equal to it", so the rule is informative in ca / wp where no option equals a written number.
4. Bars: D1 per-cell bar only on cells of 100-150 rows and pooled bars on every set except khop_blind (18-row cells;
   blind baselines are reported, and I3 / P5 measure the option-only shortcut on the model). D2 for khop_blind is
   pooled per task (18 rows cannot meet [0.17, 0.23] in a cell). Training files: cells (task, k), pooled per task.
5. Gold layout: in cell (set, k) item i puts T at (i + 3 (k - 1)) mod 5 and D1 at (p + 1 + (i // 5) mod 4) mod 5, the
   other three in a seeded shuffle. This balances the -qs twin (gold D1) and the blind cells pooled per task.
6. Eval rows also carry label_order and gold_probs (the card's 1.1 field list names them); key set = 32 + 5.
7. train_kh: k = 1 + j mod 6, order alternating in blocks of 6 rows: 502 fwd / 498 rev per task (card: "1/2 each").
   KD = the KH rows + 6,000 rows from the same rng stream (2,000 per task, same k / order rule).
8. Blind rows keep the item_id of their fwd item; order field "blind", group khop/<task>/blind/k<k>.

## 4. Jobs and exact XARGS

Training envs = copy of 663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.env with TAG, SEED = ORDERSEED, SAVE, XARGS, JOBKEY
(`4b gsmvt A2-<TAG> s<s> fsdp x1khop`) and EXTRA_EVALS (663 string + the nine khop sets) changed. EPOCHS=1, LR=3e-6,
DATA=data/student_gsm8k_tf_vt, MBS 2, GA 16, LSMOOTH 0.1, NGPU 1 asserted unchanged. Eval envs = copy of
673_4b_evb1_w663.env with WTAR, SETS (the nine khop sets), JOBKEY changed and XARGS=''. Every env was sourced under
bash and printed (test_prep_khop; manual check of 750 / 756 / 771).

| Job | Stage | Arm | Name | XARGS |
|---|---|---|---|---|
| j750 | smoke DAG (real) | BASE0 | 750_4b_evkh_base (WTAR='') | '' |
| j751 | pilot | A2Z s16 | 751_4b_evkh_w663 | '' |
| j756 | smoke (SMOKE_ONLY, name smoke_756_...) + pilot, SAVE=1 | KH s16 | 756_4b_gsmvt_A2-khop-c1lr3ch_s16_fsdp | `--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl` |
| j757 | pilot | K1 s16 | 757_4b_gsmvt_A2-khop1-c1lr3ch_s16_fsdp | `--eval-value-render vt --extra-train data/eval/khop/train_k1.jsonl` |
| j752-755 | confirm | A2Z s17-20 (w664, w665, w540, w666) | 75x_4b_evkh_w<id> | '' |
| j758-761 | confirm, PRE | KH s17-20 | 75x/76x_4b_gsmvt_A2-khop-c1lr3ch_s<s>_fsdp | as j756 |
| j762-765 | confirm, PRE | K1 s17-20 | 76x_4b_gsmvt_A2-khop1-c1lr3ch_s<s>_fsdp | as j757 |
| j766-770 | confirm, PRE | KD s16-20 | 76x/770_4b_gsmvt_A2-khopd3-c1lr3ch_s<s>_fsdp | `--eval-value-render vt --extra-train data/eval/khop/train_kd.jsonl` |
| j771 | confirm | A2Z96 (w667) | 771_4b_evkh_w667 | '' |
| j772-774 | spare, not written | | | |

PRE line on every confirmation training node: `check_pilot.sh 756_4b_gsmvt_A2-khop-c1lr3ch_s16_fsdp
757_4b_gsmvt_A2-khop1-c1lr3ch_s16_fsdp`. Resources as the b3v2 DAGs: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300,
RETRY 1, POST post_push.sh.

Order for sev-e0:
1. Bundle: `REPO_DIR=E/src_khop/repo jobs/build_bundles.sh "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop" x1khop`
   (the x1b3v2 pin read from bundle_out/x1b3v2/CODE_PIN + "-khop"); check the tar listing (section 3) and md5 both sides.
2. smoke_x1khop_4b.dag. Smoke gate: exit 0 for both; j750's tarball has the nine khop preds files (1,100 x 6, 900, 900, 324):
   `cd E/readout && python3 -I -c "import sys; sys.path.insert(0, '.'); import khop_table as T; r = T.load_run(sys.argv[1]); bad = {s: len(r['preds'].get(s, {})) for s in T.SETS if len(r['preds'].get(s, {})) != T.ROWS[s]}; print('j750 khop preds', 'OK' if not bad else bad)" <750 tgz>`
3. pilot_x1khop_4b.dag (j751, j756, j757).
4. `python3 -I E/readout/khop_pilot_gate.py --base <750 tgz> --a2z s16=<751 tgz> --kh <756 tgz> --k1 <757 tgz>
   --ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz > gate.txt`
5. GATE PASS: `python3 -I E/stage/prep_khop_confirm.py --gate-output gate.txt` writes confirm_x1khop_4b.dag (18 nodes).
   It refuses unless the last line is `GATE PASS` and every confirmation env still equals the generated text.
6. Readout: `bash E/readout/run_khop.sh` once all tarballs are in E/results.

## 5. Compute

Token totals (Qwen3.5-4B, build_report.json): gsmvt train 586,902 tokens / 5,923 rows; train_kh 595,189;
train_k1 595,011; train_kd 1,786,775. So KH / K1 double the training tokens of A2 and add 3,000 rows (279 updates
vs 186); KD has 14,923 rows (466 updates). j663 ran 30.6 min wall: 2.5 min smoke, about 12 min training at about
3.9 s per update (mostly cpu-adamw), the rest evals and the save.

| Job | Estimate | Basis |
|---|---|---|
| smoke j756 | about 0.1 GPU-h | j663 smoke 2.5 min + model load |
| j750, j751 and each eval-only job | about 0.2 GPU-h | 7,800 short rows (khop eval prompts 48-550 tokens) + 9 GB weights extract |
| j756, j757 (and each KH / K1) | about 0.75 GPU-h | 279 updates x about 4.2 s = 20 min + j663's evals (about 14 min) + khop evals + save |
| each KD | about 1.0 GPU-h | 466 updates x about 4.3 s = 33 min + evals |

Pilot: about 0.1 + 0.2 + 0.2 + 2 x 0.75 = 2.0 GPU-h (cap 4; card estimate 3.5). P7 bar 1.6 h per job.
Confirmation: 8 x 0.75 + 5 x 1.0 + 5 x 0.2 = about 12 GPU-h (card estimate 23.2; cap 25). The card's numbers are the
upper bound printed by prep_khop_confirm.py.

## 6. Risks

1. Weights j667 (A2-96k s16) for j771 and j664 / j665 / j666 for A2Z: the card says they are on staging (SAVE=1 envs,
   checked: 663-667 envs have SAVE=1); not verified on the AP from here. A missing tar fails that eval job only.
2. The sbatch smoke change touches the shared launcher; the flag-off path is byte-identical (stub-python test against
   repo.base), but the stub test is the only coverage of the new branch (the DRY_RUN xtrace tests cannot run on
   macOS bash 3.2).
3. P7 / compute estimates are from j663 timing on an A100-80GB; a 40 GB GPU or an L40S is slower. P7 is 1.6 h, about
   2x the estimate.
4. D1 cell max 0.32 against a 0.33 bar (ca k = 7, pick-min, 100 rows, reported only). It passes; the bar was set by the
   card.
5. Card deviations listed in section 3 (resample cap, stricter literals, first / last literal as "closest", blind
   bars pooled, two extra eval keys, 502 / 498 order split) and section 1 (smoke subset). A cold critic should read them.
6. Readout: khop_table.nearest_share imports build_khop.parse_prompt by path from E/src_khop/repo/scripts/; the
   readout depends on that directory staying in place (the pilot gate uses it for an INFO line only).
7. Readout: the nearest-chain share counts ties (several distractor chains at the same distance) with expected share
   |nearest| / 4 per item; it is reported only.

## Fix round (2026-10-26 00:30 CDT, after E/src_khop/critique/BUILD_CRITIC_khop.md, verdict SUBMIT AFTER FIXES)

Nothing was bundled, copied to CHTC or submitted. No cluster command was run. Card Amendment 2 was written at 00:21
CDT, before the rebuild of the data (00:22 CDT). The Amendment 2 author check ran on other seeds (train 101, eval 102),
so the real files were not seen before the amendment. Sections 3 and 5 above describe the Amendment 1 data and are
superseded by this section where they differ.

### Findings

| # | Severity | Disposition | Where |
|---|---|---|---|
| 1 | BLOCKER, magnitude leak | FIXED: shared skeleton; 2 new barred D1 lines, 1 eval-only barred line, 5 reported lines; data rebuilt | scripts/build_khop.py, tests/test_build_khop.py, card 1.1 / 2.4 / section 0 item 2 |
| 1e | optional | TAKEN: chase-sign agreement and hard-subset accuracy per run | readout/khop_table.py, test_khop_table.py |
| 2 | MODERATE, P7 allows > 25 GPU-h | FIXED: new gate line P7b | readout/khop_pilot_gate.py, test_khop_pilot_gate.py, card section 3 |
| 3 | MINOR, P0 trusts d_pass | FIXED: P0 re-checks every barred D1 value and every D2 share | readout/khop_pilot_gate.py (d_bar_violations), tests |
| 4 | MINOR, MAX_RESAMPLE 1,000,000 | FIXED: 10,000 (card value); the new rule needs about 2 draws per item | scripts/build_khop.py |
| 5 | NOTE | Stated below | this section |

### Finding 1: what changed in the generator

Cause (critic, confirmed): under Amendment 1 the five chains had independent roots in [2, 60]. A policy that chases
the target to its root, applies its x2 / x3 ops and adds +/- 5.5 per +/- hop picked the gold at 0.85-1.00 for k <= 2.

Fix (draw_arith_item, ca / wp / cag): one shared skeleton per item.
- One root value r for the five chains, one op-kind sequence for the five chains; only the constants differ.
- Hop 1 is + or - (uniform over the kinds whose constant range holds 5 values), five distinct constants without
  replacement. Without this, a x2 / x3 hop 1 makes the hop-1 set (K1 options) five equal values.
- Hops 2..k: uniform over the kinds that keep all five chains in range; each chain draws its own constant.
- Option rule, for the final-value set and the hop-1 set: distinct; no option equals a written integer (roots,
  constants, the multipliers 2 / 3); no option is 2x or 3x another (mult_pair).

Why this closes the leak: the target's root and op kinds are the same for all five options. So any policy that does not
read the target's constants cannot tell the target from a distractor. The chains are exchangeable and the gold position
is balanced, so its expected accuracy is exactly 0.20. The critic proposed roots in a band of +/- 3. I did not use that:
the chase policy still reads a root difference of 1-3 after the multipliers (x 2^k), so the leak would only shrink.

Traded rule (critic 1b): "no two options 2-9 apart" is dropped. Five options that differ only by constants from [2, 9]
cannot meet it. The leak it guarded against is still measured: one_op_parent is 0.19-0.21 in every cell.

Also changed:
- row_rule_ok checks the skeleton on every row: one root value, one op-kind sequence, hop 1 + / - with five distinct
  constants.
- draw_chain is removed (dead code).
- The hop-1 set check keeps every rule.

New D1 lines (d1_baselines in build_report.json):

| Line | Bar | What |
|---|---|---|
| chase_mult | barred (all non-blind files) | nearest option to root x the target's x2 / x3 ops (+ / - ignored) |
| chase_sign | barred (all non-blind files) | the same with every +c / -c at +/- the mean constant (5.5; cag 55) |
| qop_avoid | barred on eval sets, reported on train files | uniform over the options that are NOT query-op pair candidates |
| chase_last1 | reported | only the queried statement's constant exact (1.00 at k = 1 by construction) |
| chase_drop1 | reported | every constant exact except hop 1 |
| chase_droplast | reported | every constant exact except the last hop |
| drop_one_best | reported | per row, the best "all constants exact but one additive hop" estimate (an oracle upper bound) |
| hard | reported | share of rows where no drop-one estimate puts the gold among its nearest options |

Why qop_avoid is barred only on eval sets: on a shared root, the target of a hop-1 query (r + c_T) is never a
query-op pair candidate, because its parent r is not an option. So the "avoid" policy is about 0.26 on hop-1 option
sets: eval k = 1 cells and every train_k1 cell (0.25-0.27; pooled ca 0.256, wp 0.260, which is above the 0.25 pooled
bar). On the eval sets it is 0.18-0.26 per cell and 0.20-0.21 pooled. The only fix I found is a sum-free constraint on
the hop-1 constants, which forces them into {5..9} and distorts the task. K1 is the format control and is read on the
eval sets, so the edge on its training rows is reported, not barred. I did not add this line to the card before the
build in Amendment 1; Amendment 2 defines it.

Why the partial lines are reported, not barred (advisor review in this round): numeric options let a model that reads
k - 1 of the k constants pick the gold often (drop_one_best ca fwd: 0.36 at k = 2, 0.64 at k = 3, >= 0.86 at k >= 4).
This is not an option-design bug: the distractors differ from the target in every hop, so one wrong or skipped
constant often leaves the gold nearest. The card now says (section 0 item 2) that the word "composes" needs the hard
subset of (ca, fwd, 3) at m >= 0.80; else the verdict is "PASS (approximate)". khop_table.verdict() implements this, and
test_khop_table.test_pass_approximate covers it.

### Data (rebuilt 2026-10-26 00:22 CDT, logs/build_khop.log; the Amendment 1 log is logs/build_khop.a1.log)

Same command as section 3 (Qwen3.5-4B tokenizer, --gsm-train), 9.5 s (was 1,435 s). d_pass true, d_fail [].

| File | Rows | md5 |
|---|---|---|
| khop_ca_fwd.jsonl | 1,100 | 171f097c2f0f55ae4505d37365d95491 |
| khop_ca_rev.jsonl | 1,100 | 2f2bc7b6fc1cfd45f24451741f0e4105 |
| khop_wp_fwd.jsonl | 1,100 | c58c93ffa4d3912cfdbfd848f6eb9fb9 |
| khop_wp_rev.jsonl | 1,100 | bdad8780cc64e94e3fcdd0148833a4d7 |
| khop_vb_fwd.jsonl | 1,100 | 046c6ad382d66c02f1315ce7f8c7577d |
| khop_vb_rev.jsonl | 1,100 | 2279c09f4adb666196bdbbe636b53646 |
| khop_cag_fwd.jsonl | 900 | 3e79c1b5233e3db7dc23927e3066eba5 |
| khop_ca_fwd_qs.jsonl | 900 | 38cf266ea1490f0c42f6378baa85f4e4 |
| khop_blind.jsonl | 324 | b58f7284e1b1d88197a44d0b54eba2ae |
| train_kh.jsonl | 3,000 | a16b1520f0fb8b24ffd8fa57ef696de0 |
| train_k1.jsonl | 3,000 | 4bb15f66c8d8843013eb0410c6a87e48 |
| train_kd.jsonl | 9,000 | 2ed4ecc377e4638e35b12e8b22e37461 |
| build_report.json | - | 33e94aff1632e69f929b89533c49e35e |

The vb files changed too. vb items come from the same seeded stream after the ca / wp items, so a different number of
draws for ca / wp moves the vb values. The vb generator itself did not change.

D-lines of the real build:
- Barred D1 lines: highest cell 0.283 (khop_ca_fwd_qs k2 first_lit; bar 0.33); highest pooled 0.232 (khop_wp_fwd /
  rev chase_mult; bar 0.25).
- chase_sign per cell, ca fwd: 0.19 0.18 0.16 0.22 0.15 0.18 0.21 0.18 (k = 1..8; pooled 0.18). chase_mult ca fwd:
  0.18-0.25 (pooled 0.22).
- The critic's own script (critique/chase_only.py) on the new files: root / mult / signs policies 0.14-0.27 in every
  ca / cag cell (was 0.49-1.00).
- Reported, khop_ca_fwd k = 1..8: chase_droplast 0.19 0.72 0.54 0.61 0.55 0.61 0.60 0.64; drop_one_best 0.19 0.36 0.64
  0.86 0.89 0.90 0.94 0.94; hard share 0.72 0.53 0.28 0.10 0.05 0.07 0.03 0.02. The hard subset of (ca, fwd, 3) has 42
  items.
- Resample rate (draws per kept item): ca 1.21 / 2.45 / 2.73 / 2.33 / 2.15 / 1.99 / 1.83 / 2.01 at k = 1..8; wp
  1.18-2.54 (k = 6, 7, 8: 1.94 / 1.83 / 1.89); cag 1.06-1.29. The critic's feasibility question for k = 6, 7, 8 is
  answered: about 2 draws per item.
- D2 gold shares in [0.17, 0.23] in every barred cell (D2 lines pass). D3: 0 shared prompt hashes, 0 shared value
  keys; 47 eval items redrawn for a value-key clash (ca k1 17, ca k2 2, wp k1 21, wp k2 7; k = 1 items have fewer
  distinct values on a shared skeleton). D4: longest prompt 549 tokens.
- Token totals: train_kh 595,887; train_k1 595,865; train_kd 1,787,847 (gsmvt 586,902). Section 5 compute estimates
  stand.

### Findings 2 and 3: gate

- P7b (new line): projected confirmation = 8 t + 5 x (14,923 / 8,923) t + 5 e <= 25 GPU-h. t = mean wall of j756 and
  j757; e = max(0.2, mean wall of j750 and j751). P7 (<= 1.6 h per job) stays. Test: KH 1.55 h and K1 1.55 h pass P7
  and fail P7b alone (26.4 GPU-h). The P7 fixture (KH 1.75 h, K1 1.0 h) fails P7 alone (P7b 23.5).
- P0 now calls d_bar_violations(report). The gate keeps its own list of the 12 barred lines and of the eval-only line.
  It fails P0 when a barred line is missing, a cell of >= 100 rows is above 0.33, a pooled cell is above 0.25, or a
  gold share is outside [0.17, 0.23]. Tests edit one number of the real report each time: a cell, a pooled cell, a
  missing line, qop_avoid on an eval set (each one fails P0); qop_avoid on a train file, a reported line, and an
  18-row blind cell (each one passes).

### Finding 5 (note)

The repo.base comparison (logs/cmp_repo.log, logs/cmp_repo.base.log) covers 23 tests only: tests.test_tree_target and
tests.test_probe_slots. Both trees have the same 5 failures: 3 in test_tree_target and 2 in test_probe_slots.TestSbatch
(macOS bash 3.2 xtrace). The card names only the 3 test_tree_target failures. The 2 TestSbatch failures are on the file
this diff touches (hpc/train_arm.sbatch), so they cannot detect a regression in the smoke block. That change is covered
by tests/test_extra_train.TestSbatchSmokeSubset instead: flag off, every call equals the snapshot's sbatch (smoke and
SMOKE_ONLY included); flag on, the smoke subset, two files and a missing file. This round did not change the sbatch or
train_student.py.

### Tests (run_tests.sh, test_all.log, 2026-10-26 00:23-00:27 CDT)

| Suite | Result |
|---|---|
| repo: test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train | OK, 66 tests, 1 skip (test_frontier.TestRealData needs data/) |
| readout: test_khop_table, test_khop_pilot_gate | OK, 17 tests |
| stage: test_prep_khop | OK, 4 tests |

New or changed tests:
- test_build_khop: test_option_rule checks the new signature and the skeleton on 240 drawn items (ca / wp / cag, k = 1,
  2, 4, 8). test_skeleton_rule_on_rows checks that row_rule_ok refuses a different root and a repeated hop-1 constant.
  test_chase_lines_and_hard checks every new line on a hand row (with a tie case and a vb row). The real-file tests
  recompute every D1 line from the rows and compare it with the report, and they apply the skeleton rule to every row.
- test_khop_eval_paths reruns on the new rows and passes; every real row fits 1,536 tokens.
- test_khop_table: test_pass_approximate. TestRuns checks hard_ids / hard_curve against build_khop.hard_item, and checks
  chase_agree on a run that always picks the chase-sign option (1.0) and on the gold-picking run (< 0.6).
- test_khop_pilot_gate: the P7b case and the P0 re-check cases above.

### Envs, subs, DAGs, diff

- `python3 -I stage/prep_khop.py` was rerun. All 22 envs (750-771), both subs and both DAGs are byte-identical to the
  previous ones (md5 lists critique/env_md5_before.txt and env_md5_after.txt; 26 files). Expected: the envs give data
  paths, not md5s. `prep_khop_confirm.py --dry-run` still prints 18 jobs. Ids 772-774 stay unused.
- build.diff was regenerated (`diff -ruN -x __pycache__ -x data repo.base repo`, 1,565 lines, md5
  330868476d870bc6a50babba3316eedc). Files: hpc/train_arm.sbatch, scripts/build_khop.py, scripts/train_student.py and
  the three new tests.

### For sev-e0 (order of section 4 unchanged)

1. Rebuild the bundle from the current E/src_khop/repo and E/src/data/eval/khop, with the same command and pin as
   section 4 step 1. Any x1khop bundle built before 00:22 CDT 2026-10-26 has the Amendment 1 data; do not use it. The
   tar listing must show the 12 files plus build_report.json with the md5s above. P0 checks them against this report.
2. Then smoke, pilot, gate, and confirmation as in section 4. GATE PASS now also needs P7b.

### Remaining risks

1. The hard subset of (ca, fwd, 3) is 42 items per seed. With five seeds, m has an SE of about 0.03 at 0.8, so the
   "PASS" / "PASS (approximate)" split near 0.80 is noisy. The full-cell lines (k*, G2) do not depend on it.
2. On a shared root, ca has no root-value binding part, and the five wp people all start with the same count. This
   makes the task less like GSM. The claim scope (section 0 item 2) already lists "the option design" as a candidate
   for the gap.
3. qop_avoid is about 0.26 on train_k1 (above the 0.25 pooled bar, reported only). K1 could learn a 6-point "avoid"
   rule at hop 1. This affects only the K1 control. The KH - K1 lines are read at k = 2, 3, where the line is about
   0.20.
4. The partial-composition levels (chase_droplast about 0.55-0.72 at k >= 2) mean that a full-cell SOLVED at k = 3 can
   come from a model that reads 2 of 3 constants. The hard-subset rule only makes this visible in the claim text. It
   does not change k*.

## Amendment 3 build (2026-10-26, readout only)

Spec: card section "Amendment 3". Cause: E/src_khop/critique/FIXROUND_CRITIC_khop.md, finding 1. No cluster command
was run. The data, E/src_khop/repo, the envs, the DAGs and the bundle did not change: the md5 of 16,127 files under
src/data/eval/khop, src_khop/repo, stage, bundle_out and readout/khop_pilot_gate.py is equal before and after the
edit. Pre-change copies: critique/khop_table.pre_a3.py, critique/test_khop_table.pre_a3.py.

### What changed in readout/khop_table.py

- verdict(): the word "composes" needs m(a_KH(ca, fwd, 3 | n_add = 3)) >= 0.80 (BAR "alladd"). Else the reading is
  "PASS (approximate)". The hard subset is a reported line only. k*, the SOLVED / FAILED bars and G2 on the full cell
  do not change.
- claim(): a PASS reading prints "claim: ... composes 3 + / - steps ...", names the shared x2 / x3 steps and the
  all-+/- mean, and keeps the rc2 part of the card claim. Its last clause is now "the rc2 ceiling is not a limit on
  composing given + / - steps". A "PASS (approximate)" reading prints "selects the value of a 3-step chain" and the
  all-+/- mean. The text has no "k given operations".
- Reported lines: (wp, fwd, 3 | n_add = 3) beside the bar, and G2 on (ca, fwd, 2 | n_add = 2) with its CI. This G2
  line has the same rc2 reference as G2 and decides nothing.
- Every ca / wp / cag / ca_fwd_qs curve is also printed by n_add, and the all-+/- curve gets its own line.
- Rank probes rank_sum (signed sum of the + / - constants) and rank_h1 (signed first constant). The readout prints
  them per cell as reference lines and as agreement with each run's picks. The values equal the critic's probe.out.

### Item counts (from the data files, n_add = k)

| cell | items | card / critic |
|---|---|---|
| (ca, fwd, 3) | 48 | 48 |
| (wp, fwd, 3) | 67 | 67 |
| (ca, fwd, 2) | 63 | 63 |
| (wp, fwd, 2) | 66 | not stated |
| (cag, fwd, 3) | 50 | not stated |

The eval files are the same for every seed, so "48 items per seed" is the same 48 items.

### Tests

- test_khop_table: 21 tests, OK (12 before). The new tests cover the 0.80 flip of the all-+/- line, the hard line as
  reported only, the claim text, G2 all-+/- as a reported line, the counts from the data, the probe reference values,
  the split curves, and two end-to-end runs (PASS and PASS (approximate)).
- test_khop_pilot_gate: 5 tests, OK (unchanged file).
- Full readout suite (python3 -B -m unittest in E/readout): 104 tests, 1 skip, 1 error. The error is in test_pfx_table.
  That file is a script that needs argv and is not a unittest module. It is not related to this change.
- Critic khopcrit/test_flip.py: 15 of 18 pass. The 8 gate cases pass. Verdict.test_hard, test_g2 and test_solved_k3
  fail by design: their fixture gives only "kh_hard", so under Amendment 3 the reading is "PASS (approximate)". A copy
  with "kh_add3" set to the same values (<tmp> passes 18 of 18.

### Not implemented as the task text said

1. The task asked for a test where a signed-sum model reads "PASS (approximate)". This contradicts Amendment 3. On an
   all-+/- item the signed sum is the exact value, so rank_sum is 1.00 there, and Amendment 3 counts that as
   composing the + / - steps. The test asserts that rank_sum reads "PASS" with the "+ / - steps" claim. It also
   asserts that a first-constant reader (rank_h1, 0.375 on the 48 items) reads "PASS (approximate)".
2. The critic (fix c) suggested to drop the transfer sentence when the all-+/- G2 CI includes 0. Amendment 3 item 3
   makes that line reported only, so nothing depends on it.
3. Claim text changed only for PASS and PASS (approximate). Amendment 3 does not change the NEGATIVE / NULL / BETWEEN
   text, and khop_table.py does not print those claims.
4. The critic's advisory rank_last line is not added. Amendment 3 names only two probes.
5. khop_pilot_gate.py does not print the "composes" claim, so it did not change.
6. Remaining risks 1 and 4 above refer to the Amendment 2 hard-subset rule. Under Amendment 3 they apply to the
   48-item all-+/- subset of (ca, fwd, 3).
