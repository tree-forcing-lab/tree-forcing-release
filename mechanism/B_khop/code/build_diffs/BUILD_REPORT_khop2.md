# Build report: CARD_khop_v2 (family khop2, ids 875-899), 2026-10-26 11:00 CDT

E = <tmp>
Spec: <path> with Amendment 1. Nothing was bundled, copied to CHTC or
submitted. No cluster command was run. The card file was not edited.

The owner's 2-epoch branch runs as R3 / H3 (3 repeats inside the chain file, matched updates and gsmvt exposure). The
literal EPOCHS=2 arms (KD-2ep, KH-2ep) need a new card and a new id range. The gate, the confirmation generator and the
readout print this sentence; every message to the second author about this family must carry it (card sections 1.2, 3, 4).

## 1. What changed

Code (E/src_khop2/repo; E/src_khop2/build.diff = `diff -ruN -x __pycache__ repo.base repo`, 626 lines, md5
febf6e9b048b37178452f91fffda7209). Two new files, no edited file:

| File | Content |
|---|---|
| scripts/build_khop2.py | The card 1.2 generator. It imports scripts/build_khop.py and does not edit or call build_khop.build(). |
| tests/test_build_khop2.py | Generator, real-file and A2-step tests (section 2). |

No trainer flag and no trainer change. The A2 path of D27 / R3 / H3 is the parent's code byte for byte (D5, below).
`grep problem_id` over repo/src and repo/scripts (build_* excluded) finds nothing, so the unchanged problem_id of the
copy rows reaches no trainer path. The trainer refuses duplicate item_ids (train_student.py line 1043); the copy rows
carry "-r2" / "-r3" suffixes.

D5 (card 2.5), printed by run_tests.sh into test_all.log for both references:
`diff -rq -x __pycache__ -x data E/src_khop/repo E/src_khop2/repo` and the same against repo.base list exactly
"Only in .../repo/scripts: build_khop2.py" and "Only in .../repo/tests: test_build_khop2.py".

Generator rule (build_khop2.py, card 1.2 steps 1-4):
1. Read the 12 parent files from --src. A missing file or an md5 different from the parent report is a SystemExit.
   --out equal to --src is a SystemExit.
2. Forbidden sets: the value key (row_key: task + sorted chain value tuples parsed from the prompt, build_khop.build's
   rule) and the prompt hash of every row of the nine eval files (blind rows: hash only) and train_kh / train_k1 /
   train_kd.
3. New rows: random.Random(3), task-major over ca, wp, vb, j = 3000..8999, k = 1 + j % 6, order fwd when (j // 6) % 2
   == 0, item_id khop-tr-<task>-<j:05d>. Per row: gen_item, then the value-key check, then layout(i, k, rng) and
   row(), then the prompt-hash check. A failed check redraws the item (at most 2,000 redraws, else BuildError). i
   continues the per (task, k) cycle of the parent: it starts at 500 and advances only on a kept row.
4. train_kd27 = the train_kd lines byte for byte + the 18,000 new rows. train_kd_x3 / train_kh_x3 = copy 1 byte for byte,
   copies 2 and 3 with item_id + "-r2" / "-r3" and every other field unchanged (problem_id too).

Stage (E/stage, new files only):

| File | Content |
|---|---|
| prep_khop2.py | Writes the 15 envs 875-889, train_x1khop2_4b.sub, smoke_x1khop2_4b.dag, pilot_x1khop2_4b.dag. Refuses another env on an id of the family and any env on the spare ids 890-899. |
| prep_khop2_confirm.py | Writes confirm_x1khop2_4b.dag only after a gate output whose last line is "GATE PASS" (13 nodes) or "GATE PASS (H3 dropped: projection <x> GPU-h)" (8 nodes). Refuses anything else and any confirmation env that differs from prep_khop2.py's text. `--dry-run [--h3-dropped]` prints the table. |
| test_prep_khop2.py | Env, sub, DAG and confirmation-generator tests. |
| train_x1khop2_4b.sub | train_x1khop_4b.sub with lines 13 (CODE_TGZ), 16 (transfer_input_files) and 33 (log) changed, plus header comment lines 1 and 3 (line 3 is a comment that names the bundle file). Tested: exactly these 5 lines differ and only 13, 16, 33 are not comments. |
| smoke_x1khop2_4b.dag | smoke_875 (D27 env) and smoke_880 (R3 env), SMOKE_ONLY smoke="1": one node per new path (the 27,000-row file and the 3-copy file). |
| pilot_x1khop2_4b.dag | j875 (D27 s16, SAVE=1) and j880 (R3 s16). |
| condor/jobs_x1/875-889 envs | 15 envs (section 4). |

No eval sub, no eval_x1khop2_4b.dag: the card has no eval-only job (1.4). All reference runs are existing tarballs.

Readout (E/readout, new files only):

| File | Content |
|---|---|
| khop2_pilot_gate.py | Pilot gate P0-P9 with P7b; exit 0 for both PASS forms. Imports khop_table and khop_pilot_gate (wall_hours, exit_code, gsm_sets); keeps its own 12 barred D1 lines, its own d_bar_violations over the 4 khop2 stems and the 12 parent md5s of the Amendment 2 table. |
| khop2_table.py | Five-seed readout: curves, I1-I5, verdict() in the card's order with the BETWEEN sub-labels, branch_b() (pooled rule), log-dose slope, k* / fwd - rev, G2 / G3, N1, plain cost of the dose, rc1..rc4 transfer / harm sentences, timing. khop_table primitives by import; khop_table.py is not edited. |
| run_khop2.sh | `bash run_khop2.sh <gate output> [results dir]`: gathers every tarball (D27 875-879, R3 880-884, H3 885-889 unless the gate dropped H3, KD, KH, K1, A2Z, BASE0, A2Z96, both A2-6k sets), stops on a missing one or a gate that did not pass, logs to E/logs/khop2_readout_<date>.txt and .json. |
| test_khop2_table.py, test_khop2_pilot_gate.py | Tests (section 2). |

## 2. Tests

Runner: E/src_khop2/run_tests.sh (venv_pfx python, `python -m unittest`; reference trees in section 2.1). Output:
E/src_khop2/test_all.log. Result of the final run: see section 2.1.

| Suite | What it checks |
|---|---|
| repo tests.test_build_khop2 (13 tests, new) | Copy rule: copy 1 byte for byte, copy c = parent with item_id + "-r<c>", a reverted copy re-serializes to the parent line, copies_ok refuses a changed field or a changed copy 1; row_key = build_khop.value_key on generated items; small builds on the real parent files: determinism by seed (seed 3 twice equal, seed 4 different, copies independent of the seed), parent md5 before = after = Amendment 2 table, a changed parent md5 and --out = --src refused, the redraw cap raises BuildError. Real files: md5 and counts (1,500 per (task, k) in train_kd27, 500 per (task, order, k) in the new rows), train_kd27 starts with train_kd byte for byte, every new row's id / k / order / group / gold letter (layout with i from 500) / key set / problem_id, row_rule_ok on all 18,000 new rows, fwd topological, no shared value key or prompt hash with the 12 parent files or among the new rows, D3 per file recomputed = report = 0, 26,941 distinct keys in train_kd27, copy files = 3 x parent, ids distinct, D1 / D2 recomputed = report and within the bars, D4 recomputed with the Qwen3.5-4B tokenizer = report. A2 step: 1-rank FSDP (torchrun --local-addr 127.0.0.1, cpu-adamw, --eval-value-render vt, label smoothing 0.1) on 6 GSM rows + an 18-row 3-copy khop file: exit 0, the log line "extra-train: 18 rows ...", the total line, metrics extra_train, final step = ceil(24 / 4) = 6, finite losses. |
| repo parent tests (test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train) | Unchanged parent tests; test_extra_train.test_flag_off_identical_to_reference and test_frontier_train run against repo.base. |
| readout test_khop2_pilot_gate (6 tests, new) | A good pilot passes; P1 (khop set, GSM set), P2 (total, metrics step, log step, NaN loss, wrong file), P3, P4, P5, P6 (gsm8kx and gsm8kv2 at reference - 11), P7 (2.95 h), P8 (smoke exit 1, smoke missing), P9 each fail alone; valx and gsm1kx at reference - 6 pass; P7b: 2.30 h "GATE PASS", 2.45 h "GATE PASS (H3 dropped: projection 26.0 GPU-h)", the bound 25 / (8 + 5 x 0.521) = 2.36 h, 23.2 GPU-h at 2.9 h, fail at 3.2 h; P0 on 15 edits of the real reports (new md5, parent_md5, d_pass, nested, copies, new keys 17,999, D3, a cell, a pooled cell, a missing barred line and a gold share fail; qop_avoid on a train file and a REPORTED line pass; a missing or changed parent report fails); the log step parser on the real j766 log (467). |
| readout test_khop2_table (15 tests, new) | MANIPULATION FAILED from each of I1-I5; PASS ("chain data" on BOTH / ROWS, "chain training" on UPDATES), PASS (approximate) (0.70; missing; 0.80 flips to PASS); NEGATIVE with and without the R3 sentence, not NEGATIVE at D_dose +0.08; NULL with the printed CI upper; each BETWEEN sub-label (data-limited, dose-limited, repetition, lower, k = 3 not k = 2, unresolved); branch (b) BOTH / ROWS / UPDATES / UNRESOLVED, the pooled case (R27 and R9 each CI lower <= 0, rows_pool +7.5 with CI lower > 0), the one-level form (labels and verdict); the three GSM sentences and the plain-cost case; the log-dose slope on hand cases; I4 on 6 bad fixtures; end to end through main() with the real KD (j766-770), KH (j756, j758-761) and A2-6k tarballs plus synthetic D27 / R3 / H3 run directories (KD's GSM files, synthetic khop preds): VERDICT PASS, BOTH, R27 = +10.0 at s18, KD rc1 -7.3 / rc2 -5.5 and KH -1.1 / -1.0 against the original set (the card's parent values), the H3-dropped form, the refusals (gate not passed, H3 missing on a full pass), run_khop2.sh refusals (gate not passed, missing tarball). |
| readout parent tests (test_khop_table, test_khop_pilot_gate) and stage test_prep_khop | Unchanged; the parent's files are not touched. |
| stage test_prep_khop2 (9 tests, new) | Section 1.4 envs, the sub (5 lines), the DAGs, the confirmation generator (6 refused gate texts, a changed env, 13 nodes with PRE / POST / RETRY / PRIORITY, 8 nodes on H3 dropped, gate_mode, dry runs write nothing). |

Full-suite regression against repo.base: the repo differs from repo.base only by two new files (D5), so every other
test file runs the same code in both trees. The known parent failures stay (section 2.1).

### 2.1 Final run (test_all.log, 2026-10-26 11:01-11:08 CDT, exit 0)

| Block | Result |
|---|---|
| repo (new): tests.test_build_khop2 | Ran 13 tests, OK |
| repo (parent tests, COTD_REF_REPO = E/src_khop/repo.base) | Ran 66 tests, OK (skipped 1: test_frontier.TestRealData needs data/) |
| repo (flag-off identity, COTD_REF_REPO = E/src_khop2/repo.base): test_extra_train flag off, sbatch without the flag, test_frontier_train | Ran 23 tests, OK |
| readout: test_khop2_table, test_khop2_pilot_gate, test_khop_table, test_khop_pilot_gate | Ran 47 tests, OK |
| stage: test_prep_khop2, test_prep_khop | Ran 13 tests, OK |
| D5 against E/src_khop/repo and against repo.base | D5 OK (only scripts/build_khop2.py and tests/test_build_khop2.py) |

Reference tree for the parent tests: the first run used repo.base for all of them, and
test_extra_train.TestSbatchSmokeSubset.test_smoke_subset failed. Cause: that parent test compares the parent's new
sbatch smoke call with the sbatch before the parent's change; this family's repo.base already holds that change, so the
reference also rewrites the path. The parent tests now run against the parent's own snapshot (E/src_khop/repo.base),
where they pass, and the flag-off identity tests run against this family's repo.base. No code changed.

Regression modules (E/src_khop2/logs/cmp_repo.log): tests.test_tree_target + tests.test_probe_slots in repo give 23
tests, the same 5 failures as the parent's logs/cmp_repo.log (3 test_tree_target, 2 macOS bash 3.2 TestSbatch).

## 3. Data

Command (venv_pfx python, 44 s; log E/src_khop2/logs/build_khop2.log):

    cd E/src_khop2/repo && python scripts/build_khop2.py --src E/src/data/eval/khop --out E/src/data/eval/khop2 \
      --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
      --gsm-train E/src/data/student_gsm8k_tf_vt/train.jsonl

Files in E/src/data/eval/khop2/:

| File | Rows | md5 | Qwen3.5-4B tokens |
|---|---|---|---|
| train_kd27.jsonl | 27,000 | 92545dd42ca451c39e0e7c104aa78d1e | 5,361,925 |
| train_kd_x3.jsonl | 27,000 | f3404df79fca62d379d2754f2553e48d | 5,363,541 (3 x train_kd 1,787,847) |
| train_kh_x3.jsonl | 9,000 | 265c96cd3a8c50c3e804db39da99d386 | 1,787,661 (3 x train_kh 595,887) |
| build_report.json | - | 048941d3a848f2cd9ba6e28c764a7c3a | - |

The 12 parent files in E/src/data/eval/khop/ are unchanged: their md5s equal the Amendment 2 table (tested before and
after the build). d_pass true, d_fail [].

D-lines (build_report.json):
- D1 (12 barred lines, whole files and the 18,000 new rows alone as train_kd27_new): highest cell value 0.270
  (train_kh_x3 vb/k3 last_lit; bar 0.33); highest pooled value 0.213 (train_kh_x3 wp median; bar 0.25). The new rows
  alone: highest cell 0.246 (wp/k3 max), highest pooled 0.206.
- D2: gold share 0.20 in every (file, task, k) cell of train_kd27, train_kd_x3 and the new rows; train_kh_x3 0.198-0.205.
- D3: 0 shared prompt hashes and 0 shared value keys between each new file and the nine eval files; the new rows share
  no value key or prompt hash with any parent file and none among themselves (18,000 distinct keys). The generator's
  value key equals the key parsed from every new prompt.
- D4: longest prompt + "\n\nAnswer:" 439 tokens (train_kd27, wp k = 6); bar 1,536.
- D6: nested true (train_kd27 lines 1-9,000 = train_kd.jsonl); copies_ok true for both copy files; item_ids distinct.
- Distinct value keys: train_kd27 26,941 (the card's expected value; the 59 duplicate-key rows of train_kd stay),
  train_kd_x3 8,941, train_kh_x3 2,996.
- Redraws (value key or prompt hash already used): ca k1 263, ca k2 70, ca k3 3, wp k1 283, wp k2 59, wp k3 2, 0
  elsewhere; no row came near the cap of 2,000. These differ from the author check feas_kd27.out (244 / 77 / 1 / 267 /
  77 / 1) because that check did not call layout(), so its rng stream is different. The card says the check is not the
  build.

What the bundle must contain (jobs/build_bundles.sh copies data/eval whole and stage/condor/jobs_x1):
- data/eval/khop2/: train_kd27.jsonl, train_kd_x3.jsonl, train_kh_x3.jsonl, build_report.json (md5s above).
- data/eval/khop/: the 12 parent files and build_report.json (Amendment 2 md5s; the eval files and the 9 EXTRA_EVALS).
- condor/jobs_x1/875_* .. 889_*: the 15 envs of section 4.
- repo/ with scripts/build_khop2.py and tests/test_build_khop2.py (repo = E/src_khop2/repo).
No data/student_khop2_* directory: build_bundles.sh does not copy it (card 1.2). The new training files are under
data/eval/khop2, which is copied whole. Side effect: every bundle built from E/src/data/eval after this build also
carries these 80 MB (9.7 MB gzip-compressed: 4.15 + 4.15 + 1.38 MB); this does not change any other family's job.

## 4. Jobs and exact XARGS

Training envs = 766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.env with TAG, SEED = ORDERSEED, XARGS, SAVE, JOBKEY changed
(JOBKEY "4b gsmvt A2-<TAG> s<s> fsdp x1khop2"). Every other key equals 766 (tested): EPOCHS=1, LR=3e-6, LSMOOTH=0.1,
MBS=2, GA=16, NGPU=1, FSDP_OPTIM=cpu-adamw, EVALAT=0.5, EVALMID, EXTRA_EVALS (the 663 sets + the nine khop sets),
DATA=data/student_gsm8k_tf_vt. Every env was sourced under bash (test_prep_khop2 and by hand for 875 / 880 / 885).

| Job | Stage | Arm | Name | XARGS (as sourced) | SAVE |
|---|---|---|---|---|---|
| smoke_875 | smoke DAG | D27 env, SMOKE_ONLY | smoke_875_4b_gsmvt_A2-khopd27-c1lr3ch_s16_fsdp | as j875 | 1 |
| smoke_880 | smoke DAG | R3 env, SMOKE_ONLY | smoke_880_4b_gsmvt_A2-khopd9x3-c1lr3ch_s16_fsdp | as j880 | '' |
| j875 | pilot | D27 s16 | 875_4b_gsmvt_A2-khopd27-c1lr3ch_s16_fsdp | `--eval-value-render vt --extra-train data/eval/khop2/train_kd27.jsonl` | 1 |
| j880 | pilot | R3 s16 | 880_4b_gsmvt_A2-khopd9x3-c1lr3ch_s16_fsdp | `--eval-value-render vt --extra-train data/eval/khop2/train_kd_x3.jsonl` | '' |
| j876-879 | confirm, PRE | D27 s17-20 | 87x_4b_gsmvt_A2-khopd27-c1lr3ch_s<s>_fsdp | as j875 | '' |
| j881-884 | confirm, PRE | R3 s17-20 | 88x_4b_gsmvt_A2-khopd9x3-c1lr3ch_s<s>_fsdp | as j880 | '' |
| j885-889 | confirm, PRE (unless H3 dropped) | H3 s16-20 | 88x_4b_gsmvt_A2-khoph3x3-c1lr3ch_s<s>_fsdp | `--eval-value-render vt --extra-train data/eval/khop2/train_kh_x3.jsonl` | '' |
| j890-899 | spare, not written | re-runs only | | | |

Expected log lines (I4 / P2): D27 "extra-train: 27000 rows from data/eval/khop2/train_kd27.jsonl", R3 "27000 rows from
data/eval/khop2/train_kd_x3.jsonl", H3 "9000 rows from data/eval/khop2/train_kh_x3.jsonl"; totals 32923 / 32923 / 14923;
final step 1029 / 1029 / 467. The smoke stage trains on 16 gsmvt rows + the first 16 rows of the extra file, so R3's
smoke sees copy-1 rows only; the suffixed ids are covered by the local tests.

PRE line on every confirmation node: `check_pilot.sh 875_4b_gsmvt_A2-khopd27-c1lr3ch_s16_fsdp
880_4b_gsmvt_A2-khopd9x3-c1lr3ch_s16_fsdp`. Resources as the parent: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300,
RETRY 1, POST post_push.sh.

## 5. Compute

Measured (`python3 -I E/jobs/round_gpuh.py 750 771`): KD j766-770 1.27 / 1.03 / 0.99 / 1.25 / 0.99 h (mean 1.11 h, 467
updates); KH j756, j758-761 0.75 / 0.73 / 0.72 / 0.93 / 0.70 h; K1 0.63-0.89 h; smoke_756 0.04 h; the parent family
13.57 GPU-h in total. The card's fit (6.51 s per update + 0.26 h) gives D27 / R3 2.12 h (worst host 2.37 h) and H3
1.11 h. Tokens per update: D27 (5.95 M training tokens / 1,029 updates, about 5,800) and KD (2.37 M / 467, about 5,100)
are close, so the update-count fit applies; it is still an extrapolation to 2.2x the fitted range, and P7 / P7b measure
the real wall before any confirmation node is written.

| Stage | Jobs | Estimate | Cap |
|---|---|---|---|
| smoke | smoke_875, smoke_880 | 0.1 GPU-h | in the pilot cap |
| pilot | j875, j880 | 4.24 GPU-h (worst 4.74; 5.9 at the P7 bar of 2.9 h) | 6 |
| confirmation | 8 x 2.12 + 5 x 1.11 | 22.5 GPU-h (prep_khop2_confirm.py --dry-run prints 22.51) | 25 (P7b drops H3 when t > 2.36 h) |

## 6. Order for sev-e0

1. Bundle: `REPO_DIR=E/src_khop2/repo jobs/build_bundles.sh "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2" x1khop2`
   (bundle_out/x1khop/CODE_PIN + "-khop2"). Pre-submit checks: `tar tzf cotd_code_data_x1khop2_4b.tgz | grep data/eval/khop2/`
   lists the 3 files + build_report.json; `... | grep data/eval/khop/` lists the 12 parent files + build_report.json;
   `... | grep -E 'condor/jobs_x1/(87[5-9]|88[0-9])_'` lists 15 envs; md5 of the bundle on both sides.
2. smoke_x1khop2_4b.dag. Smoke gate: exit 0 for both nodes.
3. pilot_x1khop2_4b.dag (j875, j880).
4. `python3 -I E/readout/khop2_pilot_gate.py --d27 <875 tgz> --r3 <880 tgz> --kd s16=E/results/results_766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.tgz --ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz --report E/src/data/eval/khop2/build_report.json --parent-report E/src/data/eval/khop/build_report.json > gate.txt`
   (the smoke tarballs are found as results_smoke_875_*.tgz / results_smoke_880_*.tgz in E/results, or pass
   --smoke-d27 / --smoke-r3).
5. Last line "GATE PASS" or "GATE PASS (H3 dropped: ...)": `python3 -I E/stage/prep_khop2_confirm.py --gate-output gate.txt`
   writes confirm_x1khop2_4b.dag (13 or 8 nodes). Submit and tell the second author, with the 2-epoch sentence above.
   "GATE FAIL: ...": submit nothing; report the failing lines to the second author.
6. Readout when every tarball is in E/results: `bash E/readout/run_khop2.sh gate.txt`.

A check of the gate on real data (j766 as D27 and j767 as R3, with smoke_756 as both smokes): P0, P1, P3-P9 pass and
P2 fails on both (9,000 rows of train_kd, total 14,923, step 467), as it must for KD runs. The P6 s16 references print
59.8 (gsm8kx) and 71.1 (gsm8kv2), the card's values.

## 7. Risks and deviations

1. Sub: the card names lines 13, 16, 33 "plus the header comment". Line 3 is also a comment that names the bundle file;
   it was changed with line 1 (both are header comment lines). The test pins exactly lines 1, 3, 13, 16, 33.
2. Gate interface: the card's gate command has no smoke arguments, but P8 needs the smoke exits. The gate finds
   results_smoke_875_* / results_smoke_880_* in --results-dir (default E/results) or takes --smoke-d27 / --smoke-r3.
   A missing smoke tarball fails P8.
3. The H3-dropped gate line is "GATE PASS (H3 dropped: projection <x> GPU-h)" (the card writes "<x>"; the unit was
   added). prep_khop2_confirm.py and run_khop2.sh accept exactly this form.
4. Harm sentence wording: "chain training lowers GSM rc<j> by <|x|> points (X - best-A2 <x> [CI])", so the sentence
   does not read "lowers by -7.3". The decision rule is the card's (CI upper < 0).
5. BETWEEN "still data-limited" / "still dose-limited" keep the card's "at 27,000 rows and 1,029 updates"; the
   dose-limited form adds "(rows and updates not separated; see branch (b))" (Amendment 1 finding 3).
6. Under MANIPULATION FAILED the branch (b) contrasts are still printed (the card asks for them with every verdict),
   marked "(not read: MANIPULATION FAILED)".
7. The 1,029-update wall is extrapolated (section 5). P7 (2.9 h) and P7b handle it.
8. Two copies of one row can land in the same or nearby batches (the three copies are shuffled together). Reported by
   the card, not controlled.
9. data/eval/khop2 adds 80 MB to every bundle built from E/src/data/eval after now (section 3).
10. khop2_table.py and the gate read build_khop.py from E/src_khop/repo/scripts through khop_table.build_khop (the
    parent readout's loader); that directory must stay in place.

## 8. Codex fix round (2026-10-26)

Input: critique/CODEX_REVIEW_khop2.r1.md ("VERDICT: SUBMIT AFTER FIXES", 1 major, 1 minor). Files changed:
readout/khop2_table.py, readout/test_khop2_table.py, src_khop2/run_tests.sh. Not changed: the repo (build.diff
regenerated, same content), the gate, the stage files, the envs, the DAGs, the card bars. No card amendment: each fix
makes the code compute the card's own lines. The pre-fix readout is in critique/khop2_table.pre_codexfix.py.

Round 1 findings:
1. major, fixed. Float rounding moved verdicts. Reproduced under Python 3.9 (/usr/bin/python3, the Codex python3): the
   k3 counts (113, 113, 122, 126, 126) / 150 have m = 0.80 exactly, and the old readout read BETWEEN instead of PASS.
   Under Python 3.12 too: R27 = 60/150 - 61/150 and R9 = 64/150 - 63/150 give rows_pool = +2.8e-17 per seed with SD 0,
   and the old label was ROWS. Fix: every reading, label and I-line is decided on exact rationals.
   - The cells of the decision arms (D27, R3, H3, KD, KH) are Fraction(correct, items) from the preds (xacc). The I2
     qs pair share and the I3 blind share are exact the same way (xqs_pair, xacc).
   - XS compares the CI in squared form with t as the decimal of TCRIT: lo >= L when m - L >= 0 and
     (m - L)^2 n >= t^2 var; lo > L and hi < U the same way. The bars are Fraction("0.80") and so on (XBAR).
   - SOLVED / FAILED / k* (xsolved, xfailed, xkstar), D_dose, R27, the pooled labels, the all-+/- bar, I1-I5 and the CI
     side of the GSM three-way sentence use XS. Floats are for printing only, from the same exact values.
   - No tolerance (epsilon) is used, so no bar moves.
2. minor, fixed. The 2.3 diagnostics (all-+/- subsets, rank-probe agreement) now print for every loaded arm: D27, R3,
   H3, KD, KH, K1, A2Z, BASE0, A2Z96. BASE0 has one seed, so its CI prints nan (a reported line).

Fail-closed additions (task rule: missing or partial inputs never shrink the denominator):
- A decision arm on the command line without exactly seeds 16-20 stops main() (UNREADABLE).
- A decision arm whose preds file is absent, or misses any item of its set, stops main(). Before, T.acc dropped the
  missing items. The sets are the eight scored khop sets for D27 / R3 / H3, and khop_ca_fwd for KD / KH.
- verdict() and branch_b() stop on a decision cell without exactly seeds 16-20, on NaN / inf / None values, and on a
  missing all-+/- subset under k* >= 3. Before, a missing subset read "PASS (approximate)"; the test now expects
  SystemExit.

Not changed, on purpose: N1 and the plain-cost answer read the trainer's "acc" from metrics.json. Its count is not in
the tarball, so an exact form would be fake; these are reported lines.

Regression tests (readout/test_khop2_table.py, class TestExact plus two end-to-end tests). Run against the pre-fix
khop2_table.py, all of them fail (3 FAIL, 5 ERROR in the unit classes under 3.12 and 3.9; 2 FAIL end to end):
test_solved_at_exact_bar, test_i1_mean_at_exact_bar, test_zero_pooled_contrast_not_rows, test_ci_boundary_exact,
test_ci_forms_match_float_off_boundary (2,000 random count sets: the exact forms equal the float forms off the
boundary), test_fail_closed, test_gsm_sentence_exact, test_pass_approximate (missing subset), and end to end
test_partial_preds_unreadable and the every-arm 2.3 lines in test_end_to_end_with_h3.
run_tests.sh also runs test_khop2_table under /usr/bin/python3 (3.9.6), since the float sum differs between 3.9 and 3.12.
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 55, readout 3.9 23, stage 13, D5 OK twice).

Round 2 findings (critique/CODEX_REVIEW_khop2.r2.md, "VERDICT: SUBMIT AFTER FIXES", 2 major, 1 minor):
1. major, fixed with a card amendment. rc4 is read on 128 primary items, not 249: pfx_table.py has an rc4 cell only
   for the 128 primary items with a hint row at r_path 4. The eval files cannot give rc4 for the other 121. Card
   "Amendment (Codex fix round, 2026-10-26)" item 1 registers the population (the parent's and PFX's). xrc_cells
   replaces T.rc_cells: exact Fraction, population sizes asserted (249 / 249 / 249 / 128, else UNREADABLE), and a run
   that misses any item of a population stops the readout. The readout prints the populations.
   Tests: test_rc_cells_exact_and_complete (exact equals T.rc_cells on j766 s16; one rc4 row removed -> SystemExit),
   the population line in test_end_to_end_with_h3.
2. major, fixed. The printed k*(X, t, o) lines of 2.3 used float T.kstar and could disagree with the verdict's exact
   k*. They now use xkstar on the exact cells. Test: test_printed_kstar_matches_verdict (k3 counts 113, 113, 122,
   126, 126 / 150); with the old lines it fails under Python 3.9 (k* 2 != 3).
3. minor, fixed. I4 is computed once per run (i4r) and reused for the printed line.
gsm_sentence now converts the printed stats to float, so Fraction inputs print under Python 3.9.
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 57, readout 3.9 25, stage 13, D5 OK twice).

Round 3 findings (critique/CODEX_REVIEW_khop2.r3.md, "VERDICT: SUBMIT AFTER FIXES", 1 major):
1. major, fixed. The gate and the readout trusted the seed labels of the command line. Fix: the seed and order_seed in
   metrics.json args must equal the label (card Amendment (Codex fix round) item 3, a note: code matches card).
   - khop2_pilot_gate.py: P1 j875 / j880 need recorded (16, 16); P6 needs the label "s16", two references, both at
     (16, 16). The KD s16 INFO line prints its recorded seed (INFO only, as the card says).
   - khop2_table.py: check_seed() on every decision-arm run (D27, R3, H3, KD, KH) and on both A2-6k reference sets.
     Each reference set must hold exactly seeds 16-20. Else UNREADABLE.
   Tests: test_recorded_seed_checked (R3 s16 run reused under s16-s20; an s16 reference under the s17 label), gate
   cases "P1 j875 seed", "P6 ref seed", "P6 ref label". Against the pre-round-3 files (critique/*.pre_codexfix_r3.py)
   both tests fail. The gate fixtures now carry metrics args seeds, as the real tarballs do (j756-770, j520: seed =
   order_seed = label; eval-only runs A2Z / BASE0 / A2Z96 record none and are not checked).
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 58, readout 3.9 26, stage 13, D5 OK twice).

Review after round 3 (critique/CODEX_REVIEW_khop2.r4.md, "VERDICT: SUBMIT AFTER FIXES", 1 major, 1 minor). Both
fixed and tested. Codex did not re-review them: the loop allows 3 rounds.
1. major, fixed. P6 passed with the same reference twice (j663, j663), so the bar became j663 alone. khop2_pilot_gate.py
   P6 now needs two distinct files (realpath), the first named j520 and the second j663. Same for the readout:
   each A2-6k reference set must use the section 2.4 jobs by seed (REF_JOBS), else UNREADABLE. Tests: gate cases
   "P6 ref dup" (j520 twice) and "P6 ref id" (j766 as the second ref); readout test_reference_identity (j663 under the
   orig s16 label). Against the pre-round-4 files (critique/*.pre_codexfix_r4.py) all three fail.
2. minor, fixed (cheap, and the task rule asks for it). An absent or mistyped --hint-dir skipped G2 / G3 and rc1-rc4
   without an error. The readout now stops (UNREADABLE) when hint_plain.jsonl is not in --hint-dir. Test:
   test_hint_dir_required. The end-to-end tests now run with the hint files.
Card: Amendment (Codex fix round) item 3 names both checks (code matches card 2.4; no bar change).
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 60, readout 3.9 28, stage 13, D5 OK twice).

Not changed in this fix round: the repo (build.diff, 626 lines, same content), stage/ files, the envs, the DAGs, jobs/.
No envs or DAGs were regenerated, because no generator changed.

## Codex fix round 5 (2026-10-26)

Review: critique/CODEX_REVIEW_khop2.r5.md ("VERDICT: SUBMIT AFTER FIXES", 2 majors). Rule for this round: a missing,
partial, NaN or unreadable input fails closed. It never shrinks n or a denominator, and it never passes a line.
Pre-round files: critique/*.pre_codexfix_r5.py. Red evidence: critique/r5_tests_red.txt (every new test fails on the
pre-round code).

Findings of the review:
1. major, fixed. khop2_pilot_gate.py P6 dropped a missing reference before max(j520, j663). P6 now needs j520 and j663
   both finite on gsm8kx and gsm8kv2, and a finite pilot value; else P6 fails and prints MISSING. A NaN reference also
   shrank the bar (max([0.55, nan]) is 0.55); that case fails now too. valx / gsm1kx stay INFO.
   Test: test_p6_reference_missing_or_nan_fails (pilot 0.492 between the one-reference bar 0.488 and the bar 0.498).
2. major, fixed. khop2_table.py read the plain metrics through a None filter, so N1, best-A2 and the plain cost of the
   dose ran on four seeds. Every decision arm, KD, KH and both A2-6k sets now need all four plain metrics as finite
   floats at seeds 16-20, else UNREADABLE. best_set() needs seeds 16-20 on every set and decides on exact values.
   Tests: test_plain_missing_or_nan_unreadable (missing, NaN and string values on D27 / R3 / H3),
   test_reference_plain_missing_unreadable (orig s17 without valx).

Sweep for the same class (a value filtered out before a decision line):
3. P0 d_bar_violations read only the cells present in the report. It now needs the 21 D1 cells per stem (ca / wp / vb
   x k1-6, three pooled), an int n in each, a D2 entry for each non-pooled cell of >= 100 rows, and finite values.
   The card's "cells of >= 100 rows" skip stays. Test: test_p0_missing_cells_fail.
4. P1 counted khop preds rows only; P3-P9 (khop_table.acc / qs_pair) drop ids absent from the preds. P1 now needs the
   exact item-id set of each eval file. Test: test_p1_item_ids_must_match.
5. P1 GSM files: an env663 without its EXTRA_EVALS line gave 7 of the 55 sets. P1 now fails then.
   Test: test_p1_env_without_extra_evals_fails.
6. khop2_table.py: the nine khop files must have the CARD_khop_v1 1.2 row counts (xacc reads their ids, so a short file
   shrank every cell). Test: test_short_khop_file_unreadable.
7. khop2_table.py: G2 / G3, fwd - rev and the printed cell CIs now read the exact complete cells (xcell); the
   None-filtering cell() helper is removed. N1 and the pooled plain-cost answer are decided on exact values
   (XBAR["N1"] = 2/100), as the other readings are.
Not changed (reported lines, no decision): the all-+/- and probe-agreement lines of the non-decision arms (K1, A2Z,
BASE0, A2Z96) still use khop_table.stats. prep_khop2_confirm.py (strict regex on the gate's last line) and
run_khop2.sh (stops on a missing tarball or gate output) already fail closed.
Real-data check: all four plain metrics are finite in j520-542, j663-666, j756-770 (the readout can still run).
No card amendment: no bar, reading, arm or id changes. The repo did not change, so build.diff is unchanged.
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 67, readout 3.9 31, stage 13, D5 OK twice).

Review after round 5 (critique/CODEX_REVIEW_khop2.r6.md, "VERDICT: SUBMIT", 0 blockers, 0 majors, 1 minor):
1. minor, fixed. The readout did not print the card 1.3 cross-bundle disclosure ("This is stated in the readout").
   khop2_table.py now prints it with the closing note. Test: the end-to-end test asserts the line.
No second review run: no blocker or major remained.
Result: test_all.log, exit 0 (repo 13 + 66 + 23, readout 67, readout 3.9 31, stage 13, D5 OK twice).
