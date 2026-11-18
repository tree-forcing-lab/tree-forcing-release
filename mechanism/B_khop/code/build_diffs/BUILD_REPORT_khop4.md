# Build report: CARD_khop_v4 (family khop4, ids 1025-1049), 2026-10-27 CDT

E = <path> (the old scratchpad path .../scratchpad/exp is a symlink to it).
Spec: <path> with Amendment 1 (card critic) and Amendment 2 (this build).
Nothing was bundled into E/bundle_out, copied to CHTC or submitted. No cluster command was run.

## 1. What changed

### Code (E/src_khop4/repo)

E/src_khop4/build.diff = `diff -ruN -x __pycache__ -x data repo.base repo` (509 lines, md5 6e6a70f2d72aa57902a307a43824aa16).
Two new files. No file of the parent is edited.

| File | Content |
|---|---|
| scripts/build_khop4.py | The card 1.2 generator: byte copies of the khop3 stage-1 directories under the khop4 names and data/eval/khop4/build_report.json. |
| tests/test_build_khop4.py | Generator, refusal and real-file tests, and the two stage-2 protections on the tiny FSDP model (section 2.2). |

No trainer flag and no trainer change (D5). Both stage-2 arms use flags that exist: L changes `--lr` (env LR), R uses
`--extra-train` (CARD_khop_v1). The flag-off path is therefore identical to repo.base by construction. run_tests.sh
checks D5 against E/src_khop3/repo and against repo.base, runs the parent's flag-off identity tests against
repo.base, and runs the whole tests/ directory (section 2.1).

Generator rule (build_khop4.py, card 1.2):
1. Inputs under `--khop3-root` (the data root): student_khop3_c27/{train,val}.jsonl, student_khop3_k1x/{train,val}.jsonl,
   eval/khop/train_kh.jsonl, eval/khop/train_k1.jsonl. Each must have the card md5 and row count, and the md5 of the
   khop3 report (student files) or the khop report (train_kh, train_k1). Any other value, a missing file, a report that
   does not parse or is not an object, is a SystemExit before any write.
2. `--out-root`, `--report` or an output dir equal to an input dir is a SystemExit. An output dir or the report dir that
   holds another file is a SystemExit.
3. Outputs: byte copies to student_khop4_c27 / student_khop4_k1x. The report records the md5s and row counts of the 4
   outputs and the 6 inputs, the 4 replay equalities (train_kh = rows 0..2,999 of the c27 train file, train_k1 = rows
   0..2,999 of the k1x train file, for the khop3 inputs and again for the khop4 outputs), the file sets, inputs
   unchanged, d_fail, d_pass. No timing field, so two builds are byte-equal. main() exits 2 on a failed line.

### Stage (E/stage, new files only)

| File | md5 | Content |
|---|---|---|
| prep_khop4.py | 48f6ecfb23fe759aa95f2cc4f13b3292 | Writes the 28 envs 1025-1044, the two subs, the smoke DAG and the pilot DAG. Refuses another env on a family id (1037-1044: anything but their two candidate envs), and any env on 1045-1049. `--dry-run`, `--jobs-dir`, `--out`. |
| prep_khop4_confirm.py | 94cc00bddb07116dce21d5e7006c2b7f | Writes confirm_x1khop4_4b.dag (16 nodes) for the arm that the gate output names, only after gate_arm_text accepts the output and only when the 16 envs equal prep_khop4.py's text. `--dry-run [--arm R / L]`. |
| check_bundle_khop4.py | 9ce0554ef3031d47e7b5807e7b350c6a | The card section 5 bundle check (exact file sets, md5s, the 28 envs, the 21 top-level eval files against the x1khop3 bundle). |
| test_prep_khop4.py | | Env, sub, DAG, confirmation, 4-digit id, copy-script, lr-parse and bundle-check tests. |
| train_x1khop4_4b.sub | 5db44a37daef0ee9b36d25e8de59759e | S1 confirmation sub = train_x1khop3_4b.sub with lines 13, 16, 33 and the header comment lines 1 and 3 changed. Executable run_train5.sh. |
| train_x1khop4w_4b.sub | eb78c2da7c1463fe309a160d869016cb | S2 and smoke sub = train_x1khop3w_4b.sub with lines 9, 12, 29 and the header comment lines 1-3 changed. Executable run_train_x1verify2.sh (md5 3866fc7788af662fe94052a9d2bae4c8, unchanged). |
| smoke_x1khop4_4b.dag | 2983607e458ccf7095f04ffdb969a98b | smoke_1025 ... smoke_1028 on the S2 sub, smoke="1", no PRE. |
| pilot_x1khop4_4b.dag | 6c7fa3139506b2e241bdc2426dae20f9 | j1025-j1028, independent, PRE check_weights_khop3.sh <975 or 985 name>. |
| condor/jobs_x1/1025-1044 | | 28 envs (section 4). |

check_weights_khop3.sh (md5 11448089f57fe745f9c7a0c7c3cb11c4) and check_pilot.sh (e1a4ee9982ceb7c31e4040c7da64da1c) are
used unchanged; prep_khop4.py asserts both md5s of the runner and of check_weights_khop3.sh.

### Readout (E/readout, new files only)

| File | md5 | Content |
|---|---|---|
| khop4_pilot_gate.py | 5305d2c2b32632953cab59608ff088fb | The pilot gate: 7 common lines, 10 decision lines per arm, one INFO a-Q7 line per arm, the 0.5 readings, the 0.5 last line. Imports khop3_pilot_gate.py, khop3_table.py, khop4_table.py. |
| khop4_table.py | fadcfa5b1fef50bd57a70db649298e85 | The five-seed readout: exact cells, I1-I7 (I7 with the stage-2 k = 1 line), I4 with the all-args recipe check, N1, V, the 0.7 verdict for the arm of the gate output, the rc3 line, 2.2, 2.3, timing and compute. Imports khop_table.py, khop2_table.py, khop3_table.py. |
| run_khop4.sh | 6fd80faa7be572edf691315c217e879e | `bash run_khop4.sh <gate output> <retry-walls file> [results dir]`. |
| test_khop4_pilot_gate.py, test_khop4_table.py | | Tests (section 2). |

No parent readout or stage file is edited (khop3_pilot_gate.py, khop3_table.py, prep_khop3.py, check_weights_khop3.sh,
round_gpuh.py, build_bundles.sh, scp_envs.sh are import-only or read-only).

### Card

Amendment 2 of CARD_khop_v4.md (section 7 below lists the same items). The only substantive one: the S1 confirmation
recipe allow-list gains `eval_subq`, because train_arm.sbatch records `$DATA/val.jsonl` there, so every correct
confirmation S1 job would have failed I4 otherwise.

## 2. Tests

Runner: E/src_khop4/run_tests.sh. Output: E/src_khop4/test_all.log (ends with an EXIT= line). Python:
E/src_verify3/venv_pfx/bin/python (3.12.7, torch 2.14.1, transformers 5.20.2); the family's readout and stage tests
also run under /usr/bin/python3 (3.9.6).

| Suite | What it checks |
|---|---|
| repo tests.test_build_khop4 (16) | Real build in a temp dir: byte copies, D2 (the 3,000 replay rows equal the head of the khop4 train files, md5 = card), report recomputed from the files, two builds byte-equal and equal to E/src/data, inputs unchanged against the card values, the replay item_ids (khop-tr-*) disjoint from the 5,923 gsmvt train ids. Refusals before any write: one changed row in each of the 6 inputs, a parent report with another md5 / not an object / not parsable, a missing input, a last line without a newline, --out-root / --report / an output dir equal to an input dir, an output or report dir with another file. Tiny two-stage run (2.2). |
| repo parent tests | tests.test_build_khop3, test_build_khop2, test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train (COTD_REF_REPO = E/src_khop/repo.base). |
| repo flag-off identity | test_extra_train flag off, sbatch without --extra-train, test_frontier_train (COTD_REF_REPO = E/src_khop4/repo.base). |
| readout test_khop4_pilot_gate (11) | Good pilot -> GATE PASS ARM R (both arms pass); R failing -> ARM L; both failing -> GATE FAIL with the four 0.5 readings; each of the 10 decision lines of each arm fails alone (and the other arm is chosen); each common line fails alone (C3 only together with C1 j985, because C1 pins the round-G tarball md5); the a-Q9 120/119 and 128/127, a-Q6 2.9/3.1 and 4.9/5.1, a-Q7 0.95 h / 1.15 h (INFO), a-Q7b 23.00 / 23.01, C2 120/119, C7c edges; a missing retry-walls file is refused; a-Q2 by an extra differing key, a key on one side only, a missing key, lr 3e-06 or "7.5e-07" as a string in L, an S2 log naming another S1, an L log with an extra-train line, a wrong seed, an R job without extra_train, a metrics extra_train count off, the K1X replay file in CUR-R, L stage counts in R, save True; a NaN prob, a hint file one row short, a non-object or unparsable metrics.json (that arm's a-Q1 fails, the other arm is chosen), an "n/a" wall (a log without its start line: C7c and that arm's a-Q7b), a missing smoke; no rc / hint line; the printed line names equal khop4_table's and prep_khop4_confirm's lists and both accept the output. |
| readout test_khop4_table (16) | Every 0.7 reading and sub-label for both arms (section 0.7 order, the L and R claim texts, NEGATIVE both texts, NULL drift, "gain below the bar", exact 3/100 bars, V by I7 / N1 of K1X, Bonferroni INFO), fail-closed verdict inputs; recipe_diff (one side only, allowed keys too, False vs 0, float vs string); the S1 allow-list on the real j975 / j985 args (Amendment 2), the real j980 / j990 / j756 / j757 args; run_name from the basename; gate_text_arm. End to end through main() on the real A2-6k, j975 / j985, j980 / j990, D27, KH / K1 tarballs plus synthetic khop4 dirs: PASS for R and for L, "installed, then lost in the GSM stage (arm R)", "GSM damaged or not recovered" (L), V failing by I7 at stage 2; UNREADABLE on a missing seed, a plain row count off, a NaN loss, a wrong seed, a non-object metrics.json, a missing wall, a run under another seed's name, a reference with a wrong id and a 4-digit look-alike (results_1525_), a context run with a wrong id or a missing seed, two landed tarballs for one id; "not run as registered" by an S1 arg outside the allow-list and by an S2 log naming another S1. run_khop4.sh on a results dir where 1027_ sorts before 975_ (with 10250_, 102_, 1024_ and smoke_1027_ decoys): full run, refusals (gate not passed, a gate output the table refuses, no retry-walls file, a missing tarball, the other arm's candidate tarball landed on the same id). |
| readout parent tests | test_khop3_table, test_khop3_pilot_gate, test_khop2_table, test_khop2_pilot_gate, test_khop_table, test_khop_pilot_gate. |
| stage test_prep_khop4 (16) | Every env sourced under bash against its khop3 S2 env (key by key, key order) and, for R, against the KH / K1 env of the same seed; every S1 env against 976-979 / 986-989; names, TAGs (lr075 only on L), SAVE only on S1, MODEL / WTAR, no env on 1045-1049; both subs (exact changed lines, no x1khop3 outside comments, executables, runner and check_weights md5s); smoke and pilot DAGs; the confirmation generator (16 refused texts incl. bare "GATE PASS", khop3's real output, an ARM R output whose lines say R failed, a FAIL in a common line, a hand-given --arm; a changed or missing env; 16 nodes, PARENT / CHILD, PREs, POST, RETRY, PRIORITY for each arm; dry run); 4-digit ids (decoys 102_, 10250_, 1024_, 1000_, smoke_10450_; the 1045 / 1049 refusals, another env on 1025, a CUR-R name on a K1X id, a third candidate on 1037); round_gpuh.py 1025 1049 on a results dir with those decoys and 1050_ (lists exactly the 3 family rows); check_pilot.sh / check_weights_khop3.sh take names; scp_envs.sh / build_bundles.sh copy by glob / whole dir; no script parses lr<digit> from a name; check_bundle_khop4.py on a synthetic bundle (good, and each of 15 rules failing alone). |
| stage parent tests | test_prep_khop3, test_prep_khop2, test_prep_khop. |
| bash -n | run_khop4.sh, check_weights_khop3.sh, check_pilot.sh. |
| D5 | `diff -rq -x __pycache__ -x data` against E/src_khop3/repo and repo.base: only build_khop4.py and test_build_khop4.py. |
| full repo suite | Every tests/test_*.py module of E/src_khop4/repo except tests.test_fsdp, tests.test_tsf and tests.test_latent_slots; each failing test is re-run in repo.base and the failing sets must be equal. The three modules start torchrun without --local-addr (test_fsdp's runner), and on this Mac the rendezvous hangs: twice on 2026-10-28 a torchrun sat 8 min with no worker process, and the run was stopped. They test only the trainer, which is byte-identical to repo.base (D5). |

### 2.1 Final run

test_all.log, 2026-10-28 00:47 (first block) to 02:57 CDT, EXIT=0 (venv_pfx: Python 3.12.7, torch 2.14.1,
transformers 5.20.2; 3.9 blocks: /usr/bin/python3 3.9.6).

| Block | Result |
|---|---|
| repo (new): tests.test_build_khop4 | Ran 16 tests in 33 s, OK |
| repo parent tests (COTD_REF_REPO = E/src_khop/repo.base): test_build_khop3, test_build_khop2, test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train | Ran 92 tests in 299 s, OK (skipped 1: a parent test that needs data/) |
| repo flag-off identity (COTD_REF_REPO = E/src_khop4/repo.base) | Ran 23 tests in 99 s, OK |
| readout (3.12): test_khop4_table, test_khop4_pilot_gate, test_khop3_table, test_khop3_pilot_gate, test_khop2_table, test_khop2_pilot_gate, test_khop_table, test_khop_pilot_gate | Ran 127 tests in 1,281 s, OK |
| readout under Python 3.9.6: test_khop4_table, test_khop4_pilot_gate | Ran 27 tests in 691 s, OK |
| stage (3.12): test_prep_khop4, test_prep_khop3, test_prep_khop2, test_prep_khop | Ran 47 tests in 93 s, OK |
| stage under Python 3.9.6: test_prep_khop4 | Ran 16 tests in 86 s, OK |
| bash -n run_khop4.sh, check_weights_khop3.sh, check_pilot.sh | OK |
| D5 against E/src_khop3/repo and against repo.base | D5 OK twice (only scripts/build_khop4.py and tests/test_build_khop4.py) |
| full repo suite (all modules but test_fsdp, test_tsf, test_latent_slots) | Ran 1,002 tests in 1,075 s: 12 failures, 17 errors, 172 skipped. The same 29 tests fail in repo.base (re-run there, failing sets equal): "FULL SUITE: no new failure against repo.base". They are failures of the parent tree on this Mac, identical in repo.base; they were not diagnosed one by one (the khop2 / khop3 reports name the data/ symlink and bash 3.2 TestSbatch causes for some of them). Examples: tests.test_build_v7.TestAssemble, tests.test_tree_target, tests.test_tree_placebo, tests.test_probe_slots.TestSbatch, tests.test_extra_train.TestSbatchSmokeSubset.test_smoke_subset. The list is in test_all.log. |

An earlier run (2026-10-27 22:15) passed every block up to D5 the same way. Its full-suite block hung in
tests.test_fsdp (torchrun with no worker for 8 min) and was stopped. A second run hung in tests.test_latent_slots the
same way. Those modules are now left out (section 7 item 11).

### 2.2 The two stage-2 protections on the tiny model (card section 5)

torchrun 1 rank, --local-addr 127.0.0.1, FSDP cpu-adamw, --eval-value-render vt, label smoothing 0.1, --token-budget 4096,
seed = order seed 16, micro-bs 2, grad-accum 2, --log-every 1. The trainer is not changed; the khop3 test wrapper
(TRACER, imported from tests/test_build_khop3.py) records the examples of every training micro-batch.
- Stage 1: 18 khop rows with --save ("stage all: 18 examples per pass, 5 steps").
- (a) L: stage 2 from the saved weights on 6 GSM rows at --lr 2.5e-4 against the reference at 1e-3. The history lr
  values are exactly one quarter of the reference's at every step (a power-of-two factor is exact in floating point);
  the per-micro-batch examples (item_id, text, labels, target) equal the reference's; losses finite and different from
  the reference's; no extra-train line.
- (b) R: --extra-train of 6 further khop rows (not stage-1 rows). Log lines "extra-train: 6 rows from <path>",
  "extra-train: total training rows 12 (6 from <gsm path>)", "stage all: 12 examples per pass, 3 steps"; final step
  3 = ceil(12 / 4); metrics.json extra_train {<path>: 6}; every one of the 12 rows is seen and no other row.
- A --extra-train file that repeats a GSM training item_id stops the trainer (nonzero exit, "is already a training
  row", no metrics.json). The real replay ids are disjoint from the gsmvt ids (TestRealBuild).

## 3. Data

Command (log E/src_khop4/logs/build_khop4.log, under 1 s):

    cd E/src_khop4/repo && python3 scripts/build_khop4.py --khop3-root E/src/data --out-root E/src/data \
      --report E/src/data/eval/khop4

| File | Rows | md5 |
|---|---|---|
| E/src/data/student_khop4_c27/train.jsonl | 27,000 | 92545dd42ca451c39e0e7c104aa78d1e (= student_khop3_c27 = train_kd27) |
| E/src/data/student_khop4_c27/val.jsonl | 393 | 9f55fec541246e41401a7b5fa6c5510a (= gsmvt val) |
| E/src/data/student_khop4_k1x/train.jsonl | 27,000 | d7cc9716e5785461d16ec92156104261 (= student_khop3_k1x) |
| E/src/data/student_khop4_k1x/val.jsonl | 393 | 9f55fec541246e41401a7b5fa6c5510a |
| E/src/data/eval/khop4/build_report.json | - | ace50a741a4b3bc93673e925da33d8cb (d_pass true, d_fail [], 4 replay lines true, inputs unchanged) |

No new eval set; data/eval/khop4/ holds only build_report.json. The replay files of arm R are the parent files
data/eval/khop/train_kh.jsonl (a16b1520...) and train_k1.jsonl (4bb15f66...), not rebuilt.

Bundle placement. jobs/build_bundles.sh line 21 copies `$E/src/data/student_${v#x1}_*` for variant x1khop4, which is
exactly student_khop4_c27 and student_khop4_k1x (no student_khop3_*). Line 18 copies data/eval whole (khop4/, khop/
with train_kh / train_k1, khop2/, khop3/). The env DATA keys name data/student_khop4_c27 / data/student_khop4_k1x
(S1 confirmation) and data/student_gsm8k_tf_vt (S2), relative to the job's ~/cotd. Trial build (section 6.1 item 2
with B redirected to the scratchpad, not E/bundle_out): 46 MB, 1,211 files, check_bundle_khop4.py BUNDLE OK.

## 4. Jobs and exact XARGS

Common values (asserted against the templates): EPOCHS=1, LSMOOTH=0.1, MBS=2, GA=16, NGPU=1, FSDP_OPTIM=cpu-adamw,
OPTIM=adamw, PREC=bf16, TOKBUDGET=4096, ARMS=A2. Resources: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300, RETRY 1,
POST post_push.sh. S2 envs keep the khop3 S2 env's EVALAT=0.5, EVALMID (val,valx,gsm8kv2,gsm8kx,gs5p,bbh500) and
EXTRA_EVALS (57 sets: the 48 of the 663 env and the nine khop sets), SAVE='', DATA=data/student_gsm8k_tf_vt. S1 envs
keep khop3's EVALAT='', EVALMID='', EXTRA_EVALS (13 sets), SAVE=1, MODEL=models/Qwen3.5-4B.

| Job | Stage | Arm, seed | Name | Template | LR | XARGS (sourced) | MODEL / WTAR |
|---|---|---|---|---|---|---|---|
| smoke_1025-1028 | smoke | the 4 pilot envs, SMOKE_ONLY | smoke_<pilot name> | - | as the pilot | as the pilot | as the pilot |
| j1025 | pilot | CUR-L s16 | 1025_4b_gsmvt_A2-khop4curlw975-c1lr075ch_s16_fsdp | 980 | 7.5e-7 | `--eval-value-render vt` | models/975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp/model / 975_... |
| j1026 | pilot | K1X-L s16 | 1026_4b_gsmvt_A2-khop4k1xlw985-c1lr075ch_s16_fsdp | 990 | 7.5e-7 | `--eval-value-render vt` | 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp |
| j1027 | pilot | CUR-R s16 | 1027_4b_gsmvt_A2-khop4currw975-c1lr3ch_s16_fsdp | 980 (and = 756) | 3e-6 | `--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl` | 975_... |
| j1028 | pilot | K1X-R s16 | 1028_4b_gsmvt_A2-khop4k1xrw985-c1lr3ch_s16_fsdp | 990 (and = 757) | 3e-6 | `--eval-value-render vt --extra-train data/eval/khop/train_k1.jsonl` | 985_... |
| j1029-1032 | confirm S1 | CUR1 s17-20 | 10xx_4b_kd27_A2-khopcur1-c1lr3ch_s<s>_fsdp | 976-979 | 3e-6 | `--eval-value-render vt` | models/Qwen3.5-4B, SAVE=1, DATA=data/student_khop4_c27 |
| j1033-1036 | confirm S1 | K1X1 s17-20 | 10xx_4b_kd27k1_A2-khopk1x1-c1lr3ch_s<s>_fsdp | 986-989 | 3e-6 | `--eval-value-render vt` | as above, DATA=data/student_khop4_k1x |
| j1037-1040 (L env) | confirm S2 | CUR-L s17-20 | 10xx_4b_gsmvt_A2-khop4curlw<1029-1032>-c1lr075ch_s<s>_fsdp | 980, seed s | 7.5e-7 | `--eval-value-render vt` | the same seed's CUR1 |
| j1037-1040 (R env) | confirm S2 | CUR-R s17-20 | 10xx_4b_gsmvt_A2-khop4currw<1029-1032>-c1lr3ch_s<s>_fsdp | 980, seed s (and = 758-761) | 3e-6 | `... --extra-train data/eval/khop/train_kh.jsonl` | the same seed's CUR1 |
| j1041-1044 (L env) | confirm S2 | K1X-L s17-20 | 10xx_4b_gsmvt_A2-khop4k1xlw<1033-1036>-c1lr075ch_s<s>_fsdp | 990, seed s | 7.5e-7 | `--eval-value-render vt` | the same seed's K1X1 |
| j1041-1044 (R env) | confirm S2 | K1X-R s17-20 | 10xx_4b_gsmvt_A2-khop4k1xrw<1033-1036>-c1lr3ch_s<s>_fsdp | 990, seed s (and = 762-765) | 3e-6 | `... --extra-train data/eval/khop/train_k1.jsonl` | the same seed's K1X1 |
| j1045-1049 | - | reserved, never written | | | | | |

JOBKEY: "4b gsmvt A2-<TAG> s<s> fsdp x1khop4" (S2), "4b kd27 / kd27k1 A2-<TAG> s<s> fsdp x1khop4" (S1). The R envs
equal the parent KH / K1 env of the same seed except MODEL, WTAR, SAVE (where the parent saved: 756 only), TAG, JOBKEY.

Expected log lines (a-Q2 / I4): S2-L "== weights <staging>/weights/<S1>.tar -> models/<S1>/model (9.1G)", "stage all:
5923 examples per pass, 186 steps", no extra-train line, final step 186, metrics lr 7.5e-07, extra_train "" and no
top-level extra_train; S2-R the weights line, "extra-train: 3000 rows from <replay> (sources {'khop': 3000})",
"extra-train: total training rows 8923 (5923 from data/student_gsm8k_tf_vt/train.jsonl)", "stage all: 8923 examples per
pass, 279 steps", final step 279, metrics lr 3e-06, extra_train <replay>, top-level extra_train {<replay>: 3000};
S1 "stage all: 27000 examples per pass, 844 steps", final step 844, "== weights -> <staging>/weights/<name>.tar (9.1G)",
train data/student_khop4_<c27|k1x>/train.jsonl, eval_subq data/student_khop4_<c27|k1x>/val.jsonl.

## 5. Compute

Measured walls (round_gpuh.py on the parent tarballs, card_numbers.out): j975 1.41, j985 1.37, j980 0.55, j990 0.50,
KH 0.70-0.93, K1 0.63-0.89, khop3 smokes 0.04-0.07 h. No khop4 tarball exists (`round_gpuh.py 1025 1049`: 0 jobs).

| Stage | Jobs | Estimate | Bar / cap |
|---|---|---|---|
| smoke | smoke_1025-1028 (warm start, extract 9.1G each) | 4 x 0.07 = 0.28 | inside C7c |
| pilot | j1025 / j1026 (0.55 each), j1027 / j1028 (0.82 each) | 2.74 + smokes = about 3.0 GPU-h | C7c 6.0 measured, incl. every retry wall |
| confirmation R | 8 S1 x 1.518 + 8 S2 x 0.816 | 18.67 GPU-h (`prep_khop4_confirm.py --dry-run`) | 25; a-Q7b 4 x measured walls <= 23.0 |
| confirmation L | 8 S1 x 1.518 + 8 S2 x 0.55 | 16.54 GPU-h | 25; a-Q7b <= 23.0 |

Staging: the pilot exports nothing (S2 SAVE=''). It reads $STAGING/weights/975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tar
and 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tar (9,702,973,440 B each). The confirmation S1 jobs write 8 new tars
(1029-1036, 9.04 GiB each, 72.3 GiB).

## 6. Operator section (sev-e0)

All Mac paths are under E. The AP DAG directory is ~/cotd_chtc (the round-G DAGs ran there). The card requires before
any of this: the cold build critic, the fix round, and the Codex review loop until SUBMIT (card section 5), and the
lead's go on the amended card (a RESULTS_LOG line).

### 6.1 Before any submission

1. RESULTS_LOG: add the khop4 line (lead decision 3 names the family; card 0.2: the lead's go on the amended card is
   its acceptance).
2. Build the bundle (on the Mac):

       cd E && REPO_DIR=E/src_khop4/repo jobs/build_bundles.sh \
         "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2-khop3-khop4" x1khop4

   Output: E/bundle_out/cotd_code_data_x1khop4_4b.tgz and .md5. Pin: the string above (CODE_PIN).
3. Bundle check (exact file sets, card section 5):

       python3 -I E/stage/check_bundle_khop4.py E/bundle_out/cotd_code_data_x1khop4_4b.tgz

   Pass: every line PASS and "BUNDLE OK" (the reference of the 21 top-level eval files is
   E/bundle_out/cotd_code_data_x1khop3_4b.tgz by default). It also prints the bundle md5 and compares it with the .md5
   file.
4. Copy to the AP DAG directory and compare the md5 on both sides (md5 on the Mac, md5sum on the AP):
   - cotd_code_data_x1khop4_4b.tgz (md5 from E/bundle_out/cotd_code_data_x1khop4_4b.md5);
   - E/stage/train_x1khop4_4b.sub (5db44a37daef0ee9b36d25e8de59759e), E/stage/train_x1khop4w_4b.sub
     (eb78c2da7c1463fe309a160d869016cb);
   - E/stage/smoke_x1khop4_4b.dag (2983607e458ccf7095f04ffdb969a98b), E/stage/pilot_x1khop4_4b.dag
     (6c7fa3139506b2e241bdc2426dae20f9);
   - the 28 envs into condor/jobs_x1/ of the DAG directory: `bash E/jobs/scp_envs.sh '102[5-9]_*' '103[0-9]_*'
     '104[0-4]_*'` (it prints "ENV MD5 OK 28 files"; the glob form copies both candidate envs of 1037-1044). No PRE of
     this family reads an env file; the copy keeps the AP in line with the bundle.
5. Check on the AP that these files exist in the DAG directory with these md5s (all from round G, unchanged):
   run_train_x1verify2.sh 3866fc7788af662fe94052a9d2bae4c8, check_weights_khop3.sh 11448089f57fe745f9c7a0c7c3cb11c4
   (executable), check_pilot.sh e1a4ee9982ceb7c31e4040c7da64da1c, run_train5.sh (the copy the khop3 S1 jobs used),
   post_push.sh, chtc_settings.inc, results/.
6. What the PREs read on the AP:
   - pilot nodes j1025-j1028: check_weights_khop3.sh <975 or 985 name> reads only
     results/results_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tgz or results/results_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tgz
     (metrics.json present, last exit line 0, one weights export line of at least 8G). Both tarballs are on the AP
     since round G; check that they are there.
   - confirmation S1 nodes: check_pilot.sh reads the results tarballs of the arm's two pilot jobs and of 975 / 985.
   - confirmation S2 nodes: check_weights_khop3.sh <the seed's S1 name> reads that S1's results tarball.
   - No PRE reads an env file or $STAGING. The runner itself refuses a missing or bad weights tar (exit 1 before any
     model load).
7. Staging, read back on the AP (quota tool and du; no assumption):
   - before the smoke and the pilot: $STAGING/weights/975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tar and
     985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tar exist with 9,702,973,440 B each (never delete them);
   - before the confirmation: free space >= 77.3 GiB (8 x 9.04 + 5) and file count + 8 below the file quota (lead
     decision 1: about 257 GB free, quota 1000 GB / 10,000 files). When a line fails: stop and ask the second author; deleting a tar
     is the second author's decision; submit nothing.
8. On the Mac: the reference tarballs for the gate exist in E/results (they do now): results_520_..., results_663_...,
   results_975_..., results_985_..., results_980_..., results_990_..., results_756_..., results_757_....

### 6.2 Submit order

1. Smoke: `condor_submit_dag smoke_x1khop4_4b.dag`. Smoke gate (by hand on the 4 tarballs; the pilot gate's C8 checks
   it again): each log ends with "== exit 0", has "smoke ok A2" and the line "== weights <staging>/weights/<975 name>.tar
   -> models/<975 name>/model" (smoke_1025, smoke_1027) or the 985 line (smoke_1026, smoke_1028).
2. Pilot: `condor_submit_dag pilot_x1khop4_4b.dag` (four independent nodes).
3. Retry walls: from the job event log logs/train_x1khop4w_4b.log, one "<run name> <hours>" line per failed execution
   (every interval from event 001 to the next 004 / 005 / 009 / 012 of the same job id, except the landed execution of
   each node), smokes and pilots together; an empty file when nothing failed. Use the run name (smoke_1025_... or
   1025_...) as <node>, because the smoke and pilot DAGs both name their nodes j1025 ... j1028.
4. Gate, on the Mac after the 8 tarballs are in E/results:

       python3 -I E/readout/khop4_pilot_gate.py \
         --s1 E/results/results_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tgz E/results/results_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tgz \
         --curl E/results/results_1025_4b_gsmvt_A2-khop4curlw975-c1lr075ch_s16_fsdp.tgz \
         --k1xl E/results/results_1026_4b_gsmvt_A2-khop4k1xlw985-c1lr075ch_s16_fsdp.tgz \
         --curr E/results/results_1027_4b_gsmvt_A2-khop4currw975-c1lr3ch_s16_fsdp.tgz \
         --k1xr E/results/results_1028_4b_gsmvt_A2-khop4k1xrw985-c1lr3ch_s16_fsdp.tgz \
         --smokes E/results/results_smoke_1025_4b_gsmvt_A2-khop4curlw975-c1lr075ch_s16_fsdp.tgz,E/results/results_smoke_1026_4b_gsmvt_A2-khop4k1xlw985-c1lr075ch_s16_fsdp.tgz,E/results/results_smoke_1027_4b_gsmvt_A2-khop4currw975-c1lr3ch_s16_fsdp.tgz,E/results/results_smoke_1028_4b_gsmvt_A2-khop4k1xrw985-c1lr3ch_s16_fsdp.tgz \
         --ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz \
         --recipe-ref E/results/results_980_4b_gsmvt_A2-khopcur2w975-c1lr3ch_s16_fsdp.tgz E/results/results_990_4b_gsmvt_A2-khopk1x2w985-c1lr3ch_s16_fsdp.tgz \
                      E/results/results_756_4b_gsmvt_A2-khop-c1lr3ch_s16_fsdp.tgz E/results/results_757_4b_gsmvt_A2-khop1-c1lr3ch_s16_fsdp.tgz \
         --retry-walls E/logs/khop4_pilot_retry_walls.txt > E/logs/khop4_pilot_gate_<date>.txt

   (The defaults of --report, --khop3-report, --khop2-report, --parent-report, --khop-dir, --data-root are the E paths.)
5. Last line "GATE PASS ARM R" or "GATE PASS ARM L": `python3 -I E/stage/prep_khop4_confirm.py --gate-output
   E/logs/khop4_pilot_gate_<date>.txt` writes E/stage/confirm_x1khop4_4b.dag (16 nodes of that arm). Check the
   confirmation staging line (6.1 item 7), copy only confirm_x1khop4_4b.dag to the AP, `condor_submit_dag
   confirm_x1khop4_4b.dag`, tell the second author (he can stop it). Last line "GATE FAIL: ...": submit nothing, report the failing
   lines and the READING lines to the second author.
6. Re-runs: a node that failed after its RETRY is resubmitted under its own id, name, env and tar name (the rescue DAG,
   or a one-node DAG with the same JOB / VARS / SCRIPT / RETRY lines); never a new id (1045-1049 stay unused). Allowed
   only when the measured walls, the retry walls so far and the re-run's bar stay under the cap (6 pilot, 25
   confirmation); else stop and report. Add the failed executions to the retry-walls file.
7. Readout when all 20 tarballs of the arm are in E/results: write the retry-walls file of the smoke, pilot and
   confirmation DAGs (both subs' event logs: train_x1khop4w_4b.log and train_x1khop4_4b.log), then
   `bash E/readout/run_khop4.sh E/logs/khop4_pilot_gate_<date>.txt <retry-walls file>`. Log
   E/logs/khop4_readout_<date>.txt and .json.

## 7. Risks and deviations

1. Card gap (Amendment 2 item 1): eval_subq in the S1 confirmation allow-list. Without it every correct confirmation S1
   job reads "not run as registered". The value is checked (khop3 -> khop4 directory only).
2. The gate's C8 / C7c read each pilot log on their own (Amendment 2 item 3). A pilot with broken preds therefore fails
   only its arm's lines, and the other arm can still be chosen by 0.5. A missing or unreadable log fails C8 / C7c.
3. C3 cannot fail alone in a test: C1 pins the md5 of the real round-G j985 tarball, so any changed j985 fails C1 j985
   too. The test checks the pair.
4. khop3_table.run_name matched the first "/results_" of the whole path, so a results directory named results_X gives a
   wrong name; khop4_table.run_name reads the basename (test_run_name_from_basename). The parent file is not edited;
   in khop3's own use the results dir is E/results, which does not trigger it.
5. The readout loads j980 / j990 for both arms (recipe references) and D27 as context (Amendment 2 item 5). A missing
   one stops the readout (registered population).
6. L's GSM recovery at LR 7.5e-7 is unmeasured (card 0.3). The tiny test proves only that the LR reaches the schedule
   exactly; a-Q6 at the pilot is the first measurement.
7. The S1 run dir names of 1029-1036 (runs/A2-khopcur1-c1lr3ch-Qwen3.5-4B-student_khop4_c27-s<s>) are not in any
   fixture as such. khop_table.load_run finds metrics.json and the preds by suffix, and the tests cover flat dirs, the
   real khop3 S1 tarball shape and the synthetic runs/<x>/ shape.
8. post_push.sh exists in E/src/repo/cloud/chtc/post_push.sh (read: it takes the name as $1 and parses no id); the AP
   copy in ~/cotd_chtc is not on the Mac and is not verified here (the round-G jobs used it).
9. run_khop4.sh prints `round_gpuh.py 1025 1049` from E/results whatever [results dir] is given, and stops on an "n/a"
   row. The trial bundle (section 3) was built with a copy of build_bundles.sh writing to the scratchpad; the
   operator's bundle has another md5 (tar times), so its md5 is read from its .md5 file.
10. The gate test takes about 6 min and the table test about 5 min per Python (real tarball loads); run_tests.sh runs
    both under 3.12 and 3.9. The readout block (with the khop3 / khop2 / khop parents) takes about 24 min.
11. The full-suite block leaves out tests.test_fsdp, tests.test_tsf and tests.test_latent_slots (their torchrun has no
    --local-addr and hangs on this Mac). They cover only trainer code, which D5 shows unchanged.
12. khop4_table.py sets khop2_table.unreadable and khop3_table.unreadable to its own function, so in one Python process
    the khop2 / khop3 helper messages carry the "khop4_table: UNREADABLE" prefix (the combined readout run). No parent
    test pins that prefix (checked: the combined block passes); the standalone parent readouts are not affected.

## 8. Not done in this build step (process of round G)

- The cold build critic, the fix round and the Codex CLI review loop (card section 5).
- Bundle into E/bundle_out, copy to the AP and every cluster command: sev-e0 (section 6).

## 9. Final test run

Section 2.1 (test_all.log, EXIT=0). build.diff was regenerated after the last repo change (509 lines, md5
6e6a70f2d72aa57902a307a43824aa16).

## Fix round 1 (2026-10-28 03:42 CDT, after critique/BUILD_CRITIC_khop4.md: SUBMIT AFTER FIXES)

Card: Amendment 3 of CARD_khop_v4 (the lead accepts it with Amendment 2 before the bundle). The repo is unchanged
(D5 holds; build.diff regenerated: same 509 lines, md5 6e6a70f2d72aa57902a307a43824aa16). prep_khop4.py is unchanged
(md5 48f6ecfb...); it was rerun into a scratch jobs dir and its 28 envs, 2 subs and 2 DAGs are byte-equal to
E/stage/condor/jobs_x1 and E/stage.

| Finding | Decision | Change | Regression test (fails on the old code) |
|---|---|---|---|
| F1 MEDIUM: a mis-run R pilot printed GATE PASS ARM L | fixed | khop4_pilot_gate.py: an arm with an a-Q1 / a-Q2 FAIL or an a-Q7b FAIL on a missing wall is "not run as registered"; READING line, last line GATE FAIL, both arms printed, science readings only for arms run as registered. The same refusal in khop4_table.gate_text_arm and prep_khop4_confirm.gate_arm_text (identical code). | test_khop4_pilot_gate: test_misrun_r_never_selects_l (P2 wrong replay, P4 NaN final loss, truncated tarball, swapped slots), test_misrun_l_blocks_r, test_misrun_arm_has_no_science_reading, test_parsers_agree_on_texts; the Q1 / Q2 cases of test_each_arm_line_fails_alone, test_q2_cases and test_fail_closed_inputs now expect GATE FAIL with the READING; test_prep_khop4 test_gate_text_rule (7 new refusals, and R-Q7b over the cap still gives ARM L). A mutation copy without the rule (scratch) fails the first 3 gate tests ("GATE PASS ARM L", "GATE PASS ARM R", "('L', 'ok')"). |
| F1 side find: a truncated pilot .tgz crashed the gate (EOFError in read_log) | fixed | read_log catches EOFError and zlib.error -> unreadable | the "truncated tarball" case of test_misrun_r_never_selects_l (raised before the fix) |
| F2 LOW: CST dates stop round_gpuh.py; wall_hours ignores the zone | fixed | new readout/khop4_gpuh.py (CDT = UTC-5, CST = UTC-6, other zone -> no wall); used by the gate (C7c, a-Q7, a-Q7b), khop4_table.py (no-wall check, timing, compute) and run_khop4.sh (compute line on the given results dir, stops on n/a or a non-zero exit) | test_khop4_gpuh (zones, 01:30 CDT -> 01:30 CST = 1.00 h, other zone / missing line -> None, equal to the parent on CDT and on the real j975 / j980 logs, CLI decoys 102_ 10250_ 1024_ 1000_ 1050_ 975_, unreadable / truncated / no-log / UTC rows -> n/a and exit 1); test_cst_and_dst_walls in the gate test; test_run_khop4_sh: "over 19 jobs" from the given results dir and an unreadable j1045 tarball stops the readout |
| F3 LOW: compound array assignment masked a failed first lookup | fixed | run_khop4.sh: one `p=$(t ...)` per assignment | test_run_khop4_sh: j1030 (CUR1 s18, first of its pair) missing -> "0 tarballs for j1030", no "bad --run entry", no readout; bash 3.2 check of the old pattern: continues with rc 0 |
| F4 INFO: LR read from args only | done | INFO line "max history lr" per pilot S2 job | test_history_lr_info |
| F5 INFO: operator pre-submit items | no build change | section 6.1 items stand; Amendment 3 adds the lead's acceptance before the bundle | - |

Changed files (md5): readout/khop4_pilot_gate.py 52ea81a0, readout/khop4_table.py c4426a14, readout/khop4_gpuh.py (new)
88ca8d63, readout/run_khop4.sh fc3359fb, stage/prep_khop4_confirm.py e30f1840, readout/test_khop4_pilot_gate.py
0005e224, readout/test_khop4_table.py fbbb5d24, readout/test_khop4_gpuh.py (new) 85fea3d2, stage/test_prep_khop4.py
8a0c0a60, run_tests.sh 0017fba7 (adds test_khop4_gpuh under 3.12 and 3.9). Section 7 item 2 is withdrawn by
Amendment 3 item 1; item 9 is replaced: run_khop4.sh now prints khop4_gpuh.py 1025 1049 <results dir>.

Readings: the rule change touches only the mis-run case. A correct pilot gives the same last line as before. Every
science line, bar and the order R first are unchanged. Residual: a mis-run L now blocks a passing R until the L node is
rescued (the cost is one rescue, counted against 6.0 / 25 GPU-h).

Fix round 1 test run: run_tests.sh -> test_all.log, EXIT=0 (2026-10-28 04:58 CDT). Readout block (3.12) 137 tests OK
(includes test_khop4_gpuh), readout under 3.9.6 OK, stage OK under 3.12 and 3.9.6, bash -n OK, D5 OK against
src_khop3/repo and repo.base, full repo suite: no new failure against repo.base.
Superseded by fix round 1: the section 1 md5s of khop4_pilot_gate.py, khop4_table.py, run_khop4.sh and
prep_khop4_confirm.py (now as in the table above). In section 6 the readout prints `khop4_gpuh.py 1025 1049
<results dir>`, not `round_gpuh.py 1025 1049`. jobs/round_gpuh.py is unchanged and still handles the 4-digit ids
(test_prep_khop4).

## Codex rounds (E/jobs/codex_review.sh khop4 CARD_khop_v4.md, gpt-6-astra xhigh, read-only)

No card amendment in these rounds: no bar, arm, id, schedule, reading or verdict order changes. The repo, prep_khop4.py,
the envs, subs and DAGs are unchanged, so build.diff, the envs and the DAGs were not regenerated. Each regression test
was run against a saved pre-fix copy (scratchpad khop4_mutexp/readout, with links to the real E/results, E/src and
E/stage) and fails there.

### Round 1 (2026-10-28 05:06 CDT): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r1.md)

| Finding | Decision | Change | Regression test (fails on the pre-fix copy) |
|---|---|---|---|
| 1 major: a-Q2 read only the last history row; an earlier NaN loss with a finite last loss gave PASS R-Q2 and GATE PASS ARM R | fixed | khop4_table.history_ok: every history row is an object with an int step (not bool) and a finite loss, the steps strictly increase, the last step is the schedule's final step. Used by i4_s1 (C1 and readout I4) and i4_s2 (a-Q2 and readout I4); the detail prints "history <n> rows ok / BAD". A failure is an a-Q2 FAIL, so Amendment 3 makes it "not run as registered" and GATE FAIL for both arms. | test_khop4_pilot_gate test_history_rows_all_checked: an earlier NaN loss, a row without loss, a non-object row, a repeated step, a string step, each in j1027 (R) and j1026 (L): only a-Q2 fails, GATE FAIL, READING not run as registered, both parsers refuse; a valid two-row history still passes. Pre-fix: "Lists differ: [] != ['R-Q2']". |
| 2 minor: timing lines dropped a non-finite s_per_step and printed 0 for a missing peak_gib | fixed (cheap) | khop4_table.py timing: any NEW-run history row without a finite s_per_step and peak_gib is UNREADABLE (the readout stops). khop4_pilot_gate.py INFO line: prints "s_per_step / peak_gib UNREADABLE" (INFO only, decides nothing). | test_khop4_table test_fail_closed: "NaN s_per_step" (K1X s17) and "missing peak_gib" (CUR1 s19) -> UNREADABLE; pre-fix: "SystemExit not raised". test_khop4_pilot_gate test_timing_info_counters; pre-fix: the gate printed "s_per_step -" and "peak_gib 0.0". |
| 3 minor: the PASS claim called every non-PASS rc3 line "below the bar" | fixed | The claim now reads "rc3 does not meet the full rc3 rule (secondary reading <reading> "<sub-label>"; D_3 m [CI])". Text only; the rc3 line never changes the verdict. | test_khop4_table test_pass_claim_names_the_rc3_reading (D_3 +5, C_3 -1 -> BETWEEN "gain from extra updates, not composition" in the claim) and test_pass_texts; both fail on the pre-fix copy. |

Changed files (md5): readout/khop4_pilot_gate.py 053d59e8, readout/khop4_table.py 57084af1,
readout/test_khop4_pilot_gate.py 1a248b3f, readout/test_khop4_table.py fa2a1d97. prep_khop4_confirm.py is unchanged
(the history rule acts through the a-Q2 line that it already refuses on).

Round 1 test run: run_tests.sh -> test_all.log, EXIT=0 (Wed Oct 28 06:22:34 CDT 2026). Readout block (3.12) 140 tests OK,
readout under 3.9.6 40 OK, stage 47 OK and 16 OK under 3.9.6, D5 OK twice, full repo suite: no new failure against repo.base.

### Round 2 (2026-10-28 06:33 CDT): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r2.md)

Codex: "Both findings concern reporting; neither changes gate selection or verdict statistics."

| Finding | Decision | Change | Regression test (fails on the pre-fix copy) |
|---|---|---|---|
| 1 minor: the gate printed no j980 / j990 curves and no plain val (card 3, "Reported, not decision lines") | fixed (cheap) | khop4_pilot_gate.py: "INFO j<id> plain gsm8kx .. gsm1kx .. gsm8kv2 .. valx .. val .. (n ..)" for j1025-j1028, j980 and j990 (val from the preds already checked by load; MISSING when not readable), and "INFO j980 / j990 (khop3 S2) <set> k1 .. k8" for the five curve sets. INFO only. | test_khop4_pilot_gate test_context_info_lines (the real j980 line equals card 0.1: gsm8kx 61.7, gsm8kv2 71.3, valx 63.1, val 67.9; (ca, fwd) 100.0 / 92.0 / 64.0 / 64.7; no rc line). Pre-fix: "(1025, [])". |
| 2 minor: the readout copied only double-space "INFO  a-" lines of the unchosen arm, so the gate's single-space "INFO a-Q6 ..." and "INFO j<id> ..." lines of that pair were dropped (card 2.3, "every gate value") | fixed | khop4_table.py: also copies "INFO <other>-" lines and "INFO j<id> " lines whose id (regex, int compare) is one of the unchosen pair (1025 / 1026 or 1027 / 1028); never the chosen pair, j980 / j975 or a 5-digit decoy j10250. | test_khop4_table test_pass_r_and_l with a gate text that holds those lines for both arms and the decoys; pre-fix: "'INFO L-Q6 j1025 gsm1kx 52.9 (reference 53.8)' not found". |

Changed files (md5): readout/khop4_pilot_gate.py 50c30cdb, readout/khop4_table.py 778f84f8,
readout/test_khop4_pilot_gate.py 3a97ce32, readout/test_khop4_table.py 006c0390.

Round 2 test run: run_tests.sh -> test_all.log, EXIT=0 (Wed Oct 28 07:41:35 CDT 2026). Readout block (3.12) 141 tests OK,
readout under 3.9.6 OK, stage OK under 3.12 and 3.9.6, D5 OK twice, full repo suite: no new failure against repo.base.

### Round 3 (2026-10-28 07:48 CDT): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r3.md)

| Finding | Decision | Change | Regression test (fails on the pre-fix copy) |
|---|---|---|---|
| 1 major: "kept, GSM damaged (arm a)" was printed when only K1X-a failed its a-Q6 bar; card 0.5 needs "a-Q6 fails for CUR-a" | fixed | khop4_pilot_gate.py keeps the CUR side of a-Q6 apart (cur_gsm_fail, only with a readable reference). The a-Q6 line and the arm rule are unchanged (both sides still decide a-Q6); only the READING text follows the card. No new reading is added for a K1X-only failure (the FAIL line names it). | test_gsm_damaged_reads_cur_only: K1X-R gsm8kx at reference - 5.1 -> FAIL R-Q6 only, GATE PASS ARM L, no "kept, GSM damaged"; CUR-R gsm8kv2 - 3.1 still prints it. Pre-fix: "'READING  kept, GSM damaged' unexpectedly found". |
| 2 minor: the INFO "max history lr" dropped a NaN or missing lr row | fixed (cheap) | Prints UNREADABLE unless every history row has a finite lr; args.lr stays the decision value (a-Q2). | test_history_lr_info_counters (NaN in j1027, missing in j1028 -> UNREADABLE; 0.0 in j1025 is finite -> max 3e-06); test_history_lr_info now expects UNREADABLE for a history without lr. Pre-fix: "'INFO j1027 max history lr UNREADABLE' not found". |
| 3 minor (style): i4_s1 / i4_s2 ran history_ok and last_step twice (check and detail text) | fixed | Each is computed once per call and reused in the detail; no behavior change (the I4 / a-Q2 / C1 tests are unchanged and pass). | none needed (no behavior change) |

Changed files (md5): readout/khop4_pilot_gate.py 63c063e9, readout/khop4_table.py 419826a6, readout/test_khop4_pilot_gate.py 7c4f7bf1.

Round 3 test run: run_tests.sh -> test_all.log, EXIT=0 (Wed Oct 28 08:59:27 CDT 2026). Readout block (3.12) 143 tests OK,
readout under 3.9.6 OK, stage OK under 3.12 and 3.9.6, D5 OK twice, full repo suite: no new failure against repo.base.

### Round 4 (2026-10-28 09:04 CDT): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r4.md)

| Finding | Decision | Change | Regression test (fails on the pre-fix copy) |
|---|---|---|---|
| 1 minor: the readout accepted a confirmation S2 run without preds_subq.jsonl (subq is not in args.eval, so khop3_table.load does not ask for it); the gate's a-Q1 does | fixed | khop4_table.py main: each CUR / K1X run must hold every preds file of the 663 env's GSM / hint list (khop_pilot_gate.gsm_sets, subq included) exactly once, and each CUR1 / K1X1 run preds_subq (its eval_subq) exactly once; else UNREADABLE. Checked on the real j980: subq is the only listed set outside args.eval. | test_khop4_table test_fail_closed: "CUR s18 without preds_subq" and "K1X1 s17 without preds_subq" -> UNREADABLE "preds files ['subq'] missing"; pre-fix: "SystemExit not raised". The synthetic S1 fixture now carries preds_subq, as the real j975 / j985 tarballs do. |

Changed files (md5): readout/khop4_table.py 94a8fe47, readout/test_khop4_table.py be32b7bd.

Round 4 test run: run_tests.sh -> test_all.log, EXIT=0 (Wed Oct 28 10:26:09 CDT 2026). Readout block (3.12) 143 tests OK,
readout under 3.9.6 43 OK, stage OK under 3.12 and 3.9.6, D5 OK twice, full repo suite: no new failure against repo.base.

### Round 5 (2026-10-28 10:35 CDT): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r5.md)

| Finding | Decision | Change | Regression test (fails on the pre-fix copy) |
|---|---|---|---|
| 1 major: the timing-counter check (round 1 finding 2) ran after the VERDICT and claim lines were printed; run_khop4.sh kept stdout only, so a saved report could show PASS without the UNREADABLE line | fixed | khop4_table.py: the s_per_step / peak_gib check moved into the load loop, before any line is printed (the timing block reads the checked rows). No unreadable() call is left after the first printed line. run_khop4.sh: the table's stderr goes into the report; on a non-zero exit the report ends with "== READOUT FAILED (khop4_table.py exit N): nothing in this file is a reading" (also on stderr), the .json is removed, and the script exits with that code. | test_fail_closed: NaN s_per_step -> SystemExit and no "VERDICT" / "claim:" on stdout; pre-fix: "'VERDICT' unexpectedly found". test_run_khop4_sh: a gate text the table refuses -> the report holds the table's message and ends with READOUT FAILED; pre-fix: the message was not in the report. |
| 2 minor: the exact t = 2.78 boundary assertion in test_exact_bars never ran (its square-root condition was false) | fixed (test only) | The test now uses x_i = m + i d, i in (-3, -1, 0, 1, 3), d = m / 2.78 (sample var 5 d^2, so t = 2.78 and CI lower = 0 exactly), asserts the exact equalities and PASS unconditionally, and a d larger by 1e-9 is not PASS. | The code was already right (Codex checked it). Mutation check: with XS.lo_ge replaced by lo_gt (strict), test_exact_bars fails. |

Changed files (md5): readout/khop4_table.py d1ec57c4, readout/test_khop4_table.py f8355e8f, readout/run_khop4.sh c5ae4197.

Round 5 test run: run_tests.sh -> test_all.log, EXIT=0 (Wed Oct 28 11:55:04 CDT 2026). Readout block (3.12) 143 tests OK,
readout under 3.9.6 43 OK, stage OK under 3.12 and 3.9.6, bash -n OK, D5 OK twice, full repo suite: no new failure against repo.base.

### Round 6 (2026-10-28 12:01 CDT, the last of the 6 allowed rounds): VERDICT: SUBMIT AFTER FIXES (critique/CODEX_REVIEW_khop4.r6.md)

Codex: "No blocker or major defects found. Read-only checks confirmed data hashes, replay equality, envs, DAGs, and
statistical helpers."

| Finding | Decision | Reason / proposed fix |
|---|---|---|
| 1 minor: rc_reading(j) prints the fixed rc2 drift "-1.4" (re-train - original) in the NULL "below the original A2 set only" text also for the rc3 line, where the drift is -2.49 points | OPEN, not fixed | The loop budget is used up, and a fix made now could not be reviewed (task rule: every fix gets one more review). Effect: one number in the text of the secondary rc3 line. The rc2 text is right (65.66 - 64.28 = 1.38, card 0.6); no verdict, gate line or statistic reads it. Proposed fix (one line in khop4_table.rc_reading): print the metric's own drift, the five-seed mean of Dorig - Dre for rc j from exact counts, in place of the constant; regression test: an rc3 NULL fixture below the original set only prints its own drift. It needs one more Codex round (a lead decision on the round budget) or the lead's acceptance of the text as is before the bundle. |

State after round 6: no code was changed after the round 6 review, so every change in rounds 1-5 was reviewed at least once
(round 6 reviewed the round 5 code). test_all.log is the round 5 run (EXIT=0). build.diff, the envs, the subs and the DAGs
did not change in any round (no generator changed; D5 holds).
