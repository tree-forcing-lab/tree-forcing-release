# Build report: CARD_khop_v3 (family khop3, ids 975-999), 2026-10-27 CDT

E = <path> (the old scratchpad path .../scratchpad/exp is a symlink to it).
Spec: <path> with Amendments 1-3. Nothing was bundled, copied to CHTC or submitted.
No cluster command was run. The card file was not edited in the build; the fix round (last section) added Amendment 2
and changed readout / stage files and tests; where a statement below differs, the Fix round section holds, and the Codex rounds section holds over both.

## 1. What changed

Code (E/src_khop3/repo; E/src_khop3/build.diff = `diff -ruN -x __pycache__ -x data repo.base repo`, 811 lines, md5
4b746406cb2bcd5639327a3e1c2b2cf2).
Two new files, no edited file:

| File | Content |
|---|---|
| scripts/build_khop3.py | The card 1.2 generator. It imports build_khop.py and build_khop2.py and edits neither. |
| tests/test_build_khop3.py | Generator, replay, refusal and real-file tests, and the two-stage tiny FSDP run (section 2). |

No trainer flag and no trainer change. The A2 path of both stages is the parent's code byte for byte (D5, section 2).
The flag-off path is therefore identical to repo.base by construction: run_tests.sh checks D5 against E/src_khop2/repo
and against repo.base, and runs the parent's flag-off identity tests against repo.base.

Generator rule (build_khop3.py, card 1.2 steps 1-4):
1. Inputs. train_kd27.jsonl must have the md5 of the khop2 report and 92545dd4...; train_k1.jsonl the khop2
   report's parent_md5 and 4bb15f66...; the gsmvt val.jsonl 9f55fec5... with 393 rows; the 12 CARD_khop_v1 files the
   md5s of the khop report. Any other value is a SystemExit. An output directory equal to an input directory is a
   SystemExit. An output directory that holds a file other than train.jsonl / val.jsonl is a SystemExit.
2. student_khop3_c27/train.jsonl is train_kd27.jsonl byte for byte. Both val.jsonl files are the gsmvt val byte for byte.
3. K1X replay. The generator reads the stream parameters from the two parent reports (train seed 1, 1,000 KH and 2,000
   KD-extra items per task; khop2 seed 3, 6,000 new rows per task). It wraps build_khop.Item.row, runs
   build_khop.build_train and build_khop2.new_rows with the forbid sets that build_khop2.build computes, and stops
   unless the replay equals train_kd27 (27,000 rows) and train_k1 (3,000 rows) dict by dict. Rows 0..2,999 of K1X are
   the train_k1 lines byte for byte. Rows 3,000..26,999 are the K1 twins by the card rule (rng1 keyed by
   "khop3-k1-<item_id>", i1 continues from the train_k1 counts per (task, k)).
4. data/eval/khop3/build_report.json with every line of card 1.2 step 4. d_fail lists each failed line; main() exits 2
   on a failed line.

The replay also counts the row() calls per item_id. On the real streams every item_id is rendered once
(ids_rendered_more_than_once 0): every khop2 redraw was a value-key redraw, made before row(). So the "last call wins"
rule of the card has no effect on these files. A test checks the recorded Item of every one of the 27,000 rows.

Stage (E/stage, new files only):

| File | Content |
|---|---|
| prep_khop3.py | Writes the 21 envs 975-995, the two subs, the smoke DAG and the pilot DAG. Refuses another env on a family id and any env on the spare ids 996-999. Ids come from `re.match(r"(?:smoke_)?(\d+)_", name)`. `--dry-run` prints only; `--jobs-dir` / `--out` exist for the tests. |
| prep_khop3_confirm.py | Writes confirm_x1khop3_4b.dag (16 nodes) only when the last non-empty line of the gate output is exactly "GATE PASS", and only when every confirmation env equals prep_khop3.py's text. `--dry-run` prints the table. |
| check_weights_khop3.sh | DAG PRE of every S2 node (section 4). |
| test_prep_khop3.py | Env, sub, PRE, DAG, confirmation and 4-digit-id tests. |
| train_x1khop3_4b.sub | S1 sub: train_x1khop2_4b.sub with lines 13 (CODE_TGZ), 16 (transfer_input_files), 33 (log) and the header comment lines 1 and 3 changed. Executable run_train5.sh. |
| train_x1khop3w_4b.sub | S2 and smoke_995 sub: train_x1verify2w_4b.sub with lines 9, 12, 29 and the header comment lines 1-3 changed. Executable run_train_x1verify2.sh (md5 3866fc7788af662fe94052a9d2bae4c8, unchanged, asserted). |
| smoke_x1khop3_4b.dag | smoke_975, smoke_985 (S1 sub), smoke_995 (S2 sub), smoke="1". |
| pilot_x1khop3_4b.dag | j975 -> j980, j985 -> j990 (PARENT / CHILD), PRE check_weights_khop3.sh on j980 and j990. |
| condor/jobs_x1/975-995 | 21 envs (section 4). |

Id parsing with 4-digit ids (task rule): prep_khop3.py (regex, tested), round_gpuh.py (regex `(\d+)_`, int compare,
tested on 999 / 1000 / 1024), check_pilot.sh and check_weights_khop3.sh (take names, parse no id; tested), post_push.sh
(takes the name as $1, parses no id; read, not tested), khop3_pilot_gate.py and khop3_table.py (reference identity
regex `(^|_)<id>(_|\.tgz$)`, tested with results_1520_...), run_khop3.sh (matches results_<id>_*, tested on a dir where
1000_ / 1024_ sort first).

Readout (E/readout, new files only):

| File | Content |
|---|---|
| khop3_pilot_gate.py | Pilot gate P0-P11 with P7b and P7c. Exit 0 only for "GATE PASS". |
| khop3_table.py | Five-seed readout: exact cells, I1-I7, N1, V, the verdict of section 0 in the card's order, the rc3 line, sections 2.2 and 2.3, timing and compute. Imports khop_table.py and khop2_table.py and edits neither. |
| run_khop3.sh | `bash run_khop3.sh <gate output> <retry-walls file> [results dir]`. Gathers the 20 new and 34 reference / context tarballs, stops on a missing tarball or two tarballs for one id, prints round_gpuh.py 975 999 and the retry walls. Log E/logs/khop3_readout_<date>.txt and .json. |
| test_khop3_table.py, test_khop3_pilot_gate.py | Tests (section 2). |

## 2. Tests

Runner: E/src_khop3/run_tests.sh. Output: E/src_khop3/test_all.log. Result of the final run: section 2.1.

Python. The task names the venv_pfx python of the c19b05fb scratchpad. That venv is broken: the macOS /tmp cleanup
removed its pyvenv.cfg and files inside torch, so `import torch` fails. run_tests.sh uses a new venv with the same
versions (Python 3.12.7, torch 2.14.1, transformers 5.20.2, accelerate 1.15.0):
<tmp>
The same /tmp cleanup can remove this venv too. Set KHOP3_TEST_PY to another python with torch and transformers 5.x.

| Suite | What it checks |
|---|---|
| repo tests.test_build_khop3 (new, 13 tests) | Refusals before any write: a changed train_kd27, train_k1, val.jsonl or khop parent file, --out-root or --report equal to an input dir, an extra file in an output dir. Replay equals both files; a replay with one changed row (KH, khop2 or K1 stream) or one row short stops the build; the recorded Item of every row renders that row's statements. Real build in a temp dir: c27 and val byte-equal to the inputs, K1X rows 0..2,999 = train_k1, and for each of the 27,000 rows the statements equal the kd27 twin's, the query is the hop-1 variable by an independent parent walk, the gold is its value, the option rule (ca / wp) or the five roots (vb), one-hot teacher, no eval prompt hash; gold shares in [0.18, 0.22] per (task, k); report values recomputed from the files; two builds byte-equal and equal to the files in E/src/data; parent md5s before = after = card values. Two-stage tiny run (section 2.2). |
| repo parent tests (COTD_REF_REPO = E/src_khop/repo.base) | tests.test_build_khop2, test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train, unchanged. |
| repo flag-off identity (COTD_REF_REPO = E/src_khop3/repo.base) | test_extra_train flag off (losses, eval, preds byte-identical), sbatch without --extra-train, test_frontier_train. |
| readout test_khop3_table (new, 21 tests) | Every reading and sub-label of section 0 (list in the file docstring), the exact-bar cases (+3.0 in print but 1e-7 below 3/100 is not PASS; exactly 3/100 with t exactly 2.78 is PASS; C_2 CI lower exactly 0 is not > 0), fail-closed verdict inputs, the I4 log lines against the real j875 (export) and j851 (import) line formats, I7's loss lines (5 lines pass, 4 lines, NaN or none UNREADABLE), the retry-walls parser. End to end through main() on the real A2-6k tarballs plus synthetic CUR1 / CUR / K1X1 / K1X dirs (the original A2 set's GSM / hint preds with chosen rc2 counts): PASS, "installed, then lost", "control invalid"; UNREADABLE on a missing khop preds file, a plain file one row short, a NaN loss, a wrong seed in metrics.json, a khop file one row short, a reference with a wrong job id and a 4-digit look-alike (results_1520_...); an S2 log that names another S1 reads "not run as registered"; refusals of a non-PASS gate and a missing retry-walls file. run_khop3.sh end to end on a results dir where 1000_ / 1024_ sort before 975_, and its refusals (gate, retry walls, missing tarball, two tarballs for one id). |
| readout test_khop3_pilot_gate (new, 9 tests) | A good pilot passes (real reports and data files, real j520 / j663, synthetic pilots on j663's preds); each of P0-P11 fails alone; the P6, P7, P7c, P9 and P11 edges of the card (list in the file docstring); P7b printed; the gate prints no rc / hint line. |
| readout parent tests | test_khop2_table, test_khop2_pilot_gate, test_khop_table, test_khop_pilot_gate, unchanged. |
| readout under /usr/bin/python3 (3.9.6) | test_khop3_table and test_khop3_pilot_gate (the AP / Codex python3). |
| stage test_prep_khop3 (new, 17 tests), also under 3.9 | Every env sourced under bash against 766 and 663 (card 1.4 key sets), names, TAGs, SAVE only on S1, MODEL = models/<WTAR>/model, the 995 env on the 663 trunk, no env on 996-999; both subs (exact changed lines, CODE_TGZ = transfer_input_files, executables, runner md5); check_weights_khop3.sh on 11 bad fixtures and on the real j875 tarball; smoke and pilot DAGs; the confirmation generator (10 refused gate texts, a changed or missing env, 16 nodes with PARENT / CHILD, PREs, POST, RETRY, PRIORITY; dry run writes nothing); 4-digit ids (prep on a jobs dir with 1000_, 1024_, 9750_, 9960_ and smoke_9990_ envs; the 996 / 999 / other-975 refusals; round_gpuh.py and check_pilot.sh on a results dir where 1000_ sorts first). |
| stage parent tests | test_prep_khop2, test_prep_khop, unchanged. |
| bash -n | check_weights_khop3.sh, run_khop3.sh. |
| D5 | `diff -rq -x __pycache__ -x data` against E/src_khop2/repo and against repo.base: only build_khop3.py and test_build_khop3.py. |

### 2.1 Final run

test_all.log, 2026-10-27 12:44-13:06 CDT, exit 0 (superseded: test_all.log now holds the fix-round run, see Fix round):

| Block | Result |
|---|---|
| repo (new): tests.test_build_khop3 | Ran 13 tests, OK |
| repo parent tests (COTD_REF_REPO = E/src_khop/repo.base): test_build_khop2, test_build_khop, test_extra_train, test_khop_eval_paths, test_frontier, test_frontier_train | Ran 79 tests, OK (skipped 1: test_frontier.TestRealData needs data/) |
| repo flag-off identity (COTD_REF_REPO = E/src_khop3/repo.base) | Ran 23 tests, OK |
| readout (Python 3.12): test_khop3_table, test_khop3_pilot_gate, test_khop2_table, test_khop2_pilot_gate, test_khop_table, test_khop_pilot_gate | Ran 97 tests, OK |
| readout under Python 3.9.6: test_khop3_table, test_khop3_pilot_gate | Ran 30 tests, OK |
| stage: test_prep_khop3, test_prep_khop2, test_prep_khop | Ran 30 tests, OK |
| stage under Python 3.9.6: test_prep_khop3 | Ran 17 tests, OK |
| bash -n check_weights_khop3.sh, run_khop3.sh | OK |
| D5 against E/src_khop2/repo and against repo.base | D5 OK twice (only scripts/build_khop3.py and tests/test_build_khop3.py) |

Full-suite regression against repo.base: the repo differs from repo.base only by the two new files (D5), so every
other test module runs the same code in both trees; the known parent failures (3 test_tree_target, 2 macOS bash 3.2
TestSbatch, khop2 BUILD_REPORT section 2.1) are not touched. The whole tests/ directory was not re-run.

### 2.2 The two-stage tiny run (card section 5, Amendment 1 finding 8)

The trainer logs no per-update item ids, and D5 forbids a trainer flag. The test writes a wrapper script to a temp
dir. The wrapper loads scripts/train_student.py as a module, wraps its module-level kl_loss, and calls main(). Under
grad mode the wrapper records each training micro-batch: item_id, the full text (so the option order), labels, target
and weight. Evaluation runs under no_grad, so it is not recorded. All runs: torchrun 1 rank, --local-addr 127.0.0.1,
--fsdp --fsdp-optim cpu-adamw, --eval-value-render vt, --label-smoothing 0.1, --token-budget 4096, seed = order seed
16, micro-bs 2, grad-accum 2.
- Stage 1: 18 khop rows (build_train seed 1, 6 per task), --save. Log "stage all: 18 examples per pass, 5 steps",
  final step 5 = ceil(18 / 4), finite losses. The trace has 20 sequences: the 18 rows once, and the last update filled
  from the next shuffled pass. This is the trainer's rule for every A2 job (844 x 32 = 27,008 at S1, 186 x 32 = 5,952
  at S2).
- Stage 2 (S2): --model <stage-1 out>/model, 6 GSM rows, eval sets val + one extra set (kx), --eval-at 0.5
  --eval-mid val, --save.
- References from the same stage-1 weights without the extra set: A2w (--eval-at 0.5) and A2w_noat (no --eval-at).
  S2 equals both: the same micro-batch trace, the same per-update losses (exact, --log-every 1) and the same final
  weights (torch.equal on every tensor). The final step is 2 = ceil(6 / 4).
- A2base: an A2 run from the base tiny model with the A2w eval arguments. Its trace equals S2's trace; its losses
  differ (only the starting weights differ).

## 3. Data

Command (log E/src_khop3/logs/build_khop3.log, 30 s):

    cd E/src_khop3/repo && <python> scripts/build_khop3.py --khop2 E/src/data/eval/khop2 --khop E/src/data/eval/khop \
      --gsm-dir E/src/data/student_gsm8k_tf_vt --out-root E/src/data --report E/src/data/eval/khop3

| File | Rows | md5 |
|---|---|---|
| E/src/data/student_khop3_c27/train.jsonl | 27,000 | 92545dd42ca451c39e0e7c104aa78d1e (= train_kd27) |
| E/src/data/student_khop3_c27/val.jsonl | 393 | 9f55fec541246e41401a7b5fa6c5510a (= gsmvt val) |
| E/src/data/student_khop3_k1x/train.jsonl | 27,000 | d7cc9716e5785461d16ec92156104261 |
| E/src/data/student_khop3_k1x/val.jsonl | 393 | 9f55fec541246e41401a7b5fa6c5510a |
| E/src/data/eval/khop3/build_report.json | - | 0d1c75608f19a14fa3e99d979c28557c |

Report lines (d_pass true, d_fail []):
- Replay: 27,000 / 27,000 kd27 rows and 3,000 / 3,000 train_k1 rows equal; ids rendered more than once 0.
- K1X: statement mismatches 0, query not hop-1 0, gold not the hop-1 value 0, option-rule failures 0, vb options not
  the roots 0, build_khop.row_rule_ok errors 0, teachers not one-hot 0, key set 32 gsmvt keys, item_ids distinct.
- Gold letter shares: 0.20 in every (task, k) cell and overall (5,400 per letter).
- Eval prompt-hash overlap: K1X 0, c27 0. kd27 prompt-hash overlap: 730 K1X rows, all k = 1 (for k = 1 the K1 query is
  the kd27 query; the prompt is equal when the layout draw also matches).
- Token totals with the Qwen3.5-4B tokenizer the khop2 report records: kd27 5,361,925 (equal to the khop2 report),
  K1X 5,361,776 (-0.003%).
- Inputs unchanged (md5 before = after); each output dir holds exactly train.jsonl and val.jsonl.

Bundle placement. jobs/build_bundles.sh line 21 copies `$E/src/data/student_${v#x1}_*` for variant x1khop3, which is
`student_khop3_*`. Checked: the glob matches exactly student_khop3_c27 and student_khop3_k1x. Line 18 copies
data/eval whole, so data/eval/khop3/build_report.json, data/eval/khop2 (4 files) and data/eval/khop (13 files) are in
the bundle. The env DATA keys name data/student_khop3_c27 and data/student_khop3_k1x, relative to the job's ~/cotd.
Size: 2 x 36 MB (about 9 MB gzip).

## 4. Jobs and exact XARGS

Every training env is the 766 env with the card 1.4 keys changed (sourced under bash in test_prep_khop3). Common
values: EPOCHS=1, LR=3e-6, LSMOOTH=0.1, MBS=2, GA=16, NGPU=1, FSDP_OPTIM=cpu-adamw, OPTIM=adamw, PREC=bf16,
TOKBUDGET=4096, ARMS=A2. Resources: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300, RETRY 1, POST post_push.sh.

| Job | Stage | Arm | Name | DATA | MODEL / WTAR | SAVE | XARGS (sourced) | EVALAT |
|---|---|---|---|---|---|---|---|---|
| smoke_975 | smoke | CUR S1 env, SMOKE_ONLY | smoke_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp | student_khop3_c27 | Qwen3.5-4B / - | 1 (skipped under SMOKE_ONLY) | `--eval-value-render vt` | '' |
| smoke_985 | smoke | K1X S1 env, SMOKE_ONLY | smoke_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp | student_khop3_k1x | Qwen3.5-4B / - | as above | `--eval-value-render vt` | '' |
| smoke_995 | smoke | S2 path on the 663 trunk | smoke_995_4b_gsmvt_A2-khopcur2w663-c1lr3ch_s16_fsdp | student_gsm8k_tf_vt | models/663_.../model / 663_4b_gsmvt_A2-c1lr3ch_s16_fsdp | '' | `--eval-value-render vt` | 0.5 |
| j975 | pilot | CUR S1 s16 | 975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp | student_khop3_c27 | Qwen3.5-4B / - | 1 | `--eval-value-render vt` | '' |
| j980 | pilot | CUR S2 s16 | 980_4b_gsmvt_A2-khopcur2w975-c1lr3ch_s16_fsdp | student_gsm8k_tf_vt | models/975_.../model / 975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp | '' | `--eval-value-render vt` | 0.5 |
| j985 | pilot | K1X S1 s16 | 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp | student_khop3_k1x | Qwen3.5-4B / - | 1 | `--eval-value-render vt` | '' |
| j990 | pilot | K1X S2 s16 | 990_4b_gsmvt_A2-khopk1x2w985-c1lr3ch_s16_fsdp | student_gsm8k_tf_vt | models/985_.../model / 985_... | '' | `--eval-value-render vt` | 0.5 |
| j976-979 | confirm | CUR S1 s17-20 | 97x_4b_kd27_A2-khopcur1-c1lr3ch_s<s>_fsdp | student_khop3_c27 | Qwen3.5-4B / - | 1 | as j975 | '' |
| j981-984 | confirm | CUR S2 s17-20 | 98x_4b_gsmvt_A2-khopcur2w<976-979>-c1lr3ch_s<s>_fsdp | student_gsm8k_tf_vt | the same seed's CUR S1 | '' | as j980 | 0.5 |
| j986-989 | confirm | K1X S1 s17-20 | 98x_4b_kd27k1_A2-khopk1x1-c1lr3ch_s<s>_fsdp | student_khop3_k1x | Qwen3.5-4B / - | 1 | as j985 | '' |
| j991-994 | confirm | K1X S2 s17-20 | 99x_4b_gsmvt_A2-khopk1x2w<986-989>-c1lr3ch_s<s>_fsdp | student_gsm8k_tf_vt | the same seed's K1X S1 | '' | as j990 | 0.5 |
| j996-999 | spare | re-runs only, not written | | | | | | |

EXTRA_EVALS: S1 = gsm8kv2, gsm8kx, gsm1kx, valx and the nine khop sets (766 order, no hint sets). S2 and smoke_995 =
766's (the 663 sets + the nine khop sets). EVALMID: S1 '', S2 the 766 value. JOBKEY "4b kd27 A2-<TAG> s<s> fsdp x1khop3"
(kd27k1 for K1X S1), "4b gsmvt A2-<TAG> s<s> fsdp x1khop3" for S2. An S2 env equals the 663 env (seed keys set) except
SAVE, TAG, JOBKEY, MODEL, WTAR and the nine khop sets appended to EXTRA_EVALS (tested).

Expected log lines (I4 / P2): S1 "stage all: 27000 examples per pass, 844 steps", last step line 844, "== exit 0",
"== weights -> <staging>/weights/<S1 name>.tar (9.1G)"; S2 "== weights <staging>/weights/<S1 name>.tar ->
models/<S1 name>/model (9.1G)", "stage all: 5923 examples per pass, 186 steps", last step line 186, "== exit 0".

## 5. Compute

Measured walls (`python3 -I E/jobs/round_gpuh.py`, 2026-10-27): D27 j875-879 1.63-2.16 h (1,029 updates with khop
evals), R3 j880-884 1.59-1.86 h, H3 j885-889 0.92-1.10 h, khop2 smokes 0.03-0.05 h; A2 re-trains j663-666 0.45-0.53 h
(186 updates, no khop evals); warm-start S2-shape jobs j851 / j852 0.32 / 0.38 h, their smokes 0.08 / 0.09 h. No khop3
tarball exists yet (`round_gpuh.py 975 999`: 0 jobs).

| Stage | Jobs | Estimate (card fit) | Bar / cap |
|---|---|---|---|
| smoke | smoke_975, smoke_985, smoke_995 | 0.15-0.20 GPU-h (smoke_995 is warm-start: about 0.09 h) | in the pilot cap |
| pilot | j975, j985 (1.52 h each), j980, j990 (0.69 h each) | 4.42 + smokes = about 4.6 GPU-h | P7 2.0 / 0.9 h, P7c 6.0 GPU-h measured |
| confirmation | 8 S1 x 1.52 + 8 S2 x 0.69 | 17.68 GPU-h (`prep_khop3_confirm.py --dry-run`) | 25; P7b 4 x the pilot walls |

Staging (card section 3, Amendment 1 finding 4): each S1 writes a 9.04 GiB tar to $STAGING/weights. The pilot needs 2
tars, the confirmation 8 more. The card records that staging is probably full now. See the operator steps (section 6).

## 6. Operator section (sev-e0)

All paths on the Mac are under E. The AP DAG directory is the one the round-F DAGs ran from (RESULTS_LOG: ~/cotd_chtc).

### 6.1 Before any submission

1. Amend the Round G line for khop3 in RESULTS_LOG.md to the exact text of card section 4 (main / sev-e0; finding 5).
2. Build the bundle:

       cd E && REPO_DIR=E/src_khop3/repo jobs/build_bundles.sh \
         "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2-khop3" x1khop3

   Output: E/bundle_out/cotd_code_data_x1khop3_4b.tgz and .md5.
3. Pre-submit checks on the bundle (each an exact file set; directory entries and the "./" prefix are filtered out, so the
   counts do not depend on the tar listing format; B = E/bundle_out/cotd_code_data_x1khop3_4b.tgz):
   - `tar tzf B | grep 'data/student_khop3_' | grep -v '/$' | sort`: exactly the 4 files student_khop3_c27/train.jsonl,
     student_khop3_c27/val.jsonl, student_khop3_k1x/train.jsonl, student_khop3_k1x/val.jsonl.
   - `tar tzf B | grep 'data/eval/khop3/' | grep -v '/$'`: build_report.json only (1 file).
   - `tar tzf B | grep 'data/eval/khop2/' | grep -v '/$'`: the 4 khop2 files.
   - `tar tzf B | grep 'data/eval/khop/' | grep -v '/$'`: the 12 parent files and build_report.json (13 files).
   - `tar tzf B | grep -E 'condor/jobs_x1/(97[5-9]|98[0-9]|99[0-9])_' | grep -v '/$'`: exactly the 21 envs 975-995 (no
     996-999).
   - After extraction to a temp dir: md5 of the 4 student_khop3 files and of data/eval/khop3/build_report.json equal
     section 3; the 21 envs equal E/stage/condor/jobs_x1 byte for byte; repo/scripts/build_khop3.py exists.
   - Fix round F6: the top-level GSM eval files equal the x1khop2 bundle copies (exact file set and md5s). With
     B2 = E/bundle_out/cotd_code_data_x1khop2_4b.tgz and X3, X2 two empty temp dirs:

         tar xzf B -C X3 --include='./data/eval/*.jsonl'; tar xzf B2 -C X2 --include='./data/eval/*.jsonl'
         (cd X3/data/eval && md5 -r *.jsonl) > X3.md5; (cd X2/data/eval && md5 -r *.jsonl) > X2.md5; diff X3.md5 X2.md5

     Pass: diff prints nothing and each list has 21 lines (on 2026-10-27, E/src/data/eval against B2: 21 files, 0
     differences). The bsdtar pattern also extracts the .jsonl files of the subdirectories; `md5 -r *.jsonl` lists
     the top level only, which is the set this line checks (the subdirectories have their own lines above).
4. Copy to the AP DAG directory and check the md5 on both sides:
   - cotd_code_data_x1khop3_4b.tgz (md5 from E/bundle_out/cotd_code_data_x1khop3_4b.md5);
   - E/stage/train_x1khop3_4b.sub (md5 bfc9a184870b8302897fd20cc1ce10b0) and E/stage/train_x1khop3w_4b.sub
     (c30e4b8624e8050d58e99b3a546d8c6e);
   - E/stage/smoke_x1khop3_4b.dag (7b0a8fe7fea8e932b63710060aac14ca) and E/stage/pilot_x1khop3_4b.dag
     (c8535c640302b69ea90948540153f288);
   - E/stage/check_weights_khop3.sh (11448089f57fe745f9c7a0c7c3cb11c4); then `chmod +x check_weights_khop3.sh`;
   - the 21 envs E/stage/condor/jobs_x1/97[5-9]_*, 98[0-9]_*, 99[0-5]_* into condor/jobs_x1/ of the DAG directory.
     No PRE of this family reads an env file, but the copy keeps the AP in line with the bundle (round-F lesson).
5. Check on the AP that these files already exist in the DAG directory: run_train_x1verify2.sh with md5
   3866fc7788af662fe94052a9d2bae4c8, run_train5.sh (its SAVE block writes "== weights -> ..."; j875 used it),
   check_pilot.sh (md5 e1a4ee9982ceb7c31e4040c7da64da1c), post_push.sh, chtc_settings.inc, results/.
6. Staging lines, read back on the AP from the quota tool and du (card section 4; no assumption):
   - $STAGING/weights/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tar exists (smoke_995 reads it; never delete it).
   - Before the pilot: free space >= 23.1 GiB (2 x 9.04 + 5) and file count + 2 < 1,000.
   - Before the confirmation: free space >= 77.3 GiB (8 x 9.04 + 5) with the 2 pilot tars present, and file count + 8
     < 1,000. Space that verify3 (ids 1000-1024) plans to use counts as used.
   - When a line fails: stop and ask the second author (deletion of named old tars, candidates 651, 656, 664-669, or a quota
     raise). Deleting a tar is the second author's decision. Submit nothing.
7. Locally (Mac): the reference tarballs E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz and
   results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz exist (they do now).

What the PREs read on the AP: check_weights_khop3.sh <S1 name> reads results/results_<S1 name>.tgz only (metrics.json,
logs/<S1 name>.out: the last "== exit" line and the weights export line). check_pilot.sh reads the four pilot tarballs
in results/. Neither reads an env file or $STAGING. The S2 runner itself refuses a missing or bad weights tar (exit 1
before any model load).

### 6.2 Submit order

1. Smoke: `condor_submit_dag smoke_x1khop3_4b.dag`. Smoke gate (by hand, on the three tarballs): each log ends with
   "== exit 0", and the smoke_995 log has the line
   "== weights <staging>/weights/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tar -> models/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp/model".
   The smokes do not test the S1 weights export (run_train5.sh skips SAVE under SMOKE_ONLY).
2. Pilot: `condor_submit_dag pilot_x1khop3_4b.dag` (j975 -> j980, j985 -> j990).
3. Retry walls (fix round F3): from the job event logs of the subs (logs/train_x1khop3_4b.log and
   logs/train_x1khop3w_4b.log in the DAG directory), write one "<node> <hours>" line per failed execution: every
   interval from event 001 (executing) to the next 004 (evicted), 005 (terminated), 009 (aborted) or 012 (held) of the
   same job id, except the one interval of the landed execution of each node. The DAG RETRY lines of nodes.log are not
   enough: a backfill eviction or a GPU-preflight exit 86 restarts the job inside one DAG attempt (max_retries = 3),
   and only the event log shows it. The smoke and pilot DAGs use the same subs, so both write to the same event log
   and both name a node j975 / j985: tell them apart by the job id (cluster) that each DAG's nodes.log records for its
   node in the submit event (000 ... "DAG Node: j975"). Empty file when no node had a failed execution. Example:
   E/logs/khop3_pilot_retry_walls.txt. Write the run name
   (smoke_975_... or 975_...) as <node>, not the DAG node name: both DAGs name their nodes j975 / j985.
4. Gate, locally after the 7 tarballs are in E/results:

       python3 -I E/readout/khop3_pilot_gate.py \
         --cur1 E/results/results_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tgz \
         --cur E/results/results_980_4b_gsmvt_A2-khopcur2w975-c1lr3ch_s16_fsdp.tgz \
         --k1x1 E/results/results_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tgz \
         --k1x E/results/results_990_4b_gsmvt_A2-khopk1x2w985-c1lr3ch_s16_fsdp.tgz \
         --smokes E/results/results_smoke_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tgz,E/results/results_smoke_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tgz,E/results/results_smoke_995_4b_gsmvt_A2-khopcur2w663-c1lr3ch_s16_fsdp.tgz \
         --ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz \
         --report E/src/data/eval/khop3/build_report.json --khop2-report E/src/data/eval/khop2/build_report.json \
         --parent-report E/src/data/eval/khop/build_report.json --retry-walls E/logs/khop3_pilot_retry_walls.txt \
         > E/logs/khop3_pilot_gate_<date>.txt

5. Last line "GATE PASS": `python3 -I E/stage/prep_khop3_confirm.py --gate-output E/logs/khop3_pilot_gate_<date>.txt`
   writes E/stage/confirm_x1khop3_4b.dag (16 nodes). Check the staging line for the confirmation (6.1 item 6), copy
   the DAG to the AP, `condor_submit_dag confirm_x1khop3_4b.dag`, tell the second author (he can stop it). Last line
   "GATE FAIL: ...": submit nothing, report the failing lines to the second author. prep_khop3_confirm.py runs on the Mac (it
   imports prep_khop3.py and prep_khop.py and reads the 766 template env in E/stage/condor/jobs_x1); copy only
   confirm_x1khop3_4b.dag to the AP (fix round F5). It refuses a gate output that is not a full passing khop3 output
   (fix round F4). Re-runs (fix round F2): a node that failed after its RETRY is re-run by resubmitting the same
   node, under its own id, name, env and tar name: the rescue DAG (`condor_submit_dag` on the same DAG file picks up
   the .rescue001 file) or a one-node DAG with the same JOB / VARS / SCRIPT / RETRY lines. Never a new id: 996-999
   stay unused, and no tool reads them. Allowed only when the measured walls of the phase, the retry walls so far and
   the re-run's P7 bar stay under the cap (6 pilot, 25 confirmation); else stop and report. Add the failed
   executions of the resubmitted node to the retry-walls file.
6. Readout when all 20 tarballs are in E/results: write the retry-walls file of the pilot and confirmation DAGs, then
   `bash E/readout/run_khop3.sh E/logs/khop3_pilot_gate_<date>.txt <retry-walls file>`.

## 7. Risks and deviations

1. venv: section 2. The task's venv_pfx no longer imports torch.
2. Generator flag: the card lists --khop2, --gsm-dir, --out-root, --report. build_khop3.py also takes --khop (default
   data/eval/khop): the replay and train_k1 need the CARD_khop_v1 files, and a fixed sibling path would be a hidden
   input. No behavior of the card changes.
3. check_weights_khop3.sh "small tar" (card test list): the PRE does not stat the tar (card 1.4). It reads the size
   that run_train5.sh prints with the export line (du -sh of the model dir) and refuses a size below 8 GiB (a 4B bf16
   model dir is 9.1G). "Missing tar" is the log's "weights export ... failed" line or no export line.
4. Gate output separator: "GATE FAIL: <line>; <line>" uses "; ", because the P9 label "P9 (installed, then lost at
   s16)" holds a comma. prep_khop3_confirm.py and run_khop3.sh accept only the exact last line "GATE PASS".
5. Gate inputs: --smokes takes the three smoke tarballs in the order 975, 985, 995; each name is checked
   (results_smoke_<id>_...). A missing smoke fails P7c (no wall) and P8.
6. Plain sets are exact counts from the preds files (card 2.7), not the trainer's metrics "acc" (the parent readout's
   N1 read the floats). Each plain preds file must hold exactly the j520 file's item ids (gsm8kx 1,317, gsm1kx 1,205,
   gsm8kv2 1,317, valx 393). On j520 / j663 the exact values equal the metrics values; the P6 references are 59.42 and
   70.96, the card's values.
7. P0 requires the token totals in tokens (unit "tokens"), so a character-count build does not pass the gate.
8. K1X shares 730 prompt hashes with train_kd27, all at k = 1 (card: reported; only k = 1 rows can share). In these
   rows CUR and K1X train on the same prompt; this is the parent K1 design (about 1/6 of the rows are k = 1).
9. Re-runs (fix round F2, card Amendment 2): a re-run is a resubmission of the same node under its own id, so every
   tool reads it unchanged. prep_khop3.py still writes no env on 996-999 and refuses one; those ids stay unused.
10. Wall estimates are the card's fit (0.400 h + 4.77 s per update). S1 has no mid evaluation and fewer eval sets than
    D27 (no hint sets), so the S1 wall is probably below the fit. P7 / P7c / P7b measure the real walls before any
    confirmation node is written.
11. The synthetic pilots and runs in the tests use the real A2 tarballs for GSM / hint / plain preds. The gate and
    readout logic is tested; the real pilot outcome is not predicted.
12. run_khop3.sh prints `round_gpuh.py 975 999` from E/results (round_gpuh.py reads its own ../results), whatever
    [results dir] is given. On another results dir the compute total covers E/results only.
13. khop3_table.py sets khop2_table.unreadable to its own function, so the imported exact helpers stop with the
    "khop3_table: UNREADABLE" prefix. In one Python process this also changes the prefix of khop2_table's messages
    (the combined unittest run); the standalone khop2 readout is not affected, and no test pins that prefix.
14. The S2 weights tar path: run_train_x1verify2.sh reads $STAGING/weights/<WTAR>.tar. S1 writes
    $STAGING/weights/<S1 name>.tar. Both use the job's $STAGING (sub environment line), so the names agree.

## 8. Not done in this build step (process of round G)

- The cold build critic, the fix round and the Codex CLI review loop (card section 5) run after this build. Build
  critic and fix round: done (Fix round section). The Codex CLI review loop: not run yet.
- Bundle, copy to the AP and every cluster command: sev-e0 (section 6).

## Fix round (2026-10-27 13:20-14:45 CDT, after BUILD_CRITIC_khop3.md, verdict SUBMIT AFTER FIXES)

Every finding is fixed; none is rejected. Card: Amendment 2 of CARD_khop_v3.md (no bar, arm, env, schedule, job id,
verdict order or reading changes). The repo is unchanged (D5 holds; build.diff regenerated, byte-identical, 811 lines,
md5 4b746406...). prep_khop3.py was rerun: the 21 envs, both subs, both DAGs and check_weights_khop3.sh are
byte-identical to the build (md5s of 6.1 item 4 unchanged). No cluster command was run.

| Finding | Fix | Files | Regression test (fails on the build, passes now) |
|---|---|---|---|
| F1 MEDIUM, NaN probs / duplicate rows fail open | khop3_table.check_preds, called by load: each preds_khop_* / preds_hint_* file present once, every row finite probs with len(probs) = len(labels) > 0, gold_label and item_id, distinct ids over the raw rows, khop row count = khop_table.ROWS | E/readout/khop3_table.py | test_khop3_table test_fail_closed (all-NaN khop_ca_fwd at CUR1 s16; one khop id twice at the same row count; a duplicate hint_true_r2 row; one NaN hint_true_r3 prob); test_khop3_pilot_gate test_p1_nan_and_duplicate_preds (P1 FAIL for j975 / j980 with the UNREADABLE reason in INFO). The critic's nanprobe dir: the build's load reads it and xacc(k = 3) = 1/5; now UNREADABLE |
| F2 MEDIUM, 996-999 re-runs unusable | text: a re-run resubmits the same node under its own id (rescue DAG or one-node DAG); 996-999 reserved, unused | card header, 1.3, 3, 5, Amendment 2; this report 6.2 item 5, risk 9; prep_khop3.py docstring; khop3_table compute line | none needed (no code path changes; test_prep_khop3 still checks the 996-999 refusal) |
| F3 MINOR, retry walls undercount | text: every failed execution from the job event log (001 to 004 / 005 / 009 / 012), the landed one excluded | card 3 P7c and compute; this report 6.2 item 3 | none (operator text; no event log exists on the Mac to test a parser against) |
| F4 MINOR, any "GATE PASS" file opens the confirmation | gate_ok / gate_text_ok: also exactly one "PASS  <name>: " line per GATE_LINES entry (20, P1 / P2 with the pilot ids) and no "FAIL" line, no second GATE line | E/stage/prep_khop3_confirm.py, E/readout/khop3_table.py | test_prep_khop3 test_refuses_foreign_or_partial_gate_output (the real khop2 output E/logs/khop2_pilot_gate_20261026.txt, a khop2-shaped text, a missing / doubled / FAIL line, P1 j9750, a bare GATE PASS); test_khop3_table test_gate_and_walls_required and test_run_khop3_sh (bare GATE PASS passes run_khop3.sh's tail check and is refused by khop3_table); test_khop3_pilot_gate test_good_pilot (the gate's own PASS names equal both lists, both accept the output) |
| F5 MINOR, confirm generator files | text: runs on the Mac, only the DAG goes to the AP | card 4, prep_khop3_confirm.py docstring, this report 6.2 item 5 | none (text) |
| F6 MINOR, restored eval files unchecked | text: pre-submit line, top-level data/eval/*.jsonl set and md5s equal to the x1khop2 bundle | card 5, this report 6.1 item 3 | the command was run on E/src/data/eval against the x1khop2 bundle: 21 files, 0 differences |
| F7 MINOR, report lines not fail-closed | a missing wall of a new job is UNREADABLE at load, before any decision line (no 0 GPU-h); history_finite on the 10 reference runs | E/readout/khop3_table.py | test_khop3_table test_fail_closed (a K1X s20 log without the exit line; a j520 copy with a NaN history loss) |
| F8 INFO, staging inventory | text: the 17 exported tars of the landed logs (83, 84, 85, 540, 541, 663-670, 756, 829, 833, 875; recomputed from E/results) | card 3 | none (text) |

Checks on real inputs: the 34 reference and context tarballs that run_khop3.sh reads (j520-j542, j663-j666, D27, H3,
KD, KH, K1) all pass check_preds (11.5 s); j520 holds 0 khop and 18 hint preds files, the parent runs 9 and 33. The
reference histories have 10 rows each, all finite.

Edge cases, stated: check_preds does not require a khop or hint file to be present (the A2 references have no khop
preds); a missing set still stops in xacc / xrc_cells. The hint files have no fixed row count here; xrc_cells checks
that each rc population is complete. Smoke tarballs are read by khop_table.load_run only (exit, wall, weights line),
so check_preds does not apply to them. GATE_LINES exists twice (readout and stage, no cross-directory import); the
gate test pins both to the gate's real output, so a renamed gate line fails a test instead of silently refusing a pass.

Tests: run_tests.sh -> test_all.log (results below).
run_tests.sh, 2026-10-27 14:13-14:41 CDT (final run after the last edit; a first fix-round run 13:40-14:08 gave the
same counts), exit 0 (python 3.12.7, torch 2.14.1, transformers 5.20.2; venv_k3):
repo tests.test_build_khop3 13 OK; parent repo suites 79 OK (1 skip, a parent test, as before); flag-off identity 23 OK;
readout (khop3 table + gate and the khop2 / khop parents) 98 OK; readout under Python 3.9.6 31 OK; stage 31 OK; stage
under Python 3.9.6 18 OK; bash -n OK on both scripts; D5 OK against E/src_khop2/repo and repo.base.

## Codex rounds (2026-10-27, `jobs/codex_review.sh khop3 CARD_khop_v3.md`, gpt-6-astra xhigh, read-only)

Card: Amendment 3 of CARD_khop_v3.md. The repo is unchanged (D5), so build.diff, the 21 envs, both subs and both DAGs
are unchanged; only E/readout khop3 files, their tests and the card changed. No cluster command was run.

### Round 1 (critique/CODEX_REVIEW_khop3.r1.md, VERDICT: SUBMIT AFTER FIXES; pre-fix copies in critique/pre_codex_r1/)

| # | Finding | Outcome | Files | Regression test (red on the pre-fix code, green now) |
|---|---|---|---|---|
| 1 | major, partial GSM / hint / plain preds pass the gate and the readout | fixed: every set of metrics.json args.eval must have one preds file with exactly the eval file's item ids (E/src/<path>); args.eval must list S1_NEED (S1), s2_need() (the 663 env sets + 9 khop, S2) or PLAIN (references, context); preds_subq skipped (another row format, read by no line; P1 still requires it) | khop3_table.py (eval_ids, check_preds, load, s2_need, --env663), khop3_pilot_gate.py (load with need) | test_khop3_pilot_gate test_p1_partial_or_unlisted_sets (emptied hint_true_r2 and valx short in j980: the pre-fix gate printed GATE PASS; bbh500, S1 val, gsm1kx short; a required set not listed); test_khop3_table test_fail_closed (emptied hint_placebo_r1, bbh500 / S1 val short, deleted file, unlisted set, no args.eval, a j520 copy without hint_placebo_r1: the pre-fix readout ran past them) |
| 2 | major, rc3 transfer claim under MANIPULATION FAILED | fixed: rc3 line = ("INFO", "MANIPULATION FAILED: <sub>", numbers, "no rc3 reading and no transfer sentence") | khop3_table.verdict | test_rc3_line_same_order (all five sub-labels; the old test pinned the flagged behaviour and was replaced) |
| 3 | minor, D_2 / C_2 without SD and t | fixed: D_2, C_2, D_3, C_3 print m [CI] (SD, t, n) | khop3_table.pts_full, verdict | test_primary_lines_print_sd_and_t |
| 4 | minor, rc_reading repeats verdict's need_seeds on the same dicts | fixed: repeat removed (verdict checks all eight contrasts before both calls; rc_reading has no other caller) | khop3_table.rc_reading | none possible (a removal of a repeated check, no behaviour change); TestVerdict.test_fail_closed still shows that verdict refuses a missing seed / NaN |

Real inputs after the fix: the 34 reference and context tarballs that run_khop3.sh reads load (j520 / j525 / j530 39
sets, j540 / j542 / j663-j666 54, the khop2 / khop parents 63; each set equal to its eval file). Fixtures: the synthetic
pilot and run tarballs now carry args.eval (j663's list + the 9 khop sets for S2; val, the four plain sets and the 9 khop
sets for S1) and an S1 preds_val.
Test effect: in test_each_line_fails_alone a missing args.eval file (khop_cag_fwd, valx) now makes the run unreadable,
so P1 fails together with the lines that need the run; "P1 alone" is now a missing preds_subq of j980.

### Round 2 (critique/CODEX_REVIEW_khop3.md = CODEX_REVIEW_khop3.r2.md, VERDICT: SUBMIT)

Run on the round-1 fixes and Amendment 3 (all code and card edits were made before the run). Result: "No findings."
Codex states that the final bundle contents, the AP staging quota and GPU execution are not verified by it; the card's
pre-submit lines (section 4 and 5) still apply. The loop stops here (2 of at most 3 rounds).

Tests after the fixes: run_tests.sh -> test_all.log, 2026-10-27 15:05-15:37 CDT, exit 0: repo tests.test_build_khop3 13
OK; parent repo suites 79 OK (1 skip, as before); flag-off identity 23 OK; readout 100 OK (98 before: + 2 new tests);
readout under Python 3.9.6 33 OK; stage 31 OK; stage under Python 3.9.6 18 OK; bash -n OK on both scripts; D5 OK
against E/src_khop2/repo and repo.base.

Open items: none from Codex. Rejected findings: none.
