# Pre-registration card: transfer of the installed k-hop capacity to GSM rc2 (family khop3, parent khop2), 2026-10-27, sev-e0

Status: DRAFT 2026-10-27 11:20 CDT, written before any build, bundle or submission. No khop3 data exists. Amendment 1
(2026-10-27 11:20 CDT, after the cold critic E/src_khop3/critique/CARD_CRITIC_khop3.md, before any build) is at the end;
the sections below are updated in place. The control arm is now K1X (single-hop queries on the same 27,000 chains), not
SHUF. Amendment 2 (2026-10-27, after the cold build critic E/src_khop3/critique/BUILD_CRITIC_khop3.md, before any
bundle or submission) is at the end: re-runs reuse the node's own id (996-999 stay unused), the retry-walls file counts
every failed execution, the gate output check is exact, preds files with NaN probs or duplicate rows stop the readout.
Amendment 3 (2026-10-27 15:00 CDT, after the Codex CLI review round 1) is at the end: every eval set of a run's metrics.json
args.eval must be complete (the eval file's item ids), and the rc3 line is INFO under MANIPULATION FAILED.
Owner's words (2026-10-27, RESULTS_LOG "Round G", the go to the three follow-up decisions of HANDOFF section 16):
"那你继续". Open item (1) of HANDOFF section 16: "does chain training that keeps GSM (mixed with GSM data, or H3-size
doses) transfer to rc2?"
Family khop3, job ids 975-999 only (975-994 jobs, 995 a smoke-only env, 996-999 reserved and unused: a re-run is a
resubmission of the same node under its own id, Amendment 2). Code copy
E/src_khop3/repo (untouched snapshot E/src_khop3/repo.base; both equal E/src_khop2/repo today), variant x1khop3, bundle
cotd_code_data_x1khop3_4b.tgz. E = <path> (the old scratchpad path .../scratchpad/exp is a
symlink to it; files may carry either form).
One-pass rule (RULINGS_2026-10-23, tier ii): every arm is plain A2 training (letter CE after "Answer:"). The
curriculum is two ordinary A2 training jobs in sequence (chain rows, then the A2 GSM epoch on the saved weights). No new
mechanism, no generated tokens (G0), no recurrence; executed layers 32 of 32, one forward per item, FLOPs and latency
equal to A2 for the same prompt length. No trainer change: the A2 path is the parent's code byte for byte (D5).
Numbers of this card: E/src_khop3/critique/card_numbers.py, output card_numbers.out (walls by
E/jobs/round_gpuh.py on the parent tarballs; power from exp/logs/khop2_readout_20261027.txt).

## 0. Question, hypothesis and pre-registered readings

Question. CARD_khop_v2 (PASS, adversarial HOLDS WITH CAVEATS): after 27,000 symbolic chain rows (train_kd27, 1 epoch
mixed into the gsmvt epoch), one pass of Qwen3.5-4B composes 3-4 small-integer steps, k*(ca, fwd) = 4, and uses step
order. But GSM is damaged: rc1 -10.1, rc2 -7.6 against best A2, and N1 (plain) fails. Does the installed multi-step
capacity raise GSM rc2 / rc3 when GSM itself is not damaged?

Background (RESULTS_LOG "khop2 five-seed readout"; card_numbers.out):

| arm (parent) | chain rows x copies | updates | k*(ca, fwd) | rc2 - best A2 (n 5) | plain (N1) |
|---|---|---|---|---|---|
| KH | 3,000 x 1 | 279 | 2 | -1.0 (SD 3.3) | holds |
| H3 | 3,000 x 3 | 467 | 2 | -3.2 [-4.8, -1.7] (SD 1.2) | holds (gsm8kv2 -1.2, valx -1.8) |
| KD | 9,000 x 1 | 467 | 2 | -5.5 [-10.5, -0.4] | - |
| D27 | 27,000 x 1 | 1,029 | 4 | -7.6 [-12.9, -2.2] | fails (gsm8kx -2.4, gsm1kx -3.0, valx -3.7) |
| R3 | 9,000 x 3 | 1,029 | 3 | -9.1 | fails |

In every interleaved arm the GSM cost grows with the chain exposures, and only the arms that fail N1 reach k* >= 3.
So no interleaved dose of the parent grid gives both "capacity installed" and "GSM intact". The question needs a design
that separates the two in time.

The parent already answers the "H3-size dose" branch of the owner's item (Amendment 1, finding 5): H3 kept N1 and still
lost 3.2 rc2 points [-4.8, -1.7], at k* 2. So N1 alone does not protect the rc cells, and every parent arm moved rc2
down. These results are prior evidence for NULL or NEGATIVE here (the priors below lean that way). The run is still
worth about 22 GPU-h for one reason: no parent arm had k* >= 3 and plain GSM intact at the same time. A NULL here, with
a CI upper of about +4 to +5 rc2 points at k* >= 3 and N1 holding, closes the transfer route for this format and dose;
the parent data cannot close it. The "mixed with GSM data" branch is the interleaved design, which the parent grid
already maps (cost grows with the dose).

Arm choice (the task listed four designs; the reasons for the choice):
- Curriculum (CHOSEN, arm CUR). Stage 1 trains on the 27,000 chain rows alone; stage 2 is the A2 GSM epoch, byte for
  byte the A2 re-train env, started from the stage-1 weights. The GSM stage then sees the same rows, LR schedule, option
  orders and batch order as an A2 run of this repo at the same seed (the tiny-model test of section 5 is the evidence;
  equality with the j663 batches is not claimed). Only the starting weights differ. The dose is D27's, the one dose with
  k* >= 3 at five seeds. Warm start from saved weights exists already (round F, CARD_verify_v2 branch F:
  run_train_x1verify2.sh WTAR block; SAVE=1 exports to $STAGING/weights/<name>.tar), so no trainer change is needed.
- Interleaving with a GSM-loss weight or ratio chosen on val (NOT RUN). A per-row weight exists (cotdistill/student.py
  line 308: every example's weight is multiplied by item["weight"]), so the build is possible. But the choice on val needs
  a sweep of at least two 1,029-update jobs (about 1.8 h each) before the pilot. The sweep and the pilot do not fit
  under the 6 GPU-h pilot cap, and the parent grid already shows that interleaving trades installation against GSM.
- Chains rewritten into GSM-like surface (cag) (NOT RUN HERE, next card). cag in build_khop.py is ca with GSM-sized
  values ([10, 999]) in the same symbolic wording; D27 reached only 0.583 at (cag, fwd, 3), k* 2. A cag curriculum arm
  is a second treatment. With its control it needs 5 more seed pairs (about 11.1 GPU-h); the confirmation would then be
  28.8 GPU-h > 25 (card_numbers.out). It is the pre-registered follow-up of a NULL (section 5).
- A lower chain dose with more GSM (NOT RUN). KH and H3 kept plain but stayed at k* 2: by the parent data a lower dose
  fails the installation line by design, so it cannot answer the question.

Control (arm K1X, Amendment 1 finding 1; it replaces the shuffled-target arm SHUF of the draft). K1X is the curriculum
with stage 1 on the K1 twins of the same 27,000 chain items: the same statements and premise order, the query on the
target chain's hop-1 variable, the hop-1 option set (vb keeps the five roots), as the parent's train_k1 (CARD_khop_v1
1.2 K1 rule, build_khop.Item.row(hop=1)). Same row count, updates (844), LR schedule, template and stage 2. K1 is a
learnable task (the parent K1 arm reached 1.000 on every k = 1 cell at every seed), so the control trains, and C_2 =
CUR - K1X reads "composition of k steps, not single-hop reading of the same chains, and not 844 extra updates". In about
1/6 of the rows (k = 1) the K1 query equals the CUR query; this is the parent's K1 design.
Why not SHUF: an independent uniform target letter has its CE optimum at the uniform letter distribution (loss ln 5 =
1.61), the collapsed state of the EARLY3 arm (j648-j652). That control does not train, which is how verify2 branch R
read MANIPULATION FAILED in round F; its I6 line was true by construction. GSM duplicates add a second GSM epoch (A2
peaks at c1-c2), which CARD_khop_v2 1.2 already refused as a confound. K1X costs the same GPU time as SHUF; its build
cost is a generator replay (section 1.2).

What is new. CARD_khop_v2 measured the capacity in a model that had paid a GSM cost, so "binding, not depth" (CARD_khop_v1
section 0, hypothesis H) stayed open. This card reads GSM rc2 in a model that has the capacity (installation line at
the final model) and has not paid the cost (N1), against a matched control. That is the first direct transfer test of
installed composition in this project.

Prior expectation (no new literature search this round; sources as cited in CARD_khop_v1 section 0):
- CARD_diag_v1 (intermediate values decodable at operand tokens, not used at the answer) and CARD_b1_v1 (removing used
  premises lifts rc2 by about 10 at 96k) point to binding / premise selection, not serial depth.
- Wang et al. (NeurIPS 2024, grokked transformers): learned composition does not generalize out of distribution. The
  chain format is far from GSM wording (cag 0.58 already shows a gap at larger values).
- Sequential fine-tuning forgets (stage 2 at the A2 LR is short, 186 updates, but stage 1 is 844 updates on one format).
Priors: PASS about 0.12; NULL about 0.40; BETWEEN about 0.18; NEGATIVE about 0.10; MANIPULATION FAILED about 0.20
(of which "installed, then lost in the GSM epoch" about 0.12 and "GSM damaged" about 0.08).

Hypothesis H3T (transfer). In a model with k*(ca, fwd) >= 3 and plain GSM intact, GSM rc2 on the 249 primary items is
at least 3 points above best A2, and above the matched K1X control (single-hop queries on the same chains).

Notation (as CARD_khop_v1 / v2). a_X(t, o, k) = accuracy of arm X on khop task t, premise order o, chain length k, one
value per seed. X1 = the stage-1 model of arm X (end of stage 1), X = the final model (end of stage 2). m = five-seed
mean; CI = m +/- 2.78 x SD / sqrt(5) (df 4, seeds 16-20); t = m / (SD / sqrt(5)). SOLVED: m >= 0.80 and CI lower >= 0.70.
k*(X, t, o) = the largest k <= 6 with cells 1..k all SOLVED (0 when k = 1 is not SOLVED).
rc_j(X, s) = the rc_j cell of the 249 primary items (pfx_table.py; rc4 on its 128 items) of arm X at seed s.
D_j(s) = rc_j(CUR, s) - rc_j(bestA2, s); C_j(s) = rc_j(CUR, s) - rc_j(K1X, s); both paired over s16-20, in points.
D_j^orig and D_j^re = the same difference against each A2-6k set separately (originals, re-trains), paired by seed;
D_j = the one of the two that bestA2 names. NEGATIVE uses both (Amendment 1, finding 6).
V (control valid) = I7 (K1X trained, section 2.5) holds AND N1 (section 2.6) holds for K1X.
bestA2 = the A2-6k set with the higher five-seed mean on that cell (section 2.4; on rc2 and rc3 the original set
j520 j525 j530 j540 j542, 65.66 / 55.08 against the re-trains 64.28 / 52.60).
Selection seed: the pilot gate chooses no setting and no arm (go / no-go only). s16 is one of the five seeds everywhere
(n = 5, t_crit 2.78). Nothing is tuned on the 249 primary items, and no gate line reads them.

Power statement (pre-registered, card_numbers.out). Paired SDs of rc2 against the original A2 set in the parent: KH 3.3,
H3 1.2, KD 4.1, D27 4.3, R3 3.1, and the A2 re-train set itself 3.8. PASS needs m >= 3 and t >= 2.78. At m = 3 the SD
must be <= 2.42; at SD 3 the mean must be >= 3.72; at SD 4.3 it must be >= 5.34. So the card detects a gain of about
+4 to +5 rc2 points; a true +3 is often read BETWEEN or NULL. A NULL here excludes a gain of the printed CI upper, no
more. rc3 has SDs 1.7-4.9 (same reading).

Verdict readings, evaluated in this order; the first that applies is the verdict (order fixed by Amendment 1, findings
1, 2 and 6):
1. MANIPULATION FAILED. The design premise does not hold, so no transfer reading. Sub-labels (the first that applies):
   - "not run as registered": any of I1-I4 (section 2.5) fails at five seeds. Report the failing line.
   - "not installed": k*(CUR1, ca, fwd) < 3 (the stage-1 model did not reach the parent's capacity).
   - "installed, then lost in the GSM epoch": k*(CUR1, ca, fwd) >= 3 and k*(CUR, ca, fwd) < 3. Reading: at this LR the
     A2 epoch overwrites the chain capacity; the two compete for the same weights. This is a result about the curriculum,
     not a build fault, and it is reported with both curves. No transfer sentence.
   - "control not a control": the content check I6 fails (CUR - K1X at (ca, fwd, 3) has CI lower <= 0: single-hop
     training reached the same k = 3 accuracy, so C_2 isolates no composition).
   - "GSM damaged": N1 (section 2.6) fails for CUR. The curriculum did not keep GSM intact; D_2 is printed as INFO only.
2. PASS: D_2 has m >= +3.0 and t >= 2.78, AND C_2 has CI lower > 0, AND V holds. Claim: "in Qwen3.5-4B with plain GSM
   intact, symbolic chain training before the A2 epoch (27,000 rows, then the A2 GSM epoch) raises one-pass GSM rc2 on
   the 249 primary items by <m> [CI] points over the best A2 set and by <m> [CI] over the same schedule with single-hop
   queries on the same chains." rc3 enters the claim ("and rc3 by <m> [CI]") only when D_3 and C_3 meet the same rule;
   else the claim says "rc3: <m> [CI], below the bar". Why C_2 needs only CI lower > 0 and not "+3 and t >= 2.78": two
   trained sets differ with a paired SD near 3.8 (the A2 re-train set against the original set, card_numbers.out), so a
   +3 bar on both contrasts would need a true effect near +5 twice. The size bar sits on D_2 (against the stronger A2
   set); C_2 only has to exclude "single-hop chain reading plus the extra updates". Reading for binding vs depth:
   composition of given steps is part of the rc2 limit, and training it on clean chains reaches GSM. H of CARD_khop_v1
   ("binding, not depth") is weakened for rc2. The claim names no mechanism and stays within the symbolic format, small
   values and this schedule.
3. NEGATIVE: D_2^orig CI upper < 0 AND D_2^re CI upper < 0 (below both A2 sets, each paired over s16-20). Claim: "with
   plain GSM intact and k* >= 3 installed, the chain curriculum lowers GSM rc2 by <m> [CI] against both A2 sets". The
   sentence "the symbolic composition training interferes with GSM multi-step items" is added only when C_2 CI upper < 0
   and V holds; else the claim adds "the K1X control is not above CUR (C_2 <m> [CI]): the cost is chain training plus
   extra updates, not composition".
4. BETWEEN "gain from extra updates, not composition": D_2 meets the PASS rule (m >= 3, t >= 2.78), V holds, C_2 CI
   lower <= 0.
5. BETWEEN "gain over best A2, control invalid": D_2 meets the PASS rule and V fails. Prints which part of V failed
   (I7, or N1 for K1X with the sets) and C_2 as INFO.
6. BETWEEN "gain below the bar": D_2 CI lower > 0 and m < +3.0. Claim: "the curriculum raises GSM rc2 by <m> [CI] over
   the best A2 set, below the pre-registered +3 bar; C_2 = <m> [CI]". Evaluated before NULL, so a gain whose CI
   excludes 0 is never printed as NULL.
7. NULL: D_2 CI lower <= 0 and D_2 CI upper < +3.0. Claim: "the installed capacity (k*(ca, fwd) = <k>, plain intact) does
   not raise GSM rc2: D_2 = <m> [CI]; the data exclude a gain of more than <CI upper> points." When D_2 CI upper < 0
   (below bestA2) but NEGATIVE did not fire, the claim adds "below the original A2 set only (the A2 drift of -1.4,
   re-train - original, card_numbers.out, is not excluded)". Reading for binding vs depth: the model can compose 3+
   given steps in one pass and still does not use that on GSM items, so the rc2 limit is not the serial composition of
   given operations. This supports "binding, not depth" (CARD_khop_v1 H) for the transfer route. It does not exclude a
   surface gap (symbolic wording, values 1-99): the cag follow-up of section 5 tests that.
8. BETWEEN "unresolved": everything else (D_2 CI lower <= 0 and CI upper >= +3.0; prints the CI).
The same rule (readings 2-8 with D_3, C_3, D_3^orig, D_3^re) is printed for rc3 as a secondary line ("rc3 reading");
it never changes the verdict.

## 1. Data and arms

### 1.1 Eval sets (unchanged)

The nine khop eval files in E/src/data/eval/khop/ and the three khop2 training files in E/src/data/eval/khop2/ are not
rebuilt and not edited. Their md5s must equal E/src/data/eval/khop/build_report.json (the Amendment 2 table of
E/src_khop/BUILD_REPORT.md) and E/src/data/eval/khop2/build_report.json (train_kd27.jsonl
92545dd42ca451c39e0e7c104aa78d1e, 27,000 rows). GSM and hint eval sets: the A2-6k env's EXTRA_EVALS, unchanged. No new
eval set. E/src/data/eval/khop3/ holds only build_report.json.

### 1.2 New training directories (E/src/data/student_khop3_*; jobs/build_bundles.sh line 21 copies them for x1khop3)

Generator: a new script scripts/build_khop3.py in E/src_khop3/repo. It reads files and replays the parent generator
streams; it draws no new chain item (Amendment 1, finding 1).

1. Read train_kd27.jsonl from --khop2 (default data/eval/khop2) and stop (SystemExit) when its md5 differs from the
   khop2 build_report.json or from 92545dd4...; read data/eval/khop/train_k1.jsonl (md5 4bb15f66c8d8843013eb0410c6a87e48,
   the khop2 report's parent_md5) and data/student_gsm8k_tf_vt/val.jsonl (md5 9f55fec541246e41401a7b5fa6c5510a, 393
   rows) the same way. --out-root or --report equal to an input dir is a SystemExit.
2. data/student_khop3_c27/train.jsonl = train_kd27.jsonl byte for byte (27,000 rows, same md5).
   data/student_khop3_c27/val.jsonl = gsmvt val.jsonl byte for byte (it is only evaluated: train_arm.sbatch line 281
   always runs val=$DATA/val.jsonl).
3. data/student_khop3_k1x/train.jsonl (27,000 rows, line i = the K1 twin of line i of train_kd27):
   a. Replay. Wrap build_khop.Item.row to record the Item object of every rendered item_id (the last call per item_id
      wins, so a redrawn row maps to the accepted item). Run build_khop.build_train(random.Random(1), stats, 1000, 2000)
      (the train_kd stream of CARD_khop_v1, seed 1) and build_khop2.new_rows(3, 6000, forbid_keys, forbid_hash, cycle0)
      with the inputs build_khop2.build computes from the parent files (seed 3, 6,000 per task). Stop (SystemExit) unless
      the replayed rows, in kd27 order, are dict-equal to the 27,000 train_kd27 rows, and the replayed K1 rows of the
      first stream are dict-equal to the 3,000 train_k1 rows. A failing replay stops the build; no other generator is
      chosen after the fact (a new amendment is needed).
   b. Rows 0..8,999 of kd27 (the train_kd part): rows 0..2,999 (KH) get their train_k1 twin, the line of train_k1 byte
      for byte. Rows 3,000..26,999 (KD extra and the khop2 new rows): it.row(item_id + "-k1", order, layout(i1, k, rng1),
      hop=1, group=<the kd27 row's group>), with rng1 = random.Random(f"khop3-k1-{item_id}"), i1 = the running count of
      earlier K1X rows with the same (task, k) (it continues from the train_k1 counts), order = the kd27 row's order.
      This is the parent K1 rule: the same statements and order, query on the target chain's hop-1 variable, ca / wp
      options = the hop-1 values of the five chains (build_khop.draw_arith_item already requires option_set_ok on the
      hop-1 set of every arithmetic item, so every item has a legal twin), vb options = the five roots, gold letter
      cycle by layout.
   data/student_khop3_k1x/val.jsonl = gsmvt val.jsonl byte for byte.
4. data/eval/khop3/build_report.json: card CARD_khop_v3, generator scripts/build_khop3.py; md5 and row counts of the 4
   files and of the 3 inputs; replay results (rows equal 27,000 / 27,000 and 3,000 / 3,000); for K1X: gold letter shares
   (each in [0.18, 0.22]); per row, the statement text (prompt up to the query sentence) equal to the kd27 twin's (count
   of mismatches must be 0); the query variable equal to the target chain's hop-1 name by build_khop.parse_prompt and
   the gold option equal to its value (mismatches 0); ca / wp option sets passing build_khop.option_set_ok (failures 0);
   prompt hashes shared with the nine eval files (must be 0) and with train_kd27 (reported; only k = 1 rows can share);
   item_ids distinct; token totals of K1X and kd27 with the tokenizer recorded in khop2/build_report.json (characters
   when none is recorded), K1X within 1% of kd27 (the wall is per update, the tokens set the per-update cost).

### 1.3 Arms

Every arm is two jobs per seed: S1 (stage 1, chain rows only, SAVE=1) then S2 (stage 2, the A2 GSM epoch from the S1
weights). S2 of seed s starts only after S1 of seed s landed (DAG PARENT / CHILD, PRE check_weights_khop3.sh).

| Arm | Stage 1 (S1) | Stage 2 (S2) | Jobs S1 / S2 |
|---|---|---|---|
| CUR (curriculum) | data/student_khop3_c27, 27,000 rows, 844 updates | A2 epoch, 5,923 gsmvt rows, 186 updates, MODEL = S1 weights | S1 j975 s16 (pilot), j976-979 s17-20; S2 j980 s16 (pilot), j981-984 s17-20 |
| K1X (control) | data/student_khop3_k1x, 27,000 K1 twins of the same chains, 844 updates | as CUR | S1 j985 s16 (pilot), j986-989 s17-20; S2 j990 s16 (pilot), j991-994 s17-20 |
| smoke only | - | the j980 env with MODEL / WTAR = the 663 trunk | j995 (smoke_995 only; never a training job) |
| reused, no new job | A2-6k originals j520 j525 j530 j540 j542, re-trains j663 j664 j665 j540 j666; parent D27 j875-879, H3 j885-889, KD j766-770, KH j756 j758-761 (context lines only) | | tarballs in E/results |

Updates = ceil(rows / 32): S1 ceil(27,000 / 32) = 844; S2 ceil(5,923 / 32) = 186 (the j663 log: "stage all: 5923
examples per pass, 186 steps"). Total
1,030 per seed, against 1,029 for D27. Spare ids 996-999: reserved, never written (Amendment 2, finding F2). A re-run is
a resubmission of the same node (rescue DAG, or a one-node DAG with the same JOB id, VARS, env, name and tar name), never
a new id and never a new setting.

Why stage 2 is the A2 re-train env and not a trainer stage. The trainer's stage list (train_student.py lines 1489-1500)
could run both stages in one job, but it would need a new flag and an rng re-seed to make the GSM stage equal to A2.
Two jobs need no trainer change: S2 is the A2 env with MODEL changed, so its GSM rows, option orders, batch order, LR
schedule (warmup 0.05 + cosine over 186 updates) and fresh AdamW state are A2's by construction. Cost: one more model
load and one weights tar per seed (measured below).

### 1.4 Exact envs and flags

No new trainer flag. New generator flags (scripts/build_khop3.py only): --khop2 DIR (default data/eval/khop2), --gsm-dir
DIR (default data/student_gsm8k_tf_vt), --out-root DIR (default data), --report DIR (default data/eval/khop3).

Env generator stage/prep_khop3.py (python, values bash-escaped as stage/prep_khop2.py does).
- S1 envs (975-979, 985-989) = a copy of condor/jobs_x1/766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.env with exactly these
  keys changed (every other key asserted equal to 766: EPOCHS=1, LR=3e-6, LSMOOTH=0.1, MBS=2, GA=16, NGPU=1,
  FSDP_OPTIM=cpu-adamw, OPTIM=adamw, PREC=bf16, MODEL=models/Qwen3.5-4B):
  - SEED = ORDERSEED = s; SAVE=1.
  - DATA=data/student_khop3_c27 (CUR) or data/student_khop3_k1x (K1X).
  - XARGS="--eval-value-render vt" (no --extra-train).
  - EVALAT='' and EVALMID='' (no mid evaluation in stage 1).
  - EXTRA_EVALS = the nine khop sets and gsm8kx, gsm1kx, gsm8kv2, valx of the 766 env, in the 766 order (no hint sets:
    stage 1 is read for the khop curves and the plain sets only).
  - TAG=khopcur1-c1lr3ch (CUR) or khopk1x1-c1lr3ch (K1X). Names 975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp ...
    979_..._s20_fsdp, 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp ... 989_..._s20_fsdp. JOBKEY "4b kd27 A2-<TAG> s<s>
    fsdp x1khop3" (kd27k1 for K1X).
- S2 envs (980-984, 990-994) = a copy of the same 766 env with exactly these keys changed (every other key asserted equal):
  - SEED = ORDERSEED = s; SAVE=''; DATA=data/student_gsm8k_tf_vt; XARGS="--eval-value-render vt"; EVALAT=0.5 and
    EVALMID as 766 (the A2 values); EXTRA_EVALS as 766 (GSM, hint and the nine khop sets).
  - MODEL=models/<S1 name>/model and WTAR=<S1 name>, S1 = the same arm's S1 job of seed s (980 <- 975, ..., 994 <- 989).
  - TAG=khopcur2w<S1 id>-c1lr3ch or khopk1x2w<S1 id>-c1lr3ch. Names 980_4b_gsmvt_A2-khopcur2w975-c1lr3ch_s16_fsdp,
    990_4b_gsmvt_A2-khopk1x2w985-c1lr3ch_s16_fsdp, and so on. JOBKEY "4b gsmvt A2-<TAG> s<s> fsdp x1khop3".
  - Assertion (tested): an S2 env equals the A2 re-train env condor/jobs_x1/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.env (seed
    keys set to s) except SAVE, TAG, JOBKEY, MODEL, WTAR and the nine khop sets appended to EXTRA_EVALS.
- Smoke env 995 = the 980 env with MODEL=models/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp/model and WTAR=663_4b_gsmvt_A2-c1lr3ch_s16_fsdp
  (the 663 tar is in $STAGING/weights since round F), TAG khopcur2w663-c1lr3ch. It tests the warm-start path, the khop
  evals and the S2 sub before any S1 weights exist.
Subs:
- stage/train_x1khop3_4b.sub (S1) = stage/train_x1khop2_4b.sub with exactly three lines changed: CODE_TGZ, 
  transfer_input_files = cotd_code_data_x1khop3_4b.tgz, log = logs/train_x1khop3_4b.log, plus the header comment;
  executable run_train5.sh. Its SAVE block (E/src/repo/cloud/chtc/run_train5.sh lines 63-67) exports the weights to
  $STAGING/weights/<name>.tar when SAVE is set, the run is not a smoke, the exit code is 0 and $OUTD/model exists; OUTD
  (line 59) is built as run_train_x1verify2.sh line 100 builds it.
- stage/train_x1khop3w_4b.sub (S2 and smoke_995) = stage/train_x1verify2w_4b.sub with the same three lines changed;
  executable run_train_x1verify2.sh, unchanged (md5 3866fc7788af662fe94052a9d2bae4c8, asserted by the test and checked
  on the AP before submission).
Resources as the parent: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300, RETRY 1, POST post_push.sh.
PRE script stage/check_weights_khop3.sh <S1 name>: exit 1 unless results/results_<S1 name>.tgz exists, holds a
metrics.json and logs/<S1 name>.out with "== exit 0" and the line "== weights -> <path>/weights/<S1 name>.tar (<size>)"
(written by run_train5.sh only after the tar command succeeded). The PRE does not stat the tar itself: $STAGING is set in
the job's environment, not in DAGMan's, so a PRE cannot rely on it. The fail-closed line for the tar is the runner's own
refusal (run_train_x1verify2.sh lines 76-79: "weights tar missing" / extract failed / no config.json, exit 1 before any
model load, a few GPU-minutes). It reads no env file.

## 2. Metrics and rules

### 2.1 Primary
- D_2 = rc2(CUR) - rc2(bestA2) and C_2 = rc2(CUR) - rc2(K1X), paired over s16-20, with m, SD, t and CI, and the verdict of
  section 0 in its order.
- Secondary decision line (no verdict change): the same rule on rc3 (D_3, C_3).

### 2.2 Installation and the "lost" reading
- k*(CUR1, ca, fwd), k*(CUR, ca, fwd), and the full curves a(ca, fwd, k), k = 1..8, every seed, for CUR1, CUR, K1X1,
  K1X, with the parent D27 and K1 next to them.
- Forgetting: a(CUR, ca, fwd, k) - a(CUR1, ca, fwd, k), k = 2..4, paired, with CI.

### 2.3 Secondary (reported; no verdict depends on them)
- D_1, D_4, C_1, C_4 (rc4 on its 128 items) and K1X - bestA2 at rc1..rc4 (do 844 updates of single-hop chain reading
  help or hurt?), and D_j^orig, D_j^re for j = 1..4.
- G2 / G3 in the undamaged model: a(CUR, ca, fwd, 2) - rc2(CUR) and a(CUR, ca, fwd, 3) - rc3(CUR), per seed, with CI.
  This is the CARD_khop_v1 gap read in a model that has not paid the GSM cost (the CARD_khop_v2 caveat). Printed only
  when N1 holds.
- Plain sets of CUR1 and K1X1 (the GSM state after stage 1 alone) against bestA2.
- k* and a(t, o, 2..4) for wp, vb, cag and rev orders for CUR and CUR1; the 48 all-+/- items and the 31 order-sensitive
  items of (ca, fwd, 3) (CARD_khop_v2 adversarial check, run_all.sh lists) for CUR.
- CUR - D27 and CUR - H3 at rc1..rc4 and on the plain sets (cross-bundle, same seeds).
- Timing (wall per job, s_per_step), peak memory, weights tar size.

### 2.4 Reference A2 (best-A2 rule, fixed now; as CARD_khop_v2 2.4)
For every GSM-side line the reference is the A2-6k set (originals j520, j525, j530, j540, j542 or re-trains j663, j664,
j665, j540, j666) with the higher five-seed mean on that line's metric, computed from exact counts. That set's per-seed
values enter the paired difference. The reference is the stronger set on each cell, so D_j is conservative for PASS and
anti-conservative for NEGATIVE; this is the rule fixed before the data, not a choice made for these numbers. At the pilot (n = 1) the s16 reference is mean(j520, j663) per metric (Amendment 1, finding
3: the mean, so a max of two noisy runs does not inflate it), both files required (finite): gsm8kx 59.42, gsm8kv2 70.96.
NEGATIVE also needs the difference against each set separately (section 0, reading 3).

### 2.5 Installation and manipulation checks
Data side (before any bundle; data/eval/khop3/build_report.json and the builder's tests):
- D1 data/student_khop3_c27/train.jsonl md5 = 92545dd42ca451c39e0e7c104aa78d1e; both val.jsonl md5 = 9f55fec5...; the
  khop and khop2 parent files unchanged (md5 before and after the build).
- D2 K1X: 27,000 rows; both replays equal (27,000 / 27,000, 3,000 / 3,000); rows 0..2,999 equal train_k1 byte for
  byte; statement mismatches 0, hop-1 query and gold mismatches 0, option-rule failures 0, eval hash overlap 0, item_ids
  distinct; every teacher a one-hot on gold_label; gold letter shares in [0.18, 0.22]; token total within 1% of kd27.
- D3 the bundle holds data/student_khop3_c27/{train,val}.jsonl and data/student_khop3_k1x/{train,val}.jsonl with these
  md5s (exact file set: no other file in either dir).
- D5 Same trainer: `diff -rq -x __pycache__ -x data E/src_khop2/repo E/src_khop3/repo` lists only scripts/build_khop3.py
  and tests/test_build_khop3.py.
Model side, at five seeds (khop3_table.py; exact counts):
- I1 Format: m(a_CUR(t, o, 1)) >= 0.90 on the six (task, order) cells of ca / wp / vb.
- I2 Query read: query-swap pair accuracy at k = 1, m >= 0.85 for CUR.
- I3 No option-only shortcut: m(blind accuracy) <= 0.30 for CUR and CUR1.
- I4 Run as registered: every S1 log has the trainer's 27,000-example count from its DATA file, final step 844, a
  finite final loss and the weights export line; every S2 log has "== weights $STAGING/weights/<its S1 name>.tar ->
  models/<S1 name>/model", 5,923 training rows, final step 186; metrics.json seed / order_seed = the label's seed for
  all 20 jobs.
- I5 Installation (the manipulation line): k*(CUR, ca, fwd) >= 3 at the final model (with k*(CUR1, ca, fwd) deciding
  between "not installed" and "installed, then lost", section 0).
- I6 Content check: a(CUR, ca, fwd, 3) - a(K1X, ca, fwd, 3), paired, CI lower > 0. It can fail: the parent K1 arm (3,000
  rows) sat at 0.19-0.27 at k = 3, but K1X has 9 times the rows.
- I7 Control trained (part of V): for every K1X1 S1 log, the mean of the "loss" field of the trainer's JSON step lines
  ({"step": ..., "loss": ...}, one line every 20 updates, each the loss of that single update) over the lines with step
  >= 0.9 x 844 (steps 760, 780, ..., 840 and the final 844 line: six lines in the parent format) is <= 1.0. The collapse value is ln 5 = 1.61, the label-smoothing floor about
  0.39; D27, a mixed chain + GSM run, logged 0.49-0.69 over its last 5 lines (j875). No line, a non-finite value or fewer
  than five lines in that range is UNREADABLE. and m(a_K1X1(t, o, 1)) >= 0.98 on each of the six (task, order)
  cells of ca / wp / vb. An untrained model does not meet it: A2Z sits at 0.917-0.965 on five of the six cells (0.983 on
  vb fwd); the parent K1 arm is 1.000 on all six at every seed.
The verdict order of section 0 maps these lines to the sub-labels.

### 2.6 Necessary condition N1 (GSM intact)
- N1: for CUR, the five-seed mean on each of gsm8kx, gsm1kx, gsm8kv2, valx >= the bestA2 mean on that set - 2 points
  (bestA2 per set, exact counts). A failing N1 gives MANIPULATION FAILED ("GSM damaged") and no transfer sentence; D_2 is
  printed as INFO.
- N1 for K1X (the same rule) is part of V. PASS needs V, so a damaged control cannot inflate C_2 into a PASS: when N1
  fails for K1X and D_2 meets the PASS rule, the verdict is BETWEEN "gain over best A2, control invalid" (section 0,
  reading 5). NULL and NEGATIVE read D_2 and do not need V; their text prints V.

### 2.7 Fail-closed rules (round F lessons)
- Every decision value is computed from exact counts (Fraction(correct, items)); CIs and t are compared in squared form
  with t = 2.78; m >= 3 points is compared as m >= 3/100 exactly. A float print never decides a line.
- A missing, partial (row count off), NaN or unreadable preds file, log or metrics.json of any of the 20 jobs or of the
  10 reference jobs stops the readout (UNREADABLE). Every khop and hint preds file present is checked row by row
  (finite probs, len(probs) = len(labels) > 0, distinct item_ids, khop row counts of 1.1). Every set that a run's
  metrics.json args.eval lists must have exactly one preds file with exactly the item ids of its eval file (E/src/<path>);
  args.eval must list the S1 sets (val, the four plain sets, the nine khop sets) for CUR1 / K1X1, the 663 env's sets and
  the nine khop sets for CUR / K1X, the four plain sets for a reference or context run (Amendment 3, finding 1); a
  missing wall of a new job
  and a non-finite history loss of a reference job stop as well (Amendment 2, findings F1 / F7). It never drops a seed, an item or a set, and never shrinks a
  denominator. Population sizes must be 249 / 249 / 249 / 128 for rc1..rc4.
- Pairing is by the seed recorded in metrics.json, never by file order or name sort (names of 4-digit ids sort before
  3-digit ones).

## 3. Pilot (seed 16) and compute

Smoke DAG stage/smoke_x1khop3_4b.dag: smoke_975 (CUR S1 env, smoke="1", SMOKE_ONLY), smoke_985 (K1X S1 env) and
smoke_995 (the S2 path on the 663 trunk). Smoke gate: exit 0 for all three, and the smoke_995 log has the
"== weights ... 663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tar -> models/..." line. The smokes do not test the S1 weights export:
run_train5.sh skips its SAVE block under SMOKE_ONLY (Amendment 1, finding 9). The first test of the export is j975 /
j985 itself, read by P2 and by check_weights_khop3.sh (both fail closed); the block is the one that wrote the 663-670
tars. No new job.

Pilot DAG stage/pilot_x1khop3_4b.dag: j975 -> j980 (CUR s16), j985 -> j990 (K1X s16); PARENT / CHILD per pair; the S2
nodes have SCRIPT PRE check_weights_khop3.sh <S1 name>.

Deterministic pilot gate: E/readout/khop3_pilot_gate.py --cur1 <975 tgz> --cur <980 tgz> --k1x1 <985 tgz> --k1x <990 tgz>
--smokes <smoke_975, smoke_985, smoke_995 tgz> --ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz
--report E/src/data/eval/khop3/build_report.json --khop2-report E/src/data/eval/khop2/build_report.json
--parent-report E/src/data/eval/khop/build_report.json --retry-walls <file> (P7c; required, empty when no node
retried). Exit 0 = GATE PASS. Lines (each fails closed on a missing or
non-finite input):
- P0 Data: D1 and D2 of 2.5 re-checked from the report and the files' md5s.
- P1 Each pilot tarball has metrics.json; S1 tarballs hold the nine khop preds files (CARD_khop_v1 1.2 row counts) and
  the four plain sets; S2 tarballs hold the nine khop preds files and every GSM and hint preds file of the 663 env.
  Every set of the job's metrics.json args.eval holds exactly its eval file's item ids (2.7; Amendment 3, finding 1).
- P2 (I4 at n = 1) the S1 and S2 log lines of I4 for j975, j980, j985, j990; seeds 16.
- P3 (I1 at n = 1) a_980(t, o, 1) >= 0.85 on all six (task, order) cells.
- P4 (I2 at n = 1) query-swap pair accuracy at k = 1 >= 0.80 for j980.
- P5 (I3 at n = 1) blind accuracy <= 0.32 for j980 and j975.
- P6 Plain at n = 1 (bars from card_numbers.out, Amendment 1 finding 3). Reference = mean(j520, j663) at s16: gsm8kx
  59.42, gsm8kv2 70.96. CUR (j980): gsm8kx >= reference - 3.0 and gsm8kv2 >= reference - 3.0 (56.42 / 67.96, compared
  from exact counts). K1X (j990): the same sets >= reference - 5.0. valx and gsm1kx: INFO (valx at s16 is noisy: KD
  j766 58.0 against 64.4, CARD_khop_v2 critic finding 1). How the bars were chosen (card_numbers.py, normal
  approximation, one s16 draw): the per-seed SD of (arm - mean(original, re-train)) pooled over the six parent arms KH,
  K1, KD, D27, R3, H3 is 1.27 on gsm8kx and 0.82 on gsm8kv2. Rule: among B in {2.0, 2.5, ..., 5.0}, the B with the
  smallest false-stop + false-go among those with false-stop <= 0.15, where false-stop = P(stop) for a CUR whose
  five-seed mean sits 1 SD above the N1 edge (best - 2 + SD) and false-go = P(go) for a CUR 4 points below best on both
  sets. B = 3.0: false-stop 0.034, false-go 0.036 (B = 2.5: 0.099 / 0.007; the draft's B = 5: 0.000 / 0.757). The K1X
  bar B = 5 is lenient on purpose (false-stop 0.000): a damaged control removes only the C_2 clause (V fails), not the
  NULL / NEGATIVE readings, so it must not stop the family; its five-seed N1 is read in V.
- P7 Wall: j975 and j985 each <= 2.0 h, j980 and j990 each <= 0.9 h (pilot <= 2 x 2.9 + 3 smokes x 0.05 = 5.95 GPU-h).
- P7c Measured pilot total: the sum of the measured walls of every execution of the seven pilot nodes (3 smokes, j975,
  j980, j985, j990), RETRY executions included, <= 6.0 GPU-h. Gated on the sum, not derived from the per-job bars (the
  warm-start smokes of verify2 took 0.08-0.09 h). round_gpuh.py reads only landed tarballs, so the wall of a failed
  attempt before a RETRY comes from the DAG's nodes.log / job event log on the AP: sev-e0 reads it and passes it to the
  gate (--retry-walls <file>, one "<node> <hours>" line per failed execution; the gate refuses to run without the file,
  an empty file meaning no retry). The file lists every failed execution, not only the DAG RETRY lines (Amendment 2,
  finding F3): for each node, every execute-to-evict and execute-to-terminate interval of the job event log (the sub's
  log = logs/train_x1khop3_4b.log / logs/train_x1khop3w_4b.log: event 001 to the next 004, 005, 009 or 012 of the same
  job id), except the one interval of the landed execution; a backfill eviction and a GPU-preflight exit 86 restart the
  job inside one DAG attempt and appear only there. The smoke and pilot DAGs share the event logs and node names; the
  job id that each DAG's nodes.log records for its node tells them apart. A rescue-DAG resubmission adds its own
  failed intervals. The worst case at the fit (6.15 GPU-h) exceeds the cap; P7c then fails and the gate stops
  (Amendment 1, finding 7).
- P7b Projected confirmation = 4 x (wall 975 + 980 + 985 + 990) <= 25 GPU-h. Under P7 it is <= 23.2, so P7b alone never
  fails; it is printed. No arm is dropped: the confirmation runs CUR and K1X together or not at all (a CUR without its
  control cannot reach PASS).
- P8 smoke_975, smoke_985, smoke_995, j975, j980, j985, j990 landed with exit 0.
- P9 (installation at n = 1, Amendment 1 finding 3) a_975(ca, fwd, 3) >= 0.78 and a_980(ca, fwd, 2) >= 0.85 and
  a_980(ca, fwd, 3) >= 0.78 (exact counts on 150 items: >= 117 / 150 and >= 128 / 150). The k = 3 bar: the per-seed SD
  at (ca, fwd, 3) pooled over D27, R3, KD, H3 is 0.061 (D27 alone 0.015, which gives the same bar); with the P6 rule,
  false-stop for a model 1 SD above the I5 edge (mean 0.861) is 0.092 and false-go for a dead model at mean 0.70 is
  0.095 (draft bar 0.70: 0.004 / 0.500). The k = 2 bar 0.85 is set from the parent level (D27 and R3 >= 0.987 at every
  seed, per-seed SD 0.015), not from an edge calculation. When 975 passes and 980 fails, the gate prints "GATE FAIL:
  P9 (installed, then lost at s16)"; this is a pilot observation, not the five-seed reading.
- P10 (I6 at n = 1) a_980(ca, fwd, 3) - a_990(ca, fwd, 3) >= 0.15.
- P11 (I7 at n = 1) the j985 S1 loss mean of I7 <= 1.0, and a_985(t, o, 1) >= 0.97 on each of the six (task, order)
  cells (A2Z fails this at every seed: at least three of its six k = 1 cells are below 0.97 at each of s16-20).
No gate line reads rc1..rc4 or any hint set of the 249 primary items; the gate prints none of them. Reported, not decision
lines: every curve, timing, peak memory, tar sizes. Output: "GATE PASS" or "GATE FAIL: <lines>". GATE FAIL: nothing else
is submitted; report to the second author with the failing lines.

Compute (measured from the parent's jobs, card_numbers.out):
- Measured walls (h, mean / min / max): KH 279 updates 0.766 / 0.70 / 0.93; K1 279 0.716; KD 467 1.106 / 0.99 / 1.27; H3
  467 1.010; D27 1,029 1.804 / 1.63 / 2.16; R3 1,029 1.706 / 1.59 / 1.86; A2 re-trains 186 updates (no khop evals)
  0.492 / 0.45 / 0.53; warm-start S2-shape jobs of verify2 (851, 852) 0.32-0.38 h and their smokes 0.08-0.09 h.
- Fit on the 30 khop jobs (with khop evals): wall = 0.400 h + 4.77 s per update; largest residual +0.40 h.
- S1 (844 updates): 1.52 h (fit + largest residual 1.91 h). S2 (186 updates): max(fit 0.65, A2 0.49 + 0.15 khop evals +
  0.05 tar extract) = 0.69 h. One seed pair 2.21 GPU-h (worst 3.00).
- Pilot: 3 smokes (0.05 each) + j975 + j980 + j985 + j990 = 4.57 GPU-h (worst 6.15; 5.95 at the P7 bars). Cap 6.
- Confirmation: 8 pairs (CUR and K1X at s17-20) = 17.69 GPU-h (worst 24.02). Cap 25. K1X has the same rows, updates
  and (within 1%, D2) tokens as CUR, so the CUR walls apply.
- Family total about 22.3 GPU-h. Round G cap 60 (shared with verify3, ids 1000-1024).
- Caps count everything (Amendment 1, finding 7; Amendment 2, findings F2 / F3): every failed execution (DAG RETRY,
  condor restarts inside one attempt, rescue-DAG resubmissions) counts against 6 (pilot) and 25 (confirmation). sev-e0 submits no re-run when the measured walls of the phase plus the projected wall of the re-run
  (the P7 bar of its job type) would exceed the cap; it stops and reports to the second author instead. The readout prints the
  measured family total from `E/jobs/round_gpuh.py 975 999` plus the retry walls from the job event logs.
- Weights (Amendment 1, finding 4): each S1 exports one bf16 tar of 9,702,973,440 bytes (9.04 GiB, the "9.1G" of du;
  j663-j669) to $STAGING/weights; 10 tars = 90.4 GiB. The recorded /staging quota is 100 GB and 1,000 files (session
  notes 2026-10-14), and at least 11 weights tars are recorded present (651, 656, 663-669 and more: about 99.4 GiB)
  plus the 4B model tar (9.3 GB). The landed logs in E/results name 17 exported tars (83, 84, 85, 540, 541, 663-670,
  756, 829, 833, 875; Amendment 2, finding F8); some may be deleted already, and the AP read-back of section 4 decides,
  not this list. So staging is probably full now, and the pilot likely needs the second author's decision before
  anything is submitted. Hard pre-submit lines are in section 4. Removing a tar is an operator step that needs the second author's go
  (it deletes files). Candidates, named and not chosen: 651, 656, 664-669 (663 must stay: smoke_995 reads it). The
  one-job alternative (S1 and S2 in one job with job-local weights) is rejected: it changes the runner, so S2 would no
  longer be the A2 re-train path byte for byte (D5), and its walls are unmeasured.

## 4. Conditional confirmation (written now; sev-e0 submits it without asking when the gate prints GATE PASS)

stage/confirm_x1khop3_4b.dag, written by stage/prep_khop3_confirm.py --gate-output gate.txt. It refuses unless the last
line of gate.txt is exactly "GATE PASS", the text holds exactly one "PASS  <name>: " line for each of the 20 khop3
gate lines (P0, P1 j975 / j980 / j985 / j990, P2 j975 / j980 / j985 / j990, P3-P11, P7b, P7c) and no line that starts
with "FAIL" (Amendment 2, finding F4), and every env still equals the text that prep_khop3.py generates.
prep_khop3_confirm.py runs on the Mac (it imports prep_khop3.py and prep_khop.py and reads the 766 template env); only
confirm_x1khop3_4b.dag is copied to the AP (Amendment 2, finding F5). Nodes:
- S1: j976-979 (CUR s17-20), j986-989 (K1X s17-20). SCRIPT PRE check_pilot.sh 975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp
  980_4b_gsmvt_A2-khopcur2w975-c1lr3ch_s16_fsdp 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp
  990_4b_gsmvt_A2-khopk1x2w985-c1lr3ch_s16_fsdp.
- S2: j981-984 and j991-994, each a CHILD of its S1 (976 -> 981, ..., 989 -> 994), SCRIPT PRE check_weights_khop3.sh
  <its S1 name>.
- Every node: POST post_push.sh, RETRY 1, prio 300. 16 nodes. the second author is told at submission and can stop it.
Readout: E/readout/run_khop3.sh <gate output> <retry-walls file of the pilot and confirmation DAGs, written by sev-e0
from their job event logs, section 3 P7c> [results dir] -> E/readout/khop3_table.py on the 20 new tarballs plus both
A2-6k sets and the parent D27 / H3 / KD / KH tarballs (context), then section 0 in that order, the rc3 line, N1, the
installation and forgetting lines, and section 2.3. khop3_table.py reuses khop_table.py and khop2_table.py primitives
(load_run, curve, qs_pair, blind, solved, kstar, paired, rc_cells, plain_acc, the exact-count CI helpers) by import and
edits neither. khop_table.load_run finds metrics.json and the preds by path suffix inside the tarball, so the S2 run
directories (runs/A2-<TAG>-model-student_gsm8k_tf_vt-s<s>, MODEL basename "model", as the verify2 F jobs) and the S1 ones
(runs/A2-<TAG>-Qwen3.5-4B-student_khop3_<c27|k1x>-s<s>) both load; a test covers both shapes. A missing tarball stops
the readout. Log exp/logs/khop3_readout_<date>.txt (+ .json).
Operator steps on the AP before the smoke (sev-e0; no cluster command is run by the card writer or the builder):
the bundle and its md5 on both sides; run_train_x1verify2.sh present with md5 3866fc77...; check_pilot.sh,
check_weights_khop3.sh, post_push.sh (only copy in this tree: E/src/repo/cloud/chtc/post_push.sh; it parses no id) and
both subs in the DAG directory; $STAGING/weights/663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tar present (smoke_995); the reference tarballs j520 and j663 in
E/results. Hard staging lines (Amendment 1, finding 4), read back on the AP by sev-e0 from the quota tool and du, no
assumption: before the pilot, free /staging space >= 2 x 9.04 GiB + 5 GiB = 23.1 GiB and the file count + 2 < 1,000;
before the confirmation, free >= 8 x 9.04 GiB + 5 GiB = 77.3 GiB with the 2 pilot tars still present, and the file
count + 8 < 1,000. Space that verify3 (ids 1000-1024) plans to use in the same period counts as used. When a line
fails: stop and ask the second author (deletion of named old tars, or a quota raise); no submission. Before any submission the Round
G line of RESULTS_LOG.md for khop3 must be amended (finding 5; this card writer may not edit RESULTS_LOG.md, so it is
an operator step for main / sev-e0) to: "- khop3 (CARD_khop_v3, ids 975-999): does the symbolic multi-step capacity
installed by chain training transfer to GSM rc2 when GSM is not damaged? Curriculum: stage 1 on the 27,000 train_kd27
chain rows, stage 2 the A2 GSM epoch from the saved weights (arm CUR), against a matched control with the K1 twins of
the same chains (arm K1X); rc2 / rc3 against best A2 and against K1X with N1 required. Parent: CARD_khop_v2 (D27 k* 4
but rc2 -7.6; H3 kept N1, k* 2, rc2 -3.2, which answers the H3-dose branch)."
The readout also needs the retry-walls file of each DAG (P7c, compute caps). The confirmation generator reads the envs
locally, so the 16 confirmation envs must also be on the AP (inside the bundle and in condor/jobs_x1 of the DAG
directory) before the confirmation DAG is submitted.

## 5. Fixed before the data

- Eval files and parent training files (md5s of 1.1), the generator rule of 1.2 (the c27 copy, the K1X replay and twin
  rule with rng1 keyed by "khop3-k1-<item_id>", the val copies), file names and directories. md5s of the new files are
  outputs of the build (build_report.json), as in the parent.
- Arms, job ids 975-994 (995 smoke only, 996-999 reserved and unused; a re-run reuses its node's id), envs, XARGS,
  schedules (S1: 1 epoch of 27,000 chain rows, LR 3e-6, warmup 0.05 + cosine, label smoothing 0.1, MBS 2, GA 16, SAVE=1; S2: the A2 re-train env with MODEL /
  WTAR), subs and PRE scripts (1.3-1.4).
- The verdict order and every sub-label of section 0, the +3.0 bar and t 2.78 on D_2 with C_2 CI lower > 0, the rc3
  line, I1-I7, V, D1-D5, N1, the reference rule, the P6 / P9 bar rule of card_numbers.py, the fail-closed rules, the gate lines P0-P11 with P7b, the confirmation
  node list and the compute caps.
- Not run here, each needs a new card: (a) after a NULL, the cag curriculum (stage 1 with GSM-sized values) to test the
  surface / number-size gap; (b) after a PASS, the noise-target or GSM-duplicate controls are not needed (K1X is in
  this card); a PASS is followed by the cag curriculum to test the surface range; (c) after "installed, then lost", an
  interleaved stage 2 (GSM rows plus a chain replay share) or a lower stage-2 LR; (d) any val sweep of a chain-loss
  weight; (e) probes of the S1 weights; other model sizes.
- Bundle: `REPO_DIR=E/src_khop3/repo jobs/build_bundles.sh "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2-khop3" x1khop3`
  (the x1khop2 pin from bundle_out/x1khop2/CODE_PIN + "-khop3"). Pre-submit lines, all must hold, each an exact file
  set: `tar tzf cotd_code_data_x1khop3_4b.tgz | grep 'data/student_khop3_'` lists exactly the 4 files of D3 (plus the 2
  dirs); `... | grep data/eval/khop3/` lists exactly build_report.json; `... | grep data/eval/khop2/` the 4 khop2 files;
  `... | grep data/eval/khop/` the 12 parent files and build_report.json; `... | grep -E 'condor/jobs_x1/(97[5-9]|98[0-9]|99[0-5])_'`
  lists exactly the 21 envs 975-995; md5s after extraction as 1.1 / 1.2; the data/eval/*.jsonl file set of the
  x1khop3 bundle equals that of the x1khop2 bundle (21 files) and every md5 is equal (Amendment 2, finding F6: the
  reference A2 sets and N1 were computed on those files, and they were restored after the /tmp cleanup); md5 of the
  bundle checked on both sides.
- Ids with 4 digits (task rule; round G also uses 1000-1024). The builder checks every script in the path and tests it:
  - stage/prep_khop3.py parses ids with re.match(r"(?:smoke_)?(\d+)_", name), never name[:3] (stage/prep_khop2.py line
    126 uses f[:3].isdigit(), which reads "1000_..." as 100); test: a condor/jobs_x1 holding 1000_..., 1024_... and
    9750_... envs is not read as the family's ids, and the refusal of spare ids 996-999 still fires.
  - round_gpuh.py (regex (\d+)_, int compare), check_pilot.sh and check_weights_khop3.sh (take names, parse no id),
    post_push.sh (parses no id), khop3_pilot_gate.py and khop3_table.py (the reference-identity rule
    re.search(rf"(^|_){j}(_|\.tgz$)") of khop2_table.py line 513): a test runs each on fixture names with ids 999, 1000
    and 1024 and on a results dir whose name sort puts 1000_ before 975_.
- Builder checks before the bundle (CPU, venv_pfx python, python -m unittest, torchrun --local-addr 127.0.0.1):
  - tests/test_build_khop3.py: refuses a changed train_kd27, train_k1 or val md5 and --out-root / --report equal to an
    input; c27 files equal the inputs byte for byte; the replay equals train_kd27 (27,000 rows) and train_k1 (3,000
    rows), and a replay fixture with one row changed stops the build; K1X rows 0..2,999 equal train_k1; for every K1X
    row, statements equal the kd27 twin's, the query and gold are the hop-1 variable and value by parse_prompt, the
    option rule holds; gold shares within the bars; eval hash overlap 0; deterministic (two builds equal); parent files
    untouched (md5 before and after); report values recomputed from the files equal the report.
  - A two-stage run on the tiny test model (the tests/test_extra_train.py 1-rank FSDP pattern, cpu-adamw,
    --eval-value-render vt, label smoothing 0.1): stage 1 on 18 khop rows with --save, stage 2 with --model set to the
    saved weights on 6 GSM rows. Stage 2 runs with one extra eval set and --eval-at 0.5 (the S2 eval asymmetry:
    the nine khop sets and the mid evaluation, Amendment 1 finding 8); the reference A2 run on the same 6 rows and seed
    from the same stage-1 weights runs without that set. Assert equal per-update item_ids, option orders and losses
    (exact), and equal final weights; separately, the stage-2 batches equal those of an A2 run from the base tiny model;
    the final steps are ceil(18 / 4) and ceil(6 / 4), losses finite.
  - D5: the diff of 2.5 against E/src_khop2/repo, printed in the build report.
  - E/readout/test_khop3_table.py on synthetic tarballs: every verdict reading and sub-label of section 0 reached by a
    fixture (each MANIPULATION FAILED sub-label from its own line, including "installed, then lost", "control not a
    control" and "GSM damaged"; PASS with and without rc3 in the claim; NEGATIVE with both texts (C_2 CI upper < 0 and
    V holds, and not); a case below the original set only (D_2^orig CI upper < 0, D_2^re not) that reads NULL with the
    drift text; a gain m = 1.5 with paired SD 1.0 (CI [+0.26, +2.74]) that reads "gain below the bar", not NULL; PASS
    blocked by I7 and by N1 of K1X (each reads "gain over best A2, control invalid"); NULL printing the CI upper; each
    BETWEEN sub-label; the rc3 line in the same order); a case where D_2 rounds to 3.0 in print but is below 3/100 exactly (not PASS) and one at exactly 3/100
    with t exactly at the bar; the rc3 line; V text in NULL / NEGATIVE; fail-closed cases (a missing preds file, a row count off by
    one, a NaN loss, a wrong seed in metrics.json, a reference set with a wrong job id) each UNREADABLE; end to end
    through main() against the real A2-6k tarballs plus synthetic CUR / K1X run directories.
  - E/readout/test_khop3_pilot_gate.py: a good pilot passes; each of P0-P11 fails alone (P6: CUR passes at reference - 2.9 and fails at
    reference - 3.1 on each set, K1X passes at - 4.9 and fails at - 5.1, the reference being the mean of j520 and j663;
    P7 by 2.05 h on S1 and 0.95 h on S2; P7c by a retry wall that brings the sum over 6.0, and a missing retry-walls file;
    P9: 117 / 150 passes and 116 / 150 fails at k = 3, 128 / 150 passes and 127 / 150 fails at k = 2, in both forms with
    the "installed, then lost" text; P11 by a loss mean of 1.01, by a log with four lines in the range, and by one k = 1 cell at 0.96); valx and gsm1kx at reference
    - 6 pass; a missing or NaN reference fails P6; P7b printed.
  - E/stage/test_prep_khop3.py: every S1 env equals 766's env except the listed keys; every S2 env equals 663's env
    except the listed keys; names, TAGs, SAVE only on S1, MODEL / WTAR pairs (980 <- 975, ...), smoke 995 on the 663
    trunk; both subs differ from their sources in exactly the CODE_TGZ, transfer_input_files and log lines (plus header
    comments), transfer_input_files names the same tgz as CODE_TGZ, the S2 executable md5; check_weights_khop3.sh on
    fixtures (missing tarball, no metrics, no weights line, missing tar, small tar, good); smoke and pilot DAGs (PARENT /
    CHILD, PRE); prep_khop3_confirm.py refuses every non-"GATE PASS" text, writes 16 nodes with the PARENT / CHILD
    pairs, the PREs, POST, RETRY and PRIORITY, and refuses a changed env.
  - The full src_khop3 suite has no new failure against repo.base (the known test_tree_target and TestSbatch failures
    of the parent build report stay).
- One cold critic on this card before the build; one cold build critic on the diff, the data and the stage files before
  the bundle; then the Codex CLI review loop (exp/jobs/codex_review.sh) until SUBMIT, as in round F.

## Amendment 1 (2026-10-27 11:20 CDT, after the cold critic, before any build)

Source: E/src_khop3/critique/CARD_CRITIC_khop3.md (verdict FIX, 9 findings). Every finding is answered below; the
sections named are changed in place. New numbers: E/src_khop3/critique/card_numbers.py (Amendment 1 block, appended;
the draft lines are unchanged) and card_numbers.out; the draft output is kept as card_numbers.out.draft for the record.
No build, data or cluster command was run.

1. MAJOR, SHUF is a non-training control. FIX, option (a). The control is now K1X: the K1 twins of the same 27,000
   train_kd27 chains (same statements and order, hop-1 query, hop-1 option set; vb the five roots), built by replaying
   the two parent generator streams (build_khop seed 1 with 1,000 + 2,000 rows per task, as khop/build_report.json
   records: seeds train 1, train_kh 3,000, train_kd 9,000; build_khop2 seed 3, 6,000 per task) and asserting equality with train_kd27 and
   train_k1, so no chain item is drawn anew (1.2 step 3). Every arithmetic item already has a legal hop-1 option set
   (draw_arith_item checks option_set_ok on it), so the file has 27,000 rows and 844 updates: same GPU cost. New lines:
   D2 (replay and twin checks), I6 can now fail (CUR - K1X at (ca, fwd, 3)), I7 control trained (S1 loss mean over the logged steps >= 760 <= 1.0,
   read from the trainer's JSON step lines, and m >= 0.98 on all six k = 1 cells, a bar A2Z fails), V = I7 and N1 for K1X, P11 (I7 at n = 1). PASS needs V; a
   PASS-sized D_2 with V failing reads BETWEEN "gain over best A2, control invalid". The PASS text now names the
   single-hop control. Renamed everywhere: data/student_khop3_k1x, TAGs khopk1x1 / khopk1x2w<id>, names 985_4b_kd27k1_...
   and 990_4b_gsmvt_A2-khopk1x2w985-..., the check_pilot.sh arguments, the gate flags --k1x1 / --k1x, follow-up (b).
2. MAJOR, a significant gain printed as NULL. FIX. The verdict order of section 0 is now MANIPULATION FAILED, PASS,
   NEGATIVE, BETWEEN "gain from extra updates, not composition", BETWEEN "gain over best A2, control invalid", BETWEEN
   "gain below the bar" (D_2 CI lower > 0, m < 3, its own claim), NULL (CI lower <= 0 and CI upper < 3), BETWEEN
   "unresolved". The unclear "(and not 2-3)" is removed. The rc3 line uses the same order. Fixture: m = 1.5, SD 1.0.
3. MAJOR, gate bars looser than the five-seed lines. FIX, with bars computed, not copied. card_numbers.py prints
   false-stop / false-go tables from the parent per-seed values and picks each bar by one rule (smallest sum among the
   bars with false-stop <= 0.15). P6: CUR >= mean(j520, j663) - 3.0 on gsm8kx and gsm8kv2 (0.034 / 0.036); the reference
   is the mean, not the max (2.4 updated: 59.42 / 70.96). K1X >= reference - 5.0 (lenient on purpose, P6 text). P9:
   a_975 and a_980 (ca, fwd, 3) >= 0.78 (0.092 / 0.095), a_980 (ca, fwd, 2) >= 0.85 (from the parent level). Both
   tables are printed, so the alternatives (B = 2.5, b = 0.80) stay visible. Normal approximation; pooled arms named in
   P6 / P9. Gate tests at the new edges.
4. MEDIUM, staging space unverified. FIX. Hard pre-submit lines in section 4 (free >= 23.1 GiB and file count + 2 <
   1,000 before the pilot; >= 77.3 GiB and + 8 before the confirmation; verify3's planned use counts), read back on the
   AP by sev-e0, else stop and ask the second author. Section 3 states that staging is probably full now (about 99.4 GiB of recorded
   tars against the recorded 100 GB quota), so the pilot likely needs the second author's decision first; deletion candidates are
   named, not chosen. The one-job alternative is REJECTED: it changes the runner, so S2 would no longer be the A2 path
   byte for byte (D5), and its walls are unmeasured.
5. MEDIUM, closed line points to NULL / NEGATIVE; card and Round G log disagree. FIX in the card (section 0: the H3
   result answers the H3-dose branch, is prior evidence for NULL / NEGATIVE, and the reason the run is still worth 22
   GPU-h). The RESULTS_LOG line is outside this writer's edit scope: section 4 gives the exact replacement text as an
   operator step for main / sev-e0, required before any submission.
6. MEDIUM, NEGATIVE from A2 drift alone. FIX. NEGATIVE needs D_2 CI upper < 0 against both A2 sets, each paired; the
   "interferes" sentence needs C_2 CI upper < 0 and V as well. A case below the original set only reads NULL with the
   text "below the original A2 set only (the A2 drift of -1.4 ... is not excluded)". Fixtures for each branch.
7. MINOR, caps do not count retries and re-runs. FIX. RETRY executions and 996-999 count against 6 / 25; no re-run when
   measured + projected would exceed the cap. P7c sums every execution, with failed-attempt walls from nodes.log passed
   to the gate (--retry-walls, required). The readout prints `round_gpuh.py 975 999` plus the retry walls. The worst
   case at the fit (6.15) is above the pilot cap; P7c fails closed there.
8. MINOR, tiny test misses the S2 eval asymmetry. FIX. The tiny stage 2 runs with an extra eval set and --eval-at 0.5,
   the reference A2 without it; per-update item_ids, option orders, losses and final weights must be equal (section 5).
9. MINOR, smoke never exercises the SAVE export. FIX (text only). Section 3 says so; P2 and check_weights_khop3.sh are
   the named first test of the export, both fail closed. No new job.

Unchanged: question, CUR arm, schedules, job ids 975-999, envs other than the K1X DATA / TAG / names, compute fit and
caps, the best-A2 rule, the selection-seed rule, nothing read from the 249 primary items by any gate line.

## Amendment 2 (2026-10-27 14:20 CDT, after the cold build critic, before any bundle or submission)

Source: E/src_khop3/critique/BUILD_CRITIC_khop3.md (verdict SUBMIT AFTER FIXES, 2 MEDIUM, 5 MINOR, 1 INFO). Every finding
is answered below; the sections named are changed in place. No bar, arm, env, schedule, id of a job, verdict order or
reading changes. No data was read for this amendment; no cluster command was run.

F1. MEDIUM, NaN probs and duplicate rows in the khop and hint preds fail open. FIX (code). khop3_table.load checks every
    preds_khop_* and preds_hint_* file present with the read_plain rule: finite probs, len(probs) = len(labels) > 0, a
    gold_label and an item_id per row, distinct item_ids over the raw rows, each file once, and a khop file with exactly
    the row count of 1.1. Else UNREADABLE; the gate maps it to "run None" (P1 FAIL, and every line that needs the run).
    2.7 is updated in place. The 34 real reference and context tarballs pass the check.
F2. MEDIUM, the re-run ids 996-999 cannot be used with the delivered tools. FIX (text, the critic's smaller option). A
    re-run is a resubmission of the same node: a rescue DAG, or a one-node DAG with the same JOB id, VARS, env, name
    and tar name. Every tool already reads that name. 996-999 stay reserved and unused; prep_khop3.py keeps refusing
    them, and the bundle line keeps "no 996-999 env". The header, 1.3, 3 (compute) and 5 are changed in place. The
    rule "no re-run when the measured walls of the phase plus the P7 bar of the job type exceed the cap" stays.
F3. MINOR, retry walls from DAG RETRY lines only undercount. FIX (text). The retry-walls file lists every failed
    execution from the job event log (event 001 to the next 004, 005, 009 or 012 of the same job id), except the one
    landed execution; condor restarts inside one DAG attempt (backfill eviction, exit 86) are included. Section 3 P7c
    and the compute line are changed in place.
F4. MINOR, any file that ends with "GATE PASS" opens the confirmation and the readout. FIX (code). The gate output must
    also hold exactly one "PASS  <name>: " line for each of the 20 khop3 gate lines and no line that starts with
    "FAIL". The P1 / P2 names carry the pilot ids, so another family's output (khop2: P1 j875, which ends with GATE PASS)
    is refused. Same check in prep_khop3_confirm.py and khop3_table.py; a test checks both lists against the gate's own
    passing output. Section 4 is changed in place.
F5. MINOR, prep_khop3_confirm.py needs more files than the operator steps name. FIX (text). It runs on the Mac; only
    confirm_x1khop3_4b.dag is copied to the AP. Section 4 is changed in place.
F6. MINOR, the bundle lines do not check the restored GSM eval files. FIX (text). New pre-submit line in 5: the
    data/eval/*.jsonl set of the x1khop3 bundle equals that of the x1khop2 bundle (21 files), every md5 equal. On
    2026-10-27 the 21 files in E/src/data/eval equal the x1khop2 bundle copies (0 md5 differences).
F7. MINOR, two report-only lines are not fail-closed. FIX (code). A missing wall of a new job stops the readout (it no
    longer counts as 0 GPU-h), and the metrics.json history of the 10 reference jobs must be finite. 2.7 is changed in
    place. The real j520-j542 and j663-j666 histories pass (10 rows each, all finite).
F8. INFO, the staging inventory is incomplete. FIX (text). Section 3 lists the 17 exported tars of the landed logs. The
    AP read-back of section 4 still decides; deletion stays the second author's decision.

## Amendment 3 (2026-10-27 15:00 CDT, after the Codex CLI review round 1, before any bundle or submission)

Source: E/src_khop3/critique/CODEX_REVIEW_khop3.r1.md (verdict SUBMIT AFTER FIXES, 2 major, 2 minor). No bar, arm, env,
schedule, id of a job or verdict order changes. No data was read for this amendment; no cluster command was run.

1. major, a partial GSM / hint / plain preds file could pass the gate (an emptied hint_true_r2 or a valx one row short
   in j980 gave GATE PASS; valx is INFO in P6) and the readout (a reference without hint_placebo_r1). FIX (code). Every
   set of a run's metrics.json args.eval (the trainer's "<name>=<path>" list) must have one preds file that holds
   exactly the item ids of the eval file E/src/<path>, rows checked as in Amendment 2 F1. args.eval must list the sets
   named in 2.7 (S1, S2, reference / context). preds_subq (eval_subq, not in args.eval, another row format, read by no
   line) is not row-checked; P1 still requires it for S2. In the gate a failing run is "run None": P1 FAIL and every
   line that needs the run. 2.7 and P1 are read this way. The 34 real reference and context tarballs pass (39, 54 or 63
   sets each, every set equal to its eval file).
2. major, under MANIPULATION FAILED the readout still printed an rc3 reading with a transfer claim. FIX (code, a reading
   clarification of section 0: "no transfer reading" covers the rc3 line). Under MANIPULATION FAILED the rc3 line prints
   D_3, D_3^orig, D_3^re and C_3 as INFO with "no rc3 reading and no transfer sentence"; readings 2-8 apply to rc3 only
   when no MANIPULATION FAILED sub-label applies.
3. minor, D_2 and C_2 were printed without SD and t (2.1 names m, SD, t and CI). FIX (code). D_2, C_2, D_3 and C_3 print
   m [CI] (SD, t, n). The decision stays exact (squared form, 2.7).
4. minor, rc_reading repeated the seed / value checks that verdict makes on the same dictionaries just before. FIX
   (code). The repeat is removed; verdict checks all eight contrasts before either call.

## Submission note (2026-10-27 15:52 CDT, sev-e0, new session)

- Status: SUBMITTED (smoke first). This note changes no line of the spec.
- Bundle: cotd_code_data_x1khop3_4b.tgz, md5 f5cf97aa1396e6366eb46d76301f5b31, 1,179 files. Pin 994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2-khop3.
- The BUILD_REPORT 6.1 item 3 checks pass. The 4 student_khop3 files and khop3/build_report.json match the section 3 md5s. The 21 envs (975-995, none on 996-999) match the stage copies byte for byte. The 21 top-level eval files match x1khop2 (0 differences).
- AP copy: the md5 matches on both sides for the bundle, the 2 subs, the 2 DAGs, check_weights_khop3.sh and the 21 envs in condor/jobs_x1.
- Staging line (section 4) read back with get_quotas: 725.4 of 1000 GB used, 84 of 10,000 files. Free space (about 275 GB) is above both 23.1 and 77.3 GiB.
- The "100 GB / 1,000 files" quota in sections 3-4 is out of date. No tar is deleted, and no the second author decision is needed.
- The 663 tar is present. The AP's run_train_x1verify2.sh is 3866fc77..., check_pilot.sh is e1a4ee99..., and post_push.sh parses no ids.
- The RESULTS_LOG Round G line was amended to the section 4 text.

## Outcome (2026-10-27 18:42 CDT, sev-e0)

- The pilot gate printed GATE FAIL: P9 (installed, then lost at s16). All other lines pass. Log: exp/logs/khop3_pilot_gate_20261027.txt.
- j975 (ca, fwd, 3) = 92.7. j980 (ca, fwd, 2 / 3) = 92.0 / 64.0, against the bars 85 / 78.
- Per section 3, nothing else is submitted. This is the pilot observation that section 3 names, not the five-seed reading.
- The card's "installed, then lost" text applies as the pilot finding: at this LR, the A2 epoch overwrites the chain capacity.
- Follow-up (c), an interleaved stage 2 or a lower stage-2 LR, needs a new card. Compute: 4.01 GPU-h.
- Details: RESULTS_LOG "Round G outcome".
