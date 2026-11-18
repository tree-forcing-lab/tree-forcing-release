# Pre-registration card: k-hop dose follow-up, rows versus updates (family khop2, parent khop), 2026-10-26, sev-e0

Status: DRAFT 2026-10-26 10:05 CDT, written before any build, bundle or submission. Amendment 1 (10:05 CDT, after the
cold critic E/src_khop2/critique/CARD_CRITIC_khop2.md, verdict FIX, before any build) edits the sections in place.
Owner's words (2026-10-26 about 09:55 CDT, RESULTS_LOG "Follow-up round F"): "你继续 选择性的内容就是你全部都做就行 有分枝
你就全部弄" (run every open branch). This card runs both branches of the CARD_khop_v1 outcome: (a) the data dose and
(b) the schedule dose, in a matched form (section 1.2 says why the literal "2 epochs" arms are replaced).
Family khop2, job ids 875-899 only. Code copy E/src_khop2/repo (untouched snapshot E/src_khop2/repo.base, a copy of
E/src_khop/repo), variant x1khop2, bundle cotd_code_data_x1khop2_4b.tgz.
E = <tmp>
One-pass rule (RULINGS_2026-10-23, tier ii): every arm is plain A2 training (letter CE after "Answer:") on extra
chain rows, as in CARD_khop_v1. No new mechanism, no generated tokens (G0), no recurrence; executed layers 32 of 32,
one forward per item, FLOPs and latency equal to A2 for the same prompt length. Memory and wall time are read from the
logs and reported. No trainer change: the A2 path is the parent's code byte for byte (section 2.5, D5).

## 0. Question, hypothesis and pre-registered readings

Question. CARD_khop_v1 ended BETWEEN, "data- or schedule-limited". Does one pass of Qwen3.5-4B reach 3 hops on the
symbolic chains (ca, fwd, k = 3) with more chain data? If the dose helps, is the variable the number of distinct chain
rows or the number of updates on chain rows?

Background (RESULTS_LOG "khop five-seed readout"; per-seed values from exp/logs/khop_readout_20261026.json, recomputed
by E/src_khop2/critique/card_numbers.py, output card_numbers.out):

| (ca, fwd), seeds 16 / 17 / 18 / 19 / 20 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| KH (3,000 chain rows, 279 updates) | 95.3 92.7 92.0 88.7 86.7 | 55.3 40.7 43.3 40.0 31.3 (m 42.1) | 34.7 24.0 31.3 33.3 28.7 |
| KD (9,000 chain rows, 467 updates) | 94.7 96.7 98.7 98.7 96.0 | 60.7 70.7 66.0 75.3 66.7 (m 67.9) | 46.0 58.7 63.3 61.3 56.7 |

KD - KH at k = 3: +25.7, SD 12.5, CI [+10.2, +41.3]. The v1 contrast changed two things at once: distinct rows
(3,000 -> 9,000) and updates (279 -> 467, so also the LR schedule length). This card separates them.

Design (the dose grid). Every arm is 1 epoch over gsmvt 6k (5,923 rows) plus chain rows, MBS 2, GA 16 (32 rows per
update), LR 3e-6, warmup 0.05 + cosine to 0 over the run. Repetition is inside the chain file, so the gsmvt exposure
stays 1 epoch in every arm. Cells: (distinct chain rows, chain-row exposures, updates).

| | exposures 3,000 | exposures 9,000 | exposures 27,000 |
|---|---|---|---|
| distinct 3,000 | KH (parent, 279 upd) | H3 = KH rows x 3 (new, 467 upd) | - |
| distinct 9,000 | - | KD (parent, 467 upd) | R3 = KD rows x 3 (new, 1,029 upd) |
| distinct 27,000 | - | - | D27 (new, 1,029 upd) |

Two matched contrasts hold updates, LR schedule, chain exposures and gsmvt exposure fixed and change only the number of
distinct rows: D27 - R3 (at 1,029 updates) and KD - H3 (at 467 updates). Two contrasts hold the distinct rows fixed and
change only the repetition (updates and exposures): R3 - KD (9,000 distinct) and H3 - KH (3,000 distinct). The v1
contrast splits exactly: KD - KH = (KD - H3) + (H3 - KH), and D27 - KD = (D27 - R3) + (R3 - KD), per seed.

What is new. CARD_khop_v1 measured k* = 2 and a climbing k = 3 cell. This card measures where the k = 3 curve goes at 3x
the v1 dose, and it says which part of the dose carries the gain. A PASS here is a new measurement for this model
("3 + / - steps reachable in one pass with enough chain training"), not a claim about GSM. The rows-versus-updates split is the
part that the literature does not give for this setting.

Prior expectation (no new search this round; sources as cited in CARD_khop_v1 section 0, plus one from memory):
- Wang et al. (NeurIPS 2024, grokked transformers): composition needs training far past fit. This predicts that
  repetition helps (R3 - KD > 0, H3 - KH > 0).
- Muennighoff et al., "Scaling Data-Constrained Language Models" (NeurIPS 2023; from memory, not re-checked this round; used only for the priors and never as evidence in the readout
  text):
  up to about 4 repeats of the same data are worth nearly as much as fresh data in pretraining. This predicts small
  distinct-rows contrasts at 3 repeats (D27 - R3 and KD - H3 near 0).
- Rank probes (CARD_khop_v1 Amendment 3, probe_ref of the v1 readout): on (ca, fwd, 3) the signed-sum probe (composes
  only the + / - steps) scores 0.82 and the first-constant probe scores 0.61. So the SOLVED bar (m >= 0.80) can be met by
  a model that composes only the + / - steps. The claim words follow Amendment 3 (item 2 below).
Priors: PASS about 0.45 (log-linear extrapolation of KH -> KD gives about 0.90 at 27,000 rows; saturation is likely);
BETWEEN about 0.40; NULL about 0.10; NEGATIVE about 0.05 (power statement below). Branch (b) (Amendment 1, pooled rule):
"UPDATES" about 0.35, "BOTH" about 0.20, "ROWS" about 0.10, "UNRESOLVED" about 0.35.

Hypothesis H (dose). At 27,000 distinct chain rows (1 epoch, LR 3e-6) the cell (ca, fwd, 3) is SOLVED.

Notation (as CARD_khop_v1). a_X(t, o, k) = accuracy of arm X on task t, premise order o, chain length k, one value per
seed. m = five-seed mean; CI = m +/- 2.78 x SD / sqrt(5) (df 4, seeds 16-20). SOLVED: m >= 0.80 and CI lower >= 0.70.
FAILED: CI upper < 0.80. k*(X, t, o) = the largest k <= 6 with cells 1..k all SOLVED (0 when k = 1 is not SOLVED).
Paired differences are per seed, over the seeds that both arms have. D_dose = a_D27(ca, fwd, 3) - a_KD(ca, fwd, 3).
The selection seed: the pilot gate chooses no setting and no arm (go / no-go only, section 3), so s16 is one of the five
seeds everywhere (n = 5, t_crit 2.78). No k = 4 seed rule applies.

Power statement (pre-registered, so that a BETWEEN is not read as a near miss). The paired SD of D_dose is expected at
about 8-12 points (KD seed SD 5.5, so about 7.8 for two independent arms; KD - KH had SD 12.5). The CI half-width is then
about 10-16 points. "CI upper < +0.05" (NEGATIVE) needs a mean D_dose below about -5 to -10 points: NEGATIVE is reached
only if 27,000 rows are clearly worse than 9,000. A flat curve with a mean near 0 reads NULL, not NEGATIVE.
Branch (b) power (Amendment 1, card_numbers.out): one single contrast (R27, R9, U9 or U3) has a CI half-width of about
9.6-15.5 points; an even split of the v1 gain (+25.7) gives about +12.8 per part, at the edge. The pooled contrasts of
the label rule (two levels, four distinct arms each) have a half-width of about 6.8-11.0 points. A single-level label
(H3 dropped) has the single-contrast width, so UNRESOLVED is then more likely.

Verdict readings, evaluated in this order; the first that applies is the verdict:
1. MANIPULATION FAILED: any installation check of section 2.5 (I1-I5) fails at five seeds. No dose reading; report the
   failing line.
2. PASS: k*(D27, ca, fwd) >= 3 (so (ca, fwd, 3) is SOLVED under D27). Claim: "3 hops are reachable in one pass of
   Qwen3.5-4B at this scale with enough chain training: at 27,000 chain rows (1 epoch, 1,029 updates, LR 3e-6) one pass
   selects the value of a 3-step symbolic chain at m = <x> [CI]." Amendment 1: the words "with enough chain training"
   become "with enough chain data" only when the branch (b) label is ROWS or BOTH. Amendment 3 wording rule (unchanged): the claim says "composes 3 + / - steps"
   only when m(a_D27(ca, fwd, 3 | all three steps + / -)) >= 0.80 on the 48 all-+/- items; else the reading is "PASS
   (approximate)" and the claim says "selects the value of a 3-step chain; on the 48 all-+/- items the mean is <x>".
   The x2 / x3 steps are shared by the five chains and never decide the answer (CARD_khop_v1 Amendment 3), so no claim
   names x2 / x3. The branch (b) label (below) is added to the claim: "the gain comes from <label>".
3. NEGATIVE (capacity at this schedule): (ca, fwd, 3) is FAILED under D27 AND the D_dose CI upper < +0.05. Claim: "at 1
   epoch and LR 3e-6, 3,000 to 27,000 chain rows do not make one pass of Qwen3.5-4B select the value of a 3-step chain
   reliably, and the k = 3 cell is flat from 9,000 rows at 467 updates to 27,000 rows at 1,029 updates; capacity reading at this
   schedule" (a dose contrast: rows, updates and LR schedule length change together; Amendment 1). The claim always
   names the schedule. If R3 is also FAILED, the claim adds "and 3 repeats of 9,000 rows do not help either".
4. NULL: not 1-3, and the D_dose CI includes 0. "No detectable dose effect from 9,000 rows at 467 updates to 27,000
   rows at 1,029 updates, and the data do not exclude a gain of <D_dose CI upper> points." (Amendment 1: the CI upper
   is printed, not a fixed number.) The experiment does not locate the limit; no capacity sentence is written.
5. BETWEEN: everything else. Pre-registered sub-labels (the first that applies):
   - "still dose-limited at 27,000 rows and 1,029 updates": D_dose CI lower > 0 and (ca, fwd, 3) not SOLVED under D27.
     Amendment 1: the label is "still data-limited" only when R27 also has CI lower > 0; else it reads "still
     dose-limited (rows and updates not separated; see branch (b))".
   - "reachable with repetition, not with fresh rows": (ca, fwd, 3) SOLVED under R3 and not under D27.
   - "lower at the higher dose": D_dose CI upper < 0 and (ca, fwd, 3) not FAILED under D27.
   - "k = 3 solved, k = 2 not": (ca, fwd, 3) SOLVED under D27 while (ca, fwd, 2) is not (so k* < 3).
   - "unresolved": the rest (for example D27 neither SOLVED nor FAILED with a D_dose CI that excludes neither 0 nor
     the reading lines above).

Branch (b) label (rows versus updates; pre-registered, reported with every verdict, a secondary decision line).
Contrasts at (ca, fwd, 3), paired over s16-20, 95% CI with t 2.78:
- R27 = D27 - R3 and R9 = KD - H3 (distinct rows at matched updates and exposures).
- U9 = R3 - KD and U3 = H3 - KH (repetition at fixed distinct rows).
Pooled per seed (Amendment 1, finding 4): rows_pool = (R27 + R9) / 2, upd_pool = (U9 + U3) / 2, paired over s16-20,
t 2.78. rows = (rows_pool CI lower > 0); upd = (upd_pool CI lower > 0).
Label: "BOTH" when rows and upd; "ROWS" (distinct rows are the variable) when rows and not upd; "UPDATES" when upd and
not rows, worded "repetition gains; no detectable effect of distinct rows (rows_pool CI upper <x>)" (it does not claim
that repetition is as good as fresh rows); "UNRESOLVED" otherwise.
When H3 is dropped by P7b (section 3), rows_pool = R27 and upd_pool = U9, and the label carries "(one level only)".
Each of the four contrasts and both pooled contrasts are printed with CI and per-seed values whatever the label is. The same four contrasts
are also printed at k = 4 (reported only).

## 1. Data and arms

### 1.1 Tasks and eval sets (unchanged)

Tasks, eval files, option rule, shared skeleton, orders, gold layout, row keys: exactly CARD_khop_v1 section 1.1 with
Amendments 2 and 3. The nine eval files in E/src/data/eval/khop/ are not rebuilt and not edited. Their md5s must equal
the Amendment 2 table of E/src_khop/BUILD_REPORT.md ("Data (rebuilt 2026-10-26 00:22 CDT)"):
khop_ca_fwd 171f097c..., khop_ca_rev 2f2bc7b6..., khop_wp_fwd c58c93ff..., khop_wp_rev bdad8780..., khop_vb_fwd
046c6ad3..., khop_vb_rev 2279c09f..., khop_cag_fwd 3e79c1b5..., khop_ca_fwd_qs 38cf266e..., khop_blind b58f7284...;
train_kh a16b1520..., train_k1 4bb15f66..., train_kd 2ed4ecc3... (full md5s in that table; the gate compares full md5s).
No new eval set. E/src/data/eval/khop2/ holds only the new training files and their report.

### 1.2 New training files (E/src/data/eval/khop2/)

Generator: a new script scripts/build_khop2.py in E/src_khop2/repo. It imports scripts/build_khop.py and does not edit
it. It must not call build_khop.build(): build_eval() redraws eval items against the training keys, so a larger training
set changes the eval files, and the task-major loop of build_train() reorders the stream when n_kd_extra grows, so the
result would not start with train_kd. build_khop2.py extends instead:

1. Read the 12 parent files from --src (E/src/data/eval/khop) and stop (SystemExit) when any md5 differs from the
   parent's build_report.json.
2. Forbidden sets: the value keys (build_khop.value_key on the parsed chains, the row_key rule of build_khop.build) and
   the prompt hashes (phash) of every row of the nine eval files (blind rows: hash only) and of train_kh, train_k1,
   train_kd.
3. New rows: random.Random(--seed 3), task-major over ("ca", "wp", "vb"), j = 3000..8999 per task (6,000 per task,
   18,000 rows), k = 1 + j mod 6, order "fwd" if (j // 6) mod 2 == 0 else "rev" (the build_train rule), item_id
   "khop-tr-<task>-<j:05d>" (continues the parent ids; train_kd ends at 02999), it = build_khop.gen_item(task, k, rng,
   stats); redraw while the value key or the prompt hash is forbidden or already used by a new row; at most 2,000
   redraws per row, else BuildError. Gold layout: build_khop.layout(i, k, rng) with i continuing the per (task, k)
   cycle of the parent (i starts at 500, the train_kd count of every (task, k) cell). Row = it.row(iid, order, slots)
   with the gsmvt key set (32 keys), as the parent's training rows.
4. Write:

| File | Rows | Content |
|---|---|---|
| train_kd27.jsonl | 27,000 | lines 1-9,000 = train_kd.jsonl byte for byte; then the 18,000 new rows (ca, wp, vb blocks) |
| train_kd_x3.jsonl | 27,000 | train_kd rows three times: copy 1 byte for byte, copy 2 and 3 with item_id + "-r2" / "-r3" (problem_id unchanged, every other field unchanged) |
| train_kh_x3.jsonl | 9,000 | train_kh rows three times, same rule |
| build_report.json | - | card CARD_khop_v2, generator scripts/build_khop2.py, seed 3; md5 of the 3 new files and of the 12 parent files; counts per (task, k, order); D1 (all barred and reported lines of CARD_khop_v1 2.4, with build_khop.d_lines) per (task, k) cell and pooled per task, on train_kd27 whole and on its 18,000 new rows alone; D2 gold shares; D3 (shared prompt hashes and value keys of each new file against the nine eval files, both 0); distinct value keys of train_kd27 (expected 26,941: the parent's 59 duplicate-key rows in train_kd stay, the new rows add none); redraws per (task, k); nested check; Qwen3.5-4B token totals per file |

Report schema: build_report.json in data/eval/khop2 uses the parent report's keys (md5, counts, gold_shares,
d1_baselines, d4_tokens, d3_disjointness, bars, d_fail, d_pass, train_tokens) with the stems train_kd27, train_kd_x3,
train_kh_x3 (the new-rows-only D1 lines under the stem train_kd27_new), so build_khop.d_lines, cell_of, barred and
khop_pilot_gate.d_bar_violations work on it unchanged. Extra keys: nested, copies_ok, new_distinct_keys, redraws.

Author check (E/src_khop2/critique/feas_kd27.py, output feas_kd27.out; same generator, seed 3, not the build): all
18,000 new rows found new value keys; redraws ca k1 244, ca k2 77, wp k1 267, wp k2 77, at most 1 elsewhere; no cell
hit the cap. The eval files hold 4,200 value keys, 0 shared with train_kd. train_kd has 59 rows in duplicated value keys
(ca k1 41, ca k2 22, wp k1 46, wp k2 8 rows involved); this card does not change train_kd.
The k = 1 / 2 redraw rate (about 20% at ca / wp k = 1) is a mild selection on k = 1 values; k = 3 is not affected.

Why the training files are in data/eval/khop2/ and not in data/student_khop2_<name>/ (deviation from the task text):
jobs/build_bundles.sh lines 18-20 copy only data/eval, data/student_gsm8k_tf_vt and data/student_gsm8k_tf_si. A
data/student_khop2_* directory would not reach the cluster, and jobs/ is not ours to edit. data/eval is copied whole.

Why repetition is in the file and not EPOCHS=2 (the task text's KD-2ep and KH-2ep). EPOCHS=2 also runs the 5,923 gsmvt
rows twice. Then the plain and rc readout mixes the chain dose with a second GSM epoch (A2 peaks at c1-c2 on GSM), and
the 2-epoch arms match no other arm in updates (KD-2ep 933, KH-2ep 558). Repetition inside the chain file matches
updates, LR schedule and gsmvt exposure exactly (the grid in section 0), and it needs no trainer change. The literal
EPOCHS=2 arms are not run (section 5); they can follow in a new card if the second author wants the literal schedule.
Amendment 1 (finding 7): this is a change to a branch that the owner named ("27,000 chain rows and 2 epochs"). The
2-epoch branch runs as R3 / H3 (3 repeats in the file, matched updates). The literal EPOCHS=2 arms need a new card and
a new id range (spares 890-899 are re-runs only). The submission message to the second author (sections 3, 4) says this.
Known difference from a true multi-epoch run: the three copies are shuffled together, so two copies of one row can
fall in the same or nearby batches. A true epoch shows each row once per pass. This is reported, not controlled.

### 1.3 Arms

| Arm | Chain rows (distinct x copies) | Training rows | Updates | Jobs |
|---|---|---|---|---|
| D27 (data dose) | 27,000 x 1, train_kd27 | 32,923 | 1,029 | j875 s16 (pilot, SAVE=1), j876 s17, j877 s18, j878 s19, j879 s20 |
| R3 (KD rows x 3) | 9,000 x 3, train_kd_x3 | 32,923 | 1,029 | j880 s16 (pilot), j881 s17, j882 s18, j883 s19, j884 s20 |
| H3 (KH rows x 3) | 3,000 x 3, train_kh_x3 | 14,923 | 467 | j885 s16, j886 s17, j887 s18, j888 s19, j889 s20 (confirmation) |
| reused, no new job | KH j756, j758-761; KD j766-770; K1 j757, j762-765; A2Z j751-755; BASE0 j750; A2Z96 j771; A2-6k originals j520 j525 j530 j540 j542 and re-trains j663 j664 j665 j540 j666 | | | tarballs in E/results |

Updates = ceil(rows / 32) (the parent logs: KH step 279, KD step 467). Spare ids 890-899: re-runs only (the same env
under a new id); never a new setting. Smoke runs use the pilot envs with smoke="1" (names smoke_875_..., smoke_880_...).

Cross-bundle pairing. D27 - KD, R3 - KD, KD - H3 and H3 - KH pair new jobs with the parent's tarballs (bundle x1khop).
Same seeds, same env keys, same trainer (D5 proves the code is the same). Host and GPU type differ per job in both
rounds; that was already true inside v1. This is stated in the readout.

### 1.4 Exact envs and flags

No new trainer flag. New generator flags (scripts/build_khop2.py only): --src DIR (default data/eval/khop), --out DIR
(default data/eval/khop2), --seed INT (default 3), --new-per-task INT (default 6000), --repeat INT (default 3),
--tokenizer DIR (optional, token totals and D4), --gsm-train PATH (optional reference token total).

Env generator stage/prep_khop2.py (python, values bash-escaped as stage/prep_khop.py does). Each training env is a copy of
condor/jobs_x1/766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.env with exactly these keys changed (every other key asserted
equal to 766, so EPOCHS=1, LR=3e-6, LSMOOTH=0.1, MBS=2, GA=16, NGPU=1, EVALAT=0.5, EVALMID, EXTRA_EVALS with the nine
khop sets, DATA=data/student_gsm8k_tf_vt stay):
- SEED = ORDERSEED = s.
- D27: TAG=khopd27-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop2/train_kd27.jsonl".
- R3: TAG=khopd9x3-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop2/train_kd_x3.jsonl".
- H3: TAG=khoph3x3-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop2/train_kh_x3.jsonl".
- SAVE=1 for j875 only (D27 s16 weights for a later probe); SAVE='' elsewhere.
- JOBKEY "4b gsmvt A2-<TAG> s<s> fsdp x1khop2". Name <id>_4b_gsmvt_A2-<TAG>_s<s>_fsdp, for example
  875_4b_gsmvt_A2-khopd27-c1lr3ch_s16_fsdp, 880_4b_gsmvt_A2-khopd9x3-c1lr3ch_s16_fsdp,
  885_4b_gsmvt_A2-khoph3x3-c1lr3ch_s16_fsdp.
Sub: stage/train_x1khop2_4b.sub = stage/train_x1khop_4b.sub with exactly three lines changed (Amendment 1, finding 5):
line 13 environment CODE_TGZ=cotd_code_data_x1khop2_4b.tgz, line 16 transfer_input_files = cotd_code_data_x1khop2_4b.tgz,
line 33 log = logs/train_x1khop2_4b.log; plus the header comment. Tested. No eval sub: there is no eval-only
job. Resources as the parent: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300, RETRY 1, POST post_push.sh.
Smoke note: the sbatch smoke takes the first 16 rows of the extra file, so the smoke of R3 trains on copy-1 rows only.
The suffixed ids are covered by the local tests (section 5).

## 2. Metrics and rules

### 2.1 Primary
- The verdict of section 0: (ca, fwd, 3) under D27 (SOLVED / FAILED), k*(D27, ca, fwd), and D_dose with its CI.
- Reported with the primary: the full curves a(ca, fwd, k), k = 1..8, every seed, for D27, R3, H3, KD and KH.

### 2.2 Branch (b), secondary decision line
- The label of section 0 from the pooled contrasts rows_pool and upd_pool at (ca, fwd, 3) (Amendment 1); R27, R9, U9, U3
  printed. The same contrasts at k = 4, reported.

### 2.3 Secondary (reported; no verdict depends on them)
- All-+/- subsets (Amendment 3): (ca, fwd, 3 | n_add = 3), 48 items, for every arm; (wp, fwd, 3), 67 items; probe
  agreement (rank_sum, rank_h1) per arm at k = 2, 3. A rise in rank_sum agreement with no rise on the items with a x
  step is the "+ / - steps only" pattern.
- Log-dose slope (reported, not a decision line): per seed, the least-squares slope of a(ca, fwd, 3) on log3(distinct
  rows) over KH, KD, D27 (points 0, 1, 2), with its five-seed mean and CI. It mixes rows and updates; the matched
  contrasts separate them.
- k*(X, t, o) and a(t, o, 2..4) for every task and order (wp, vb, cag, rev) for D27, R3, H3; fwd - rev at k = 2, 3.
- G2 / G3 for D27 and R3 as in CARD_khop_v1 section 0 (a(ca, fwd, 2) - rc2_ref, a(ca, fwd, 3) - rc3_ref; reference
  candidates: the two A2-6k sets and the arm itself).
- GSM side: plain sets gsm8kx, gsm1kx, gsm8kv2, valx and the rc1..rc4 cells of the 249 primary items (pfx_table.py)
  for D27, R3, H3, with paired differences against the best A2-6k set (2.4) and against KD (the dose effect on GSM).
- Timing (wall, s_per_step), peak memory per job.

### 2.4 Reference A2 (best-A2 rule, fixed now; as CARD_khop_v1 2.3)
For every GSM-side line the reference is the A2-6k set (originals j520, j525, j530, j540, j542 or re-trains j663, j664,
j665, j540, j666) with the higher five-seed mean on that line's metric; that set's per-seed values enter the paired
difference. At the pilot (n = 1) the s16 reference is max(j520, j663) per metric.

### 2.5 Installation and manipulation checks
Data side (before any bundle; build_report.json in data/eval/khop2 and the builder's tests):
- D1 Every barred D1 line of CARD_khop_v1 2.4 (12 lines) on train_kd27, train_kd_x3, train_kh_x3: <= 0.33 per (task, k)
  cell and <= 0.25 pooled per task; qop_avoid and the five partial lines are reported. Computed on the whole file and on
  the 18,000 new rows alone.
- D2 Gold share per (file, task, k) cell in [0.17, 0.23]; options distinct; gold in options.
- D3 0 shared prompt hashes and 0 shared value keys between each new file and the nine eval files. The 18,000 new rows
  add no value key already in train_kd or among themselves.
- D4 Longest prompt + options < 1,536 Qwen3.5-4B tokens.
- D5 Same trainer. `diff -r -x __pycache__ -x data E/src_khop/repo E/src_khop2/repo` lists only scripts/build_khop2.py
  and tests/test_build_khop2.py (new files; the tiny-model A2 step test of section 5 lives in tests/test_build_khop2.py). So the A2 path of D27 / R3 / H3 is the parent's code byte for byte.
- D6 Nesting and copies: train_kd27 lines 1-9,000 equal train_kd.jsonl; train_kd_x3 / train_kh_x3 copy 1 equals the
  parent file and copies 2-3 differ from it only in item_id; all item_ids distinct (the trainer refuses duplicates,
  train_student.py line 1042).
Model side, at five seeds (khop2_table.py):
- I1 Format: m(a_X(t, o, 1)) >= 0.90 for X in D27, R3, H3 and all six (task, order) cells of ca / wp / vb.
- I2 Query read: query-swap pair accuracy at k = 1, m >= 0.85 for D27, R3, H3.
- I3 No option-only shortcut: m(blind accuracy) <= 0.30 for D27, R3, H3.
- I4 Dose installed (the manipulation check): every D27 / R3 / H3 log has "extra-train: 27000 rows from
  data/eval/khop2/train_kd27.jsonl" / "27000 rows from data/eval/khop2/train_kd_x3.jsonl" / "9000 rows from
  data/eval/khop2/train_kh_x3.jsonl", "extra-train: total training rows 32923 / 32923 / 14923", the final step line
  "step": 1029 / 1029 / 467, and metrics.json "extra_train" agrees.
- I5 Chain training took: m(a_X(ca, fwd, 2)) >= 0.85 for X in D27, R3, H3 (KH 91.1, KD 97.0). This line does not
  compare with KD at k >= 3: a flat k = 3 cell is a result, not a fault.
I1-I5 all pass, or the verdict is MANIPULATION FAILED.

### 2.6 Necessary condition for the GSM-side reading
- N1 Plain not down: for D27, R3 and H3, the seed mean on each of gsm8kx, gsm1kx, gsm8kv2, valx >= best-A2 reference
  - 2 points. N1 does not block the verdict (the k = 3 cell is a property of the trained model). If N1 fails for an arm,
  its line says "plain cost" and no transfer sentence is written for it.
- Plain cost of the dose (reported): D27 - KD per plain set, paired, with CI. The answer to "does more chain data hurt
  plain?" is "yes" only when the pooled plain D27 - KD CI upper < 0, and "no detectable cost" when it includes 0.
- Transfer or harm (reported, no verdict; Amendment 1, finding 2): for each of rc1, rc2, rc3, rc4 on the 249 primary
  items and for each of D27, R3, H3: X - best-A2 and X - KD, paired, with CI. Three-way sentence per (arm, cell):
  - CI lower > 0 against best-A2 and N1 holds: transfer sentence "chain training raises GSM rc<j> by <x> [CI]".
  - CI upper < 0 against best-A2: harm sentence "chain training lowers GSM rc<j> by <x> [CI]".
  - Else: "no detectable change of GSM rc<j> at this dose (CI <lo, hi>)".
  Printed next to it: KD - best-A2 and KH - best-A2 at rc1..rc4 from the parent tarballs (parent readout, against the
  original A2 set: KD rc1 -7.3, rc2 -5.5; KH rc1 -1.1, rc2 -1.0). The best-A2 rule (2.4) takes the A2-6k set with the
  higher mean on each cell, so a harm line is measured against the stronger reference. This is the rule fixed before
  the data, not a choice made for these numbers.

## 3. Pilot (seed 16) and compute

Smoke DAG stage/smoke_x1khop2_4b.dag: smoke_875 (j875's env, smoke="1", SMOKE_ONLY) and smoke_880 (j880's env,
smoke="1", SMOKE_ONLY). Smoke gate: exit 0 for both.

Pilot DAG stage/pilot_x1khop2_4b.dag: j875 (D27 s16, SAVE=1) and j880 (R3 s16). H3 is not in the pilot: its code path
(an extra file of copies) is the one R3 uses, and the pilot stays under the 6 GPU-h cap with margin.

Deterministic pilot gate: E/readout/khop2_pilot_gate.py --d27 <875 tgz> --r3 <880 tgz>
--kd s16=E/results/results_766_4b_gsmvt_A2-khopd3-c1lr3ch_s16_fsdp.tgz
--ref s16=E/results/results_520_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz,E/results/results_663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.tgz
--report E/src/data/eval/khop2/build_report.json --parent-report E/src/data/eval/khop/build_report.json.
Exit 0 = GATE PASS. Lines:
- P0 Data: the khop2 report has d_pass true; the md5 of the 3 new files equal that report; the md5 of the 12 parent
  files equal the parent report (and the Amendment 2 table); the gate re-checks every barred D1 value and every D2 share
  of the khop2 report itself (khop_pilot_gate.d_bar_violations, its own list of the 12 barred lines); D3 counts 0; D6
  nested and copy flags true; 18,000 new distinct value keys.
- P1 Each pilot tarball has metrics.json, the nine khop preds files with the CARD_khop_v1 1.2 row counts, and every GSM
  and hint preds file of the 663 env.
- P2 (I4 at n = 1) j875 and j880 logs: the extra-train line with 27,000 rows from the right file, total 32,923, final
  step 1,029, final loss finite.
- P3 (I1 at n = 1) a(t, o, 1) >= 0.85 on all six (task, order) cells for j875 and j880.
- P4 (I2 at n = 1) query-swap pair accuracy at k = 1 >= 0.80 for j875 and j880.
- P5 (I3 at n = 1) blind accuracy <= 0.32 for j875 and j880.
- P6 Plain catastrophe line at n = 1 (Amendment 1, finding 1): gsm8kx and gsm8kv2 of j875 and j880 >= max(j520, j663)
  - 10 (s16 references 59.8 and 71.1). valx and gsm1kx are printed as INFO. Reason: valx at s16 is noisy (KD j766 58.0
  against 64.4, KH j756 60.6), and the verdict does not use GSM; the gate stops only a broken model. KD s16 is -0.5 /
  -0.9 on the two gated sets, and the A2-6k, KH and KD runs span 58.2-61.6 on gsm8kx.
- P7 Wall time of j875 and j880 each <= 2.9 h (Amendment 1, finding 8: 2 x 2.9 + 2 smokes of 0.05 = 5.9 GPU-h, under
  the cap of 6).
- P7b Projected confirmation <= 25 GPU-h. t = mean wall of j875 and j880; main = 8 t (D27 and R3 at s17-20);
  h3 = 5 x max(1.11, 0.521 t) (H3 has KD's shape, 467 updates; 1.11 h is KD's measured mean; 0.521 is the fitted
  467 / 1,029 wall ratio of section 3, compute). main + h3 <= 25: "GATE PASS". Else main <= 25: "GATE PASS (H3 dropped:
  projection <main + h3>)" and the confirmation has no H3 nodes (branch (b) then uses R27 and U9 only). Else GATE FAIL.
  Under P7 (t <= 2.9) main <= 23.2, so P7b alone never fails the gate; H3 runs when t <= about 2.36 h (25 / (8 + 5 x 0.521)).
- P8 smoke_875, smoke_880, j875 and j880 landed with exit 0.
- P9 (I5 at n = 1) a(ca, fwd, 2) >= 0.80 for j875 and j880 (KD s16 94.7, KH s16 95.3). No gate line compares D27 or
  R3 with KD at k >= 3.
Output: "GATE PASS", "GATE PASS (H3 dropped: projection <x>)" or "GATE FAIL: <lines>". Reported, not decision lines:
every curve, a(ca, fwd, 3) of j875 / j880 against j766 (KD s16), the all-+/- subset, probe agreement, GSM rc cells
(pfx_table.py), timing and peak memory. GATE FAIL: nothing else is submitted; report to the second author with the failing lines.
Every message to the second author about this family (pilot submission, gate result, confirmation submission) says that the owner's
2-epoch branch runs as R3 / H3 (repetition in the file) and that the literal EPOCHS=2 arms need a new card and id range.

Compute (measured from the parent's jobs with E/jobs/round_gpuh.py; fit and projections in
E/src_khop2/critique/card_numbers.py, output card_numbers.out):
- Measured: KD j766-770 1.27 / 1.03 / 0.99 / 1.25 / 0.99 h, total 5.54 GPU-h, mean 1.11 h (467 updates). KH j756,
  j758-761 mean 0.77 h (279 updates); K1 mean 0.72 h; smoke_756 0.04 h. Updates dominate (cpu-adamw opt_s about 4.4 s of
  the 5.4-7.1 s per step in the j756 / j766 logs).
- Fit wall = a + U x s: s = 6.51 s per update, a = 0.26 h (KH and KD means). Worst host seen (j766 s16, A100-80GB
  gpu2005): 7.08 s per update, a = 0.35 h.
- D27 and R3 (1,029 updates): 2.12 h each (worst 2.37). H3 (467 updates): 1.11 h (worst 1.27). The 1,029-update value is
  an extrapolation: the line is fitted on two means (279 and 467 updates) and used at 2.2x the upper end. P7 and P7b
  measure the real wall at the pilot before any confirmation node is written.
- Pilot: smoke_875 + smoke_880 (0.05 each) + j875 + j880 = 4.34 GPU-h (worst 4.85; 5.9 at the P7 bar). Cap 6.
- Confirmation: 8 x 2.12 (D27, R3 s17-20) + 5 x 1.11 (H3 s16-20) = 22.5 GPU-h (worst 25.35, which P7b handles by
  dropping H3). Cap 25.
- Family total about 26.9 GPU-h (round cap 120, RESULTS_LOG "Follow-up round F").
- Order if not everything fits: D27 and R3 first. D27 carries the pre-registered verdict (data dose); R3 is the matched
  partner that makes D27 - R3 and R3 - KD readable at all. H3 adds the second matched level at 467 updates; it is the
  one arm whose loss leaves the verdict intact. The literal EPOCHS=2 arms (KD-2ep about 1.95 h, KH-2ep about 1.27 h per
  job, 16.1 GPU-h for 5 seeds of both) do not fit under the 25 GPU-h cap next to D27 and R3 (section 1.2 says why
  they are replaced, not only deferred).

## 4. Conditional confirmation (written now; sev-e0 submits it without asking when the gate prints GATE PASS)

stage/confirm_x1khop2_4b.dag, written by stage/prep_khop2_confirm.py --gate-output gate.txt. It refuses unless the last
line of gate.txt starts with "GATE PASS", and every env still equals the text that prep_khop2.py generates. Nodes:
- D27 s17-20: j876, j877, j878, j879.
- R3 s17-20: j881, j882, j883, j884.
- H3 s16-20: j885, j886, j887, j888, j889, unless the gate line is "GATE PASS (H3 dropped: ...)", which writes no H3
  node.
Every node: SCRIPT PRE check_pilot.sh 875_4b_gsmvt_A2-khopd27-c1lr3ch_s16_fsdp 880_4b_gsmvt_A2-khopd9x3-c1lr3ch_s16_fsdp,
POST post_push.sh, RETRY 1, prio 300. the second author is told at submission and can stop it; the message carries the 2-epoch
branch sentence of section 3.
Readout: E/readout/run_khop2.sh -> E/readout/khop2_table.py on the new tarballs (D27, R3, H3; s16 from the pilot) plus the
parent's KH, KD, K1, A2Z, BASE0, A2Z96 and both A2-6k sets, then the section 0 verdict in that order, the branch (b)
label (pooled rule), and the GSM transfer / harm lines of 2.6 (rc1..rc4, KD and KH next to them). khop2_table.py reuses khop_table.py primitives (load_run, curve, alladd_curve, qs_pair, blind, stats, solved,
failed, kstar, paired, rc_cells, probe_agree, plain_acc) by import; it does not edit khop_table.py. A missing tarball
stops the readout (it never reads a partial set), except the H3 tarballs when the gate dropped H3 (the gate output file
is an input of run_khop2.sh). Log exp/logs/khop2_readout_<date>.txt (+ .json).

## 5. Fixed before the data

- Tasks, eval files (md5s of section 1.1), generator rule, seed 3, row counts, ids, copy rule and file names (1.2).
- Arms, job ids 875-889 (spares 890-899 re-runs only), envs and XARGS, schedule (1 epoch, LR 3e-6, label smoothing 0.1,
  MBS 2, GA 16, NGPU 1, warmup 0.05 + cosine), SAVE on j875 only (1.3-1.4).
- SOLVED / FAILED bars, k*, D_dose, the order of the verdict readings and the BETWEEN sub-labels, the power statement,
  the branch (b) contrasts, the pooled label rule, the Amendment 3 "composes" rule, I1-I5, D1-D6, N1, the rc1..rc4 transfer / harm sentences, the reference rule, the
  pilot gate lines P0-P9 with P7b, the confirmation node list and the compute caps.
- Not run here, each needs a new card: the literal EPOCHS=2 arms (KD-2ep, KH-2ep), 3,000 rows x 9 (the 3,000-distinct
  point at 1,029 updates), more than 27,000 rows, a higher LR, k > 6 in training, other model sizes, a 96k mix, any
  probe of the j875 weights.
- Bundle: `REPO_DIR=E/src_khop2/repo jobs/build_bundles.sh "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2" x1khop2`
  (the x1khop pin from bundle_out/x1khop/CODE_PIN + "-khop2"). Pre-submit lines, all must hold:
  `tar tzf cotd_code_data_x1khop2_4b.tgz | grep data/eval/khop2/` lists train_kd27.jsonl, train_kd_x3.jsonl,
  train_kh_x3.jsonl and build_report.json; `... | grep data/eval/khop/` lists the 12 parent files and build_report.json
  (md5s as section 1.1 after extraction); `... | grep -E 'condor/jobs_x1/(87[5-9]|88[0-9])_'` lists the 15 envs;
  md5 of the bundle checked on both sides.
- Builder checks before the bundle (CPU, venv_pfx python, python -m unittest, torchrun --local-addr 127.0.0.1):
  - tests/test_build_khop2.py: determinism by seed; refuses a changed parent md5; does not touch the parent files (md5
    before and after); train_kd27 starts with train_kd byte for byte; 18,000 new rows with k / order / id rules of
    1.2; no new row shares a value key or prompt hash with the eval files, train_kh, train_k1, train_kd or another new
    row; every CARD_khop_v1 1.1 rule (build_khop.row_rule_ok, skeleton, option rule) holds on every new row; key set =
    the gsmvt key set; copy files equal 3 x parent apart from the item_id suffixes; all item_ids distinct; D1-D4 and
    D6 recomputed from the rows equal build_report.json and pass the bars.
  - In tests/test_build_khop2.py: one A2 step on the tiny test model with GSM rows + a 3-copy extra file (the tests/test_extra_train.py 1-rank FSDP
    pattern, --eval-value-render vt) runs end to end and logs the extra-train line with 3 x the base row count (the trainer itself does not know about copies).
  - D5: the diff of section 2.5 against E/src_khop/repo, printed in the build report.
  - E/readout/test_khop2_table.py on synthetic tarballs: every verdict reading (MANIPULATION FAILED from each of I1-I5,
    PASS, PASS (approximate), NEGATIVE with and without the R3 sentence, NULL, each BETWEEN sub-label) and every branch
    (b) label (BOTH, ROWS, UPDATES, UNRESOLVED, and the one-level form without H3) reached by a fixture through the
    pooled rule (including a case where each single contrast has CI lower <= 0 and the pooled one has CI lower > 0);
    the BETWEEN "data-limited" versus "dose-limited" wording on R27; the PASS "chain data" versus "chain training"
    wording on the label; the NULL sentence prints the CI upper; each of the three GSM rc sentences (transfer, harm,
    no detectable change) reached by a fixture; the log-dose
    slope on a hand case; end to end through main() against the real parent tarballs plus synthetic D27 / R3 / H3.
  - E/readout/test_khop2_pilot_gate.py: a good pilot passes; each of P0-P9 fails alone (P6 by gsm8kx at reference
    - 11; P7 by a wall of 2.95 h); a pilot with valx at reference - 6 and gsm1kx at reference - 6 passes; P7b gives PASS, PASS (H3
    dropped) and the t bound at 2.36 h.
  - E/stage/test_prep_khop2.py: every env equals 766's env except the listed keys; names, XARGS, SAVE only on j875;
    sub differs from train_x1khop_4b.sub in exactly the CODE_TGZ, transfer_input_files and log lines (plus the header
    comment), and transfer_input_files names the same tgz as CODE_TGZ; smoke, pilot DAGs; prep_khop2_confirm.py refuses a
    non-PASS gate, writes 13 nodes on GATE PASS and 8 on "GATE PASS (H3 dropped ...)".
  - The full src_khop2 suite has no new failure against repo.base (the known test_tree_target and TestSbatch failures
    of the parent build report stay).
- One cold critic on this card before the build; one cold build critic on the diff, the data and the stage files before
  the bundle.

## Amendment 1 (2026-10-26 10:05 CDT, after the cold critic, before any build)

Critic: E/src_khop2/critique/CARD_CRITIC_khop2.md (verdict FIX). Nothing was built, bundled or submitted before this
amendment. The sections above are edited in place; this list says what changed and where. Pre-amendment copy:
E/src_khop2/critique/CARD_khop_v2.pre_a1.md. New numbers: E/src_khop2/critique/card_numbers.py (rerun, output
card_numbers.out). Plain values per seed: the critic's script scratchpad/kd_plain.py, rerun by the author.

1. FIX (high). Section 3, P6; section 5, pilot gate test. P6 is now a catastrophe line only: gsm8kx and gsm8kv2 (the
   larger sets) of j875 and j880 >= max(j520, j663) - 10. valx and gsm1kx are INFO. Reason: KD s16 valx is 58.0
   against 64.4 and would fail the old line, while N1 (2.6) says GSM does not block the verdict. KD s16 is -0.5 / -0.9
   on the two gated sets. Test: P6 fails alone at gsm8kx reference - 11; a pilot with valx and gsm1kx at reference - 6
   passes.
2. FIX. Section 2.6 (transfer line replaced), section 4 (readout), section 5 (fixed list, readout tests). Three-way
   sentence per arm (D27, R3, H3) and per cell rc1..rc4: transfer (CI lower > 0 against best-A2, N1 holds), harm
   ("chain training lowers GSM rc<j> by <x> [CI]", CI upper < 0) or "no detectable change". KD - best-A2 and
   KH - best-A2 at rc1..rc4 are printed next to it (parent, against the original A2 set: KD rc1 -7.3, rc2 -5.5). The
   harm line is reported only; no verdict depends on it.
3. FIX. Section 0 ("What is new", PASS, NEGATIVE, NULL, BETWEEN). D_dose is worded as a dose (27,000 rows at 1,029
   updates against 9,000 rows at 467 updates). The BETWEEN sub-label is "still data-limited" only when R27 has CI lower
   > 0; else "still dose-limited (rows and updates not separated; see branch (b))". The PASS claim says "with enough
   chain training", and "with enough chain data" only when the branch (b) label is ROWS or BOTH.
4. FIX. Section 0 (priors, power statement, branch (b) rule), section 2.2 by reference, section 5 tests. The label
   decides on the per-seed pooled contrasts rows_pool = (R27 + R9) / 2 and upd_pool = (U9 + U3) / 2 (paired, t 2.78).
   The four single contrasts and the two pooled ones are all printed. UPDATES is worded "repetition gains; no
   detectable effect of distinct rows (rows_pool CI upper <x>)"; it no longer claims equivalence. Pooled half-width
   about 6.8-11.0 points against 9.6-15.5 for a single contrast. Priors restated: UPDATES 0.35, BOTH 0.20, ROWS 0.10,
   UNRESOLVED 0.35. With H3 dropped, rows_pool = R27 and upd_pool = U9, "(one level only)".
5. FIX. Section 1.4; section 5, test_prep_khop2.py. The sub differs in exactly three lines (13 CODE_TGZ, 16
   transfer_input_files, 33 log) and the header comment. The test asserts these three lines and that
   transfer_input_files names the same tgz as CODE_TGZ.
6. FIX. Section 0, NULL. The sentence prints the D_dose CI upper ("do not exclude a gain of <CI upper> points").
7. FIX. Sections 1.2, 3, 4. The card states that the owner-named 2-epoch branch runs as R3 / H3 and that the literal
   EPOCHS=2 arms need a new card and a new id range (spares 890-899 stay re-runs only). Every message to the second author about
   this family carries that sentence.
8. FIX. Section 3, P7 bar 2.9 h per pilot job (2 x 2.9 + 0.1 = 5.9 GPU-h, under the cap of 6). The P7b sentence now
   reads main <= 23.2 at t <= 2.9; the H3 bound of about 2.36 h is unchanged. The gate test fails P7 at 2.95 h.
9. FIX. Section 3, compute. The 1,029-update wall is marked as an extrapolation (fit on 279 and 467 updates, used at
   2.2x the upper end); P7 / P7b measure the real wall before any confirmation node.
10. FIX (note only). Section 0, priors. Muennighoff et al. stays "not re-checked" and is marked "used only for the
    priors and never as evidence in the readout text". No decision depends on it.

Unchanged: one-pass rule, arms, files, generator rule and seed, envs and XARGS, schedule, job ids 875-889 (spares
890-899), SOLVED / FAILED bars, the order of the verdict readings, I1-I5, D1-D6, N1, the reference rule, P0-P5, P7b,
P8, P9, the confirmation node list and the compute caps.

## Amendment (Codex fix round, 2026-10-26)

Written before any round-F data; no result was seen and nothing was tuned on the 249 primary items.

1. CLARIFY. Section 2.6, transfer or harm, and section 2.3, GSM side. The rc4 cell is read on the 128 primary items
   that have an rc4 cell, not on all 249. Reason: pfx_table.py builds an rc4 cell only for an item with a hint row at
   r_path 4 (128 of the 249 primary items; the other 121 have cp 4 or 5 and no rc4 row). The unchanged eval files cannot
   supply rc4 for the other items. This is the same population as the parent readout (khop_table.rc_cells) and the
   PFX readouts. rc1, rc2 and rc3 stay on all 249 items. The readout prints the four populations, stops (UNREADABLE)
   when a population size differs from 249 / 249 / 249 / 128, and stops when a run misses any item of a population.
   This line has no verdict; no bar, reading or arm changes.
2. NOTE (code matches card, no change of rule). The readout decides every reading, label and I-line on exact counts
   (Fraction(correct, items), CI compared in squared form with t = 2.78). A float rounding can no longer move a cell
   across the 0.80 / 0.70 bars or a pooled contrast across 0. A decision arm without exactly seeds 16-20, or with an
   incomplete preds file, stops the readout instead of shrinking the denominator.
3. NOTE (code matches card). The seed and order_seed recorded in each run's metrics.json must equal the seed of its
   label: s16 for the two pilots and both pilot references (gate lines P1 and P6), and s16-s20 for every decision arm
   and both A2-6k reference sets in the readout (else UNREADABLE). A wrong tarball path can no longer pass the gate or
   enter a paired contrast. No bar changes. The references must also be the jobs of section 2.4: the gate's P6 needs
   j520 and j663 as two distinct files; the readout needs the five original and five re-train jobs by seed. The
   readout stops (UNREADABLE) without the hint files, since G2 / G3 and the rc1-rc4 lines need them.

## Submission note (2026-10-26 16:52 CDT, sev-e0)
Codex CLI review VERDICT: SUBMIT (critique/CODEX_REVIEW_khop2.r6.md). Bundle cotd_code_data_x1khop2_4b.tgz md5 31f339b9db277d1776117f1ba5df7d6e (pin x1khop + "-khop2"); pre-submit lines: data/eval/khop2 4 files, data/eval/khop 13 files, envs 875-889 15 files, train_kd27 / kd_x3 / kh_x3 md5 as registered; md5 equal on the AP. Step 1 smoke_x1khop2_4b.dag (j875 D27, j880 R3, smoke=1); step 2 pilot_x1khop2_4b.dag after the smoke lands with exit 0.

## Outcome (2026-10-27 01:57 CDT, sev-e0)
VERDICT PASS by script: k*(D27, ca, fwd) = 4; (ca, fwd, 3) 0.907 [0.889, 0.924]; branch (b) BOTH; GSM rc2 -7.6 vs best A2. Adversarial check pending. Details: RESULTS_LOG "khop2 five-seed readout".
Adversarial check (02:09 CDT 2026-10-27): HOLDS WITH CAVEATS. Order-sensitive items 0.910 vs 0.151 for the order-free sum; no leakage; scope limits (cag 0.58, wp k* 3, values 1-99); GSM rc2 -7.6 reported with the claim. Erratum (fact, no rule change): the "x2 / x3 steps do not decide the answer" premise holds for 119 of 150 (ca, fwd, 3) items, not all.
