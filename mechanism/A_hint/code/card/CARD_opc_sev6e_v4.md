# Pre-registration card: OPC v4, oracle-prefix capacity of one forward pass (2026-10-21, sev-6e)

AMENDMENT 2026-10-21 20:45 CDT, after Stage 1 (jobs 500-513, DAG 6906464) and before any Stage 2 job: the Stage 2
content pair is GFPQ-c2 / GFPQR-c2 (yes/no nodes, ARMS GFPQ / GFPQR, XARGS "--tree-sources gsm8k_tf --eval-value-render
vt", TAG c2lr3, as card B) instead of GFPQM-c2 / GFPQMR-c2, because GFPQM-c4 failed M2 (placebo - plain -9.1) and the
read check (-22.5) in Stage 1, so a GFPQM pair could not be read. P3 bar and seeds unchanged. Stage 1 gate: PASS on
A2-c4 (seed mean) and A2-c12, so Stage 2 runs without SIB (SIB env files kept, jobs_opc2_sib_4b.dag, conditional).
Stage 1 readout: opc/stage1_readout_20261021.txt; log entry: opc/results_log_entry_j500_513.md.


## 0. Hypothesis (one sentence, one primary)

On cue-free gsm8kx items with n >= 4 worked steps whose shown values equal no option, writing the first n-1 true steps
into the prompt raises the accuracy of the answer-only student (A2 at its valx-chosen epoch, seeds 16-18) by at least
15 points, every seed by at least 10.

## 1. Why this is the most informative next experiment

- Every ledger line that tried to make one pass compute intermediate values (LS, LSLI, TSF, PTAD) or use node
  supervision (GFPQ, GFPQVT) failed its content control. Nothing measures what one pass can do when the values are
  supplied as text, nor how many steps of remaining work it finishes. All depth evidence is between items.
- Stage 0 artifact (opc/stage0_force_depth.txt, j303 / j307 preds): forced teacher STATES for steps 1-2 (TSF) change
  gsm8kv2 accuracy by +3.9 (d2), +4.7 (d3), -0.3 / -2.2 / +1.1 (d4-d6+); TSF6 (answer step included) 92-98% at d2+.
  Supplied states help when about one step remains. One seed, state channel.
- Not already answered (critics' R1): the A2 depth profile (75.7 at d1-2 vs 39.6 at d5) is between items; card B's
  +0.42 is node supervision without supplied values; TSF is a 2-slot state channel. None gives H or the within-item curve.
- Waste condition: H < 5 and D < 5 with the checks passed. The ledger does not give this result.

## 2. Claims under each outcome (GSM8K test with worked steps, cue-free options; gsm1kx has no steps)

- PASS (H and D >= 15): "A single pass is a one-step machine on GSM8K: given the first n-1 results as text, an n-step
  item is solved like a one-step item; the student cannot produce those results itself (LS / TSF write side)."
  Survives MC cues (x options untouched), epoch (valx rule), format regularisation (controls), selection seed (s12-14
  pilot only), depth (within-item curve).
- NONE (H < 5, checks passed): "Even with every earlier result written and used (M3, M3b, M4), the student does not
  finish the last step in one pass. A latent-value method that only supplies step values to a single MC pass has no
  ceiling to chase at 4B; recurrent compute that also performs the last step is untested here."
- Checks fail on the Stage 2 students, pass on base (base valid): "Answer-only fine-tuning closes the in-context text
  channel for worked steps; a student must be trained to read them (SIB)." Made only from Stage 2 (seeds 16-18), never
  from the pilot. Stage 1b then measures H on SIB-c2.
- Checks fail on base and students (base valid): "Neither the post-trained base model nor an answer-only student
  uses a worked-steps block in this MC format at 4B in one pass; the text channel needs training (SIB)." No H.
- Base invalid (plain accuracy < 40 on the 1,129 items without the chat template): the base row decides nothing;
  "format not measurable without training"; SIB-c2 is the reader.
- P3 (secondary, seeds 16-18): positive -> "node-content supervision helps only inside the one-step capacity";
  negative -> node content closed inside capacity too. GFPQM-c2 / GFPQMR-c2 are a treatment / control PAIR for this
  secondary question; non-goal 6 allows old arms only as named controls, so this pair needs the second author's leave or P3 stays
  a Stage 1 pilot reading.

## 3. Arms

Stage 1, PILOT, evaluation only (saved weights; listing opc/staging_weights_used.txt, 13 tars verified 17:30): base
(models/Qwen3.5-4B); A2-c4 s12-14 (83, 84, 85); A2-c12 s12 (41); GFPQ-c4 s12-14 (42, 70, 73); GFPQM-c4 s12-14 (164,
172, 173); GFPQMR-c4 s12-14 (165, 174, 186). Seeds 12-14 decide go / no-go and the checks; no confirmation claim.
Stage 2, CONFIRMATION, training jobs with the 18 hint sets in EXTRA_EVALS (train_arm.sbatch :268-269), fresh seeds
16, 17, 18 (card B used 13-15), data student_gsm8k_tf_vt, NGPU=1 in every env file (as batch 14 ran): A2-c1 and A2-c2
(the P2 student; valx picks the epoch), GFPQM-c2 = ARMS=GFPQ with XARGS "--tree-sources gsm8k_tf --node-format mc
--eval-value-render vt", TAG c2lr3mc (as the saved GFPQM runs 183 / 191 / 192), GFPQMR-c2 = ARMS=GFPQR with the same
XARGS (as 165). The mc node format with --eval-value-render vt on _vt data has not run before: smoke_opc2_4b.dag
smokes it (and SIB) first. Env files built: opc/stage2_build/condor/jobs_opc2 (15, ids 360-374 placeholders),
jobs_opc2_4b.dag, build_stage2_bundle_ap.sh (batch-14 fsdp1 tree + hint sets, built on the AP by the owner). A fresh
replication of card B's A2 rows, not paired with 311-325. The A2 jobs run in every branch except "fail on all three".
Stage 1b, conditional: SIB-c2 at s16-18 (train_arm.sbatch :352, --si-remove back: the block shows Step 1 .. Step n-m,
the hint prefix format; SISTAGES=4; XARGS="--tree-sources gsm8k_tf --eval-value-render vt" as env 313; SI jobs 161 /
210 ran on this data with tree_sources gsm8k_tf). Its last SI stage trains with every step removed (si_removed), so
SIB may not read either: M1-M4 are read on SIB-c2 too, and if they fail there the study ends with the format claim.
Pre-registered cost: SIB-c2 69.5 vs A2-c1 70.5 pooled original pools at s12 (scout note). Env files are in the same
Stage 2 set (3 of the 15).

## 4. Data and eval sets (built and tested: opc/build_hint_evals.py, opc/test_build_hint_evals.py, 11 checks)

Source: gsm8k_mc_v2_xprob.jsonl (cue-free, 1,317) joined to gsm8k_mc_v2_steps.jsonl by test index; equal question text
asserted (1,317 pairs, 0 mismatches). Items with n >= 2 steps: 1,129 (n = 2: 252, 3: 301, 4: 244, 5: 174, 6: 83,
7+: 75). Options byte-identical to the x options (tested): prompt interventions on an existing set (non-goal 1).
18 sets in data/eval/hint (MANIFEST.json, md5 per file):
- hint_true_r<r>: first k = n - r true steps as "\n\nWorked steps:\nStep 1: 5 - 2 = 3 ..." before "\nOptions:\n"
  (chain_block :1557, insert_chain :1564); r = 1: 1,129; 2: 877; 3: 576; 4: 332; 5: 158; 6: 75. r >= 7 (44) not built.
- hint_full: all n steps (answer step included), 1,129. Sanity ceiling.
- hint_placebo_r<r>, r = 1..4: k sentences of the item's own question (student._pl_sentences; placebo_lines :3481
  raises on gsm8k_v2 items). Length / position control at every r of the curve.
- hint_other_r<r>, r = 1, 2: k lines of another item's chain (teacher_partners :3490). Value-distraction control.
- hint_wrongres_r<r>, r = 1, 2 (1,111 / 861): a CONSISTENT wrong chain: every result wrong by at least one tier (the
  operator slip when it gives a clean number, else +-max(2, 10%)), every operand linked to an earlier step (a_from /
  b_from) rewritten to that step's wrong value; no true result of steps 1..k appears in the block (tested per line).
  18 / 16 items with no valid far-wrong chain are dropped from this set. The M3 control.
- hint_wrongres_near_r<r>, r = 1, 2 (1,129 / 877): true operands kept, results off by +-1..2. Secondary.
- hint_plain: the 1,129 items unchanged.
Fields: the x item's fields plus group (kind, n, r), n, r, k, kind, shown, shown_equals_option (ANY shown value
equals an option text), partner, wrong_values. Clean = not shown_equals_option (true r = 1: 918 of 1,129; n >= 4 at
r = 1: 440 of 576). In 4 far-chain rows (test-44, -210, -227, -505) a true result stands as an unlinked operand that
is not a question number; they are reported, not dropped (1 of them has n >= 4). Slices by n (the depth label differs
from n on 332 of 1,152 chained items).

## 5. Metrics (opc/hint_table.py, opc/test_hint_table.py pass)

- Accuracy (argmax over letters); the unit of every test is the seed.
- PRIMARY P2: H = acc(true r = 1) - acc(plain) on the clean items with n >= 4 (440; plain on the same items), A2 at
  its valx epoch, s16-18. The unfiltered H (576) is reported and decides nothing.
- Secondary P1: on items n >= 5 (332): D = acc(r = 1) - acc(r = 4); the curve acc(r); the shift check acc(r | n >= r+3)
  vs plain acc(n = r); placebo at the same r.
- Secondary P3: GFPQM-c2 - GFPQMR-c2 on the 2,006 r in {1, 2} true variants, s16-18, paired t, df 2.
- Reported: copy rate (share of predictions equal to the last shown value where it is an option; a pure copier scores
  100%, tested), every curve, GFPQ-c4 - A2-c4, A2-c12, base, sensitivity rows.

## 6. Epoch rule, seeds, n

A2 epoch in Stage 2: the 3-seed valx mean over s16-18 (1 or 2, tie -> 1; the card-B rule). GFPQM / GFPQMR at 2 epochs.
Stage 1 epochs are the saved weights'. Confirmation n = 3 (s16-18), t_crit 4.30. Seeds 12-14 never pool with Stage 2.

## 7. Kill lines and decisions (before data)

Checks, read on base, A2-c4 and A2-c12 in Stage 1 and again on the Stage 2 A2. Three-seed arms are judged on the
SEED MEAN (hint_table "seed-mean gate"; false-stop rate of M2 below 1%); single models (base, A2-c12) on their own
run, whose false-stop rate at r = 4 (332 items) is 2-9% for a 15-25% placebo-plain disagreement rate, stated with
the result.
- M1 acc(full) >= 90. M2 |placebo - plain| <= 3 pooled over r = 1..4 AND <= 5 at every r (hard stop, no pass-through).
  M3 acc(true) - acc(wrongres, far chain) >= 10 at r = 1 on the 1,111 shared items (no-leak check; a far chain the
  model discounts as noise makes M3 equal to H, which is why M3b exists). M3b acc(true) - acc(wrongres_near) >= 10
  at r = 1 (value-following check on a plausible chain). M4 acc(true) - acc(other) >= 10 at r = 1 on the shared
  items. Read check on n = 2 items: true r = 1 minus plain >= +5 = reads; below +2 while base >= +5 = unlearned.
  Base validity: base acc(plain) >= 40.
- Stage 2's A2 jobs run in every branch except "fail on all three" (the pilot never halts the study otherwise). The
  pilot decides only whether Stage 1b (SIB-c2) is added: added when the checks fail on both A2-c4 and A2-c12 and the
  base is valid. If the checks then fail on the Stage 2 A2 too, the branch-3 / branch-4 claims apply and H is read on
  SIB (after SIB's own checks).
- P2 is read only if M1, M2, M3, M3b and M4 hold on the Stage 2 A2 seed mean; otherwise the claims of section 2
  branches 3-5 apply to Stage 2, never to the pilot.
- P2: PASS mean H >= 15 AND every seed >= 10 AND paired t >= 4.30. NONE mean < 5. Between: limited, reported.
- P1: D >= 15 capacity-limited; D < 5 not step-limited (depth theory fails here); between: report.
- P3: mean >= +1.5 AND all seeds > 0 AND t >= 4.30 -> bounded claim; mean < +0.5 -> closed; between: report.

## 8. Controls

Length / position: placebo r = 1..4 (M2). Value use vs copying: far-chain wrongres (M3), the near-miss set
(secondary), clean-only primary, copy rate, r = 2 apart. Value distraction: other-item lines (M4; a drop is reading,
not a confound). Content: GFPQMR. Epoch: valx rule; GFPQM / GFPQMR aligned. MC cues: x options only. Base row: the
format without training; the read check separates "cannot read" from "unlearned".

## 9. Code (no training-code change; all of it exists and its tests pass)

- opc/build_hint_evals.py, run as: PYTHONPATH=<repo>/src <venv python> build_hint_evals.py --steps <bundle>/data/eval/
  gsm8k_mc_v2_steps.jsonl --x <bundle>/data/eval/gsm8k_mc_v2_xprob.jsonl --out data/eval/hint, where <bundle> =
  .../b14_x/inner (the data sits next to repo/, not inside it). Tests: test_build_hint_evals.py <bundle> <builder>
  (11 checks: never the last step unless full; exactly k numbered lines; prompt identical outside the block; options
  byte-identical; near-miss and far-chain rules per line; partner; placebo sentences; the flag; counts; deterministic
  md5; the step-text formatter reproduces all 4,343 step texts). Run as scripts with their arguments; under pytest
  collection they exit with their usage text instead of failing.
- opc/hint_table.py + test_hint_table.py: set accuracies, curve, M1-M4 and M3b per run and as the seed-mean gate per
  arm (a 20% random-noise test must not stop it), read-check verdict, H clean and unfiltered, D, copy rate, seed rule
  for P2, P3 pairs, WAITING on missing sets; tgz or directory input.
- Evaluation: scripts/eval_student.py (--model, --eval, --out, --bs; preds carry item_id, group, labels, probs,
  gold_label) through ~/cotd_chtc/run_eval_iv.sh + train_iv_4b.sub (jobs 276-280: 15,000 prompts in 189-317 s).
  No smoke stage exists; the base job (5 min) is the smoke and M1 on it proves the sets load.
- Stage 1 package: opc/stage1_build (bundle cotd_code_data_hint_4b.tgz md5 8487cf14, 14 env files, DAG ids 340-353,
  train_hint_4b.sub, README_stage1.md). Stage 2 package: opc/stage2_build (15 env files, jobs_opc2_4b.dag,
  smoke_opc2_4b.dag, build_stage2_bundle_ap.sh). Stage 0: opc/stage0_force_depth_ap.py.

## 10. Compute

Stage 1: 14 eval-only jobs, 5-6 min each, about 1.5 GPU-h; the smallest pilot is base + A2-c4 s12 (10 min).
Stage 2: A2-c1 x3 (25 min), A2-c2 x3 (40 min), GFPQM-c2 x3 and GFPQMR-c2 x3 (about 75 min): about 11 GPU-h, 1 GPU
each. Stage 1b: SIB-c2 x3, about 1.5 h each. Total at most about 17 GPU-h; decision within 2 days.

## 11. Risk register

1. Copying the last shown value: x options, clean-only primary, copy rate, M3 on a chain that leaks no true value,
   r = 2 apart.
2. Length or position read as headroom: placebo at every r, pooled and per-r bars.
3. Overfit 4-epoch pilot models: Stage 1 decides go / no-go only; every claim comes from Stage 2 at the valx epoch.
4. Format never seen by the student: base row with a validity gate; read check with thresholds; SIB-c2 as reader,
   itself checked (its final SI stage shows zero steps); each branch has a pre-registered claim.
5. M3 failing because the model recomputes from leaked operands: blocked by the far chain (operands rewritten).
6. gsm8kx vs gsm1kx: GSM8K-test only, stated in every table. Depth: slices by n.
7. Other-item lines hurt because values are read: pre-registered as M4.

## 12. Job naming

Stage 1: <id>_4b_hint_<tag> (base, a2c4s12 ..., gfpqmrc4s14, a2c12), env condor/jobs_hint/<id>_4b_hint_<tag>.env
(MODEL, WTAR, SETS, JOBKEY). Stage 2: <id>_4b_gsmvt_<ARMS>-<TAG>_s<seed>_fsdp as batch 14 (A2-c1lr3ch, A2-c2lr3ch,
GFPQ-c2lr3mc, GFPQR-c2lr3mc, SIB-c2lr3ch; seeds 16-18). Ids assigned by the CHTC owner (sev-42; sev-20 holds 326+).
Submission only on the second author's word.
