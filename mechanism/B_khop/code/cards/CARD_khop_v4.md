# Pre-registration card: keep the installed k-hop capacity through the GSM stage, then read transfer (family khop4, parent khop3), 2026-10-27, sev-e0

Status: DRAFT 2026-10-27 20:51 CDT, written before any build, bundle or submission. No khop4 data exists. Amended by
Amendment 1 (end of the card, after the cold critic E/src_khop4/critique/CARD_CRITIC_khop4.md, before any build) and
Amendment 2 (2026-10-27 22:22 CDT, the builder, during the build; no bar, arm, id, schedule or reading changes); the
sections below are updated in place.
Owner's words (2026-10-27 20:51 CDT, RESULTS_LOG "Lead decisions after round G"; the second author: "继续" to the four
recommendations). Lead decision 3: "khop4 gets a new card: follow-up (c) of CARD_khop_v3. Stage 2 starts from the j975
(CUR) and j985 (K1X) weights and has two arms: a lower stage-2 LR; GSM rows interleaved with a chain replay share. Ids
are 1025-1049. The process is the same as round G." Lead decision 1: the S1 weight tars 975 and 985 stay on staging as
the warm start for khop4.
Family khop4, job ids 1025-1049 only (4 digits). Code copy E/src_khop4/repo (untouched snapshot E/src_khop4/repo.base;
both equal E/src_khop3/repo today, checked with `diff -rq -x __pycache__`), variant x1khop4, bundle
cotd_code_data_x1khop4_4b.tgz. E = <path> (the old scratchpad path .../scratchpad/exp is a symlink
to it; files may carry either form).
One-pass rule (MILO_RULINGS_2026-10-23, tier ii): every arm is plain A2 training (letter CE after "Answer:"). The
curriculum is two ordinary A2 training jobs in sequence (chain rows, then a GSM stage on the saved weights). No new
mechanism, no generated tokens (G0), no recurrence; executed layers 32 of 32, one forward per item, FLOPs and latency
equal to A2 for the same prompt length. No trainer change (D5).
Numbers of this card: E/src_khop4/critique/card_numbers.py, output card_numbers.out (walls by E/jobs/round_gpuh.py on
the parent tarballs; plain sets, training histories and the khop3 gate log). That script reads no rc cell and no hint
set of the 249 primary items.

## 0. Question, hypothesis and pre-registered readings

### 0.1 What khop3 found (n = 1, s16, exp/logs/khop3_pilot_gate_20261027.txt)

| model | (ca, fwd) k1 / k2 / k3 / k4 | gsm8kx / gsm8kv2 / valx / val | note |
|---|---|---|---|
| j975 CUR1 (27,000 chain rows, 844 updates, LR 3e-6) | 100.0 / 100.0 / 92.7 / 85.3 | 38.9 / 37.8 / 38.9 / 39.7 | installed; GSM letter readout far below A2 |
| j980 CUR (j975 + A2 GSM epoch, 186 updates, LR 3e-6) | 100.0 / 92.0 / 64.0 / 64.7 | 61.7 / 71.3 / 63.1 / 67.9 | P9 FAIL: "installed, then lost" |
| j985 K1X1 (K1 twins, 844 updates) | 100.0 / 20.7 / 15.3 / 20.7 | 36.9 / 39.7 / 37.4 / 38.9 | control trains (P11), chance at k >= 2 |
| j990 K1X (j985 + A2 GSM epoch) | 99.3 / 54.7 / 23.3 / 24.0 | 56.7 / 68.8 / 63.1 / 63.1 | P10 +40.7 at (ca, fwd, 3) |
| s16 reference mean(j520, j663) | - | 59.42 / 70.96 / 64.0 / 69.0 | best-A2 rule at n = 1 |

Two facts decide this card's design (card_numbers.out sections 6-8):
- Stage 2 must do two jobs at once. After S1 the plain GSM sets sit near 38 (5 options, chance 20). The A2 epoch
  restores them, and fast: at update 93 of 186 (81% of the schedule's LR mass), j980 already has gsm8kx 62.9 and val
  64.4. In the same epoch (ca, fwd, 3) falls from 92.7 to 64.0.
- The khop2 grid shows that chain rows inside the GSM epoch keep the chain capacity (D27, interleaved, 0.907 at
  (ca, fwd, 3)), and that a one-third chain share from the base model keeps plain GSM (KH: 3,000 chain rows + 5,923 GSM
  rows, 279 updates; N1 holds at five seeds, +1.2 to +2.3 points over the N1 bar on the four plain sets; card_numbers.out
  section 10).

### 0.2 Question

Can the GSM stage keep the installed capacity (k*(ca, fwd) >= 3 at the final model) and keep GSM intact (N1), and if
both hold, does the capacity raise GSM rc2 / rc3 on the 249 primary items against the best A2 set and against a matched
single-hop control? This is CARD_khop_v3's question. khop3 could not read it because its premise failed at s16.

What the confirmation buys (Amendment 1, finding 1). Under the most likely truth (no transfer, D_2 near 0) the primary
most likely prints BETWEEN "unresolved": at the parent paired SD 3.3 the reading is "unresolved" 0.63 and NULL 0.32;
at SD 4.3, 0.75 and 0.20 (card_numbers.out section 11). Over all outcomes the prior is "unresolved" 0.39 (section 14).
The confirmation is run for the readings that do not depend on D_2, and a five-seed D_2 is a by-product:
1. The manipulation line at five seeds: whether a stage-2 protection keeps k*(ca, fwd) >= 3 at the final model with N1
   intact (I5 and N1 for CUR-a). No khop line has that model: KH and KD hold N1 with k* 2; D27 and R3 have k* >= 3 and
   fail N1; khop3 lost the capacity at s16. A pass is the precondition of every transfer card on this route; a
   "lost" or "GSM damaged" at five seeds closes the sequential route at this scale (0.5 failure text).
2. The forgetting line (2.2) with CI: how much of the installed capacity each k loses in the GSM stage under the
   protection.
3. Arm R: CUR-R - KH and K1X-R - K1 at a fixed stage-2 recipe (2.3), the effect of stage 1 at zero extra compute.
4. D_2 with its CI. "Unresolved" still bounds the gain by the printed CI upper. An extension on new seeds would be a new
   card with its own pooled rule (as verifyext); this card does not pre-register one.
The statistical rule of 0.7 is not changed to make NULL easier to reach. This card registers 1-3 as the value of the
confirmation (about 18.7 GPU-h for R, 16.6 for L, by estimate), so section 4 stays automatic on GATE PASS ARM R / L. The
acceptance is decided once, before the build: the lead's go on this amended card (a RESULTS_LOG line) is the
acceptance; a rejection withdraws the card before any build or data, and no decision about the confirmation is taken
after the gate.

### 0.3 Arms (two stage-2 protections, each with its K1X twin)

- L (lower stage-2 LR). Stage 2 = the khop3 S2 env (the A2 GSM epoch, 5,923 rows, 186 updates, warmup 0.05 + cosine)
  with LR 7.5e-7 instead of 3e-6. One value, fixed now. Rule (Amendment 1, finding 3): the largest factor in
  {1/2, 1/3, 1/4} whose projected a(CUR-L, ca, fwd, 3) clears the I5 SOLVED mean bar 0.80 (= the a-Q9 bar 120 / 150)
  by at least 0.04 (6 / 150 items, about 1.3 binomial SDs at 0.84; a plain choice, stated here). Projection: forgetting
  linear in the stage-2 LR mass (sum of the per-update LR), anchored at the one s16 drop 92.7 -> 64.0. 1/2 gives 0.783
  (margin -0.017), 1/3 gives 0.831 (margin +0.031), 1/4 gives 0.855 (margin +0.055). So L uses 1/4: LR 7.5e-7
  (card_numbers.out section 12). This is a one-point extrapolation, not a measurement: forgetting that is fast early
  (as the GSM recovery is) would give less retention than the projection.
  The cost side, written as a prior: L's total LR mass is 0.250 of A2's and 0.309 of the mass A2 spends by update 93.
  At update 93 j980 had gsm8kx 62.9 (above its final 61.7) but val 64.4 and gsm8kv2 68.3 (3-4 points below its final
  67.9 / 71.3), so the GSM recovery was well advanced, not complete. No A2 run at LR 7.5e-7 exists in this project's
  recorded x1 envs (stage and bundle_out condor/jobs_x1: LR=3e-6 or 1e-5 only, checked 2026-10-27), so the GSM recovery
  of L is unmeasured. L therefore trades "installed, then lost" against "GSM damaged (not recovered)"; at 1/4 the second
  risk is larger than at 1/3, and this is accepted so that L is not designed to land at the a-Q9 / I5 edge.
- R (chain replay share). Stage 2 = the khop3 S2 env with the parent KH replay appended:
  `--extra-train data/eval/khop/train_kh.jsonl` for CUR-R and `--extra-train data/eval/khop/train_k1.jsonl` for K1X-R.
  LR 3e-6 (A2's). Rows 5,923 GSM + 3,000 chain = 8,923, 279 updates. The share 3,000 / 8,923 = 0.336 is fixed now.
  Reasons:
  - Share rule (Amendment 1, finding 2; plain sets, khop cells and compute only, no rc cell): the smallest existing
    replay file whose rows are stage-1 rows and whose from-base run holds N1 at five seeds. KH (3,000) holds N1 with the
    smallest margin +1.17 points (gsm1kx); KD (9,000; train_kd.jsonl = rows 0..8,999 of train_kd27, md5
    2ed4ecc377e4638e35b12e8b22e37461) also holds N1, with the smallest margin +0.65 (gsm1kx; valx +1.13 against KH's
    +2.15); D27 fails N1 (card_numbers.out section 10). KH is chosen over KD because its N1 margins are larger on three
    of four sets, the chain share stays a minority (0.336 against 0.603, so GSM rows remain two thirds of the rows), and
    it needs 279 updates against 467 (about 0.8 h against about 1.0 h per S2 job). The rc cells of KH / KD did not decide
    the share (INFO: they are printed in the parent readouts and are not used here).
  - The replay rows are stage-1 rows, byte for byte: train_kh.jsonl (md5 a16b1520f0fb8b24ffd8fa57ef696de0) equals rows
    0..2,999 of train_kd27 (`head -3000 train_kd27.jsonl | md5` = a16b1520..., checked 2026-10-27), and train_k1.jsonl
    (4bb15f66c8d8843013eb0410c6a87e48) equals rows 0..2,999 of student_khop3_k1x/train.jsonl (khop3 D2;
    `head -3000 ... | md5` = 4bb15f66..., checked 2026-10-27). Both are
    balanced over task (ca / wp / vb) and k (1..6), 166-167 rows per cell. No new data file is needed for R.
  - The GSM rows keep A2's LR exposure: with the 279-update cosine, the GSM rows' expected LR mass is 0.994 of A2's
    (card_numbers.out section 7), so R does not under-train GSM the way L may.
  - Updates: 279, not 186. Keeping 186 updates would need dropping 3,000 GSM rows (the A2 epoch would see half of its
    rows, and N1 and the best-A2 reference assume the full epoch) or a replay of 0. The 93 extra updates are chain rows.
    The K1X-R twin has the same 279 updates, so C_2 = CUR-R - K1X-R holds updates fixed; against best A2 the extra
    updates are part of the treatment (as in khop3, whose S1 added 844).
  - The GSM rows are shuffled with the chain rows, so the GSM batch order is not A2's. The parent KH run (j756,
    j758-761) is the same stage-2 recipe from the base model: CUR-R - KH holds the stage-2 recipe fixed and isolates
    stage 1 (section 2.3).

Why both arms at the pilot and one at the confirmation. The ids 1025-1049 hold 25 jobs. One confirmed arm needs 20
(4 pilot S2 + 8 S1 + 8 S2), two need 28 (card_numbers.out section 4). So the pilot runs both arms at s16 (S2 only, from
the 975 / 985 tars), and the gate sends at most one arm to the confirmation by a fixed rule (0.5). The other arm's
confirmation S2 (8 jobs; L about 4.4 GPU-h, R about 6.5) is the named follow-up: it reuses the 8 confirmation S1 tars
(SAVE=1), so it needs only new ids and a lead decision, not a new S1.

### 0.4 Prior expectation

khop3's j980 is direct evidence that the A2 LR overwrites the capacity in 186 updates. D27 (interleaved, 0.907) is
direct evidence that chain rows inside the GSM epoch keep it. CARD_diag_v1 and CARD_b1_v1 point to binding / premise
selection, not serial depth, as the rc2 limit; Wang et al. (NeurIPS 2024) find that learned composition does not
generalize out of distribution. The chain format is far from GSM wording (cag 0.58 at k = 3 in khop2).
Pilot priors (Amendment 1: a-Q9 bar 120 / 150, L at LR 7.5e-7): R passes the gate about 0.45 (installation kept about
0.60, N1 about 0.80); L passes about 0.25 (installation kept about 0.65, N1 about 0.40; these two are anti-correlated
through the LR); neither passes about 0.40.
Five-seed priors, given a confirmation (card_numbers.out section 14: the judgment weights are stated there, the
reading rates come from the section 11 simulation at the KH paired SD 3.3): MANIPULATION FAILED about 0.23 (of which
"installed, then lost" about 0.10, "GSM damaged" about 0.10); PASS about 0.09; BETWEEN "extra updates" / "control
invalid" about 0.04; BETWEEN "gain below the bar" about 0.02; NULL about 0.18; NEGATIVE at most 0.05; BETWEEN
"unresolved" about 0.39. Given that the manipulation holds, "unresolved" is the most likely reading (about 0.51).

Hypothesis H4T (transfer, as H3T of CARD_khop_v3). In a model with k*(ca, fwd) >= 3 at the final model and plain GSM
intact, GSM rc2 on the 249 primary items is at least 3 points above best A2, and above the matched K1X twin of the same
stage-2 protection.

### 0.5 Pilot outcome and the arm rule (deterministic, fixed now; reads no rc cell and no hint set)

An arm "passes" when every decision line of its block (section 3: a-Q1 ... a-Q6, a-Q7b, a-Q9 ... a-Q11; the INFO line
a-Q7 is not read) prints PASS and every common line
prints PASS.

| R block | L block | gate prints | confirmation | what it says (n = 1, s16) |
|---|---|---|---|---|
| pass | pass | GATE PASS ARM R | R | both protections keep the capacity and GSM at s16; R goes first (reasons below); L's s16 pair is INFO, its confirmation S2 is the named follow-up |
| pass | fail | GATE PASS ARM R | R | replay keeps the capacity and GSM; the lower LR alone does not (the failing L lines name which side) |
| fail | pass | GATE PASS ARM L | L | the lower LR keeps both; the one-third replay does not (failing R lines name which side) |
| fail | fail | GATE FAIL: <lines> | none | see the failure readings below |
| any | any, with a common line failing | GATE FAIL: <lines> | none | not run as registered, or the s16 S1 premise or a smoke failed |

Why R first when both pass (fixed before data):
1. R has a five-seed from-base twin for each of its stage-2 recipes in E/results: KH (j756, j758-761) for CUR-R and K1
   (j757, j762-765) for K1X-R, same env except MODEL, SAVE and the run name. So CUR-R - KH and K1X-R - K1 read the
   effect of stage 1 at a fixed stage-2 recipe at zero compute. L has no A2-at-7.5e-7 run from the base model, so for L,
   D_2 against best A2 mixes the LR change with the warm start (C_2 is clean for both arms: each twin shares its LR).
2. R keeps the A2 LR exposure of the GSM rows (0.994), so D_2 is not biased down by an under-trained GSM stage.
A fixed order is used and not a margin comparison, so that the pilot seed selects nothing among arms; s16 stays one
of the five seeds of the primary (n = 5, t_crit 2.78), as in khop3. The gate reads no rc cell or hint set. Residual
effect, reported: the five-seed installation line I5 includes s16, which passed the gate's installation line by
construction; I5 is also printed over s17-20 alone (n = 4, INFO). The same holds for N1: s16 passed a-Q6, so N1 for
CUR-a and for K1X-a is also printed over s17-20 alone (n = 4, INFO; Amendment 1 finding 4). The gate's installation line
has an accepted edge-zone error at n = 1: an arm whose five-seed mean at (ca, fwd, 3) is 0.78-0.80 passes a-Q9 with
probability 0.37-0.50 (a-Q9 in section 3, card_numbers.out section 12); the five-seed I5 then decides.

Pilot failure readings (n = 1; pilot observations, not five-seed readings; each printed by the gate from its lines):
- "installed, then lost (both arms)": both a-Q9 lines fail while the s16 S1 premise (common line C2) passes. Reading:
  at s16 neither a quarter of the LR nor a one-third replay keeps 3-step composition through the GSM stage. Follow-up (c)
  is closed at this dose and these settings. The sequential curriculum cannot read the transfer question at this scale;
  the interleaved route (khop2 grid) pays GSM cost in proportion to the dose. No further khop card without the second author.
- "kept, GSM damaged" (per arm): a-Q9 passes and a-Q6 fails for CUR-a. For L: "GSM not recovered at LR 7.5e-7"; the
  capacity and the GSM recovery compete for the same LR mass. For R: the replay costs GSM after a chain stage although
  KH kept GSM from the base model.
- "lost, GSM intact" (per arm): a-Q9 fails and a-Q6 passes. For L: a quarter of the LR still overwrites the capacity, so
  forgetting is fast early (the linear projection was wrong). For R: a one-third share does not maintain the capacity.
- "control invalid" (per arm): a-Q10 or a-Q11 fails (the twin reaches the same k = 3 accuracy, or loses single-hop
  reading). The arm cannot reach PASS, so it does not go on.
GATE FAIL: nothing else is submitted; report to the second author with the failing lines.

### 0.6 Notation (as CARD_khop_v3)

a = the confirmed arm (R or L). a_X(t, o, k) = accuracy of model X on khop task t, premise order o, chain length k, one
value per seed. CUR1 / K1X1 = the stage-1 models (s16: j975 / j985 from khop3; s17-20: j1029-1032 / j1033-1036).
CUR-a / K1X-a = the final models of arm a. m = five-seed mean; CI = m +/- 2.78 x SD / sqrt(5) (df 4, seeds 16-20);
t = m / (SD / sqrt(5)). SOLVED: m >= 0.80 and CI lower >= 0.70. k*(X, t, o) = the largest k <= 6 with cells 1..k all
SOLVED (0 when k = 1 is not SOLVED). rc_j(X, s) = the rc_j cell of the 249 primary items (pfx_table.py; rc4 on its 128
items). D_j(s) = rc_j(CUR-a, s) - rc_j(bestA2, s); C_j(s) = rc_j(CUR-a, s) - rc_j(K1X-a, s); both paired over s16-20,
in points. D_j^orig and D_j^re = the same against each A2-6k set separately (originals j520 j525 j530 j540 j542,
re-trains j663 j664 j665 j540 j666). bestA2 = the set with the higher five-seed mean on that cell (on rc2 and rc3 the
original set: 65.66 / 55.08 against 64.28 / 52.60). V (control valid) = I7 and N1 for K1X-a.

Power (as CARD_khop_v3, card_numbers.out of khop3): paired SDs of rc2 against the original A2 set in the parent 1.2-4.3.
PASS needs m >= 3 and t >= 2.78: at m = 3 the SD must be <= 2.42; at SD 4.3 the mean must be >= 5.34. The card detects a
gain of +5 rc2 points with probability 0.91 / 0.70 / 0.50 at paired SD 2.4 / 3.3 / 4.3, and a gain of +3 with 0.40 /
0.29 / 0.21. Under no transfer (D_2 = 0) the reading is NULL 0.53 / 0.32 / 0.20 and "unresolved" 0.42 / 0.63 / 0.75 at
the same SDs (card_numbers.out section 11; Monte Carlo of the 0.7 rule on D_2, 100,000 draws per cell). So a true null
reads "unresolved" more often than NULL at the parent SDs (0.2, "What the confirmation buys"). A NULL excludes a gain of
the printed CI upper, no more.

### 0.7 Five-seed readings, evaluated in this order; the first that applies is the verdict

Section 0 of CARD_khop_v3 with CUR -> CUR-a and K1X -> K1X-a, and the arm named in every claim. Order and texts:
1. MANIPULATION FAILED. No transfer reading (the rc3 line prints D_3, D_3^orig, D_3^re, C_3 as INFO with "no rc3 reading
   and no transfer sentence"). Sub-labels, the first that applies:
   - "not run as registered": any of I1-I4 fails. What it says: the build or the cluster run did not do what the card
     says; fix and re-run under the same ids (rescue), nothing is read.
   - "not installed": k*(CUR1, ca, fwd) < 3. What it says: stage 1 did not reproduce the parent's capacity at five seeds
     (j975 had 92.7 at k3 at s16); the premise is false at this dose.
   - "installed, then lost in the GSM stage (arm a)": k*(CUR1, ca, fwd) >= 3 and k*(CUR-a, ca, fwd) < 3. What it says:
     the protection that kept the capacity at s16 does not keep it at five seeds; the capacity and the GSM stage still
     compete. Reported with both curves and the forgetting line (2.2). No transfer sentence.
   - "control not a control": I6 fails (CUR-a - K1X-a at (ca, fwd, 3) has CI lower <= 0). What it says: single-hop
     training of the same chains reached the same k = 3 accuracy, so C_2 isolates no composition.
   - "GSM damaged" (for L: "GSM damaged or not recovered"): N1 fails for CUR-a. What it says: the protection kept the
     capacity at the cost of GSM; D_2 is printed as INFO.
2. PASS: D_2 has m >= +3.0 and t >= 2.78, AND C_2 has CI lower > 0, AND V holds. Claim (R): "in Qwen3.5-4B with plain
   GSM intact and k*(ca, fwd) = <k> kept, symbolic chain training (27,000 rows) before an A2 GSM epoch with a one-third
   replay of those chains raises one-pass GSM rc2 on the 249 primary items by <m> [CI] points over the best A2 set and
   by <m> [CI] over the same schedule with single-hop queries on the same chains." Claim (L): the same with "before an
   A2 GSM epoch at one quarter of the A2 LR". rc3 enters the claim only when D_3 and C_3 meet the same rule. Why C_2 needs
   only CI lower > 0: CARD_khop_v3 section 0, reading 2 (two trained sets differ with paired SD near 3.8). What it says:
   composition of given steps is part of the rc2 limit and training it on clean chains reaches GSM; H of CARD_khop_v1
   ("binding, not depth") is weakened for rc2. The claim names no mechanism and stays within the symbolic format, small
   values and this schedule. For R it also prints CUR-R - KH (2.3) to show whether stage 1 or the replay carries the gain.
3. NEGATIVE: D_2^orig CI upper < 0 AND D_2^re CI upper < 0. Claim: "with plain GSM intact and k* >= 3 kept, the chain
   curriculum (arm a) lowers GSM rc2 by <m> [CI] against both A2 sets". "The symbolic composition training interferes
   with GSM multi-step items" is added only when C_2 CI upper < 0 and V holds; else "the K1X twin is not above CUR-a
   (C_2 <m> [CI]): the cost is chain training plus extra updates, not composition". What it says: the installed
   capacity, kept, costs GSM multi-step accuracy.
4. BETWEEN "gain from extra updates, not composition": D_2 meets the PASS rule, V holds, C_2 CI lower <= 0. What it
   says: the gain comes with the chain stage, but single-hop reading of the same chains gives it too.
5. BETWEEN "gain over best A2, control invalid": D_2 meets the PASS rule and V fails (prints which part, and C_2 as
   INFO). What it says: a gain exists, but the control cannot rule out "any chain training plus updates".
6. BETWEEN "gain below the bar": D_2 CI lower > 0 and m < +3.0. Claim: "arm a raises GSM rc2 by <m> [CI] over the best
   A2 set, below the pre-registered +3 bar; C_2 = <m> [CI]". Evaluated before NULL. What it says: a small positive
   effect, not the size the card set out to detect.
7. NULL: D_2 CI lower <= 0 and D_2 CI upper < +3.0. Claim: "the kept capacity (k*(ca, fwd) = <k>, plain intact) does
   not raise GSM rc2: D_2 = <m> [CI]; the data exclude a gain of more than <CI upper> points." Below the original set
   only: the drift text of CARD_khop_v3 reading 7. What it says: the model composes 3+ given steps in one pass and does
   not use that on GSM items, so the rc2 limit is not the serial composition of given operations; this supports
   "binding, not depth" for the transfer route. It does not exclude a surface gap (symbolic wording, values 1-99): the
   cag curriculum (CARD_khop_v3 section 5 (a)) tests that.
8. BETWEEN "unresolved": everything else (D_2 CI lower <= 0 and CI upper >= +3.0; prints the CI). What it says: the run
   cannot tell a null from a gain of the target size.
The same rule (2-8 with D_3, C_3, D_3^orig, D_3^re) is printed for rc3 as a secondary line; it never changes the
verdict. Multiplicity: the primary tests one arm (the gate's fixed rule picks it), so no correction applies. If a later
card confirms the other arm, that card states its correction; this card prints, as INFO, whether the arm also clears
the Bonferroni bar for two arms (t >= 3.50, df 4).

## 1. Data and arms

### 1.1 Eval sets and parent files (unchanged)

The nine khop eval files (E/src/data/eval/khop/), the khop2 files (E/src/data/eval/khop2/), data/eval/khop/train_kh.jsonl
(a16b1520f0fb8b24ffd8fa57ef696de0, 3,000 rows) and data/eval/khop/train_k1.jsonl (4bb15f66c8d8843013eb0410c6a87e48,
3,000 rows) are not rebuilt and not edited; their md5s must equal E/src/data/eval/khop/build_report.json and
E/src/data/eval/khop2/build_report.json. GSM and hint eval sets: the A2-6k env's EXTRA_EVALS, unchanged. No new eval set.
E/src/data/eval/khop4/ holds only build_report.json.

### 1.2 New training directories (E/src/data/student_khop4_*; jobs/build_bundles.sh line 21 copies them for x1khop4)

The confirmation S1 jobs need the khop3 stage-1 data, but build_bundles.sh copies only `student_${v#x1}_*`, so the x1khop4
bundle would not carry student_khop3_*. Generator scripts/build_khop4.py (new, E/src_khop4/repo):
1. Inputs: E/src/data/student_khop3_c27/{train,val}.jsonl and student_khop3_k1x/{train,val}.jsonl. Stop (SystemExit)
   unless their md5s equal E/src/data/eval/khop3/build_report.json and these values: c27 train
   92545dd42ca451c39e0e7c104aa78d1e (27,000 rows), k1x train d7cc9716e5785461d16ec92156104261 (27,000 rows), both val
   9f55fec541246e41401a7b5fa6c5510a (393 rows); and unless train_kh.jsonl and train_k1.jsonl have the md5s of 1.1.
   --out-root or --report equal to an input dir, or an output dir holding any other file, is a SystemExit.
2. data/student_khop4_c27/{train,val}.jsonl and data/student_khop4_k1x/{train,val}.jsonl: byte copies.
3. data/eval/khop4/build_report.json: card CARD_khop_v4, generator, md5 and row counts of the 4 outputs and of the 6
   inputs; the two replay equalities (train_kh = kd27 rows 0..2,999 byte for byte; train_k1 = k1x rows 0..2,999 byte for
   byte; both must be true); d_pass / d_fail. main() exits 2 on a failed line.

### 1.3 Arms and job ids (1025-1049)

| Id | Stage | Arm / seed | Name | DATA | MODEL / WTAR | XARGS / LR |
|---|---|---|---|---|---|---|
| 1025 | S2 pilot | CUR-L s16 | 1025_4b_gsmvt_A2-khop4curlw975-c1lr075ch_s16_fsdp | student_gsm8k_tf_vt | 975 S1 | `--eval-value-render vt`; LR=7.5e-7 |
| 1026 | S2 pilot | K1X-L s16 | 1026_4b_gsmvt_A2-khop4k1xlw985-c1lr075ch_s16_fsdp | as 1025 | 985 S1 | as 1025 |
| 1027 | S2 pilot | CUR-R s16 | 1027_4b_gsmvt_A2-khop4currw975-c1lr3ch_s16_fsdp | as 1025 | 975 S1 | `--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl`; LR=3e-6 |
| 1028 | S2 pilot | K1X-R s16 | 1028_4b_gsmvt_A2-khop4k1xrw985-c1lr3ch_s16_fsdp | as 1025 | 985 S1 | `--eval-value-render vt --extra-train data/eval/khop/train_k1.jsonl`; LR=3e-6 |
| 1029-1032 | S1 confirm | CUR1 s17-20 | 10xx_4b_kd27_A2-khopcur1-c1lr3ch_s<s>_fsdp | student_khop4_c27 | Qwen3.5-4B, SAVE=1 | as khop3 S1 |
| 1033-1036 | S1 confirm | K1X1 s17-20 | 10xx_4b_kd27k1_A2-khopk1x1-c1lr3ch_s<s>_fsdp | student_khop4_k1x | Qwen3.5-4B, SAVE=1 | as khop3 S1 |
| 1037-1040 | S2 confirm | CUR-a s17-20 | 10xx_4b_gsmvt_A2-khop4cur<l\|r>w<1029-1032>-c1lr<075\|3>ch_s<s>_fsdp | student_gsm8k_tf_vt | the same seed's CUR1 | as 1025 (L) or 1027 (R) |
| 1041-1044 | S2 confirm | K1X-a s17-20 | 10xx_4b_gsmvt_A2-khop4k1x<l\|r>w<1033-1036>-c1lr<075\|3>ch_s<s>_fsdp | student_gsm8k_tf_vt | the same seed's K1X1 | as 1026 (L) or 1028 (R) |
| 1045-1049 | - | reserved, never written | | | | |
| reused, no new job | | 975 / 985 (s16 S1, khop3; tars on staging), A2-6k originals and re-trains, KH j756 j758-761, K1 j757 j762-765, khop3 j980 / j990 (context) | | | | tarballs in E/results |

The s16 S1 pair (975, 985) lives outside 1025-1049: khop3 trained it under the same recipe, and this family reads it as
its s16 stage 1. Its compute (2.78 GPU-h) was counted in khop3's 4.01.
Ids 1037-1044 have two candidate envs each, one per arm (the arm letter in the TAG tells them apart: khop4curl /
khop4k1xl for L, khop4curr / khop4k1xr for R). Both sets are written and bundled now, so the bundle does not change
after the gate; the confirmation DAG names exactly one set. Only one env per id ever runs, so one id has at most one
landed tarball (the readout stops on two). prep_khop4.py refuses any other env on a family id (and any env on 1045-1049).
Updates (ceil(rows / 32)): S1 844; S2-L 186 ("stage all: 5923 examples per pass, 186 steps"); S2-R 279 ("extra-train:
total training rows 8923 (5923 from data/student_gsm8k_tf_vt/train.jsonl)", "stage all: 8923 examples per pass, 279
steps", as j756). A re-run is a resubmission of the same node (rescue DAG, or a one-node DAG with the same JOB id,
VARS, env, name and tar name), never a new id and never a new setting.

### 1.4 Exact envs and flags

No new trainer flag. Generator flags (scripts/build_khop4.py only): --khop3-root DIR (default data), --out-root DIR
(default data), --report DIR (default data/eval/khop4).
Env generator stage/prep_khop4.py (python, values bash-escaped as prep_khop3.py does). Ids parsed with
`re.match(r"(?:smoke_)?(\d+)_", name)` and an int compare, never a slice.
- S2 envs (1025-1028, 1037-1044) = the khop3 S2 env of the same arm kind (E/stage/condor/jobs_x1/980_... for CUR-*,
  990_... for K1X-*) with seed keys set to s and exactly these keys changed; every other key asserted equal:
  - L: LR=7.5e-7; TAG; JOBKEY "4b gsmvt A2-<TAG> s<s> fsdp x1khop4"; MODEL=models/<S1 name>/model; WTAR=<S1 name>.
  - R: XARGS="--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl" (CUR-R) or ".../train_k1.jsonl"
    (K1X-R); TAG; JOBKEY; MODEL; WTAR.
  - Unchanged and asserted: SAVE='', DATA=data/student_gsm8k_tf_vt, EPOCHS=1, LSMOOTH=0.1, MBS=2, GA=16, NGPU=1,
    FSDP_OPTIM=cpu-adamw, OPTIM=adamw, PREC=bf16, TOKBUDGET=4096, ARMS=A2, EVALAT=0.5, EVALMID (A2's), EXTRA_EVALS (the
    663 sets plus the nine khop sets, the khop3 S2 order).
  - Second assertion for R: a CUR-R env equals the parent KH env of the same seed (756 for s16, 758-761 for s17-20)
    except MODEL, WTAR, SAVE, TAG and JOBKEY; a K1X-R env equals the parent K1 env (757, 762-765) the same way.
  - TAGs: khop4curlw<S1 id>-c1lr075ch, khop4k1xlw<S1 id>-c1lr075ch, khop4currw<S1 id>-c1lr3ch, khop4k1xrw<S1 id>-c1lr3ch.
    "lr075" names LR 7.5e-7 so the name does not claim 3e-6. The builder greps every script in the path for a parse of
    "lr<digit>" from a name and reports none (or the card is amended).
- S1 envs (1029-1036) = khop3's 976-979 (CUR1) and 986-989 (K1X1) envs of the same seed with exactly these keys changed:
  DATA=data/student_khop4_c27 or data/student_khop4_k1x, JOBKEY "... fsdp x1khop4". TAG stays (khopcur1-c1lr3ch,
  khopk1x1-c1lr3ch). SAVE=1, EVALAT='', EVALMID='', EXTRA_EVALS the khop3 S1 list. Every other key asserted equal.
- Smoke envs: none new. The smoke DAG runs the four pilot envs with smoke="1" (SMOKE_ONLY), node names smoke_1025 ...
  smoke_1028. Each smoke extracts its weights tar, so smoke_1025 / smoke_1027 test the 975 tar and smoke_1026 /
  smoke_1028 the 985 tar.
Subs:
- stage/train_x1khop4w_4b.sub (S2 and smokes) = stage/train_x1khop3w_4b.sub with exactly the CODE_TGZ,
  transfer_input_files and log lines changed (cotd_code_data_x1khop4_4b.tgz, logs/train_x1khop4w_4b.log), plus the
  header comment; executable run_train_x1verify2.sh, unchanged (md5 3866fc7788af662fe94052a9d2bae4c8, asserted by the
  test and checked on the AP before submission).
- stage/train_x1khop4_4b.sub (S1 confirmation) = stage/train_x1khop3_4b.sub with the same three lines changed
  (logs/train_x1khop4_4b.log); executable run_train5.sh (its SAVE block exports $STAGING/weights/<name>.tar).
Resources as the parent: ngpu 1, cpus 8, mem 80GB, gpumem 40000, prio 300, RETRY 1, POST post_push.sh.
PRE scripts: stage/check_weights_khop3.sh <S1 name>, unchanged (md5 asserted equal to the khop3 copy; it takes a name and
parses no id). On every S2 node: the pilot nodes name 975 / 985; the confirmation S2 nodes name their seed's S1.
LR and XARGS reach the trainer through the env (run_train_x1verify2.sh line 91 sources it with set -a;
repo/hpc/train_arm.sbatch passes $LR and $XARGS). The smokes do not prove that (SMOKE_ONLY prints only the weights line,
"smoke ok A2", "smoke passed" and "== exit 0", smoke_995 log); line a-Q2 on the pilot jobs is the first check, and it
fails closed.

## 2. Metrics and rules

### 2.1 Primary
- For the confirmed arm a: D_2 = rc2(CUR-a) - rc2(bestA2) and C_2 = rc2(CUR-a) - rc2(K1X-a), paired over s16-20, each
  printed as m [CI] (SD, t, n), and the verdict of 0.7 in its order.
- Secondary decision line (no verdict change): the same rule on rc3 (D_3, C_3).

### 2.2 Installation, the "kept" reading and forgetting
- k*(CUR1, ca, fwd), k*(CUR-a, ca, fwd), and the curves a(ca, fwd, k), k = 1..8, every seed, for CUR1, CUR-a, K1X1,
  K1X-a, with khop3's j980 / j990 (s16) and the parent D27 next to them.
- Forgetting: a(CUR-a, ca, fwd, k) - a(CUR1, ca, fwd, k), k = 2..4, paired, with CI.
- I5 over s17-20 alone (n = 4, INFO), because s16 passed the gate's installation line (0.5); N1 for CUR-a and K1X-a over
  s17-20 alone (n = 4, INFO), because s16 passed a-Q6 (Amendment 1 finding 4).

### 2.3 Secondary (reported; no verdict depends on them)
- D_1, D_4, C_1, C_4 (rc4 on its 128 items), K1X-a - bestA2 at rc1..rc4, D_j^orig and D_j^re for j = 1..4.
- Arm R only: CUR-R - KH (j756, j758-761) and K1X-R - K1 (j757, j762-765), paired by seed, at rc1..rc4, on the plain sets
  and on (ca, fwd, k = 2..4). Same stage-2 recipe from the base model, so this is the effect of stage 1 at a fixed
  stage-2 recipe.
- Arm L only: CUR-L - j980 and K1X-L - j990 at s16 (n = 1, INFO): the effect of the LR alone at s16.
- The unchosen arm's s16 pair: every gate value, printed as INFO. No rc cell of it is read for any decision.
- G2 / G3 in the kept model: a(CUR-a, ca, fwd, 2) - rc2(CUR-a) and a(CUR-a, ca, fwd, 3) - rc3(CUR-a), per seed, with CI;
  printed only when N1 holds.
- Plain sets of CUR1 and K1X1 against bestA2; k* and a(t, o, 2..4) for wp, vb, cag and rev orders for CUR-a and CUR1;
  the 48 all-+/- items and the 31 order-sensitive items of (ca, fwd, 3) for CUR-a.
- Timing (wall per job, s_per_step), peak memory, weights tar size; the Bonferroni line of 0.7.

### 2.4 Reference A2 (best-A2 rule, fixed now; as CARD_khop_v3 2.4)
For every GSM-side line the reference is the A2-6k set with the higher five-seed mean on that line's metric, computed from
exact counts; its per-seed values enter the paired difference. NEGATIVE needs both sets separately (0.7, reading 3).
At the pilot (n = 1) the s16 reference is mean(j520, j663) per metric, both files required and finite: gsm8kx 59.42,
gsm8kv2 70.96.

### 2.5 Installation and manipulation checks
Data side (before any bundle; data/eval/khop4/build_report.json and the builder's tests):
- D1 the four student_khop4 files have the md5s of 1.2 step 1; the inputs are unchanged (md5 before and after).
- D2 train_kh.jsonl = rows 0..2,999 of student_khop4_c27/train.jsonl and train_k1.jsonl = rows 0..2,999 of
  student_khop4_k1x/train.jsonl, byte for byte.
- D3 the bundle holds data/student_khop4_c27/{train,val}.jsonl and data/student_khop4_k1x/{train,val}.jsonl with these
  md5s and no other file in either dir; data/eval/khop/train_kh.jsonl and train_k1.jsonl with the md5s of 1.1.
- D5 Same trainer: `diff -rq -x __pycache__ -x data E/src_khop3/repo E/src_khop4/repo` lists only
  scripts/build_khop4.py and tests/test_build_khop4.py.
Model side, at five seeds (khop4_table.py; exact counts):
- I1 Format: m(a_CUR-a(t, o, 1)) >= 0.90 on the six (task, order) cells of ca / wp / vb.
- I2 Query read: query-swap pair accuracy at k = 1, m >= 0.85 for CUR-a.
- I3 No option-only shortcut: m(blind accuracy) <= 0.30 for CUR-a and CUR1.
- I4 Run as registered (all 20 jobs of the arm: CUR1, K1X1, CUR-a, K1X-a at s16-20):
  - S1 logs: "stage all: 27000 examples per pass, 844 steps", final step 844, a finite final loss, the weights export
    line, and the train path of the job's DATA (data/student_khop3_* for 975 / 985, data/student_khop4_* for
    1029-1036; the readout accepts exactly these two path forms by id).
  - S2 logs: "== weights <staging>/weights/<the same seed's S1 name>.tar -> models/<S1 name>/model", the counts of 1.3
    for the arm (L: 5,923 / 186; R: the two extra-train lines, 8,923 / 279), final step 186 or 279.
  - Recipe, all args: metrics.json args of every job equal those of its recipe reference over ALL keys except an explicit
    allow-list. CUR-L: j980 except {lr, out}; K1X-L: j990 except {lr, out}; CUR-R: j980 except {extra_train, out};
    K1X-R: j990 except {extra_train, out}; CUR-R also equals j756 except {model, out, save}, K1X-R j757 the same way.
    Confirmation S2 jobs: the pilot job of the same arm kind except {seed, order_seed, model, out}. Confirmation S1 jobs:
    j975 (CUR1) or j985 (K1X1) except {seed, order_seed, train, out, eval, eval_subq}, where train must be
    data/student_khop4_<c27|k1x>/train.jsonl, eval must equal the reference list with only the val entry's directory
    changed from student_khop3_* to student_khop4_*, and eval_subq must equal the reference value with the same change
    (Amendment 2 item 1). Values: lr exactly 7.5e-07 (L) or 3e-06 (R and S1); extra_train
    exactly "" (L) or the 1.3 path (R); metrics.json extra_train exactly {<path>: 3000} for R and absent for L.
    Keys present on one side only fail the line.
  - metrics.json seed / order_seed = the label's seed for all 20 jobs.
- I5 Installation kept (the manipulation line): k*(CUR-a, ca, fwd) >= 3 at the final model (k*(CUR1, ca, fwd) decides
  between "not installed" and "installed, then lost", 0.7).
- I6 Content check: a(CUR-a, ca, fwd, 3) - a(K1X-a, ca, fwd, 3), paired, CI lower > 0.
- I7 Control trained (part of V): CARD_khop_v3 I7 on each K1X1 S1 log (mean of the logged "loss" values with step >=
  760 <= 1.0, at least five lines, all finite; else UNREADABLE) and m(a_K1X1(t, o, 1)) >= 0.98 on each of the six
  (task, order) cells; plus, new, m(a_K1X-a(t, o, 1)) >= 0.90 on each of the six cells (the twin keeps single-hop
  reading after stage 2; j990 had 98.7-100.0).

### 2.6 Necessary condition N1 (GSM intact)
- N1: for CUR-a, the five-seed mean on each of gsm8kx, gsm1kx, gsm8kv2, valx >= the bestA2 mean on that set - 2 points
  (bestA2 per set, exact counts). A failing N1 gives MANIPULATION FAILED ("GSM damaged"; for L "GSM damaged or not
  recovered") and no transfer sentence; D_2 is printed as INFO.
- N1 for K1X-a (the same rule) is part of V; NULL and NEGATIVE read D_2 and do not need V; their text prints V.

### 2.7 Fail-closed rules (rounds F and G)
- Every decision value comes from exact counts (Fraction(correct, items)); CIs and t are compared in squared form with
  t = 2.78; m >= 3 points is compared as m >= 3/100 exactly. A float print never decides a line.
- A missing, partial (row count off), NaN or unreadable preds file, log, history or metrics.json; a metrics.json that is
  not a JSON object; a missing counter (never defaulted to 0); a non-finite value that a max / min / mean would drop:
  each stops the gate line or the readout (UNREADABLE). Every khop and hint preds file present is checked row by row
  (finite probs, len(probs) = len(labels) > 0, distinct item_ids, khop row counts of CARD_khop_v1 1.2). Every set that a
  run's metrics.json args.eval lists must have exactly one preds file with exactly the item ids of its eval file;
  args.eval must list the S1 sets for CUR1 / K1X1, the 663 env's sets plus the nine khop sets for S2 jobs, the four plain
  sets for a reference or context run.
- The registered population is fixed: 20 jobs of the arm, 10 reference jobs, the context jobs named in 2.3. A missing
  one stops the readout; the readout never drops a seed, an item or a set and never shrinks n or a denominator.
  Population sizes: 249 / 249 / 249 / 128 for rc1..rc4, 150 items per khop cell at k <= 6.
- Pairing is by the seed recorded in metrics.json, never by file order or name sort (4-digit ids sort before 3-digit
  ones: "1025_" < "975_").
- A missing wall stops the line that needs it: round_gpuh.py prints "n/a" for a job with no start / exit line and leaves
  it out of its TOTAL, so the gate and the readout parse every row of the id range and stop on any "n/a" row.
- Verdict statistics from exact counts; selection only from the selection seed (here: no selection reads any seed; the
  arm rule of 0.5 reads only pass / fail of manipulation and compute lines).

## 3. Pilot (seed 16) and compute

Smoke DAG stage/smoke_x1khop4_4b.dag: smoke_1025, smoke_1026, smoke_1027, smoke_1028 (the pilot envs, smoke="1"). Smoke
gate: each log ends with "== exit 0", has "smoke ok A2", and has the line "== weights <staging>/weights/<975 or 985
name>.tar -> models/<that name>/model" (the S1 of its arm kind).
Pilot DAG stage/pilot_x1khop4_4b.dag: four independent nodes j1025, j1026, j1027, j1028, each with SCRIPT PRE
check_weights_khop3.sh <975 or 985 name> (the results tarballs of 975 / 985 are on the AP since round G). No PARENT /
CHILD among them.

Deterministic pilot gate: E/readout/khop4_pilot_gate.py --s1 <975 tgz> <985 tgz> --curl <1025 tgz> --k1xl <1026 tgz>
--curr <1027 tgz> --k1xr <1028 tgz> --smokes <4 smoke tgz> --ref s16=<j520 tgz>,<j663 tgz> --recipe-ref <j980 tgz>
<j990 tgz> <j756 tgz> <j757 tgz> --report E/src/data/eval/khop4/build_report.json --khop3-report
E/src/data/eval/khop3/build_report.json --khop2-report ... --parent-report ... --retry-walls <file> (required; empty when
no node retried). It reuses khop3_pilot_gate.py functions by import and edits neither file. Each line fails closed on a
missing, partial, non-object or non-finite input. Exact count bars on 150 items.
Common lines:
- C0 Data: D1, D2 of 2.5 from the report and the files' md5s.
- C1 S1 pair re-check: khop3's P1 and P2 for j975 and j985, and the tarball md5s equal the round-G copies in E/results:
  results_975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp.tgz 453e235627b7455143b85b6460b84a5a and
  results_985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.tgz 30a47caa409e67014527471726f56c69 (computed 2026-10-27).
- C2 Installed at s16: a_975(ca, fwd, 3) >= 120 / 150 (the I5 SOLVED mean bar 0.80, Amendment 1 finding 3; the value
  is known: 92.7 = 139 / 150 in round G).
- C3 Control S1 trained: khop3's P11 on j985 (loss mean of I7 <= 1.0; a_985(t, o, 1) >= 0.97 on six cells).
- C8 smoke_1025-1028 pass the smoke gate; j1025-j1028 landed with exit 0.
- C7c Measured pilot total: the walls of every execution of the eight pilot nodes (4 smokes, j1025-j1028), RETRY
  executions and condor restarts included (khop3 P7c rule, retry-walls file from the job event logs:
  logs/train_x1khop4w_4b.log), <= 6.0 GPU-h. The s16 S1 pair is not counted here (khop3).
Per-arm lines, a in {L, R}; CUR-a = j1025 (L) / j1027 (R), K1X-a = j1026 / j1028. The block is exactly these 11 lines:
a-Q1, a-Q2, a-Q3, a-Q4, a-Q5, a-Q6, a-Q7b, a-Q9, a-Q10, a-Q11 (10 decision lines; there is no a-Q8, exits are common
line C8), plus one INFO line a-Q7 per arm that never fails the gate (Amendment 1 finding 7).
The common block is exactly these 7 lines: C0, C1 j975, C1 j985, C2, C3, C8, C7c.
- a-Q1 Completeness: each of the two tarballs has metrics.json (a JSON object), the nine khop preds files with the
  CARD_khop_v1 row counts, every GSM and hint preds file of the 663 env, and every set of args.eval holds exactly its
  eval file's item ids.
- a-Q2 Run as registered at n = 1: the I4 S2 lines (weights line naming 975 for CUR-a and 985 for K1X-a; counts; final
  step), the all-args recipe check of I4 with its allow-lists (including the second R check against j756 / j757), lr
  and extra_train values, seed / order_seed (16, 16).
- a-Q3 (I1 at n = 1) a_CUR-a(t, o, 1) >= 0.85 on all six (task, order) cells.
- a-Q4 (I2 at n = 1) query-swap pair accuracy at k = 1 >= 0.80 for CUR-a.
- a-Q5 (I3 at n = 1) blind accuracy <= 0.32 for CUR-a.
- a-Q6 N1 at n = 1 (khop3 P6 form and bars): CUR-a gsm8kx >= 59.42 - 3.0 and gsm8kv2 >= 70.96 - 3.0 (56.42 / 67.96,
  exact counts against the exact reference); K1X-a the same sets >= reference - 5.0 (54.42 / 65.96). valx and gsm1kx
  INFO. Bar derivation: CARD_khop_v3 P6 (false-stop 0.034, false-go 0.036 at B = 3.0).
- a-Q7 (INFO, printed as "INFO  a-Q7: ...", never PASS / FAIL) Wall per job against L 0.9 h / R 1.1 h (KH max 0.93 +
  0.05 extract = 0.98; card_numbers.out), with the GPU type of the execution. A slow placement is hardware, not the arm;
  the decision lines on compute are a-Q7b and C7c.
- a-Q7b Projected confirmation of arm a = 4 x (wall 975 + wall 985 + wall CUR-a + wall K1X-a) <= 23.0 GPU-h (cap 25
  minus one S1 bar of 2.0 h as retry headroom, Amendment 1 finding 5), from measured walls (no n/a row).
- a-Q9 Installation kept on the S2 model: a_CUR-a(ca, fwd, 2) >= 128 / 150 (khop3 P9) and a_CUR-a(ca, fwd, 3) >= 120 /
  150 (0.80, the I5 SOLVED mean bar; raised from khop3's 117, Amendment 1 finding 3). Accepted error rates at n = 1
  (card_numbers.out section 12, pooled per-seed SD 0.061): a model with five-seed mean 0.78 / 0.79 / 0.80 passes 0.37 /
  0.44 / 0.50 (at 117: 0.50 / 0.57 / 0.63); a dead model at 0.70 passes 0.05; a model at 0.861 stops 0.16 (at 117:
  0.09). At the D27 SD 0.015 the edge rates are 0.09 / 0.25 / 0.50 and both tails are 0.000. On failure the line prints
  "(installed, then lost at s16, arm a)".
- a-Q10 (I6 at n = 1) a_CUR-a(ca, fwd, 3) - a_K1X-a(ca, fwd, 3) >= 0.15 (difference of counts >= 23 / 150).
- a-Q11 (I7 at n = 1, stage 2) a_K1X-a(t, o, 1) >= 0.85 on each of the six cells.
No gate line reads rc1..rc4 or any hint set of the 249 primary items, and the gate prints none of them. Reported, not
decision lines: every curve (CUR-a, K1X-a, with j975 / j980 / j985 / j990), plain sets incl. val, timing, peak memory.
Printed line names (Amendment 2 item 2): the common lines as above; a-Q1 ... a-Q11 as "R-Q1" ... "R-Q11" and "L-Q1" ...
"L-Q11" (R block first); a-Q7 as "INFO  R-Q7: " / "INFO  L-Q7: "; the 0.5 readings as "READING  <text>".
Output: one "PASS  <line>: ..." or "FAIL  <line>: ..." per common line and per arm decision line, and one
"INFO  a-Q7: ..." line per arm (both arms always evaluated and printed), the pilot failure readings of 0.5 as text, then
the last line: "GATE PASS ARM R", "GATE PASS ARM L" or "GATE FAIL: <lines>". The last line follows 0.5 exactly and reads
decision lines only (common lines all PASS; R decision lines all PASS -> R; else L decision lines all PASS -> L; else
FAIL). Exit 0 only for a GATE PASS line.

Compute (measured from the parent jobs, card_numbers.out):
- Walls (h): khop3 S1 j975 1.41, j985 1.37; khop3 S2 j980 0.55, j990 0.50 (186 updates, warm start, khop evals); smokes
  0.04-0.07. KH (279 updates from base, khop evals) mean 0.766, max 0.93 (n 5); K1 mean 0.716, max 0.89. Fit of the 30
  khop2 jobs: 0.400 h + 4.77 s per update, largest residual +0.40 h.
- Estimates per job: S2-L 0.55 (the LR does not change the cost); S2-R 0.816 (max of KH mean + 0.05 extract, j980 +
  93 x 3.54 s, fit(279)); worst 1.17. S1 1.52 (fit(844); measured 1.37-1.41), worst 1.92.
- Pilot: 4 smokes + j1025-j1028 = 3.01 GPU-h (at the bars 4.60). Cap 6.0.
- Confirmation, one arm (8 S1 + 8 S2): R 18.67 GPU-h (at the bars 24.80; P7b form from the measured s16 walls 17.65);
  L 16.55 (at the bars 23.20; P7b form 15.32). Cap 25. Both arms: 23.07 by estimate (32.00 at the bars), and 28 jobs
  do not fit 25 ids, so one arm only (0.3). Retry headroom (Amendment 1 finding 5): a-Q7b uses 23.0 = 25 - one S1 bar
  (2.0 h). One automatic S1 retry at the bars would give R 26.80 / L 25.20; DAG RETRY 1 runs without a new submission,
  so such an overrun is accepted, counted and reported (it is not a re-run that sev-e0 submits). On the measured walls
  (P7b form 17.65 / 15.32) one retry fits under 25.
- Installation between S1 and S2 of a confirmation seed (Amendment 1 finding 6): no extra PRE. If S1 of one of s17-20
  does not install, that seed's S2 pair still runs (R about 1.6 GPU-h, L about 1.1 by estimate); this cost is accepted.
  A PRE that dropped a seed would shrink the registered population (2.7), and the five-seed reading "not installed"
  reads k*(CUR1) at all five seeds anyway.
- Family total about 21.7 GPU-h (R) or 19.6 (L), plus khop3's 2.78 for the reused s16 S1 pair. No round cap was given
  for this round; the family caps 6 / 25 apply. verify3 is closed (lead decision 4), so no other family shares the ids.
- Caps count everything (CARD_khop_v3 rule): every failed execution (DAG RETRY, condor restarts inside one attempt,
  rescue resubmissions) counts against 6 and 25. sev-e0 submits no re-run when the phase's measured walls plus the
  re-run's bar would exceed the cap; it stops and reports to the second author. The readout prints `E/jobs/round_gpuh.py 1025 1049`
  plus the retry walls, and stops on any "n/a" row.
- Staging: the pilot exports nothing (S2 SAVE=''). The confirmation S1 jobs export 8 tars of 9,702,973,440 B (9.04 GiB
  each, 72.3 GiB); line: free >= 77.3 GiB and file count + 8 < the file quota, read back on the AP before the
  confirmation (lead decision 1: about 257 GB free after round G; quota 1000 GB / 10,000 files, read back 2026-10-27).
  Deleting any tar needs the second author.

## 4. Conditional confirmation (written now; sev-e0 submits it without asking when the gate prints GATE PASS ARM R or L)

stage/confirm_x1khop4_4b.dag, written by stage/prep_khop4_confirm.py --gate-output gate.txt. It refuses unless:
- the last non-empty line is exactly "GATE PASS ARM R" or "GATE PASS ARM L";
- the text holds exactly one "PASS  <name>: " line for each common line (C0, C1 j975, C1 j985, C2, C3, C8, C7c) and for
  each of the 10 decision lines of the named arm (a-Q1 ... a-Q6, a-Q7b, a-Q9 ... a-Q11), exactly one "INFO  a-Q7: " line
  per arm, and exactly one "PASS" or "FAIL" line for each line of the
  other arm; no common or named-arm line starts with "FAIL";
- the named arm equals the 0.5 rule recomputed from those lines (both arms all PASS -> R);
- every one of the 16 envs it names still equals the text prep_khop4.py generates.
It runs on the Mac; only confirm_x1khop4_4b.dag is copied to the AP. Nodes (16):
- S1: j1029-1032 (CUR1 s17-20), j1033-1036 (K1X1 s17-20), sub train_x1khop4_4b.sub. SCRIPT PRE check_pilot.sh with
  four names: the named arm's CUR-a and K1X-a pilot names (1025 / 1026 for L, 1027 / 1028 for R),
  975_4b_kd27_A2-khopcur1-c1lr3ch_s16_fsdp and 985_4b_kd27k1_A2-khopk1x1-c1lr3ch_s16_fsdp.
- S2: j1037-1040 (CUR-a), j1041-1044 (K1X-a), each a CHILD of its seed's S1 (1029 -> 1037, ..., 1036 -> 1044), sub
  train_x1khop4w_4b.sub, SCRIPT PRE check_weights_khop3.sh <its S1 name>.
- Every node: POST post_push.sh, RETRY 1, prio 300. the second author is told at submission and can stop it.
Pre-submit operator lines (sev-e0; no cluster command is run by the card writer or the builder): the bundle and its md5
on both sides; run_train_x1verify2.sh md5 3866fc77...; check_pilot.sh, check_weights_khop3.sh, post_push.sh and both
subs in the DAG directory; the 975 and 985 weights tars on staging with 9,702,973,440 B each and their results tarballs
in results/ on the AP (pilot PREs); the 28 envs (1025-1036 and both candidate sets of 1037-1044) in the bundle and in
condor/jobs_x1 on the AP; the staging line of section 3 before the confirmation; a RESULTS_LOG line for khop4 (lead
decision 3 already names the family; an operator step for sev-e0).
Readout: E/readout/run_khop4.sh <gate output> <retry-walls file of the smoke, pilot and confirmation DAGs> [results dir]
-> E/readout/khop4_table.py on the 20 tarballs of the arm (975, 985, the arm's two pilot jobs, 1029-1044), both A2-6k
sets, KH / K1 (R) or j980 / j990 (L) and the parent D27 (context); then 0.7 in that order, the rc3 line, N1, the
installation and forgetting lines, and 2.3. khop4_table.py imports khop3_table.py, khop2_table.py and khop_table.py
primitives and edits none of them. khop_table.load_run finds metrics.json and the preds by path suffix, so S1 run dirs
(runs/A2-<TAG>-Qwen3.5-4B-student_khop{3,4}_<c27|k1x>-s<s>) and S2 run dirs (runs/A2-<TAG>-model-student_gsm8k_tf_vt-s<s>)
both load; a test covers the three shapes. The arm is read from the gate output's last line, not given by hand. Log
exp/logs/khop4_readout_<date>.txt (+ .json).

## 5. Fixed before the data

- Arms L and R and their only free values: LR 7.5e-7 for L (rule of 0.3, Amendment 1 finding 3), the replay files train_kh / train_k1 (3,000 rows,
  share 0.336, 279 updates) for R; neither was chosen on any rc cell or hint set. The numbers used to fix them are plain
  sets, val, training histories and khop cells of parent jobs (card_numbers.out sections 6-8).
- Job ids 1025-1044 as 1.3, 1045-1049 reserved and unused; names, TAGs, envs, subs, PREs, the smoke and pilot DAGs, the
  confirmation node list and its PREs; the generator rule of 1.2 and the data md5s.
- The arm rule of 0.5 (R first), the pilot failure readings, the gate lines C0-C8 and a-Q1-a-Q11 with their bars, the
  five-seed readings of 0.7 in order with every sub-label, the +3.0 bar and t 2.78 on D_2 with C_2 CI lower > 0, the rc3
  line, I1-I7, V, D1-D5, N1, the reference rule, the fail-closed rules and the compute caps.
- Not run here, each needs a new decision or card: (a) the other arm's confirmation S2 (8 jobs on new ids, reusing the
  1029-1036 S1 tars); (b) after a NULL, the cag curriculum (CARD_khop_v3 section 5 (a)); (c) any LR or share sweep;
  (d) probes of the S1 / S2 weights; (e) other model sizes.
- Bundle: `REPO_DIR=E/src_khop4/repo jobs/build_bundles.sh "994ff5026e54cffe260d88709ce33afd4533d326-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop-khop2-khop3-khop4" x1khop4`.
  Pre-submit lines, each an exact file set: `tar tzf ... | grep 'data/student_khop4_'` lists exactly the 4 files of D3
  (plus the 2 dirs) and no student_khop3_ path; `... | grep data/eval/khop4/` exactly build_report.json; data/eval/khop
  the 12 parent files and build_report.json (train_kh / train_k1 md5s of 1.1); `... | grep -E
  'condor/jobs_x1/10(2[5-9]|3[0-9]|4[0-9])_'` lists exactly the 28 envs of 1.3 (none on 1045-1049); the
  data/eval/*.jsonl file set and md5s equal those of the x1khop3 bundle (21 files); md5 of the bundle on both sides.
- 4-digit ids. Every script in the path parses ids with a regex and an int compare, and is tested on fixture names with
  the decoys 102_..., 10250_..., 1024_... (verify3's last id) and 1000_..., and on a results dir whose name sort puts
  1025_ before 975_: prep_khop4.py (a jobs dir holding those decoys is not read as the family's ids; the 1045-1049 and
  "another env on a family id" refusals fire; the two candidate envs per id 1037-1044 are accepted only in the arm
  pairs of 1.3), prep_khop4_confirm.py, round_gpuh.py (`round_gpuh.py 1025 1049` on the decoy dir lists none of
  them), check_pilot.sh and check_weights_khop3.sh (take names, parse no id), post_push.sh (parses no id),
  khop4_pilot_gate.py and khop4_table.py (reference identity `re.search(rf"(^|_){j}(_|\.tgz$)")`), run_khop4.sh
  (results_<id>_* with the id followed by "_").
- Builder checks before the bundle (CPU, E/src_verify3/venv_pfx/bin/python, python -m unittest, torchrun --local-addr
  127.0.0.1):
  - tests/test_build_khop4.py: the refusals of 1.2; byte equality of the 4 outputs; D2; report values recomputed from
    the files; deterministic; inputs untouched.
  - A tiny two-stage run (the khop3 tiny test pattern): stage 1 with --save on 18 khop rows; stage 2 from the saved
    weights (a) with --lr at one quarter of the reference and (b) with --extra-train of 6 khop rows. Assert: (a) the
    history lr values equal one quarter of the reference run's at every step, the per-update item_ids equal the
    reference's; (b) the log lines "extra-train: 6 rows from ...", "total training rows <6 + n>", final step
    ceil((n + 6) / 4), metrics.json extra_train {path: 6}, finite losses; a duplicate item_id in --extra-train is a
    SystemExit (the khop-tr-* ids and the GSM ids are disjoint in the real files: checked by the test).
  - D5 against E/src_khop3/repo and repo.base, printed in the build report.
  - E/readout/test_khop4_pilot_gate.py: a good pilot passes with "GATE PASS ARM R"; R failing one line gives "ARM L"
    when L passes; both failing gives GATE FAIL with the 0.5 readings ("installed, then lost (both arms)", "kept, GSM
    damaged", "lost, GSM intact", "control invalid"); each common line and each per-arm decision line fails alone; edges: a-Q9
    120 / 119 and 128 / 127 of 150; a-Q6 CUR at reference - 2.9 / - 3.1 and K1X at - 4.9 / - 5.1; a-Q7 at 0.95 h (L) and
    1.15 h (R) prints INFO and does not change the last line; a-Q7b at 23.0 / 23.01 GPU-h; C2 at 120 / 119; C7c over 6.0 by a retry wall and a missing retry-walls
    file; a-Q2 by one extra differing arg key, by a key present on one side only, by lr 3e-06 in an L job, by an R job without extra_train, by an S2 log naming
    another S1; a non-object metrics.json, a NaN prob, a hint file one row short, an "n/a" wall row; the gate prints no
    rc / hint line.
  - E/readout/test_khop4_table.py: every reading and sub-label of 0.7 reached by a fixture for both arms (incl.
    "installed, then lost in the GSM stage (arm a)" and "GSM damaged or not recovered" for L); PASS with and without
    rc3; NEGATIVE both texts; NULL with the drift text; m = 1.5 with SD 1.0 reads "gain below the bar"; D_2 1e-7 below
    3/100 is not PASS, exactly 3/100 with t exactly at the bar is PASS; V failing by I7 at stage 2; the arm read from the
    gate output; UNREADABLE on each fail-closed case of 2.7 (missing tarball, row count off, NaN loss, wrong seed, a
    reference with a wrong job id, a 4-digit look-alike, two tarballs for one id, an "n/a" wall); end to end through
    main() on the real A2-6k, KH / K1 and khop3 tarballs plus synthetic khop4 runs.
  - E/stage/test_prep_khop4.py: every S2 env against its khop3 S2 env and (R) against the KH / K1 env, key by key; every
    S1 env against khop3's 976-979 / 986-989 env; names, TAGs, SAVE only on S1, MODEL / WTAR pairs; both subs differ
    from their sources in exactly the three lines (plus header comments), executables and the runner md5; the smoke and
    pilot DAGs (PREs, no PARENT / CHILD in the pilot); the confirmation generator refuses every non-matching gate text
    (incl. "GATE PASS", khop3's passing output, an "ARM R" output whose lines say R failed, a FAIL in a common line) and
    writes 16 nodes with the PARENT / CHILD pairs, PREs, POST, RETRY and PRIORITY for each arm.
  - Two candidate envs per id (1037-1044): the builder checks that jobs/scp_envs.sh (copies by glob, compares md5 per
    file name), jobs/build_bundles.sh (copies condor/jobs_x1 whole), check_pilot.sh, post_push.sh and run_khop4.sh accept
    two env files with the same leading id and act on the name, and that run_khop4.sh and khop4_table.py stop on two landed
    tarballs for one id (round_gpuh.py lists both and is not edited; the readout's check is the one that stops).
  - The full src_khop4 suite has no new failure against repo.base (the known test_tree_target and TestSbatch failures
    stay).
- Process: one cold critic on this card before the build; one cold build critic on the diff, the data and the stage
  files before the bundle; then the Codex CLI review loop (exp/jobs/codex_review.sh, at most 6 rounds, every fix
  re-reviewed) until SUBMIT; smoke; pilot; gate; confirmation on GATE PASS ARM R / L.

## Amendment 1 (2026-10-27 20:51 CDT, after the cold critic, before any build)

Critic: E/src_khop4/critique/CARD_CRITIC_khop4.md (verdict FIX). No build, bundle, data or cluster command exists for
khop4 at this time. New numbers: card_numbers.py sections 10-14 (card_numbers.out); the script still reads no rc cell and
no hint set (the paired SDs of the power table are constants taken from the parent readout logs, cited in section 11).
Every finding is answered below; the sections named are updated in place.

1. HIGH, "under no transfer the primary most likely reads unresolved". FIXED. card_numbers.py section 11 simulates the
   0.7 rule on D_2 at paired SD 1.2 / 2.4 / 3.3 / 4.3 and true D_2 in {-3, 0, +3, +5}; the critic's numbers reproduce
   (D_2 = 0, SD 3.3: NULL 0.32, "unresolved" 0.63). The 0.4 five-seed priors are replaced by section 14 (judgment weights
   stated there, rates from section 11): "unresolved" about 0.39, NULL about 0.18, PASS about 0.09. The 0.6 power
   sentence is replaced by the simulated rates. 0.2 gains "What the confirmation buys": the five-seed manipulation
   reading (k*(ca, fwd) >= 3 kept with N1 intact, a model no khop line has), the forgetting line, CUR-R - KH, and a
   bounded D_2. The statistical rule is not changed. The card registers that value; the lead's go on this amended card,
   before the build, is its acceptance, and a rejection withdraws the card before any data (0.2). Section 4 stays
   automatic on GATE PASS, so no decision is left for after the gate.
2. MEDIUM, "the replay share is justified with rc2 cells". FIXED. card_numbers.py section 10 computes the five-seed
   plain sets of KH, KD, H3 and D27 against best A2 from exact counts. KD holds N1 too (smallest margin +0.65 on gsm1kx;
   KH +1.17), so "the one parent dose with N1 holding" was wrong. The share rule in 0.3 R now reads plain sets, the chain
   share and compute only: the smallest stage-1-row replay file whose from-base run holds N1; KH over KD for larger N1
   margins on three of four sets, a minority chain share (0.336 against 0.603) and 279 against 467 updates. The rc2
   numbers are struck from 0.1 and 0.3; section 5 is now true as written.
3. MEDIUM, "pilot installation bar 0.78 below the SOLVED bar 0.80". FIXED. a-Q9 k = 3 bar raised to 120 / 150 (0.80); C2
   raised the same way (j975 has 139 / 150, so C2 is unchanged in effect). The accepted false-go / false-stop rates are
   printed in a-Q9 (section 12): false-go 0.37-0.50 for a model at mean 0.78-0.80 at the pooled SD 0.061 remains; n = 1
   cannot separate the edge zone, and the five-seed I5 is the reading that decides. The L margin rule is anchored at
   0.80: 1/3 (0.831) fails the 0.04 margin and 1/4 (0.855) meets it, so L uses LR 7.5e-7 (names: c1lr075ch; lr exactly
   7.5e-07 in a-Q2 / I4). The GSM-recovery risk of L is larger at 1/4 and is stated in 0.3 and in the pilot priors
   (L passes about 0.25).
4. LOW, "s16 passes a-Q6 by construction and enters N1". FIXED. N1 for CUR-a and K1X-a is also printed over s17-20
   (n = 4, INFO) in 0.5 and 2.2. N1 itself stays over s16-20 (the population is not shrunk).
5. LOW, "no retry headroom at the bars". FIXED. a-Q7b <= 23.0 GPU-h (25 minus one S1 bar of 2.0 h). Section 3 states that
   an automatic DAG RETRY above the cap is an accepted overrun, counted and reported.
6. LOW, "no installation check between S1 and S2 of a confirmation seed". Cost accepted, stated in section 3 (about 1.6
   GPU-h for R, 1.1 for L, per non-installing seed). No new PRE: a PRE that drops a seed would shrink the registered
   population, and a logging-only PRE adds a script on the AP that decides nothing.
7. LOW, "a-Q7 can fail an arm for hardware". FIXED. a-Q7 is an INFO line ("INFO  a-Q7: ...", with the GPU type); the arm
   block has 10 decision lines; a-Q7b and C7c are the compute decision lines. The confirmation generator (section 4)
   and the gate tests (section 5) are updated to that line set.
8. INFO, "the L projection has one measured point". Recorded, not fixed: adding khop sets to EVALMID would break the
   all-args recipe identity with j980 / j756. The pilot j1025 is the first measured point of the LR curve.

## Amendment 2 (2026-10-27 22:22 CDT, builder, during the build; no bar, arm, id, schedule, reading or verdict order changes)

The build found one gap in the card's recipe check and fixed the interfaces that the card leaves open. Nothing here
was chosen on data: no khop4 job exists. Card 0.2 takes the lead's go on the Amendment-1 card as the acceptance; item 1
changes a check, so the lead accepts this amendment before the bundle (a RESULTS_LOG line), or the build goes back. E/src_khop4/BUILD_REPORT.md section 7 lists the same items.
1. S1 confirmation allow-list (2.5 I4). The khop3 S1 jobs record `eval_subq = data/student_khop3_<c27|k1x>/val.jsonl`
   (train_arm.sbatch passes `$DATA/val.jsonl`). The jobs 1029-1036 run on data/student_khop4_*, so their eval_subq is
   data/student_khop4_<c27|k1x>/val.jsonl. With the card's allow-list {seed, order_seed, train, out, eval} every
   confirmation S1 job would fail I4 by construction ("not run as registered"). The allow-list is now
   {seed, order_seed, train, out, eval, eval_subq}, and eval_subq must equal the reference value with only
   `data/student_khop3_` changed to `data/student_khop4_` (the same rule as the val entry of eval). Test:
   readout/test_khop4_table.py TestHelpers.test_real_args (the real j975 / j985 args; without eval_subq in the
   allow-list the correct khop4 args fail on exactly that key). No other key differs: the S2 recipe pairs were checked
   on the real metrics.json files (j980 vs j756 and j990 vs j757 differ only in {extra_train, model, out, save}).
2. Gate line names. The per-arm lines a-Q1 ... a-Q11 print as "L-Q1" ... "L-Q11" and "R-Q1" ... "R-Q11"; the INFO line
   prints as "INFO  L-Q7: " / "INFO  R-Q7: ". The R block is printed before the L block. The 0.5 failure readings print
   as "READING  <text>" lines before the last line.
3. Gate inputs. `--s1 <975 tgz> <985 tgz>`, `--smokes` takes the four smoke tarballs comma separated in the order
   1025, 1026, 1027, 1028 (each name is checked: results_smoke_<id>_...), `--recipe-ref <980> <990> <756> <757>` (each
   must name its id). C1 also checks the basename results_<975 or 985 name>.tgz. C8 and C7c read each pilot's log on
   its own (exit and wall lines), so a pilot whose preds or metrics.json are broken fails that arm's a-Q1 / a-Q2 and
   the arm lines that need the run, not the common lines; a missing or unreadable log fails C8 / C7c.
4. a-Q1 completeness is khop3's P1 form: the nine khop sets with the eval ids, every set of the 663 env including
   preds_subq present, and khop3_table.load's check of every args.eval set (exact item ids, finite probs, distinct
   ids). The seed check is in a-Q2.
5. Readout population (2.7). For both arms the readout also loads j980 and j990 (the recipe references of the s16 pilot
   jobs); for R it loads KH and K1 at five seeds; for both arms the parent D27 (j875-879). The I4 recipe reference of a
   confirmation S2 job (1037-1044) is the arm's s16 pilot job of the same kind. `--results-dir` (run_khop4.sh passes
   it) stops the readout when any id of the population has two landed tarballs; run_khop4.sh also stops on two
   tarballs for one id and selects the arm's TAG on 1037-1044 (results_<id>_4b_gsmvt_A2-khop4<cur|k1x><l|r>w*).
6. Generator flags (1.4). `--khop3-root` is the data root that holds student_khop3_* and eval/khop and eval/khop3 (the
   6 inputs and the 2 parent reports); the report holds no timing field, so two builds are byte-equal.
7. Bundle check (section 5). stage/check_bundle_khop4.py runs the exact-file-set lines of section 5 on the built tarball
   (pin, student_khop4 files and md5s, no student_khop3 path, eval/khop4, eval/khop, the 28 envs byte-equal to
   stage/condor/jobs_x1, the 21 top-level eval files against the x1khop3 bundle, build_khop4.py and its test).
8. Compute print. prep_khop4_confirm.py --dry-run prints L 16.54 GPU-h (8 x 1.518 + 8 x 0.55 = 16.544); the card's 16.55
   is the same value rounded up. R prints 18.67.

## Amendment 3 (2026-10-28 03:42 CDT, builder, fix round after the cold build critic; no bar, arm, id, schedule or verdict order changes)

The cold build critic (E/src_khop4/critique/BUILD_CRITIC_khop4.md) returned SUBMIT AFTER FIXES. Finding F1 changes
what the gate prints on a mis-run pilot, so the lead accepts this amendment together with Amendment 2 before the
bundle (a RESULTS_LOG line), or the build goes back. No khop4 job exists; nothing here was chosen on data.
1. Run integrity in the gate (F1). Amendment 2 item 3 moved the pilot run checks from C8 to a-Q1 / a-Q2. With the 0.5
   rule as written, a run fault of R (wrong replay file, a NaN final loss, a truncated tarball) failed only R lines,
   and the gate printed "GATE PASS ARM L". That made the arm choice depend on a build or cluster fault. The 0.5 table
   puts "not run as registered" in the row "GATE FAIL ... none". The rule is now:
   - When any of R-Q1, R-Q2, L-Q1, L-Q2 FAILs, or an a-Q7b line FAILs because a wall is missing ("a wall is missing"),
     arm a is "not run as registered". The rule is the same for both arms.
   - The gate then prints "READING  not run as registered (arm a): <lines> FAIL; rescue the node under the same id
     (card 0.5); nothing of this arm is read and no arm is chosen", and the last line is "GATE FAIL: <lines>".
   - A mis-run L blocks a passing R too: the pilot is rescued under the same ids (the rescue walls count against 6.0
     and 25, section 3), and the gate runs again.
   - Both arms are still evaluated and printed (27 PASS / FAIL lines). The 0.5 failure readings ("kept, GSM damaged",
     "lost, GSM intact", "control invalid") are printed only for an arm run as registered; "installed, then lost (both
     arms)" only when both arms are.
   - The arm rule of 0.5 then reads only the lines Q3-Q6, Q9-Q11 and the measured-compute part of a-Q7b (a projection
     over 23.0 GPU-h fails the arm as before, and the other arm can still be chosen).
   - stage/prep_khop4_confirm.py (gate_arm_text) and readout/khop4_table.py (gate_text_arm) refuse a gate output in
     which any of R-Q1, R-Q2, L-Q1, L-Q2 FAILs, an a-Q7b line FAILs on a missing wall, or a "READING  not run as
     registered" line appears. Section 4's refusal list gains this line.
   This withdraws BUILD_REPORT section 7 item 2 ("the other arm can still be chosen").
   In practice an R pilot without a wall also fails C7c (the pilot walls are in C7c), so the a-Q7b clause is a second
   guard; the 975 / 985 walls are pinned by C1.
2. Zone-aware walls (F2). The AP logs say CDT until 2026-11-01 02:00 and CST after. jobs/round_gpuh.py raises on CST,
   and khop_pilot_gate.wall_hours ignores the zone word. New readout/khop4_gpuh.py reads each stamp with its offset
   (CDT = UTC-5, CST = UTC-6); another zone word, or a missing start or exit line, gives no wall (fail closed, never
   0 h). The gate (C7c, a-Q7, a-Q7b) and khop4_table.py (the "no wall" check and the timing / compute lines) use it.
   On CDT logs the value equals the parent rule exactly (test). run_khop4.sh now prints `khop4_gpuh.py 1025 1049
   <results dir>` in place of `round_gpuh.py 1025 1049`. It reads the results dir given to run_khop4.sh, not always
   E/results, and it stops on an n/a row or a non-zero exit (an unreadable or truncated tarball is an n/a row).
   The measured family compute is still reported only; no verdict reads it.
3. Readout lookups (F3). run_khop4.sh makes one tarball lookup per assignment, so a missing first tarball of a pair
   stops the script at the lookup (before, the status of the compound assignment was that of the last lookup only).
4. INFO (F4). The gate prints "INFO j<id> max history lr <value> (args.lr <value>; ...)" per pilot S2 job. No line
   reads it; the LR decision stays in a-Q2.
5. Fail closed (found by the F1 tests). The gate's own log reader now treats a truncated .tgz (EOFError, zlib error)
   as unreadable. Before, a truncated pilot tarball raised an exception and the gate stopped without output.

## Submission note (2026-10-28 12:25 CDT, sev-e0)
Amendments 2 and 3 accepted by the lead (RESULTS_LOG "khop4 build accepted"). Codex round 7: VERDICT: SUBMIT. Bundle d19170e8, BUNDLE OK, AP copies md5-checked. Smoke DAG smoke_x1khop4_4b.dag submitted next; the pilot follows only on the smoke gate of section 3.

## Confirmation note (16:12 CDT, sev-e0)
Pilot gate: GATE PASS ARM R (exp/logs/khop4_pilot_gate_20261028.txt). Arm L failed L-Q6 ("kept, GSM damaged"). Per section 4, the arm R confirmation (j1029-1044, confirm_x1khop4_4b.dag md5 85e08a0e) is submitted without asking. Ids 1045-1049 stay unused.

## Readout note (2026-10-28 20:10 CDT, sev-e0)

The five-seed readout of arm R ran once, as pre-registered (exp/logs/khop4_readout_20261028.txt). VERDICT (arm R): NEGATIVE. D_2 = -5.7 [-9.4, -2.0] (t -4.27, n 5); C_2 = -1.6 [-5.7, +2.5]. rc3 reading NULL. Installation and N1 hold. Confirmation compute 22.03 GPU-h, under the cap of 25. No re-run is planned. Ids 1045-1049 stay unused.
