# Adversarial check: CARD_khop_v2, arm D27 (jobs 875-879, seeds 16-20)

Date: 2026-10-27. Checker: an independent adversarial sub-agent. Inputs: the raw preds in the result tarballs and the
eval and training files. The checker did not change the card, RESULTS_LOG or any readout script.

Claim under test (from the readout): after 1 epoch on 27,000 chain rows, Qwen3.5-4B "composes 3 + / - steps" in one
pass, with a(ca, fwd, 3) = 0.907 [0.889, 0.924] and k* = 4.

## Verdict: HOLDS WITH CAVEATS

All the headline numbers recompute exactly. No shortcut, leak or options-only rule explains the result. The strongest
attack fails. On the items where the order of operations decides the answer, an order-free sum scores 0.151 and D27
scores 0.910. The caveats limit the scope of the claim (small numbers, symbolic form, GSM cost). They do not limit its
truth.

Run all checks: `bash <tmp> (11 s, standard library, `python3 -I`). Each
script writes `<script>.out` next to it.

## Check 1: recompute (adv1_recompute.py)

Integrity checks:

- The md5 of all 9 eval files and of train_kd27.jsonl agree with the two build reports.
- The own parser agrees with build_khop.parse_prompt on all 6,500 arithmetic rows (0 mismatches). On each row, gold equals the queried chain's value.
- Each tarball has 1,100 ca_fwd preds. The gold labels agree with the eval file (0 mismatches).
- The argmax accuracy equals metrics.json for each seed (for example, s16: 0.8100).
- metrics.args seed / order_seed = 16..20 for j875..879 and for the paired KD j766..770.
- The log shows `extra-train: 27000 rows from data/eval/khop2/train_kd27.jsonl`.

| line | recomputed | readout |
|---|---|---|
| (ca, fwd, 3) per seed | 92.7 / 89.3 / 89.3 / 90.7 / 91.3 | same |
| (ca, fwd, 3) | m 0.907 CI [0.889, 0.924] | same |
| k* (SOLVED: m >= 0.80 and CI lower >= 0.70) | 4 (k4 0.873 [0.849, 0.897]; k5 0.763 [0.695, 0.830] fails) | 4 |
| D_dose = D27 - KD at k = 3, paired | +22.8 [+14.9, +30.7] | same |
| all-+/- subset (n 48) | D27 0.996 [0.984, 1.007]; KD 0.875 | same |

## Check 2: shortcuts (adv2_shortcuts.py)

The five chains share one root and one op-kind sequence. Thus a reader that adds the target chain's signed constants,
in any order and without the x steps, finds the gold on every all-+/- item. The 48-item all-+/- line (0.996) therefore
does not show serial composition. The test that decides the claim is on the other items.

Order-sensitive items (OS). On these items the rank_sum probe does not single out the gold. A x2 / x3 between + / -
hops re-weights the earlier constants, so the order-free sum gives a wrong option or a tie.

| subset of (ca, fwd, 3) | n | order-free reader | D27 | KD |
|---|---|---|---|---|
| all | 150 | 0.824 | 0.907 [0.889, 0.924] | 0.679 |
| n_add = 3 (all + / -) | 48 | 1.000 | 0.996 [0.984, 1.007] | 0.875 |
| n_add < 3 (has x) | 102 | - | 0.865 [0.841, 0.888] | 0.586 |
| OS | 31 | **0.151** | **0.910 [0.876, 0.943]** | 0.535 |
| hard (v1 Amendment 2, build_khop.hard_item) | 42 | - | 0.881 [0.845, 0.917] | 0.652 |
| OS and hard | 9 | - | 0.778 [0.582, 0.973] | 0.511 |

Two limits apply to the OS line:

- Its CI is over seeds, with the 31 items fixed. Item sampling adds about 0.05 (binomial SE at p = 0.91, n = 31). The band is then about 0.81-1.00. It still clears 0.80.
- No OS item at k = 3 has n_add = 3, because an all-+/- chain is order-free by construction. The serial-order evidence thus comes from mixed chains (1-2 + / - hops and a x step). The all-+/- line shows binding and summing of 3 constants; it does not show serial order.

A text-position proxy (weight a constant by where it occurs in the prompt) does not explain the OS line. In rev order the text position flips, and D27 on (ca, rev, 3) OS is still 0.845.

The same pattern holds at other k. OS items at k = 4: order-free reader 0.252, D27 0.810 [0.744, 0.875] (n 41). At
k = 5: 0.152 against 0.673. At k = 6: 0.143 against 0.575. On (ca, rev, 3) OS, D27 scores 0.845. On (wp, fwd, 3) OS,
D27 scores 0.745 (n 22).

Heuristic tracking on (ca, fwd, 3), five seeds. "h wrong" means the heuristic has a unique pick that is not the gold.

| heuristic | data acc | P(D27 = h \| h wrong) | P(D27 = gold \| h wrong) |
|---|---|---|---|
| rank_sum (order-free signed sum) | 0.824 | 0.022 | 0.922 |
| rank_h1 (first constant) | 0.607 | 0.068 | 0.868 |
| rank_last (last + / - constant) | 0.379 | 0.043 | 0.887 |
| rank_last2 | 0.658 | 0.013 | 0.947 |
| drop first / last constant | 0.341 / 0.576 | 0.047 / 0.080 | 0.879 / 0.844 |
| max / min / median option | 0.167 / 0.273 / 0.213 | 0.002 / 0.050 / 0.025 | 0.899 / 0.894 / 0.919 |
| centroid / farthest option | 0.180 / 0.180 | 0.025 / 0.017 | 0.923 / 0.891 |

No heuristic is followed against the true value. When a heuristic is wrong, D27 picks the gold 84-95 % of the time.

Error shape:

- D27 makes 70 wrong picks at k = 3 (5 seeds x 150 items).
- 56 of the 70 are the option next to the gold in sorted order. A uniform wrong pick gives 30 such picks. The other 14 are 2 ranks away.
- Thus the errors are near-miss magnitude errors, not wrong-chain picks.
- When D27 is wrong, it picks the min option 46 % of the time (chance 25 %). The rank_h1 and drop_last picks show a similar share (45 %), but these rest on only about 4 wrong picks per seed. Report this as a lean, not as a policy.

## Check 3: leakage and options-only rules (adv3_leakage.py)

build_khop2.py drops a training row when its prompt hash or its whole-item value key matches a row of the 9 eval files.
The overlap counts against train_kd27 (27,000 rows) are:

- Prompt body: 0. Statement set (same sentences in any order): 0. Whole-item value key: 0, on all 6 arithmetic sets.
- Target chain (same root, kinds and constants as a chain or chain prefix in training): 359 of 1,100 ca_fwd rows. Most of these are k = 1, 2. At k = 3 the count is 59 of 150, because the chain space is small.
- Seen target chains do not help. D27 on (ca, fwd, 3) scores 0.885 [0.844, 0.925] on the 59 seen items and 0.921 [0.906, 0.936] on the 91 unseen items.
- Option set (same five values for the same task and k): 117 ca rows. Training never shows the chain values, so the model cannot learn a value-to-option lookup.

Options-only rules:

- The gold's rank among the sorted options is about uniform in eval and in training.
- A rule that picks the most frequent training gold rank per (task, k) scores 0.133-0.227 per cell.
- The gold letters are exactly balanced (220 each). D27's picked letters are flat.
- Blind (statements removed): D27 scores ca 0.172, wp 0.154 and vb 0.185 (108 items per task per seed; chance 0.20). The options alone do not identify the gold.

## Check 4: consistency (adv4_consistency.py)

| cell, D27 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| ca rev | 0.999 | 0.903 [0.870, 0.936] | 0.895 |
| wp fwd | 0.992 | 0.859 [0.832, 0.886] | 0.755 |
| wp rev | 0.995 | 0.933 | 0.861 |
| ca_fwd_qs (query on distractor D1) | 0.996 | 0.900 [0.873, 0.927] | 0.857 |
| cag (values 10-999) | 0.853 | **0.583 [0.485, 0.680]** | 0.476 |

- fwd - rev, ca: +0.4 [-3.0, +3.8] at k = 3. Premise order has no detectable effect on symbolic chains.
- fwd - rev, wp: -7.5 [-10.2, -4.7] at k = 3 and -10.7 at k = 4. The word form is worse in forward order.
- The all-+/- subset of (wp, fwd, 3) is 0.863, against 0.996 in the symbolic form.
- qs twins at k = 3: fwd - qs is +0.7 [-3.7, +5.0]. Only 2.4 % of qs picks equal the original target's value. The model reads the query; no artifact is tied to the target chain.

## Check 5: GSM cost (adv5_gsm.py)

The rc cells use an own copy of the pfx_table rule (249 / 249 / 249 / 128 items). The plain sets come from
preds_<set>.jsonl and agree with metrics.json. best-A2 is the A2-6k set with the higher five-seed mean. This is ORIG for
all lines except valx (RETRAIN). Seed 19 uses j540 in both sets.

| line | D27 - best-A2 | D27 - KD |
|---|---|---|
| rc1 | -10.1 [-13.8, -6.5] | -2.8 [-7.4, +1.8] |
| rc2 | -7.6 [-12.9, -2.2] | -2.1 [-6.6, +2.4] |
| rc3 | -5.9 [-9.3, -2.6] | -1.8 [-6.6, +2.9] |
| rc4 | -6.7 [-13.3, -0.1] | -3.3 [-9.0, +2.5] |
| gsm8kx | -2.4 [-4.0, -0.7] | -1.9 [-3.6, -0.1] |
| gsm1kx | -3.0 [-4.9, -1.1] | -1.7 [-4.0, +0.7] |
| gsm8kv2 | -1.7 [-2.5, -0.9] | -1.4 [-2.5, -0.3] |
| valx | -3.7 [-5.6, -1.7] | -2.8 [-7.5, +1.9] |

All lines agree with the readout to 0.1 point.

## Findings against the card text (no effect on the verdict)

1. The card's premise is false: "x2 / x3 steps are shared by the five chains and never decide the answer" (v1 Amendment 3, v2 section 0 item 2). A x step between + / - hops changes the weights of the constants. On 31 of the 150 (ca, fwd, 3) items the order-free sum gives a wrong option (18 items) or a tie (13 items), and it scores 0.151 there. The error makes the card more cautious than it needs to be. D27 also handles these items (0.910).
2. The headline 0.907 rests mostly on items an order-free reader can solve (119 of 150). The OS line (n 31) is the evidence that serial order is used. That line comes from this check and is not in the readout. Its n is small. The OS-and-hard intersection (n 9, 0.778 [0.582, 0.973]) cannot carry a claim.
3. The readout does not print the hard subset for D27. D27 on it scores 0.881 [0.845, 0.917] (n 42).

## Wording

These statements can be reported:

- "After 1 epoch on 27,000 chain rows, one forward pass of Qwen3.5-4B selects the value of a 3-step symbolic chain at 0.907 [0.889, 0.924] (five seeds), k* = 4; on the 48 all-+/- items it is 0.996."
- "It composes 3 + / - steps." This is the card's Amendment 3 wording, and the check supports it.
- "This is not an order-free shortcut. On the 31 mixed-op items where a x step re-weights earlier constants, an order-free sum scores 0.15 and the model 0.91 (seed CI [0.88, 0.94]; with item sampling about 0.81-1.00)." Say that this evidence is on mixed chains, not on the all-+/- items.
- "No leak or options-only rule explains it." Use the overlap, blind and options-only numbers above.
- "The chain dose costs GSM accuracy": rc1 -10.1 and plain -1.7 to -3.7 points against best-A2. Always report this next to the claim.

These statements cannot be reported:

- "Composes 3 steps" without a scope. The result is for small numbers (values 1-99, constants 2-9) in symbolic form. With GSM-sized numbers (cag), k = 3 is 0.583. In word form (wp, fwd), the all-+/- subset is 0.863 and k* = 3.
- Any claim that this gives GSM multi-step reasoning. The rc lines go down, not up.
- "Exact" composition. The errors are near misses (56 of 70 wrong picks are the neighboring option). This fits an approximate magnitude computation as well as exact arithmetic with noise.
- Any claim about the hard and order-sensitive intersection (n 9).
- Any claim that the gain comes from distinct rows alone. The readout labels branch (b) BOTH (readout value; this check did not recompute it).

## Files

All files are in <tmp> (E = <tmp>

- adv_common.py (loader, own parser, stats: t = 2.78 at n = 5, sample SD)
- adv1_recompute.py, adv2_shortcuts.py, adv3_leakage.py, adv4_consistency.py, adv5_gsm.py, and their .out files
- run_all.sh
