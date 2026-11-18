# B. Serial steps in one forward pass (k-hop chains)

This folder holds the material for group B of the mechanism section: the k-hop runs of October 2026 (rounds khop, khop2, khop3, khop4).
`results/make_tables.py` computes every number below from the preds files of the jobs. No number is copied from a log.
Paths are relative to this folder.

## 1. Design

**Question.** How many serial arithmetic steps can Qwen3.5-4B do in one forward pass? Does chain training help GSM8K items that need more serial steps?

**Readout.** Every item is a five-option prompt. The model reads the prompt plus `\n\nAnswer:`. The prediction is the arg max of the next-token probabilities over the letters A-E, so the model writes no reasoning ("one-pass accuracy"). Chance is 20%, because the gold letter is balanced in every cell.

**Chain items** (`code/generators/build_khop.py`). One item has five chains. k ("hops") is the number of operations from a root to the queried variable.

- The five chains share one root value and one operation-kind sequence. Only the constants differ, so a reader that ignores the constants is at chance.
- Hop 1 is `+c` or `-c`. Later hops are `+c`, `-c`, `x2` or `x3`. Constants are 2-9 and values stay in 1-99.
- The options are the five chains' values at the queried depth. An item is drawn again when two options are equal, when an option equals a number in the text, or when one option is 2x or 3x another.
- `fwd` is a random interleaving of the five chains. `rev` is the exact reverse of the same item.

Example (`khop_ca_fwd`, k = 3, gold B: 39 + 5 - 8 = 36, x2 = 72). `results/prompt_examples.md` has one real prompt for every set and condition below.

```
Question: za = 39. zu = 39. te = za + 9. fo = zu + 2. pe = te − 5. ca = pe × 2. ci = fo − 8. zo = 39. lu = 39. li = lu + 4. bo = 39. da = ci × 2. ki = bo + 6. jo = zo + 5. je = ki − 4. gu = li − 4. po = jo − 8. ve = je × 2. fe = gu × 2. cu = po × 2. What is the value of cu?
Options:
A) 82
B) 72
C) 86
D) 66
E) 78
```

**Eval sets** (`data/eval/khop/`; 150 items per k for k = 1-6, 100 for k = 7, 8):

| Set | What it is |
|---|---|
| khop_ca_fwd / khop_ca_rev | Symbolic chains in forward / reverse order (1,100 each). `ca_fwd` is the main set. |
| khop_wp_fwd / khop_wp_rev | The same structure in words ("Iris has 5 more marbles than Paul."). |
| khop_vb_fwd / khop_vb_rev | Pointer chasing ("ki = bo."), no arithmetic. The options are the five roots. |
| khop_cag_fwd | GSM-size numbers (values 10-999, constants 11-99), k = 1-6. Never trained. |
| khop_ca_fwd_qs | The `ca_fwd` prompt with the query on a distractor chain, k = 1-6. It checks that the model reads the question. |
| khop_blind | Question and options only, no statements (324 items). It measures option-only shortcuts. |

**Training rows** come from the same generator with another seed: k = 1-6, tasks ca, wp and vb, fwd and rev. No training row shares a prompt or a value set with an eval item (`data/eval/khop/build_report.json`, line D3). The chain rows go into the same shuffled epoch as the 5,923 GSM8K rows (`--extra-train`). **K1** rows have the KH statements, but the query is the target's hop-1 variable. So K1 sees the same tokens and never needs composition. K1X is the same control at 27,000 rows.

**GSM cells.** Plain sets: `gsm8k` (GSM8K test, 1,260 items), `gsm8kv2` (GSM8K test, 1,317 items, distractors from partial computations), `gsm8kx` (the same 1,317 items, other problems' answers as options), `gsm1k`, `gsm1kx`, `valx` (393 validation items). The rc cells use 249 primary items of the group A hint files (`code/pfx_table.py`). In cell rc_j the prompt shows the true worked steps up to the point where j serial steps are left. rc2 was the pre-registered transfer cell.

**Models.** Qwen3.5-4B. Every run: 1 epoch, LR 3e-6 (arm L 7.5e-7), 32 rows per update (micro-batch 2 x grad-accum 16), label smoothing 0.1, bf16, 1 GPU, FSDP with cpu-adamw, target = the gold letter. Seeds 16-20. The full job environments are `code/envs/<job>.env`, run by `code/trainer/train_arm.sbatch`. Section 2.1 lists the arms. Weights exist only for jobs with `SAVE=1`: 756, 875, 975, 985 and 1029-1036, plus the A2 weights 540 and 663-667. They are on the CHTC AP at `/staging/<netid>/weights/<job name>.tar` (listed on 2026-11-17).

## 2. Results

Full tables: `results/tables.md`. Every cell per seed: `results/cells_per_seed.csv`. Seed means with 95% seed-t intervals: `results/cells_summary.csv`. GSM: `results/gsm_per_seed.csv` and `results/gsm_summary.csv`. 418 contrasts: `results/contrasts.csv`.

### 2.1 Arms

| Arm | Training data | Updates | Jobs (s16, s17-20) |
|---|---|---|---|
| A2 orig / A2 re-train (references) | 5,923 GSM rows | 186 | 520 525 530 540 542 / 663 664 665 540 666 |
| A2Z | eval only: the A2 re-train weights on the k-hop sets | 186 | 751-755 |
| K1 / KH | A2 rows + 3,000 one-hop / chain rows | 279 | 757 762-765 / 756 758-761 |
| H3 | A2 rows + the 3,000 KH rows x 3 | 467 | 885-889 |
| KD | A2 rows + 9,000 chain rows | 467 | 766-770 |
| R3 | A2 rows + the 9,000 KD rows x 3 | 1,029 | 880-884 |
| D27 | A2 rows + 27,000 chain rows (26,941 distinct value sets) | 1,029 | 875-879 |
| CUR1 / K1X1 | stage 1: 27,000 chain / one-hop rows, no GSM rows | 844 | 975 1029-1032 / 985 1033-1036 |
| CUR-R / K1X-R | stage 2 from CUR1 / K1X1: A2 rows + 3,000 KH / K1 replay rows | 844 + 279 | 1027 1037-1040 / 1028 1041-1044 |

Single-seed arms (s16): BASE0 (750, no training), A2Z96 (771, A2-96k weights), CUR-A2 / K1X-A2 (980 / 990, stage 2 without replay), CUR-L / K1X-L (1025 / 1026, stage 2 at LR 7.5e-7). Table 8 of `results/tables.md` gives the run directory, LR, data file and start weights of every run.

### 2.2 Curve data

`results/curve_ca_fwd.csv`: x = hops 1-8, one column per training setting (seed mean, percent) and a chance column (20). `results/curve_ca_fwd_long.csv` is the same data in long form, with the seed-t interval, the item count and a flag for trained hops (k <= 6).

### 2.3 One-pass accuracy on (ca, fwd), seed mean, percent

| Arm | Seeds | k1 | k2 | k3 | k4 | k5 | k6 | k7 | k8 |
|---|---|---|---|---|---|---|---|---|---|
| BASE0 (no training) | 1 | 29.3 | 18.7 | 18.0 | 20.7 | 14.7 | 16.0 | 18.0 | 24.0 |
| A2Z (answer-only, 0 chain rows) | 5 | 93.6 | 42.0 | 27.9 | 23.1 | 22.7 | 21.1 | 23.8 | 25.0 |
| K1 (3,000 one-hop rows) | 5 | 100.0 | 42.8 | 22.3 | 28.4 | 21.6 | 22.0 | 17.2 | 24.2 |
| KH (3,000 chain rows) | 5 | 99.9 | 91.1 | 42.1 | 30.4 | 22.8 | 26.0 | 21.8 | 23.6 |
| H3 (3,000 x 3) | 5 | 100.0 | 95.3 | 53.1 | 41.3 | 34.3 | 31.1 | 31.6 | 27.4 |
| KD (9,000) | 5 | 99.9 | 96.9 | 67.9 | 57.2 | 41.7 | 37.5 | 32.4 | 32.2 |
| R3 (9,000 x 3) | 5 | 100.0 | 99.3 | 84.4 | 79.9 | 64.5 | 59.5 | 46.4 | 37.8 |
| D27 (27,000) | 5 | 100.0 | 99.9 | 90.7 | 87.3 | 76.3 | 67.2 | 55.0 | 47.4 |
| CUR1 (stage 1 only) | 5 | 100.0 | 99.6 | 89.6 | 84.3 | 70.0 | 66.7 | 53.4 | 44.4 |
| K1X1 (stage 1, one-hop) | 5 | 100.0 | 17.9 | 18.9 | 21.9 | 22.9 | 22.3 | 16.8 | 21.2 |
| CUR-R (stage 2, replay) | 5 | 100.0 | 98.7 | 85.3 | 79.5 | 63.3 | 59.2 | 53.6 | 40.8 |
| K1X-R (stage 2, replay) | 5 | 99.9 | 41.3 | 24.9 | 22.7 | 22.7 | 22.7 | 16.0 | 29.2 |

k = 7, 8 are not in any training file. Seed-t intervals, per-seed values and the single-seed arms are in Tables 1-3 of `results/tables.md`. Blind accuracy at k = 3 (options only) is 11.9-23.7% for every arm. Table 5 has the other eval sets.

### 2.4 What the k = 3 cell measures (5 seeds)

| Subset of (ca, fwd, 3) | Items | D27 | A2Z |
|---|---|---|---|
| all | 150 | 90.7 | 27.9 |
| all steps + / - (an order-free signed sum is exact) | 48 | 99.6 | 22.9 |
| order-sensitive (an order-free signed sum does not give the gold) | 31 | 91.0 | 31.0 |
| hard (no drop-one-constant estimate gives the gold) | 42 | 88.1 | 31.9 |

At k >= 4 the hard subset is small: 15 items at k4, 10 at k6, 2 at k8. Table 4 of `results/tables.md` gives every subset at every k.

### 2.5 GSM, seed mean, percent (rc cells on the 249 primary items)

| Arm | gsm8kv2 | gsm8kx | gsm8k | valx | rc1 | rc2 | rc3 | rc4 |
|---|---|---|---|---|---|---|---|---|
| A2 orig | 71.4 | 60.4 | 66.5 | 62.9 | 83.6 | 65.6 | 55.1 | 49.2 |
| A2 re-train | 71.3 | 59.8 | 66.3 | 63.8 | 83.5 | 64.3 | 52.6 | 47.7 |
| KH | 71.7 | 59.8 | 67.2 | 64.0 | 82.5 | 64.7 | 52.9 | 47.7 |
| D27 | 69.7 | 58.0 | 65.0 | 60.2 | 73.5 | 58.1 | 49.2 | 42.5 |
| CUR1 (stage 1 only) | 35.7 | 35.2 | - | 35.8 | - | - | - | - |
| CUR-R | 70.0 | 58.5 | 64.2 | 62.7 | 74.2 | 59.9 | 50.8 | 43.8 |
| K1X-R | 71.2 | 59.8 | 66.1 | 63.3 | 76.9 | 61.5 | 51.5 | 45.0 |

The stage-1 jobs (CUR1, K1X1) did not evaluate gsm8k, gsm1k or the hint sets. Table 6 of `results/tables.md` has every arm.

### 2.6 Key contrasts (A - B, points, seed-paired, n = 5)

Method: `code/paired_stats.py` (a byte copy of `scripts/paired_stats.py`), function `paired_stats`. Seed-t is the 95% Student-t interval on the five per-seed differences. Two-level is the bootstrap over seeds, then over items with indices shared by all seeds (10,000 draws, boot seeds 0 and 1). "Both exclude 0" is the rule of `paired_stats.py`. Seed 16 was the pilot seed of every round, so the last column repeats the contrast on s17-20 only.

| Contrast | Cell | Items | Mean | Seed-t 95% | Two-level 95% | Both exclude 0 | s17-20 only |
|---|---|---|---|---|---|---|---|
| KH - K1 | ca_fwd k2 | 150 | +48.3 | [+36.0, +60.5] | [+38.1, +58.3] | yes | +44.8 [+33.5, +56.2] |
| KD - KH | ca_fwd k3 | 150 | +25.7 | [+10.2, +41.3] | [+14.1, +36.7] | yes | +30.8 [+21.3, +40.4] |
| D27 - KD | ca_fwd k3 | 150 | +22.8 | [+14.9, +30.7] | [+15.3, +30.5] | yes | +20.5 [+13.7, +27.3] |
| D27 - R3 (more distinct rows, same updates) | ca_fwd k3 | 150 | +6.3 | [+2.7, +9.8] | [+2.4, +10.8] | yes | +5.2 [+2.5, +7.8] |
| R3 - KD (more updates, same rows) | ca_fwd k3 | 150 | +16.5 | [+10.0, +23.1] | [+9.2, +23.6] | yes | +15.3 [+7.0, +23.7] |
| D27 - A2Z | ca_fwd k3 | 150 | +62.8 | [+57.3, +68.3] | [+55.2, +70.1] | yes | +62.8 [+54.7, +71.0] |
| D27 - A2Z | ca_fwd k3 order-sensitive | 31 | +60.0 | [+51.7, +68.3] | [+42.6, +75.5] | yes | +58.9 [+47.5, +70.3] |
| CUR-R - K1X-R | ca_fwd k3 | 150 | +60.4 | [+55.3, +65.5] | [+53.3, +67.5] | yes | +59.5 [+53.0, +66.0] |
| D27 - A2 orig | rc2 | 249 | -7.6 | [-12.8, -2.3] | [-13.3, -2.0] | yes | -8.3 [-15.5, -1.2] |
| D27 - A2 orig | gsm8kv2 | 1,317 | -1.7 | [-2.5, -0.9] | [-3.4, -0.0] | yes | -1.9 [-2.5, -1.3] |
| CUR-R - A2 orig | rc2 | 249 | -5.7 | [-9.4, -2.0] | [-10.6, -0.6] | yes | -5.7 [-11.2, -0.2] |
| CUR-R - A2 re-train | rc2 | 249 | -4.3 | [-8.4, -0.3] | [-9.5, +1.0] | no | -3.7 [-9.2, +1.7] |
| CUR-R - A2 orig | gsm8kv2 | 1,317 | -1.4 | [-2.5, -0.3] | [-3.1, +0.3] | no | -1.4 [-3.0, +0.2] |
| K1X-R - A2 orig | rc2 | 249 | -4.1 | [-6.5, -1.7] | [-8.4, +0.2] | no | -3.9 [-7.4, -0.5] |
| CUR-R - K1X-R | rc2 | 249 | -1.6 | [-5.7, +2.5] | [-6.3, +3.5] | no | -1.8 [-7.8, +4.2] |
| CUR-R - KH | rc2 | 249 | -4.7 | [-7.3, -2.2] | [-9.2, -0.3] | yes | -4.0 [-6.4, -1.6] |

### 2.7 Reading

1. Answer-only training gives one hop (93.6) but not two (42.0). Three or more hops are near chance.
2. More chain rows and more updates both add hops. At 27,000 rows the model reaches 90.7 at k = 3. It uses the step order: 91.0 on the 31 order-sensitive items.
3. The capacity does not help GSM. No five-seed chain arm is above both A2 sets on rc2. KH is 64.7 against 65.6 (A2 orig) and 64.3 (A2 re-train).
4. CUR-R keeps the capacity (85.3 at k = 3) and is 5.7 points below A2 orig on rc2. The one-hop twin K1X-R pays almost the same cost (CUR-R - K1X-R -1.6, interval includes 0). So the cost comes with the extra training, not with composition.

## 3. Conditions run

| Round | Card | Arm (jobs) | Status |
|---|---|---|---|
| khop v1 | CARD_khop_v1 | BASE0 (750), A2Z (751-755), A2Z96 (771): eval only | ran |
| | | KH (756, 758-761), K1 (757, 762-765), KD (766-770) | ran. Verdict BETWEEN: KH reaches k = 2; k = 3 is data-limited. |
| khop v2 | CARD_khop_v2 | D27 (875-879), R3 (880-884), H3 (885-889) | ran. Verdict PASS on capacity, with a GSM cost (rc2 -7.6). |
| khop v3 | CARD_khop_v3 | CUR1 s16 (975), K1X1 s16 (985), then a plain A2 epoch: CUR-A2 (980), K1X-A2 (990) | pilot only. GATE FAIL: the A2 epoch lost the capacity (k3 from 92.7 to 64.0). |
| | | confirmation 976-979, 981-984, 986-989, 991-994 | never run, because the gate failed. |
| | | 995 (stage 2 started from the A2 weights 663) | smoke test only. |
| khop v4 | CARD_khop_v4 | arm L (LR 7.5e-7): CUR-L (1025), K1X-L (1026) | pilot only. It failed the GSM line: gsm8kv2 65.2 against a bar of 68.0. |
| | | arm L confirmation (the `curl` / `k1xl` envs of 1037-1044) | never run. |
| | | arm R: CUR1 and K1X1 s17-20 (1029-1036), CUR-R (1027, 1037-1040), K1X-R (1028, 1041-1044) | ran. Verdict NEGATIVE: the capacity is kept, rc2 -5.7. |

Design changes and branches that did not run:

- khop v1, Amendment 2: the first data build had a leak. The five chains had independent roots, so a reader that chased the target to its root scored 0.85-1.00 at k <= 2. The data were rebuilt with one shared root and one shared operation sequence before any job ran. The shipped files are the rebuilt ones.
- khop v1, Amendment 3: the word "composes" needs 0.80 or more on the all-+/- items of (ca, fwd, 3). This changed the readout only.
- khop v2: the second author asked for a "2 epochs" branch. It ran as R3 and H3: rows repeated 3 times in one epoch, with matched updates. A literal 2-epoch arm never ran.
- khop v3: the first draft of the card mixed chain rows into the GSM epoch. The card dropped it for the two-stage design before the build.
- The other amendments of the four cards (`code/cards/`) changed gates, readouts and tests. Each card dates them before the first job of that card.
- khop v4: ids 1045-1049 were not used. Pilot job 1027 was evicted once and restarted (0.27 GPU-h).
- Smoke jobs (`results_smoke_*`) are short pipeline tests. They are not results.

## 4. Differences from the first author's reading

| the first author's reading | This table | Why |
|---|---|---|
| Answer-only, 5 seeds: 1 hop 93.6, 2 hops 42.0, 3 hops 27.9, 4+ hops about 23 (chance 20) | A2Z: 93.6, 42.0, 27.9. k4-k8: 23.1, 22.7, 21.1, 23.8, 25.0. | No difference. A2Z never saw the chain format. K1 sees the format with one-hop queries and has the same curve (100.0, 42.8, 22.3, 28.4). So the 2-hop limit is not a format effect. |
| About 27k unique chains: 3 hops 91, 4 hops 87, 6 hops 67, 8 hops 47 | D27: 90.7, 87.3, 67.2, 47.4 | The numbers agree. Four additions follow in the list below this table. |
| No transfer to GSM8K (original test -1.5) | rc2: D27 - A2 orig -7.6; CUR-R - A2 orig -5.7 | The transfer test was rc2, and rc2 goes down. I did not find the source of "-1.5". The candidates are in the list below. |
| Very different training steps (186 vs 1,029) | A2 186; D27 and R3 1,029 | Correct for A2 against D27. The other counts and the controls for updates are in the list below. |

Additions to the 27k row:

1. D27 has 27,000 chain rows (26,941 distinct value sets). They are mixed into one epoch with the 5,923 GSM rows.
2. Training covers k = 1-6, so 7 and 8 hops are extrapolation.
3. At k >= 4, partial readers score high on this option design. A reader with every constant but one exact scores 0.86-0.95 (`drop_one_best` in `build_report.json`). A reader that skips the last constant scores 0.55-0.64.
4. So the clean evidence for composition is at k = 3: all-+/- items 99.6, order-sensitive items 91.0, hard items 88.1. At k >= 4 only 2-15 hard items exist per cell.
5. RESULTS_LOG says "KH (k=1-3 khop data)". That is wrong: `train_kh.jsonl` holds k = 1-6.

Candidates for the "-1.5" GSM number (no log line or results-repo note that I searched gives -1.5 for a plain set):

1. D27 - A2 orig on `gsm8k` (the GSM8K test, 1,260 items): -1.6, seed-t [-3.6, +0.4].
2. D27 - A2 re-train on `gsm8kv2`: -1.6, seed-t [-3.1, -0.1].
3. CUR-R - A2 orig on `gsm8kv2`: -1.4, seed-t [-2.5, -0.3], two-level [-3.1, +0.3].
4. K1X-R - K1 on rc2: -1.5 (khop4 readout line). This is not a transfer number.

The paper must name the arm and the set when it gives a plain-set number.

Update counts and their controls:

1. K1 and KH 279, KD and H3 467, CUR1 844, CUR-R 844 + 279 = 1,123.
2. R3 has the same 1,029 updates as D27 with 9,000 distinct rows.
3. The K1X twins have the same rows, tokens and updates as CUR1 / CUR-R, with a one-hop query. K1X-R pays about the same rc2 cost as CUR-R (-4.1 against -5.7). CUR-R - K1X-R is -1.6 [-5.7, +2.5].

Rounding note: KD k2 is 96.93 here and 97.0 in the October log. The values are the same; the log rounded differently.

## 5. Code

- Base commit: `994ff5026e54cffe260d88709ce33afd4533d326` (994ff50) of the code repository. The results repo names it (`paper_inputs/2026-11-02/README.md`). the second author's GitHub account cannot see that repository, so I could not open the commit from here.
- `code/patches_vs_994ff50/khop.patch` to `khop4.patch`: one patch per round against 994ff50, copied from `paper_inputs/2026-11-02/patches/` of the results repo. Apply one patch to a clean checkout (`git apply --binary`). Check made here: each patch reverses cleanly from its round's code tree. All four reversed trees are identical, and `khop4.patch` re-applied gives the khop4 tree back.
- Code pins: the k-hop jobs print `code pin: 994ff50...-fsdp1-pfx-loop-x1-df-headlr-b3v2-khop`, with `-khop2`, `-khop3`, `-khop4` added in later rounds. The A2 references ran older trees: 520, 525 and 530 `-fsdp1`; 540 and 542 `-fsdp1-pfx`; 663-667 `-fsdp1-pfx-loop-x1` (patches `opc_fsdp1`, `pfx` and `x1base` in the same folder). Every env has `COMMIT=''`, so `metrics.json` records the commit as unknown.
- `code/repo_snapshot_khop4.tgz`: the khop4 code tree, without data. `code/build_diffs/`: the diff of each round against the tree before it, the cumulative diff from the pre-khop tree (pin `...-b3v2`), and each round's BUILD_REPORT. The only trainer change is the `--extra-train` flag in `scripts/train_student.py` and the smoke subset in `hpc/train_arm.sbatch`. Rounds khop2-khop4 add generators and tests only.
- `code/generators/`: `build_khop.py` (eval sets, train_kh, train_k1, train_kd), `build_khop2.py` (train_kd27, train_kd_x3, train_kh_x3), `build_khop3.py` and `build_khop4.py` (stage-1 data directories: byte copies plus the K1X twin). `code/tests/` holds their tests.
- `code/envs/`: the 86 job environments of ids 750-771, 875-889, 975-995 and 1025-1044, run and not run.
- `code/cards/`: the four pre-registered cards with amendments, and the khop2 adversarial check.
- `code/readout_original/`: the October readout scripts, for reference only.

## 6. Data

`data/md5.txt` lists every file, its md5, the md5 of the content that the jobs read, and the row count. Files over 20 MB are gzipped (`gzip -n`). The md5 of each uncompressed file equals the md5 in the build reports.

Rebuild check, run on 2026-11-17 in a scratch folder:

```
python3 code/generators/build_khop.py --out <tmp>/khop
python3 code/generators/build_khop2.py --src <tmp>/khop --out <tmp>/khop2
python3 code/generators/build_khop3.py --khop2 <tmp>/khop2 --khop <tmp>/khop --gsm-dir <dir with the gsmvt val.jsonl> --out-root <tmp> --report <tmp>/khop3
python3 code/generators/build_khop4.py --khop3-root <tmp> --out-root <tmp> --report <tmp>/khop4
```

All 23 generated .jsonl files had the md5 in `data/md5.txt`. The generators need no network and no tokenizer. The tokenizer only adds token counts to `build_report.json`.

## 7. How to rerun the tables

```
cd results
python3 -I make_tables.py           # about 50 s; needs numpy; reads item_preds.json.gz and ../data
python3 -I make_prompt_examples.py  # writes prompt_examples.md
python3 -I extract_preds.py --results-dir <dir with results_<id>_*.tgz>          # optional: rebuilds item_preds.json.gz
python3 -I crosscheck_readouts.py <dir with the October readout .json files>     # optional
```

- `item_preds.json.gz` holds the one-pass pick of 71 jobs on every item of the k-hop, hint and plain GSM sets. It also holds each job's code pin, update count, LR and start weights.
- `extract_preds.py` builds it from the result tarballs. It checks item ids, gold labels and NaN probabilities against `data/`.
- Two runs of `make_tables.py` give byte-equal outputs. A copy of this folder in another place gave the same md5s.
- `crosscheck_readouts.py` found 0 differences in 3,540 per-seed curve cells and 60 per-seed rc differences against the October readout files.

## 8. Not recoverable

- The commit 994ff50 itself: I could not reach the code repository from the second author's account. the first author can check that the patches apply.
- Weights of every run without `SAVE=1`, that is every arm except jobs 756, 875, 975, 985 and 1029-1036. A new readout on those models needs a retrain.
- The results repo holds one `A2-c1lr3ch-...-s<seed>` directory per seed: the re-train runs 663-666 and j540. The original A2 runs 520, 525, 530 and 542 are only in the local tarballs (`exp/results/` on the second author's Mac) and in `item_preds.json.gz`.
- The hint files in `data/eval/hint/` come from group A. Their generator is part of group A's material.
