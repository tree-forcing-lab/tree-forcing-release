# Pre-registration card: k-hop serial capacity of one pass on symbolic chains (idea #4, family khop), 2026-10-25, sev-e0

Status: DRAFT 2026-10-25 22:27 CDT, written before any build, bundle or submission. Amendment 1 (22:27 CDT, after the
cold critic CARD_CRITIC_khop.md, before any build) is merged into the sections below and listed at the end. The cold build
critic (BUILD_CRITIC_khop.md) ran on the Amendment 1 build. Amendment 2 (2026-10-26 00:21 CDT, after the cold build critic BUILD_CRITIC_khop.md, before
the rebuild of the data and before any bundle or submission) is merged into the sections below and listed at the end.
Amendment 2 items 1 and 3 (shared skeleton, the "PASS (approximate)" reading) have had no cold review yet.
Owner's words: "可以那你这几个都做吧 自己写内容排任务" and "你安排好了自动排队就行 能跑的全部跑" (2026-10-25, HANDOFF section 14).
Family khop, job ids 750-774 only. Code copy E/src_khop/repo (untouched snapshot E/src_khop/repo.base), variant x1khop,
bundle cotd_code_data_x1khop_4b.tgz. E = <tmp>
One-pass rule (RULINGS_2026-10-23, tier ii): every arm is plain A2 training (letter CE after "Answer:") on extra
rows. No new mechanism, no generated tokens (G0), no recurrence; executed layers 32 of 32, one forward per item,
FLOPs and latency equal to A2 for the same prompt length. Memory and wall time are read from the logs and reported.

## 0. Question, hypothesis and pre-registered readings

Question. On clean symbolic chains in a format close to the GSM items, how many serial hops k can one forward pass of
Qwen3.5-4B compose after a small direct fine-tune? Is that number above what the GSM rc curve shows at the same depth?

Background in the ledger. A2-6k sits at about rc1 83 / rc2 65 / rc3 55 on the 249 primary items (chance 20).
More data (CARD_cN_v1), a latent loop (CARD_loop_v1, CARD_loop96_v1), depth forcing (CARD_df_v1) and a prefix-LM mask
(CARD_bidir96_v1) did not move rc2. CARD_diag_v1: intermediate values are decodable at the operand tokens, but the
answer position does not use them. CARD_b1_v1: removing used premises lifts rc2 by about 10 at 96k (premise
interference). These results point to two different limits: serial depth of one pass, or parsing and premise binding.
This card separates the two with a task where parsing is trivial and depth is set exactly.

What is new here (Amendment 1, finding 7). CARD_diag_v1 and CARD_b1_v1 already point to binding, so a PASS mostly
confirms them. The NEGATIVE reading and the fwd - rev contrast carry the new information. The readout says this and
does not present a PASS as a new finding.

Prior expectation (Amendment 1, finding 6). The claim is a measurement on this model and this training set, not a
novelty claim. Sources checked by web search on 2026-10-25 (abstracts only):
- Lin et al., "Implicit Reasoning in Transformers is Reasoning through Shortcuts" (Findings of ACL 2025, arXiv
  2503.07604): GPT-2 trained from scratch on multi-step math reasons step by step implicitly when the premise order
  is fixed; with unfixed premise order it overfits to patterns (shortcuts). This predicts fwd >> rev and is the reason
  for the option rule of 1.1.
- Yang et al., "Do Large Language Models Latently Perform Multi-Hop Reasoning?" (ACL 2024): strong evidence for the
  first hop, moderate for the second hop, and no clear scaling of the second hop with model size.
- Wang et al., "Grokked Transformers are Implicit Reasoners" (NeurIPS 2024): two-hop composition is learned only
  with training far past overfitting, and it does not generalize out of distribution.
- Ye et al., Physics of Language Models Part 2.1 (iGSM): not checked here; cite only after a check.
- Yao et al. (EMNLP 2025, cited in CARD_cN_v1): data needs grow fast with k.
Prior: k = 2 in fwd order is likely SOLVED (about 0.7, in-context chains with explicit ops, 4B pretrained model). k = 3
in fwd order at 1 epoch and 3,000 rows is about even (about 0.45), given Wang et al. and Yao et al. on the data and
schedule needs. rev is likely below fwd at k >= 2 (Lin et al.). Prior for PASS about 0.35; for NEGATIVE about 0.25
(it also needs KD flat); the rest BETWEEN or NULL.

Hypothesis H (the "binding, not depth" reading). After the fine-tune, one pass solves symbolic chains of k = 3 in
reading order (cell accuracy >= 0.80), and at k = 2 it is at least 15 points above the GSM rc2 cell of the same model.

Notation. a_X(t, o, k) = accuracy of arm X on task t, premise order o, chain length k, one value per seed.
m = five-seed mean; CI = m +/- 2.78 x SD / sqrt(5) (df 4). A cell is SOLVED when m >= 0.80 and the CI lower bound
>= 0.70. A cell is FAILED when the CI upper bound < 0.80. k*(X, t, o) = the largest k <= 6 with cells 1..k all SOLVED
(0 when k = 1 is not SOLVED). rc2_ref(s) and rc3_ref(s) = the rc2 / rc3 cell of the 249 primary items (pfx_table.py)
of the reference model at seed s: the model with the highest five-seed mean on that cell among KH itself, the
A2-6k originals and the A2-6k re-trains (section 2.3). This is "the same model" when KH is best and the best-A2 rule
otherwise; both make G2 smaller. G2(s) = a_KH(ca, fwd, 2)(s) - rc2_ref(s); G3(s) = a_KH(ca, fwd, 3)(s) - rc3_ref(s).

Readings, evaluated in this order; the first that applies is the verdict:
1. MANIPULATION FAILED: any installation check of section 2.4 fails at five seeds. No depth reading; report the
   failing line.
2. PASS (H holds): k*(KH, ca, fwd) >= 3 AND m(G2) >= +0.15 with the G2 CI lower bound > 0.
   Claim (Amendment 1, finding 4): "one pass of Qwen3.5-4B composes at least 3 given operations on small integers in
   reading order, while the GSM rc2 cell stays at least 15 points lower; the rc2 ceiling is not a limit on composing
   given operations." Amendment 2: the word "composes" also needs m(a_KH^hard(ca, fwd, 3)) >= 0.80, the five-seed mean
   on the hard subset of the cell (2.4, D1 reported lines: the items where every constant of the target chain is
   needed). Else the verdict is "PASS (approximate)" and the claim says "selects the value of a 3-op chain; on the
   items that need every constant the mean is <x>", with no "composes". The claim names no cause. The listed candidates for the gap are: operation and operand selection
   (GSM steps are not written), number size (ca values <= 99; the reported ca-gsm set of 1.1 checks this), the option
   design (5-option accuracies of two tasks are not on one scale), and parsing / premise binding. Sub-readings (reported, no new claim): WP (word-problem chains) k* >= 3 too -> the cost is
   GSM-specific (quantity and operation selection), not the wording of chains; WP k* <= 2 -> natural-language binding
   is the cost. k*(KH, ca, rev) <= 1 while fwd >= 3 -> the hops are composed spread across the reading positions,
   not all after the last statement, which matches CARD_diag_v1 (values at the operand tokens).
3. NEGATIVE (H fails, the capacity reading; Amendment 1, findings 2-3): cell (ca, fwd, 3) is FAILED under KH, AND
   cell (ca, fwd, 3) is FAILED under KD, AND the paired difference a_KD(ca, fwd, 3) - a_KH(ca, fwd, 3) over s16-20 has
   its CI upper bound < +0.05 (dose-flat). KD always runs (section 4). Claim: "at this data budget and schedule
   (3,000 and 9,000 chain rows, 1 epoch, LR 3e-6), direct training on clean symbolic chains does not make one pass of
   Qwen3.5-4B compose 3 hops reliably, and tripling the chain rows does not move it; the GSM rc2 / rc3 ceiling is
   consistent with a serial-capacity limit at this schedule." If cell (ca, fwd, 2) is also FAILED (KH and KD), the
   claim is k* <= 1. The claim text always carries the budget and schedule; more epochs or a higher LR need a new card.
4. NULL: not 1-3, and the G2 CI includes 0. The symbolic k = 2 level and GSM rc2 do not differ, so the diagnostic
   does not locate the limit. If this happens because KH's own rc2 rose, that is a transfer observation for a new card,
   not a claim here.
5. BETWEEN: everything else. Examples: cell (ca, fwd, 3) neither SOLVED nor FAILED; k* >= 3 with G2 CI lower > 0 but
   m(G2) < 0.15; KH fails k = 3 and KD solves it ("data-limited at 3,000 rows, not capacity-limited"); KH and KD
   both fail k = 3 but KD - KH has CI upper >= +0.05 ("data- or schedule-limited": the curve still climbs with dose).

## 1. Data and arms

### 1.1 Tasks (generator scripts/build_khop.py in E/src_khop/repo; outputs in E/src/data/eval/khop/)

Every item is a 5-option multiple-choice prompt in the GSM eval layout: "Question: <statements> <query>\nOptions:\nA) v\nB) v\nC) v\nD) v\nE) v".
No "Answer:" in the prompt (the trainer and eval_student.py add the cue). Glyphs as in the hint sets: "+", "−" (U+2212), "×" (U+00D7).
Variable names: two-letter consonant-vowel syllables ("bo", "ki", "ze"; 19 consonants x 5 vowels, minus English
words and minus "is", "to", "do", "no", "so", "go", "me", "we", "be", "he"), distinct within an item. All values are
integers in [1, 99].

- ca (chained arithmetic, the PRIMARY task; option rule changed by Amendment 1, finding 1). Five chains (target T,
  distractors D1..D4), each of length k: a root "bo = 7." (root value uniform in [2, 60]) and k op statements
  "ki = bo + 5." / "ki = bo − 3." / "ki = bo × 2." / "ki = bo × 3." (addend / subtrahend uniform in [2, 9]).
  Shared skeleton (Amendment 2, build critic finding 1): the five chains of an item have one root value r (uniform in
  [2, 60]) and one op-kind sequence; only the constants differ. Hop 1 is + or − (uniform over the two kinds whose
  constant range holds five values), with five distinct hop-1 constants drawn without replacement. Hops 2..k are uniform
  over the kinds that keep the value of every chain in [1, 99]; each chain draws its own constant. Query "What is the
  value of <T_k name>?". Options = the five final values {T_k, D1_k, .., D4_k}: kinds gold, cross, cross, cross, cross.
  Why: the chains differ only in the constants, so the target's root and op kinds are the same for all five options.
  A policy that chases the target to its root and applies its op kinds without reading its constants is at chance by
  symmetry (the build critic's "chase + estimate" policy was 0.85-1.00 at k <= 2 under Amendment 1). ca now has no
  root-value binding part; pointer chasing alone is the vb task.
  Resample the whole item when any of these holds, for the final-value set and for the hop-1 set {T_1, D1_1, .., D4_1}
  (the K1 options): two options are equal; an option equals a number written in the text (roots, constants and the
  multipliers 2 / 3 of "× 2" / "× 3"); one option is 2x or 3x another. Traded (Amendment 2): the Amendment 1 rule
  "|g - h| in [2, 9]" is dropped. Five options that differ only in constants from [2, 9] cannot meet it. Its leak is
  measured instead: the statement-free one-op-parent line is 0.19-0.21 on every cell of the author check below. The
  shared op kinds make "query-op consistent" chance by construction.
  The generator stops with an error after 10,000 resamples of one item and build_report.json records the resample
  rate per (task, k). Author check for Amendment 2 (E/src_khop/critique/sim_a2.py, the generator at the real row counts on
  other seeds, train 101 and eval 102; output sim_a2.out): d_pass, about 1.2-2.7 draws per item at every k (k = 6, 7,
  8: 2.05 / 2.03 / 1.91 for ca, 2.06 / 2.12 / 1.82 for wp), chase-mult 0.13-0.31 and chase-sign 0.14-0.31 per cell
  (pooled 0.18-0.23) on ca / wp / cag / -qs and the ca rows of train_kh / train_k1, every old line in [0.07, 0.32]
  on these sets. Superseded:
  sim_khop_a1.py (Amendment 1 rule).
- ca-gsm (reported only; Amendment 1, finding 4, optional part). ca with GSM-sized numbers: roots in [10, 400], addends
  in [11, 99], values in [10, 999], the shared skeleton and the option rule of ca (Amendment 2). fwd only, k = 1..6 x
  150. Author check sim_a2.out: every barred D1 line in [0.14, 0.28] per cell. No verdict line uses it; it bounds the
  "small integers" scope of the PASS claim.
- wp (word-problem chains, the GSM-format bridge). The ca structure in words: root "Mia has 7 marbles."; ops "Leo has
  5 more marbles than Mia." / "... 3 fewer marbles than Mia." / "... twice as many marbles as Mia." / "... three
  times as many marbles as Mia." One object noun per item (pool of 20 plural nouns), people from a pool of 60 first
  names (single words, none a prefix of another). Query "How many marbles does <T_k name> have?". Five chains, the
  shared skeleton (five people with the same root count), options, kinds and the resample rule as ca.
- vb (variable binding / pointer chasing, no arithmetic). Five chains of length k: root "bo = 47." (distinct values in
  [10, 99]), copies "ki = bo.". Query "What is the value of <chain-1 last name>?". Options = the five roots (gold +
  four cross).

Premise order o (the shuffled-order manipulation of the idea, taken to its extreme):
- fwd: a uniformly random topological order of all statements of the five chains, interleaved. Every
  statement comes after the statement it reads. Compute-while-reading is possible.
- rev: the exact reverse of the same item's fwd order. Every statement comes before the statement it reads. Under
  causal attention the hops cannot be composed while the statements are read: they must be composed after the last
  statement of the chain (the root statement), at the positions of the root statement, the query, the options and
  "Answer:". The DeltaNet layers also carry a state over these positions.
fwd and rev eval rows are the same items (same item_id, field "order"), so fwd - rev is paired by item.

Gold letter: balanced per (set, k) cell by cycling the five positions, then a seeded shuffle of the other four.
Row fields: item_id, source "khop", task, order, k, group "khop/<task>/<order>/k<k>", type "choice", prompt, labels
[A..E], label_order, gold_label, gold_probs null, option_kinds (a list aligned with options, as in the gsmvt rows),
options, depth = k. Training rows carry every key of the gsmvt train rows (data/student_gsm8k_tf_vt/train.jsonl, 32
keys; Amendment 1, finding 10): item_id, source "khop", group, type "choice", prompt, labels, gold_label, teacher
(one-hot gold), options, option_kinds, recipe "khop", depth k, dep_level k, subqs [], random_subqs [], rationale "",
augment null, has_tree false, tree_fix false, row_kind "final", problem_id = item_id, node_status "ok", licence_class
"PERM", licence "synthetic (build_khop.py)", label_basis "gold", node_basis "program", pool_rank 0, n_steps k,
tree_source "khop", value_nodes [], worked_steps [], verify_nodes []. Eval rows carry the same keys plus task, order, k.
tests/test_build_khop.py asserts the key set equals the gsmvt key set (plus the three eval keys).

### 1.2 Files

| File | Rows | Content |
|---|---|---|
| khop_ca_fwd.jsonl, khop_ca_rev.jsonl | 1,100 each | k = 1..6 x 150, k = 7, 8 x 100 (extrapolation, reported only) |
| khop_wp_fwd.jsonl, khop_wp_rev.jsonl | 1,100 each | as ca |
| khop_vb_fwd.jsonl, khop_vb_rev.jsonl | 1,100 each | as ca |
| khop_cag_fwd.jsonl | 900 | ca-gsm, fwd only, k = 1..6 x 150 (reported only) |
| khop_ca_fwd_qs.jsonl | 900 | query-swap twin of khop_ca_fwd k = 1..6: same statements and options, query on D1_k, gold D1_k; item_id "<id>-qs" |
| khop_blind.jsonl | 324 | 3 tasks x k = 1..6 x 18: the first 18 items per (task, fwd, k) cell; their options and query, statements removed ("Question: What is the value of ki?") |
| train_kh.jsonl | 3,000 | KH: 1,000 per task, order fwd / rev 1/2 each, k uniform 1..6 |
| train_k1.jsonl | 3,000 | K1: the same 3,000 prompts' statements and order as train_kh, query on the target chain's hop-1 variable; ca / wp options {T_1, D1_1, .., D4_1} (the hop-1 set of 1.1); vb keeps the five-roots option set and queries chain 1's first copy |
| train_kd.jsonl | 9,000 | KD: train_kh's 3,000 rows + 6,000 new rows from the same generator (seed continues) |
| build_report.json | - | counts per cell, md5 per file, gold-letter shares, D1 lines (section 2.4: 12 barred, 1 barred on eval sets only, 5 reported), resample rate per (task, k), token totals per train file, disjointness counts |

Seeds: train generator seed 1, eval generator seed 2. Disjointness: no eval item shares a prompt hash or a (task,
sorted chain value tuples) key with any training row of any of the three train files.

### 1.3 Arms (all from the base Qwen3.5-4B except the eval-only rows)

| Arm | What | Training data | Jobs |
|---|---|---|---|
| BASE0 | eval-only, untrained base (WTAR='') | none | j750 |
| A2Z | eval-only on the A2-6k weights (zero-shot on khop) | gsmvt 6k (already trained) | j751 s16 (w663), j752 s17 (w664), j753 s18 (w665), j754 s19 (w540), j755 s20 (w666) |
| KH | A2 on gsmvt 6k + train_kh (3,000 chain rows, k 1..6) | 8,923 rows, 1 epoch | j756 s16 (pilot), j758-761 s17-20 |
| K1 | format control: A2 on gsmvt 6k + train_k1 (same 3,000 prompts, hop-1 queries) | 8,923 rows, 1 epoch | j757 s16 (pilot), j762-765 s17-20 |
| KD | dose check: A2 on gsmvt 6k + train_kd (9,000 chain rows) | 14,923 rows, 1 epoch | j766-770 s16-20 (always, with the confirmation; Amendment 1, finding 3) |
| A2Z96 | eval-only on the A2-96k s16 weights (w667) | GSM8K-Aug 96k | j771 (with the confirmation, reported only) |

Spare ids 772-774 (re-runs only, same env with a new id; never a new setting). 772 is reserved for the fallback
replay of section 5 (tests/test_khop_eval_paths.py).

K1 is the matched control that separates format learning from hop learning: identical statements, order and token
exposure, identical option structure, only the queried hop differs. A2Z separates both from GSM-only training.

### 1.4 Exact flags and envs

New trainer flag (src_khop only): --extra-train PATH[,PATH] in scripts/train_student.py. The rows of each PATH are
read with read_jsonl and appended to train_items right after line 1028 (`train_items = read_jsonl(args.train)`),
before any filter, shuffle or order-seed step. The trainer prints one line "extra-train: <n> rows from <path>
(sources {...})" and stores {"extra_train": {path: n}} in metrics.json. Duplicate item_id between --train and
--extra-train rows is a SystemExit. Default empty = byte-identical to repo.base (a test).

Training envs: generator stage/prep_khop.py (python, values bash-escaped as stage/prep_b3v2.py does), copy of
663_4b_gsmvt_A2-c1lr3ch_s16_fsdp.env with these keys changed:
- SEED = ORDERSEED = s; EPOCHS=1; LR=3e-6; DATA=data/student_gsm8k_tf_vt (unchanged: same val.jsonl, EVALMID valid).
- KH: TAG=khop-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl".
- K1: TAG=khop1-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop/train_k1.jsonl".
- KD: TAG=khopd3-c1lr3ch, XARGS="--eval-value-render vt --extra-train data/eval/khop/train_kd.jsonl".
- EXTRA_EVALS = the 663 string (all GSM and hint sets, unchanged) + " khop_ca_fwd=data/eval/khop/khop_ca_fwd.jsonl
  khop_ca_rev=... khop_wp_fwd=... khop_wp_rev=... khop_vb_fwd=... khop_vb_rev=... khop_cag_fwd=... khop_ca_fwd_qs=...
  khop_blind=...".
- SAVE=1 for j756 only (KH s16 weights for a later probe); SAVE='' elsewhere. JOBKEY "4b gsmvt A2-<TAG> s<s> fsdp x1khop".
- Names <id>_4b_gsmvt_A2-<TAG>_s<s>_fsdp (e.g. 756_4b_gsmvt_A2-khop-c1lr3ch_s16_fsdp).
Eval-only envs: copy 673_4b_evb1_w663.env; WTAR = the weights name ('' for j750), SETS = the nine khop sets only,
XARGS='' (as 673), names <id>_4b_evkh_w<weights id> (j750: 750_4b_evkh_base).
Subs: stage/train_x1khop_4b.sub (train_x1b3v2_4b.sub with CODE_TGZ=cotd_code_data_x1khop_4b.tgz, log
logs/train_x1khop_4b.log), stage/eval_x1khop_4b.sub (eval_x1b2_4b.sub with the same CODE_TGZ, log logs/eval_x1khop_4b.log,
executable run_eval_b2.sh unchanged). Resources as the b3v2 DAG: ngpu 1, cpus 8, mem 80GB, gpumem 40000.

## 2. Metrics and rules

### 2.1 Primary
- k*(KH, ca, fwd) and G2 (seed-paired over s16-20, n = 5, t_crit 2.78, 95% CI). s16 is not used to choose any
  setting or any arm (the pilot gate is go / no-go only; KD runs unconditionally since Amendment 1), so s16 is one of
  the five seeds.
- Reported with the primary: the full curve a_KH(ca, fwd, k), k = 1..8, every seed.

### 2.2 Secondary (reported, seed-paired where two arms share seeds; no verdict depends on them)
- Order: a(ca, fwd, k) - a(ca, rev, k) at k = 2, 3 for KH (paired by item and seed); k*(KH, ca, rev).
- Installed vs native: KH - K1 and KH - A2Z at k = 2, 3 on ca fwd and ca rev.
- Error pattern (Amendment 1, finding 1c; every distractor is now another chain's final value, so the short / cross
  split is gone). Reported: the shape of the KH curve over k (a smooth fall with k points to depth; a flat level below
  1.0 that does not fall with k points to binding), fwd - rev per k, and the share of wrong picks that are the chain
  whose last statement is nearest in the text to the target's last statement (chance 0.25; above chance = a
  position-based binding error).
- Task gradient: k* and a(t, fwd, 2..3) for wp and vb; G3.
- GSM side of KH and K1: plain sets gsm8kx, gsm1kx, gsm8kv2, valx and the rc1..rc4 cells (pfx_table.py) against the
  reference (2.3). KH rc2 - rc2_ref is the transfer line (no claim beyond a pointer to a new card).
- Extrapolation k = 7, 8; BASE0 and A2Z96 curves.

### 2.3 Reference A2 (best-A2 rule, fixed now)
Two A2-6k sets exist for s16-20: the originals (j520, j525, j530, j540, j542) and the SAVE re-trains (j663, j664,
j665, j540, j666). The re-trains differ from the originals by up to 6.9 points at rc2 (j666). For every GSM-side line
(plain conditions, transfer), the reference is the A2-6k set with the higher five-seed mean on that line's metric;
the per-seed values of that set enter the paired difference. For G2 / G3 the candidates are these two sets and KH
itself (Notation, section 0). For G2 this is the conservative choice (it makes G2
smaller). At the pilot (n = 1) the s16 reference is max(j520, j663) per metric. The 2.4 lines that compare with A2Z use
the eval-only jobs j751-755 (the re-train weights; the originals have no saved weights except j540). A2Z s19 (j754)
uses the j540 original weights; A2Z s16-18 and s20 use the re-trains j663, j664, j665, j666 (Amendment 1, finding 11).

### 2.4 Installation checks (data-side D-lines before any submission; I-lines at five seeds)
- D1 Shortcut baselines on every eval set and every k (build_report.json, deterministic, no GPU, recomputed from the
  rows by tests/test_build_khop.py): pick-max, pick-min, pick-median, pick-the-option-closest-to-a-number-in-the-text,
  pick-first-literal-in-text, pick-last-literal-in-text, pick-an-option-not-written-in-the-text, and (Amendment 1,
  finding 1b) query-op pair (options g with another option h such that g = h <op> c for the queried statement's op and
  constant), query-op consistent (options consistent with the queried statement's op: ×2 even, ×3 divisible by 3,
  +c greater than c; uniform among them) and statement-free one-op parent (options one grammar op from another
  option), and (Amendment 2, build critic finding 1) chase-mult (the option nearest the target's root with its ×2 / ×3
  ops applied, + / − ignored) and chase-sign (the same with every + c / − c replaced by +/− the mean constant: 5.5 for
  ca / wp, 55 for ca-gsm). These 12 lines are barred. Bars: each line <= 0.33 in every (set, k) cell of 100-150 rows and <= 0.25 pooled over the k = 1..6
  cells of a set (at true chance 0.20 each bar is passed by chance with p < 1e-4; the author check gives 0.16-0.24).
  The -qs twin and the K1 train file get the same lines. A rule with no candidate picks uniformly. A failing line stops
  the build: report it, no new generator seed.
  Barred on the eval sets only (Amendment 2): query-op avoid (uniform over the options that are not query-op pair
  candidates). On a shared root the target of a hop-1 query is never a candidate, so this line is about 0.26 at k = 1
  and on the K1 train file (0.25-0.27 per cell, author check). It is reported on the training files; on the eval sets
  it is 0.19-0.26 per cell and 0.21 pooled.
  Reported, no bar (Amendment 2; partial-composition lines, high by design): chase-last1 (only the queried statement's
  constant exact; 1.00 at k = 1 by construction), chase-drop1 (every constant exact but hop 1), chase-droplast (every
  constant exact but the last hop; about 0.50-0.75 at k >= 2), drop-one-best (per item the best "all constants exact
  but one" estimate; about 0.30 at k = 2, 0.49-0.63 at k = 3, 0.72-0.81 at k = 4, >= 0.82 at k >= 5) and the hard share
  (items where no drop-one estimate finds the gold; ca / wp about 0.60 at k = 2, 0.28-0.30 at k = 3, 0.14-0.16 at k = 4). A model that reads k - 1 of the k
  constants can therefore reach these levels. The hard subset is the cell restricted to the hard items; khop_table.py
  reports it per run and the PASS claim uses it (section 0 item 2). khop_table.py also reports per run the agreement
  of the model's picks with the chase-sign pick.
- D2 Gold-letter share per (set, k) cell within [0.17, 0.23]; options distinct in every row; gold in options.
- D3 Disjointness: 0 shared prompt hashes and 0 shared value keys between eval and training rows.
- D4 Prompt token length per (task, k) printed; the longest prompt + options < 1,536 tokens (max-len).
- I1 Format learned: m(a_KH(t, o, 1)) >= 0.90 and m(a_K1(t, o, 1)) >= 0.90 for all six (task, order) cells.
- I2 Query read: KH query-swap pair accuracy (both the item and its -qs twin right) at k = 1, m >= 0.85.
- I3 No option-only shortcut: m(blind accuracy) <= 0.30 for KH, K1 and A2Z.
- I4 Rows seen: every KH / K1 / KD training log has the extra-train line with 3,000 / 3,000 / 9,000 rows, and
  metrics.json "extra_train" agrees.
I1-I4 all pass, or the verdict is MANIPULATION FAILED.

### 2.5 Necessary condition for the GSM-side reading
- N1 Plain not down: KH seed mean on each of gsm8kx, gsm1kx, gsm8kv2, valx >= reference - 2. N1 does not block the
  depth reading (k* is a property of the trained model) and G2 already uses the better reference. If N1 fails, the
  verdict line says "plain cost" and no transfer sentence is written.

## 3. Pilot (seed 16) and compute

Smoke DAG stage/smoke_x1khop_4b.dag (before the pilot DAG): j756's env with smoke="1" (the --extra-train path under
the 1-GPU fsdp trainer), and j750 (base eval-only; it is a real pilot job and also the eval-path smoke). Smoke gate:
exit 0 for both (the smoke keeps no logs on success, as CARD_b3_v2 section 4; the row-count line is P2 on the real
j756); j750 tarball has all nine khop preds files with the row counts of 1.2.

Pilot DAG stage/pilot_x1khop_4b.dag: j751 (A2Z s16), j756 (KH s16), j757 (K1 s16). A2Z s17-20 (j752-755) run with
the confirmation: no gate line needs them, and the pilot stays under the 4 GPU-h cap.

Deterministic pilot gate: E/readout/khop_pilot_gate.py --base <750 tgz> --a2z s16=<751> --kh <756>
--k1 <757> --ref s16=<results_520 tgz>,<results_663 tgz> [--report <build_report.json>]. Exit 0 = GATE PASS. Lines:
- P0 build_report.json: D1-D4 pass; md5 of the nine eval files and three train files equal the report; the gate
  re-checks every barred D1 value and every D2 share of the report against the bars of 2.4 itself (Amendment 2, build
  critic finding 3), with its own list of barred lines, so a report with a missing line or a wrong bar fails P0.
- P1 Every pilot tarball has metrics.json and every khop preds file with the 1.2 row counts; j756 / j757 also have
  every GSM and hint preds file of the 663 env.
- P2 j756 and j757 logs: extra-train line with 3,000 rows; total training rows 8,923; final loss finite.
- P3 (I1 at n = 1) a_KH(t, o, 1) >= 0.85 and a_K1(t, o, 1) >= 0.85 on all six (task, order) cells.
- P4 (I2 at n = 1) KH query-swap pair accuracy at k = 1 >= 0.80.
- P5 (I3 at n = 1) blind accuracy <= 0.32 for j756, j757 and j751.
- P6 Plain at n = 1: j756 and j757 gsm8kx and valx >= max(j520, j663) - 4.
- P7 Wall time of j756 and j757 each <= 1.6 h (Amendment 1, finding 9: 0.3 + 2 x 0.2 + 2 x 1.6 = 3.9 GPU-h, under
  the 4 GPU-h cap).
- P7b (Amendment 2, build critic finding 2) Projected confirmation <= 25 GPU-h: 8 x t + 5 x (14,923 / 8,923) x t + 5 x e,
  t = mean wall of j756 and j757, e = max(0.2, mean wall of j750 and j751). P7 alone allowed 27.2 GPU-h at t = 1.6 h;
  P7b holds for t <= about 1.47 h.
- P8 j750 and j751 landed with exit 0.
- P9 Dead training (Amendment 1, finding 5): GATE FAIL when |a_KH(ca, fwd, k) - a_A2Z(ca, fwd, k)| < 0.03 at both
  k = 2 and k = 3 AND a_KH(ca, fwd, 3) < 0.80 (j756 vs j751). The khop rows did not change the model, so a
  confirmation would read an untrained curve as capacity. On this line the report says "new card for the schedule".
Output lines: "GATE PASS" or "GATE FAIL: <lines>". Reported, not decision lines: every curve, error mix, order contrast, the GSM rc cells of j756 /
j757 (pfx_table.py), timing and peak memory.
GATE FAIL: nothing else is submitted; report to the second author with the failing lines.

Compute (estimates; the builder replaces them with token totals from build_report.json. Rate (Amendment 1, finding 9):
186 updates are one epoch of the 5,923-row A2 job, which ran in 23-31 min wall (j646, j648, j651, j656). Five chains
per item make ca / wp prompts about 5(k + 1) statements; D4 prints the lengths):
- Smoke j756 about 0.3 GPU-h; eval-only jobs about 0.2 GPU-h each (7,800 short rows plus a 9 GB weights extract);
  j756 and j757 about 1.4 GPU-h each (8,923 rows, chain rows shorter than GSM rows, plus the khop evals).
  Pilot total: 0.3 + 2 x 0.2 (j750, j751) + 2 x 1.4 = about 3.5 GPU-h (cap 4).
- Confirmation: j758-765 8 x 1.4 + j766-770 5 x 2.2 + j752-755 and j771 5 x 0.2 = about 23.2 GPU-h (with the
  measured 23-31 min per 6k epoch, likely about half). Hard cap without the second author: 25 GPU-h.

## 4. Conditional confirmation (written now; run without asking when the gate prints GATE PASS)

stage/confirm_x1khop_4b.dag, generated by stage/prep_khop.py --confirm, every training child with
SCRIPT PRE check_pilot.sh 756_4b_gsmvt_A2-khop-c1lr3ch_s16_fsdp 757_4b_gsmvt_A2-khop1-c1lr3ch_s16_fsdp:
- KH s17-20: j758, j759, j760, j761.
- K1 s17-20: j762, j763, j764, j765.
- A2Z s17-20: j752, j753, j754, j755 (eval-only; no PRE line needed, they do not depend on the pilot).
- A2Z96: j771 (eval-only, w667).
- KD s16-20: j766, j767, j768, j769, j770, always (Amendment 1, finding 3; the old KD rule is deleted). KD s16 has
  the same PRE line as the other training children.
Readout: E/readout/run_khop.sh (khop_table.py on the 20 training tarballs plus j750-755 and j771; pfx_table.py
for the GSM cells of KH and K1 and both reference sets), then the verdict from section 0 in that order. the second author is told
at submission and can stop it.

## 5. Fixed before the data

- Tasks, generators, option sets and kinds, orders, row counts, seeds, disjointness, glyphs and layout (section 1).
- Arms, data, flags, schedule (1 epoch, LR 3e-6, label smoothing 0.1, MBS 2, GA 16, NGPU 1) and job ids (section 1.3-1.4).
- SOLVED / FAILED bars, k*, G2, the reference rule, the order of the readings, I1-I4, N1, the pilot gate lines P0-P9
  and the compute cap.
- Not run here: other model sizes, more epochs, k > 6 in training, a 96k mix, any probe. Each needs a new card.
- Bundle constraint: jobs/build_bundles.sh copies only E/src/data/eval (plus gsmvt, gsmsi and the aug sets) and
  drops any data dir inside the repo copy. All khop files live in E/src/data/eval/khop/ and are passed by path
  (--extra-train, EXTRA_EVALS). Pre-submit line: `tar tzf cotd_code_data_x1khop_4b.tgz` lists all 12 khop files and
  CODE_PIN = the pin of the x1b3v2 bundle (src_khop/repo is a copy of the merged src_b3/repo) + "-khop"; the builder
  reads that pin from bundle_out/x1b3v2/CODE_PIN and writes the new one in the build report. (Amendment 1: 12 khop files, 9 eval + 3 train.)
- Builder checks before the bundle (CPU, venv_pfx, python -m unittest, torchrun --local-addr 127.0.0.1):
  tests/test_build_khop.py (determinism by seed; gold in options; options distinct; fwd is a topological order and
  rev its exact reverse; K1 rows share KH statements; disjointness; the resample rule of 1.1 holds in every row;
  the row key set equals the gsmvt key set; D1-D4 (every D1 line of 2.4, Amendment 2) recomputed from rows equal the
  report and pass the bars);
  tests/test_extra_train.py (flag off: the training row list and the first batch equal repo.base; flag on: rows
  appended, count logged, duplicate item_id refused; one A2 step on the tiny test model with mixed GSM + khop rows
  runs end to end under 1-rank fsdp with --eval-value-render vt); E/readout/test_khop_table.py and
  E/readout/test_khop_pilot_gate.py on synthetic preds (every reading of section 0 and P9 reached by a fixture);
  tests/test_khop_eval_paths.py (Amendment 1, finding 8): on the tiny test model and 20 khop eval rows, the in-job
  eval of train_student.py (EXTRA_EVALS path) and eval_student.py (the run_eval_b2.sh path, XARGS '') build the same
  prompt token ids and give the same letter logits (max abs difference <= 1e-5). The test must pass before the bundle;
  if it cannot be made to pass, the KH - A2Z lines and P9 use an eval-only replay of the j756 SAVE weights instead
  (spare id 772, reserved now, bar |replay - in-job| <= 0.01 per khop set). The full
  src_khop suite passes with no new failures against repo.base (the 3 known test_tree_target failures need data/).
- One cold critic on the card, the diff and the generated data before the bundle.

## Amendment 1 (2026-10-25 22:27 CDT, after the cold critic, before any build)

Critic: E/src_khop/critique/CARD_CRITIC_khop.md (verdict FIX). Nothing was built, bundled or submitted before this
amendment. The sections above are edited in place; this list says what changed and where. Pre-amendment copy:
E/src_khop/critique/CARD_khop_v1.pre_a1.md.

1. FIX (blocker, option leak). Sections 1.1, 1.2, 2.2, 2.4. Five chains per item for ca and wp; options = the five final
   values; hop-1 and hop-k op kinds shared by the five chains; resample when two options are one grammar op apart or
   an option is a written number, for the final-value set and the hop-1 set (K1). The kinds short and cross_short are
   gone. Three new D1 baselines (query-op pair, query-op consistent, statement-free one-op parent) join the old six.
   The bars are <= 0.33 per cell and <= 0.25 pooled per set; the old 0.30 bar would fail by chance at about 450 cells.
   Author check E/src_khop/critique/sim_khop_a1.py (outputs sim_khop_a1.out, sim_khop_a1_gsm.out): all nine baselines
   at chance (0.16-0.24). The builder's test recomputes them from the real rows. The error mix is now the k-curve
   shape, fwd - rev and a nearest-chain binding share (2.2).
2. FIX. Section 0 item 3 and item 5. NEGATIVE also needs KD - KH at (ca, fwd, 3) with paired CI upper < +0.05; else
   BETWEEN "data- or schedule-limited". The claim text names the budget and schedule (1 epoch, LR 3e-6).
3. FIX. Sections 1.3, 2.1, 3, 4, 5. KD s16-20 (j766-770) always runs with the confirmation. The KD rule, "KD: RUN /
   SKIP" and the --kd flag are deleted. Confirmation estimate 23.2 GPU-h, under the 25 GPU-h cap.
4. FIX. Section 0 item 2, section 1.1. The PASS claim is "composes at least 3 given operations on small integers in
   reading order ...; the rc2 ceiling is not a limit on composing given operations". Parsing / premise binding is one
   of four listed candidates, not the claim. The optional part is taken: khop_cag_fwd.jsonl (ca-gsm, values to 999,
   addends 11-99, fwd, 900 rows), reported only. Not taken: a GSM-matched op-selection control (needs a new card).
5. FIX in part. Section 3, P9 (accuracy dead-training line, j756 vs j751). REJECTED: the khop-row loss line. It needs
   a per-source loss log in the training loop beyond --extra-train, and a loss drop over about 94 chain updates at
   label smoothing 0.1 is noisy. P9 already detects a model that the khop rows did not change.
6. FIX. Section 0, "Prior expectation". Four sources checked by web search (abstracts); Physics of LMs 2.1 is marked
   not checked. Priors stated for k = 2, k = 3, rev, PASS and NEGATIVE. No novelty claim.
7. FIX. Section 0, "What is new here".
8. FIX. Section 5: tests/test_khop_eval_paths.py (same prompt ids and letter logits on both eval paths). Fallback if
   it cannot pass: an eval-only replay of the j756 SAVE weights as j772 (reserved in 1.3), bar 0.01 per khop set.
9. FIX. Section 3: P7 <= 1.6 h per job (pilot <= 3.9 GPU-h); rate note corrected (186 updates = one epoch of the
   5,923-row job, 23-31 min wall).
10. FIX. Section 1.1: the 32 gsmvt keys and their khop values are named; option_kinds is a list as in the gsmvt rows
    (the draft said {letter: kind}); blind set is 18 per cell, 324 rows (1.2); the rev sentence says "after the last
    statement" and names the positions and the DeltaNet state.
11. FIX. Section 2.3 states that A2Z s19 (j754) uses the j540 original weights.

Unchanged: one-pass rule, arms other than KD timing, schedule, job ids 750-774, readings order, K1 design, N1, SOLVED /
FAILED bars, reference rule. File count for the bundle check: 12 (9 eval + 3 train).

## Amendment 2 (2026-10-26 00:21 CDT, after the cold build critic, before the rebuild of the data)

Critic: E/src_khop/critique/BUILD_CRITIC_khop.md (verdict SUBMIT AFTER FIXES). Nothing was bundled or submitted. The
Amendment 1 data in E/src/data/eval/khop/ was built and is replaced by the rebuild after this amendment. The sections
above are edited in place; this list says what changed and where. Pre-amendment copy:
E/src_khop/critique/CARD_khop_v1.pre_a2.md. Builder's fix-round notes: E/src_khop/BUILD_REPORT.md, "Fix round".

1. FIX (blocker, magnitude leak). Sections 1.1, 1.2, 2.4. Under Amendment 1 the five chains had independent roots in
   [2, 60], so "chase the target to its root, apply its ×2 / ×3 ops and +/- 5.5 per +/- hop" picked the gold at
   0.85-1.00 for k <= 2 and 0.63-0.87 at k = 4 (critic, critique/chase_only.py) without reading a constant. Now the five
   chains share one skeleton: one root value, one op-kind sequence, hop 1 + / − with five distinct constants; only the
   constants differ. This goes past the critic's "roots within a narrow band" (any root difference is still read by
   the chase policy after the multipliers). Every policy that does not read the target's constants is at chance by
   symmetry. Two barred D1 lines (chase-mult, chase-sign), one line barred on the eval sets (query-op avoid) and five
   reported lines (chase-last1, chase-drop1, chase-droplast, drop-one-best, hard share) join the list. The barred lines
   are recomputed from the rows by tests/test_build_khop.py and re-checked by the gate (P0).
2. Traded (section 1.1). The rule "no two options 2-9 apart" cannot hold on a shared skeleton, so it is dropped; the
   rule keeps "distinct", "no written number" and "no 2x / 3x pair". Its target leak (one-op parent) stays a barred D1
   line and is 0.19-0.21 in the author check.
3. NEW reported limit and claim wording (section 0 item 2, section 2.4). Numeric options let a model that reads k - 1 of
   the k constants pick the gold often (drop-one-best about 0.62 at k = 3). The claim word "composes" needs the hard
   subset of (ca, fwd, 3) at m >= 0.80; else "PASS (approximate)". The SOLVED / FAILED bars, k* and G2 stay on the full
   cell.
4. FIX (minor, finding 4). MAX_RESAMPLE is 10,000 again (card value). The shared-skeleton rule needs about 1.2-2.7
   draws per item at every k (k = 6, 7, 8 about 2), so the accepted items are not a strong selection on the values.
5. FIX (finding 2). Section 3, new gate line P7b: projected confirmation <= 25 GPU-h from the measured pilot wall times.
6. FIX (finding 3). Section 3, P0 re-checks the D1 values and D2 shares of the report against the bars.
7. Optional part of finding 1 taken: khop_table.py reports per run the agreement of the picks with the chase-sign
   policy and the hard-subset accuracy per cell.
8. Finding 5 (note): the build report states that the full-suite comparison with repo.base covers 23 tests and that the
   two test_probe_slots.TestSbatch failures also fail in repo.base.

Unchanged: one-pass rule, arms, schedule, job ids 750-774, row counts, seeds, orders, gold layout, vb, readings order,
SOLVED / FAILED bars, k*, G2, the reference rule, I1-I4, N1, P1-P6, P8, P9, the compute cap.

## Submission note (2026-10-26 00:43 CDT, sev-e0)

First wave: smoke_x1khop_4b.dag (s756 SMOKE_ONLY + j750 base eval). The pilot DAG goes in only after both exit 0 and a clean cold critic of Amendment 2. Bundle cotd_code_data_x1khop_4b.tgz md5 d3c7005c94f759cc3b1efffd8d131688 (CODE_PIN ...-b3v2-khop), md5 checked on both sides. Nothing else in the card changes. Round cap and tie-break: RESULTS_LOG "Idea round 2026-10-26".

## Amendment 3 (2026-10-26 00:55 CDT, sev-e0, before any pilot number; readout only)

Fix-round cold critic (src_khop/critique/FIXROUND_CRITIC_khop.md): HOLD on the readout, data and bundle fine. With one shared root and one op sequence for all five chains, every x2 / x3 step moves all five options the same way and never decides the answer. The model must compose only the + / - steps. A probe that sums the signed constants and ignores x scores 0.82 on (ca, fwd, 3) and 0.80 on its hard subset. Changes, all in the confirmation readout (khop_table.py); the pilot gate lines P0-P9 do not change:
1. The claim drops "k given operations". It names the number of + / - steps composed.
2. The word "composes" needs m >= 0.80 on the items of (ca, fwd, 3) where all three steps are + / - (48 items per seed; (wp, fwd, 3), 67 items, reported beside it). This replaces the Amendment 2 hard-subset rule. Else "PASS (approximate)".
3. G2 is also reported on the 63 all-+/- items of (ca, fwd, 2) (reported line, not a decision line).
4. Every curve is also split by the number of + / - steps. The two rank probes (signed-sum, first-constant) are printed as reference lines.
The SOLVED / FAILED bars, k*, G2 on the full cell, the reference rule and the compute cap do not change.

## Confirmation submission (2026-10-26 02:27 CDT, sev-e0)

Pilot gate P0-P9 GATE PASS (RESULTS_LOG). prep_khop_confirm.py wrote confirm_x1khop_4b.dag (18 jobs, 23.2 GPU-h by the card rule, 12.8 projected). Same bundle md5 d3c7005c. Readout under Amendment 3.

## Outcome (2026-10-26 05:44 CDT, sev-e0)
VERDICT BETWEEN, data- or schedule-limited. k*(KH, ca, fwd) = 2; G2 +25.4 [+19.5, +31.4]; KH k = 3 42.1, KD k = 3 67.9, KD - KH +25.8 [+10.2, +41.3]. Full numbers: RESULTS_LOG "khop five-seed readout"; log exp/logs/khop_readout_20261026.txt.
