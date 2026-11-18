# Stage 2 verdict (jobs 520-533), recomputed from preds

Epoch rule (valx, 393 items): A2-c1 62.60 (63.61 / 61.32 / 62.85), A2-c2 61.41 (61.32 / 62.60 / 60.31) -> the P2 student is A2-c1.

## Checks (seed means) and primary numbers per arm

| arm | M1 full | M2 placebo-plain pooled (r1/r2/r3/r4) | M3 true-far | M3b true-near | M4 true-other | read (n=2) | gate | H clean n>=4 (per seed) | H seed-t 95% | H two-level 95% | D n>=5 (per seed) | D seed-t 95% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2-c1 | 95.6 | -0.9 (+0.1 / -1.4 / -1.0 / -2.7) | +35.0 | +23.9 | +32.7 | +4.5 | PASS | +36.4 (+38.2 / +34.1 / +37.0) | [+31.2, +41.7] | [+31.7, +41.4] | +35.9 (+34.6 / +38.6 / +34.6) | [+30.3, +41.6] |
| A2-c2 | 97.6 | -0.8 (-0.3 / -1.1 / -0.6 / -1.9) | +30.3 | +23.0 | +31.4 | +4.9 | PASS | +34.2 (+35.0 / +30.5 / +37.0) | [+25.8, +42.5] | [+28.6, +39.5] | +31.1 (+30.7 / +28.9 / +33.7) | [+25.1, +37.2] |
| GFPQ-c2 | 96.1 | -3.2 (-2.4 / -4.1 / -2.9 / -3.8) | +37.7 | +26.9 | +37.6 | +6.0 | FAIL | +38.6 (+42.0 / +35.5 / +38.2) | [+30.3, +46.8] | [+33.3, +43.8] | +38.3 (+40.4 / +34.6 / +39.8) | [+30.4, +46.1] |
| GFPQR-c2 | 96.9 | -1.0 (-0.7 / -0.9 / -1.5 / -1.2) | +32.4 | +24.0 | +34.9 | +4.8 | PASS | +36.2 (+36.6 / +31.1 / +40.9) | [+24.0, +48.4] | [+29.8, +42.6] | +34.9 (+35.2 / +31.0 / +38.6) | [+25.6, +44.3] |

## Decisions

- Gate on the P2 student A2-c1 (seed means): PASS. P2 is read only if the gate passes.
- P2 (primary): H = +36.4 on 440 clean items with n >= 4, seeds +38.2 / +34.1 / +37.0, seed t 29.89 (crit 4.30) -> **PASS**.
- P1: D = acc(r1) - acc(r4) on 332 items with n >= 5 = +35.9 -> capacity-limited.
- P3 (secondary) GFPQ-c2 - GFPQR-c2 on 2006 r in {1, 2} true rows: +1.7 / +5.0 / +1.6, mean +2.8, seed t 2.46, seed-t 95% [-2.1, +7.6], two-level 95% [+0.7, +5.1] -> between.
- P3 (secondary) GFPQ-c2 - GFPQR-c2 on 1672 r in {1, 2} true rows (clean only): +1.3 / +4.2 / +1.3, mean +2.3, seed t 2.30, seed-t 95% [-2.0, +6.5], two-level 95% [+0.2, +4.5] -> between.
