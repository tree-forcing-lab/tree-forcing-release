# One item in all 18 conditions: gsm8k_v2-test-1231

n = 9 worked steps (atomic operations), depth field = 2, gold = B.
In every set, k = number of lines in the block and r = n - k. For true_r<r>, the block holds the FIRST k = n - r
true steps, so r is the number of steps the model must still compute. Prompts are copied byte for byte.

## hint_plain (n 9, k 0, r 9)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r1 (n 9, k 8, r 1)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Step 5: 310 + 90 = 400
Step 6: 400 + 100 = 500
Step 7: 500 + 70 = 570
Step 8: 570 + 60 = 630
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r2 (n 9, k 7, r 2)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Step 5: 310 + 90 = 400
Step 6: 400 + 100 = 500
Step 7: 500 + 70 = 570
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r3 (n 9, k 6, r 3)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Step 5: 310 + 90 = 400
Step 6: 400 + 100 = 500
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r4 (n 9, k 5, r 4)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Step 5: 310 + 90 = 400
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r5 (n 9, k 4, r 5)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_true_r6 (n 9, k 3, r 6)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_full (n 9, k 9, r 0)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 130
Step 2: 130 + 80 = 210
Step 3: 210 + 60 = 270
Step 4: 270 + 40 = 310
Step 5: 310 + 90 = 400
Step 6: 400 + 100 = 500
Step 7: 500 + 70 = 570
Step 8: 570 + 60 = 630
Step 9: 630 ÷ 9 = 70
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_placebo_r1 (n 9, k 8, r 1)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 2: Find her mean score.
Step 3: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 4: Find her mean score.
Step 5: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 6: Find her mean score.
Step 7: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 8: Find her mean score.
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_placebo_r2 (n 9, k 7, r 2)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 2: Find her mean score.
Step 3: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 4: Find her mean score.
Step 5: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 6: Find her mean score.
Step 7: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_placebo_r3 (n 9, k 6, r 3)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 2: Find her mean score.
Step 3: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 4: Find her mean score.
Step 5: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 6: Find her mean score.
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_placebo_r4 (n 9, k 5, r 4)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 2: Find her mean score.
Step 3: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Step 4: Find her mean score.
Step 5: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60.
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_other_r1 (n 9, k 8, r 1, partner = gsm8k_v2-test-295)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 100 + 50 = 150
Step 2: 150 + 20 = 170
Step 3: 100 + 20 = 120
Step 4: 50 − 10 = 40
Step 5: 2 × 20 = 40
Step 6: 40 + 40 = 80
Step 7: 80 + 120 = 200
Step 8: 200 + 170 = 370
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_other_r2 (n 9, k 7, r 2, partner = gsm8k_v2-test-295)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 100 + 50 = 150
Step 2: 150 + 20 = 170
Step 3: 100 + 20 = 120
Step 4: 50 − 10 = 40
Step 5: 2 × 20 = 40
Step 6: 40 + 40 = 80
Step 7: 80 + 120 = 200
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_wrongres_r1 (n 9, k 8, r 1, wrong_values = ['4000', '320000', '19200000', '768000000', '69120000000', '6912000000000', '483840000000000', '29030400000000000'])

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 4000
Step 2: 4000 + 80 = 320000
Step 3: 320000 + 60 = 19200000
Step 4: 19200000 + 40 = 768000000
Step 5: 768000000 + 90 = 69120000000
Step 6: 69120000000 + 100 = 6912000000000
Step 7: 6912000000000 + 70 = 483840000000000
Step 8: 483840000000000 + 60 = 29030400000000000
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_wrongres_r2 (n 9, k 7, r 2, wrong_values = ['4000', '320000', '19200000', '768000000', '69120000000', '6912000000000', '483840000000000'])

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 4000
Step 2: 4000 + 80 = 320000
Step 3: 320000 + 60 = 19200000
Step 4: 19200000 + 40 = 768000000
Step 5: 768000000 + 90 = 69120000000
Step 6: 69120000000 + 100 = 6912000000000
Step 7: 6912000000000 + 70 = 483840000000000
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_wrongres_near_r1 (n 9, k 8, r 1)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 131
Step 2: 130 + 80 = 211
Step 3: 210 + 60 = 271
Step 4: 270 + 40 = 311
Step 5: 310 + 90 = 401
Step 6: 400 + 100 = 501
Step 7: 500 + 70 = 571
Step 8: 570 + 60 = 631
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```

## hint_wrongres_near_r2 (n 9, k 7, r 2)

```
Question: Sally received the following scores on her math quizzes: 50, 80, 80, 60, 40, 90, 100, 70, and 60. Find her mean score.

Worked steps:
Step 1: 50 + 80 = 131
Step 2: 130 + 80 = 211
Step 3: 210 + 60 = 271
Step 4: 270 + 40 = 311
Step 5: 310 + 90 = 401
Step 6: 400 + 100 = 501
Step 7: 500 + 70 = 571
Options:
A) 72
B) 70
C) 94
D) 95
E) 90
```
