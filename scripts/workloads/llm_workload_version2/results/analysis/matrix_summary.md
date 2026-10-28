| Config | Rank 0 \| Rank 1 | Reps | Load time (min) | Estimate (min) | Rank 0 (min) | Rank 1 (min) | Agg BW (MiB/s) | Framing mean / p95 (ms) | Payload mean / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| All F | FFFF \| FFFF | 3 | 3.20 ± 0.00 | 2–2.5 | 3.16 ± 0.01 | 3.20 ± 0.00 | 89.5 ± 0.0 | 9 / 63 | 367 / 689 |
| All S | SSSS \| SSSS | 3 | 12.87 ± 0.01 | 8–10 | 12.87 ± 0.01 | 12.83 ± 0.04 | 22.3 ± 0.0 | 28 / 176 | 1555 / 2818 |
| Unbalanced | FFFF \| SSSS | 3 | 6.67 ± 0.02 | 8–10 | 1.72 ± 0.03 | 6.67 ± 0.02 | 42.9 ± 0.1 | 8 / 53 | 506 / 1387 |
| Balanced | FFSS \| FFSS | 3 | 8.02 ± 0.02 | 5–6.3 | 8.02 ± 0.02 | 8.02 ± 0.03 | 35.7 ± 0.1 | 18 / 109 | 964 / 2758 |
| One file fast | FSSS \| SSSS | 3 | 11.40 ± 0.01 | 8–10 | 10.24 ± 0.02 | 11.40 ± 0.01 | 25.1 ± 0.0 | 27 / 167 | 1292 / 2759 |
| Default pool D | DDDD \| DDDD | 3 | 11.25 ± 0.02 | 7.5–8.5 | 11.25 ± 0.02 | 11.24 ± 0.03 | 25.5 ± 0.0 | 23 / 128 | 1366 / 2773 |

± is one standard deviation across reps. Latency columns are the mean over reps of each run's per-read mean and p95.
