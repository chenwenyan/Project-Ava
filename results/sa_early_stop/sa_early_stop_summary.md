# SA self-consistency progressive early stopping

- Questions analyzed: **60** (VideoMME-long, 601+)
- Full budget: 8 rounds x nodes (= 8x13 = 104 responses at 13 SA nodes)
- Baseline = top1 after the full budget; early-stop derived from the same trajectory.

## Sweep over stability_rounds (S)

| S | baseline acc | early-stop acc | Δacc | agree w/ baseline | % early-stopped | mean stop round | mean saving |
|---|---|---|---|---|---|---|---|
| 1 | 55.0% | 53.3% | -1.7% | 86.7% | 100.0% | 1.00 | 87.5% |
| 2 | 55.0% | 60.0% | +5.0% | 91.7% | 93.3% | 4.53 | 43.3% |
| 3 | 55.0% | 58.3% | +3.3% | 96.7% | 75.0% | 6.13 | 23.3% |
| 4 (default) | 55.0% | 56.7% | +1.7% | 98.3% | 45.0% | 7.18 | 10.2% |
| 5 | 55.0% | 55.0% | +0.0% | 100.0% | 31.7% | 7.72 | 3.5% |
| 6 | 55.0% | 55.0% | +0.0% | 100.0% | 15.0% | 7.88 | 1.5% |
| 7 | 55.0% | 55.0% | +0.0% | 100.0% | 8.3% | 7.97 | 0.4% |

## Headline (S=4, --sa-stability-rounds 4)

- **Mean response savings vs fixed 104: 10.2%** (mean stop at round 7.18/8).
- **Accuracy impact: +1.7%** (baseline 55.0% -> early-stop 56.7%).
- Early-stop fired for 45.0% of questions; the derived answer matched the full-budget answer 98.3% of the time.
