# Why did generation and approval succeed or fail?

Offline diagnostics of the frozen objective comparison. These summaries do not change candidate selection, approval, or deployment.

384 valid checkpoint entries represent 260 distinct actors, reused across some runs/conditions. 4 approved run-level decisions represent 4 distinct condition/checkpoint pairs. These are not independent demonstrations of different repair mechanisms.

## Candidate generation and selection

| Run | Valid / proposed | Nominal-retaining candidates | Best development completion gain with retention, pp | Selection gain, pp | Approved |
| --- | ---: | ---: | ---: | ---: | --- |
| compound-llm-completion-seed0 | 12/12 | 5 | +0.00 | — | False |
| compound-llm-completion-seed1 | 11/12 | 2 | -1.56 | — | False |
| compound-llm-completion-seed2 | 1/1 | 0 | — | — | False |
| compound-llm-reward-seed0 | 10/12 | 4 | +3.12 | — | False |
| compound-llm-reward-seed1 | 12/12 | 0 | — | — | False |
| compound-llm-reward-seed2 | 12/12 | 8 | +3.12 | — | False |
| compound-scripted-completion-seed0 | 12/12 | 1 | -3.12 | — | False |
| compound-scripted-completion-seed1 | 12/12 | 7 | +0.00 | — | False |
| compound-scripted-completion-seed2 | 12/12 | 0 | — | — | False |
| compound-scripted-reward-seed0 | 12/12 | 0 | — | — | False |
| compound-scripted-reward-seed1 | 12/12 | 1 | +0.00 | — | False |
| compound-scripted-reward-seed2 | 12/12 | 0 | — | — | False |
| crosswind-llm-completion-seed0 | 12/12 | 8 | +9.38 | +15.62 | True |
| crosswind-llm-completion-seed1 | 12/12 | 0 | — | — | False |
| crosswind-llm-completion-seed2 | 0/0 | 0 | — | — | False |
| crosswind-llm-reward-seed0 | 1/1 | 1 | +0.00 | — | False |
| crosswind-llm-reward-seed1 | 12/12 | 11 | +10.94 | +14.06 | True |
| crosswind-llm-reward-seed2 | 12/12 | 8 | +4.69 | +10.94 | True |
| crosswind-scripted-completion-seed0 | 12/12 | 0 | — | — | False |
| crosswind-scripted-completion-seed1 | 12/12 | 5 | +1.56 | +9.38 | False |
| crosswind-scripted-completion-seed2 | 12/12 | 0 | — | — | False |
| crosswind-scripted-reward-seed0 | 12/12 | 0 | — | — | False |
| crosswind-scripted-reward-seed1 | 12/12 | 4 | +6.25 | +6.25 | False |
| crosswind-scripted-reward-seed2 | 12/12 | 0 | — | — | False |
| weak_main-llm-completion-seed0 | 10/12 | 5 | +6.25 | — | False |
| weak_main-llm-completion-seed1 | 5/5 | 5 | +9.38 | +6.25 | True |
| weak_main-llm-completion-seed2 | 11/11 | 7 | +6.25 | — | False |
| weak_main-llm-reward-seed0 | 12/12 | 2 | +0.00 | +10.94 | False |
| weak_main-llm-reward-seed1 | 12/12 | 2 | -1.56 | — | False |
| weak_main-llm-reward-seed2 | 11/12 | 10 | +10.94 | +15.62 | False |
| weak_main-scripted-completion-seed0 | 12/12 | 0 | — | — | False |
| weak_main-scripted-completion-seed1 | 12/12 | 3 | +9.38 | +12.50 | False |
| weak_main-scripted-completion-seed2 | 12/12 | 0 | — | — | False |
| weak_main-scripted-reward-seed0 | 12/12 | 0 | — | — | False |
| weak_main-scripted-reward-seed1 | 12/12 | 3 | +9.38 | +12.50 | False |
| weak_main-scripted-reward-seed2 | 12/12 | 0 | — | — | False |

## Provider failures

4 of 18 LLM searches stopped after a provider/budget error. These runs and any valid partial candidate pools remain in the primary comparison. They were not replaced or retried. Unequal realized search effort limits attribution of arm differences to generation guidance alone.

- `compound-llm-completion-seed2`: provider_or_budget_error:RuntimeError:Prime inference HTTP 429
- `crosswind-llm-completion-seed2`: provider_or_budget_error:RuntimeError:Prime inference HTTP 429
- `crosswind-llm-reward-seed0`: provider_or_budget_error:RuntimeError:Prime inference HTTP 429
- `weak_main-llm-completion-seed2`: provider_or_budget_error:RuntimeError:Prime inference HTTP 429

## Independent approval

- `crosswind-llm-completion-seed0`: task gain +9.38 pp, adjusted interval [+1.17, +17.97]; nominal gain +10.94 pp, adjusted interval [+1.93, +20.31]. Decision: approved.
- `crosswind-llm-reward-seed1`: task gain +9.38 pp, adjusted interval [+2.34, +17.19]; nominal gain +15.23 pp, adjusted interval [+7.42, +23.44]. Decision: approved.
- `crosswind-llm-reward-seed2`: task gain +13.28 pp, adjusted interval [+5.08, +21.88]; nominal gain +14.06 pp, adjusted interval [+6.64, +22.27]. Decision: approved.
- `crosswind-scripted-completion-seed1`: task gain +4.30 pp, adjusted interval [-2.37, +11.72]; nominal gain +16.02 pp, adjusted interval [+8.98, +23.83]. Decision: insufficient_completion_improvement.
- `crosswind-scripted-reward-seed1`: task gain +3.12 pp, adjusted interval [-3.93, +10.16]; nominal gain +10.55 pp, adjusted interval [+3.12, +18.36]. Decision: insufficient_completion_improvement.
- `weak_main-llm-completion-seed1`: task gain +7.42 pp, adjusted interval [+0.78, +14.84]; nominal gain +5.08 pp, adjusted interval [-3.12, +12.91]. Decision: approved.
- `weak_main-llm-reward-seed0`: task gain +2.34 pp, adjusted interval [-4.69, +10.16]; nominal gain +4.30 pp, adjusted interval [-2.73, +12.13]. Decision: insufficient_completion_improvement.
- `weak_main-llm-reward-seed2`: task gain +3.52 pp, adjusted interval [-3.12, +10.16]; nominal gain +5.86 pp, adjusted interval [-2.34, +14.06]. Decision: insufficient_completion_improvement.
- `weak_main-scripted-completion-seed1`: task gain +3.52 pp, adjusted interval [-3.15, +10.57]; nominal gain +14.06 pp, adjusted interval [+6.25, +21.88]. Decision: insufficient_completion_improvement.
- `weak_main-scripted-reward-seed1`: task gain +3.52 pp, adjusted interval [-3.15, +10.57]; nominal gain +14.06 pp, adjusted interval [+6.25, +21.88]. Decision: insufficient_completion_improvement.

Approval intervals use the frozen approximate paired-bootstrap procedure and adjust across 72 comparisons. Development peaks are searched scores, not independent improvement estimates. A rejected candidate's deployment falls back to the base controller.

## Independent final test of approved patches

| Run | Base → patch completions | Gain, pp [descriptive 95% interval] |
| --- | --- | --- |
| crosswind-llm-completion-seed0 | 25/256 → 40/256 | +5.86 [+0.78, +11.33] |
| crosswind-llm-reward-seed1 | 25/256 → 36/256 | +4.30 [-0.39, +8.98] |
| crosswind-llm-reward-seed2 | 25/256 → 36/256 | +4.30 [+0.00, +8.98] |
| weak_main-llm-completion-seed1 | 207/256 → 216/256 | +3.52 [-0.39, +7.42] |

Only the completion-guided crosswind patch has a final descriptive 95% interval wholly above zero. The other approved patches' final intervals include zero. The final gains are smaller than their approval gains. Approval is not a guarantee of future performance; these final intervals are not adjusted for multiplicity and do not measure variation across target networks or new searches.

![Fresh-test effects and all paired outcomes](figures/approved-final-test.png)

## Campaign-specific inference ledger

265 requests; 4 unknown outcomes retain reservations. Estimated committed cost: $0.302177. This filters the ledger to this campaign's run paths, excluding other campaigns; estimates are not invoice reconciliation.
