# Does completion-guided generation repair harder landing conditions?

Matched experiment, 3 November 2026 (Asia/Kolkata). Frozen GPT-OSS-120B edits one pretrained 4,996-parameter actor; no training.

Reward-guided and completion-guided runs share 64 development starts, edit tools, aggregate/trajectory feedback, 12-proposal and 28-turn caps, and three search seeds. Completion guidance also specifies nominal retention. Scripted search changes adaptive ranking under the same objective contrast.

All 36 planned searches close before independent selection (64 starts), approval (256), and final testing (256). Both generation objectives use the same completion-based gate. Approval bounds adjust across 72 task/retention comparisons. No approval retries or outcome-driven extra runs.

## All search and deployment outcomes

| Condition | Generator / objective | Seed | Proposals | Selected | Approved | Final completions (base → deployed) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| compound | llm-completion | 0 | 12 | — | False | 6/256 → 6/256 |
| compound | llm-completion | 1 | 12 | — | False | 6/256 → 6/256 |
| compound | llm-completion | 2 | 1 | — | False | 6/256 → 6/256 |
| compound | llm-reward | 0 | 12 | — | False | 6/256 → 6/256 |
| compound | llm-reward | 1 | 12 | — | False | 6/256 → 6/256 |
| compound | llm-reward | 2 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-completion | 0 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-completion | 1 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-completion | 2 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-reward | 0 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-reward | 1 | 12 | — | False | 6/256 → 6/256 |
| compound | scripted-reward | 2 | 12 | — | False | 6/256 → 6/256 |
| crosswind | llm-completion | 0 | 12 | 5 | True | 25/256 → 40/256 |
| crosswind | llm-completion | 1 | 12 | — | False | 25/256 → 25/256 |
| crosswind | llm-completion | 2 | 0 | — | False | 25/256 → 25/256 |
| crosswind | llm-reward | 0 | 1 | — | False | 25/256 → 25/256 |
| crosswind | llm-reward | 1 | 12 | 0 | True | 25/256 → 36/256 |
| crosswind | llm-reward | 2 | 12 | 1 | True | 25/256 → 36/256 |
| crosswind | scripted-completion | 0 | 12 | — | False | 25/256 → 25/256 |
| crosswind | scripted-completion | 1 | 12 | 9 | False | 25/256 → 25/256 |
| crosswind | scripted-completion | 2 | 12 | — | False | 25/256 → 25/256 |
| crosswind | scripted-reward | 0 | 12 | — | False | 25/256 → 25/256 |
| crosswind | scripted-reward | 1 | 12 | 3 | False | 25/256 → 25/256 |
| crosswind | scripted-reward | 2 | 12 | — | False | 25/256 → 25/256 |
| weak_main | llm-completion | 0 | 12 | — | False | 207/256 → 207/256 |
| weak_main | llm-completion | 1 | 5 | 0 | True | 207/256 → 216/256 |
| weak_main | llm-completion | 2 | 11 | — | False | 207/256 → 207/256 |
| weak_main | llm-reward | 0 | 12 | 5 | False | 207/256 → 207/256 |
| weak_main | llm-reward | 1 | 12 | — | False | 207/256 → 207/256 |
| weak_main | llm-reward | 2 | 12 | 10 | False | 207/256 → 207/256 |
| weak_main | scripted-completion | 0 | 12 | — | False | 207/256 → 207/256 |
| weak_main | scripted-completion | 1 | 12 | 5 | False | 207/256 → 207/256 |
| weak_main | scripted-completion | 2 | 12 | — | False | 207/256 → 207/256 |
| weak_main | scripted-reward | 0 | 12 | — | False | 207/256 → 207/256 |
| weak_main | scripted-reward | 1 | 12 | 5 | False | 207/256 → 207/256 |
| weak_main | scripted-reward | 2 | 12 | — | False | 207/256 → 207/256 |

## Matched objective comparisons

| Condition | Generator | Reward: selected / approved | Completion: selected / approved | Mean deployed gain from completion guidance, pp |
| --- | --- | --- | --- | ---: |
| weak_main | llm | 2/3 / 0/3 | 1/3 / 1/3 | +1.17 |
| weak_main | scripted | 1/3 / 0/3 | 1/3 / 0/3 | +0.00 |
| crosswind | llm | 2/3 / 2/3 | 1/3 / 1/3 | -0.91 |
| crosswind | scripted | 1/3 / 0/3 | 1/3 / 0/3 | +0.00 |
| compound | llm | 0/3 / 0/3 | 0/3 / 0/3 | +0.00 |
| compound | scripted | 0/3 / 0/3 | 0/3 / 0/3 | +0.00 |

Approved 4/36 run-level repairs. A rejected run deploys the original checkpoint: a zero deployment effect is fallback behavior, not an estimate of the rejected candidate's effect.

This tests whether changing generation guidance and ranking helps within the existing edit space and proposal budget. Three search seeds and one target cannot establish a broad model capability or an optimal search method. Provider seed support does not guarantee identical sampling. The historical six-start reward campaign is not a matched comparator.

Estimated ledger increase during this experiment: $0.302177; token-rate estimates include conservative reservations for unknown outcomes, not invoice reconciliation. Cumulative cap remains $20.

Full candidate scores, delivered prompts/feedback, usage, and failures are retained in search/. The independent gate report and per-episode measurements are in ../control-objective-comparison-gate-v1/. Historical publication reports and measurements are unchanged.
