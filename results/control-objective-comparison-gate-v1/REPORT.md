# Completion-first approval experiment

Matched reward/completion-guided candidate pools; independent selection/approval/test episodes.
The source candidate experiment and its evidence remain unchanged.

## Frozen acceptance contract

Success requires no crash, body asleep, and both grounded legs within the
helipad bounds by 1,000 steps. This is a new goal metric; historical safe-landing
and reward≥200 results have not been redefined. Reward is only a ranking tie-breaker.

Selection: 64 new paired starts; reject candidates with >5pp nominal loss.
Approval: 256 independent paired starts, >=5pp observed goal gain,
positive task-gain lower bound, and nominal-gain lower bound >=−5pp.
Bounds use 20,000 paired-seed bootstrap resamples at confidence 99.93% per comparison (Bonferroni-style adjustment across
all registered task and retention comparisons). These are approximate statistical
screens, not formal safety guarantees. One candidate per run reaches approval.
No retries on approval data. Rejection deploys original weights.

Reversed actuators require the opposite commands. Those edits are accepted only
for that explicit context, and deployment on original physics restores the base
checkpoint. They are a sanity check, not a universal skill-preserving repair.

## Decisions before final test

| Run | Selected candidate | Accepted | Reason |
| --- | ---: | --- | --- |
| compound-llm-completion-seed0 | — | False | no_candidate_passed_selection |
| compound-llm-completion-seed1 | — | False | no_candidate_passed_selection |
| compound-llm-completion-seed2 | — | False | no_candidate_passed_selection |
| compound-llm-reward-seed0 | — | False | no_candidate_passed_selection |
| compound-llm-reward-seed1 | — | False | no_candidate_passed_selection |
| compound-llm-reward-seed2 | — | False | no_candidate_passed_selection |
| compound-scripted-completion-seed0 | — | False | no_candidate_passed_selection |
| compound-scripted-completion-seed1 | — | False | no_candidate_passed_selection |
| compound-scripted-completion-seed2 | — | False | no_candidate_passed_selection |
| compound-scripted-reward-seed0 | — | False | no_candidate_passed_selection |
| compound-scripted-reward-seed1 | — | False | no_candidate_passed_selection |
| compound-scripted-reward-seed2 | — | False | no_candidate_passed_selection |
| crosswind-llm-completion-seed0 | 5 | True | changed context with original-physics regression check |
| crosswind-llm-completion-seed1 | — | False | no_candidate_passed_selection |
| crosswind-llm-completion-seed2 | — | False | no_candidate_passed_selection |
| crosswind-llm-reward-seed0 | — | False | no_candidate_passed_selection |
| crosswind-llm-reward-seed1 | 0 | True | changed context with original-physics regression check |
| crosswind-llm-reward-seed2 | 1 | True | changed context with original-physics regression check |
| crosswind-scripted-completion-seed0 | — | False | no_candidate_passed_selection |
| crosswind-scripted-completion-seed1 | 9 | False | insufficient_completion_improvement |
| crosswind-scripted-completion-seed2 | — | False | no_candidate_passed_selection |
| crosswind-scripted-reward-seed0 | — | False | no_candidate_passed_selection |
| crosswind-scripted-reward-seed1 | 3 | False | insufficient_completion_improvement |
| crosswind-scripted-reward-seed2 | — | False | no_candidate_passed_selection |
| weak_main-llm-completion-seed0 | — | False | no_candidate_passed_selection |
| weak_main-llm-completion-seed1 | 0 | True | changed context with original-physics regression check |
| weak_main-llm-completion-seed2 | — | False | no_candidate_passed_selection |
| weak_main-llm-reward-seed0 | 5 | False | insufficient_completion_improvement |
| weak_main-llm-reward-seed1 | — | False | no_candidate_passed_selection |
| weak_main-llm-reward-seed2 | 10 | False | insufficient_completion_improvement |
| weak_main-scripted-completion-seed0 | — | False | no_candidate_passed_selection |
| weak_main-scripted-completion-seed1 | 5 | False | insufficient_completion_improvement |
| weak_main-scripted-completion-seed2 | — | False | no_candidate_passed_selection |
| weak_main-scripted-reward-seed0 | — | False | no_candidate_passed_selection |
| weak_main-scripted-reward-seed1 | 5 | False | insufficient_completion_improvement |
| weak_main-scripted-reward-seed2 | — | False | no_candidate_passed_selection |

## Fresh deployment test

256 fresh paired starts per condition. Test outcomes never affect approval.

| Run | Original goal completion | Deployed goal completion | Deployed patch |
| --- | ---: | ---: | --- |
| compound-llm-completion-seed0 | 2.3% | 2.3% | base fallback |
| compound-llm-completion-seed1 | 2.3% | 2.3% | base fallback |
| compound-llm-completion-seed2 | 2.3% | 2.3% | base fallback |
| compound-llm-reward-seed0 | 2.3% | 2.3% | base fallback |
| compound-llm-reward-seed1 | 2.3% | 2.3% | base fallback |
| compound-llm-reward-seed2 | 2.3% | 2.3% | base fallback |
| compound-scripted-completion-seed0 | 2.3% | 2.3% | base fallback |
| compound-scripted-completion-seed1 | 2.3% | 2.3% | base fallback |
| compound-scripted-completion-seed2 | 2.3% | 2.3% | base fallback |
| compound-scripted-reward-seed0 | 2.3% | 2.3% | base fallback |
| compound-scripted-reward-seed1 | 2.3% | 2.3% | base fallback |
| compound-scripted-reward-seed2 | 2.3% | 2.3% | base fallback |
| crosswind-llm-completion-seed0 | 9.8% | 15.6% | approved weights |
| crosswind-llm-completion-seed1 | 9.8% | 9.8% | base fallback |
| crosswind-llm-completion-seed2 | 9.8% | 9.8% | base fallback |
| crosswind-llm-reward-seed0 | 9.8% | 9.8% | base fallback |
| crosswind-llm-reward-seed1 | 9.8% | 14.1% | approved weights |
| crosswind-llm-reward-seed2 | 9.8% | 14.1% | approved weights |
| crosswind-scripted-completion-seed0 | 9.8% | 9.8% | base fallback |
| crosswind-scripted-completion-seed1 | 9.8% | 9.8% | base fallback |
| crosswind-scripted-completion-seed2 | 9.8% | 9.8% | base fallback |
| crosswind-scripted-reward-seed0 | 9.8% | 9.8% | base fallback |
| crosswind-scripted-reward-seed1 | 9.8% | 9.8% | base fallback |
| crosswind-scripted-reward-seed2 | 9.8% | 9.8% | base fallback |
| weak_main-llm-completion-seed0 | 80.9% | 80.9% | base fallback |
| weak_main-llm-completion-seed1 | 80.9% | 84.4% | approved weights |
| weak_main-llm-completion-seed2 | 80.9% | 80.9% | base fallback |
| weak_main-llm-reward-seed0 | 80.9% | 80.9% | base fallback |
| weak_main-llm-reward-seed1 | 80.9% | 80.9% | base fallback |
| weak_main-llm-reward-seed2 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-completion-seed0 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-completion-seed1 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-completion-seed2 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-reward-seed0 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-reward-seed1 | 80.9% | 80.9% | base fallback |
| weak_main-scripted-reward-seed2 | 80.9% | 80.9% | base fallback |

Accepted 4/36 proposed run-level repairs. Rejected edits
are not described as improvements. A rejection can mean insufficient evidence,
not proof that a candidate is harmful. The gate changes which edits are deployed;
it does not create capabilities absent from the candidate pool.

Candidate generation uses matched reward/completion objectives; editors remain frozen, with no training.
The original target is still one pretrained actor, so cross-model transfer is untested.

Protocol hash: `797d900f7471582935e880d7a9c09222d888f3679901269e97431afe882486b2`.
Machine-readable selection, approval decisions, frozen weights, phase locks, and
per-episode measurements accompany this report. No source measurement was overwritten.
