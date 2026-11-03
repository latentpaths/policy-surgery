# Completion-first approval experiment

Retrospective candidate pool, prospective selection/approval/test episodes. No new model calls.
The source candidate experiment and its evidence remain unchanged.

## Frozen acceptance contract

Success requires no crash, body asleep, and both grounded legs within the
helipad bounds by 1,000 steps. This is a new goal metric; historical safe-landing
and reward≥200 results have not been redefined. Reward is only a ranking tie-breaker.

Selection: 64 new paired starts; reject candidates with >5pp nominal loss.
Approval: 256 independent paired starts, >=5pp observed goal gain,
positive task-gain lower bound, and nominal-gain lower bound >=−5pp.
Bounds use 20,000 paired-seed bootstrap resamples at confidence 99.75% per comparison (Bonferroni-style adjustment across
all registered task and retention comparisons). These are approximate statistical
screens, not formal safety guarantees. One candidate per run reaches approval.
No retries on approval data. Rejection deploys original weights.

Reversed actuators require the opposite commands. Those edits are accepted only
for that explicit context, and deployment on original physics restores the base
checkpoint. They are a sanity check, not a universal skill-preserving repair.

## Decisions before final test

| Run | Selected candidate | Accepted | Reason |
| --- | ---: | --- | --- |
| compound-llm-glm-seed1 | — | False | no_candidate_passed_selection |
| compound-llm-seed0 | — | False | no_candidate_passed_selection |
| compound-scripted-seed0 | — | False | no_candidate_passed_selection |
| crosswind-llm-glm-seed1 | — | False | no_candidate_passed_selection |
| crosswind-llm-seed0 | 3 | False | insufficient_completion_improvement |
| crosswind-scripted-seed0 | — | False | no_candidate_passed_selection |
| mirrored_thrusters-llm-seed0 | 0 | True | changed context only; base checkpoint for original mapping |
| mirrored_thrusters-scripted-seed1 | 0 | True | changed context only; base checkpoint for original mapping |
| weak_main-llm-seed0 | — | False | no_candidate_passed_selection |
| weak_main-scripted-seed0 | — | False | no_candidate_passed_selection |

## Fresh deployment test

256 fresh paired starts per condition. Test outcomes never affect approval.

| Run | Original goal completion | Deployed goal completion | Deployed patch |
| --- | ---: | ---: | --- |
| compound-llm-glm-seed1 | 5.1% | 5.1% | base fallback |
| compound-llm-seed0 | 5.1% | 5.1% | base fallback |
| compound-scripted-seed0 | 5.1% | 5.1% | base fallback |
| crosswind-llm-glm-seed1 | 9.0% | 9.0% | base fallback |
| crosswind-llm-seed0 | 9.0% | 9.0% | base fallback |
| crosswind-scripted-seed0 | 9.0% | 9.0% | base fallback |
| mirrored_thrusters-llm-seed0 | 0.0% | 88.3% | approved weights |
| mirrored_thrusters-scripted-seed1 | 0.0% | 88.3% | approved weights |
| weak_main-llm-seed0 | 79.7% | 79.7% | base fallback |
| weak_main-scripted-seed0 | 79.7% | 79.7% | base fallback |

Accepted 2/10 proposed run-level repairs. Rejected edits
are not described as improvements. A rejection can mean insufficient evidence,
not proof that a candidate is harmful. The gate changes which edits are deployed;
it does not create capabilities absent from the candidate pool.

This is exploratory reranking: candidate generation used the old reward-oriented prompts. It does not test a completion-trained model or completion-guided candidate generator.
The original target is still one pretrained actor, so cross-model transfer is untested.

Protocol hash: `32e51e6543a76204ab542132f16d7fa05a41c60b0f1c56cba59e33e51123e53d`.
Machine-readable selection, approval decisions, frozen weights, phase locks, and
per-episode measurements accompany this report. No source measurement was overwritten.
