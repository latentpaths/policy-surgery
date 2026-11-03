# Training-free controller edits on LunarLander

The off-the-shelf language-model editor stays frozen. The target was trained
upstream; here it receives direct weight edits, without fine-tuning or policy gradients.
No editor, conversation, or observation/action wrapper is used at deployment.

## Frozen evaluation

Each row uses the same 64 previously unseen initial seeds for its condition.
Primary metric: mean undiscounted episode return. 'Solved' means reward ≥200;
safe landing is measured separately. Selection uses six development and twelve
validation seeds, with a maximum of twelve proposals per search. Only the top
three development candidates enter validation; test data never reach the editor.

| Condition | Method / run | Return | Solved | Safe landing | Δ return vs original [95% paired CI] | Nominal Δ return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | unchanged | 247.5 | 98.4% | 98.4% | — | — |
| mirrored_thrusters | unchanged | -685.3 | 0.0% | 0.0% | — | — |
| mirrored_thrusters | llm / 0 | 247.5 | 98.4% | 98.4% | +932.7 [+896.2, +971.3] | -932.7 |
| mirrored_thrusters | scripted / 1 | 247.5 | 98.4% | 98.4% | +932.7 [+896.2, +971.3] | -932.7 |
| weak_main | unchanged | 220.8 | 89.1% | 95.3% | — | — |
| weak_main | Gym heuristic (external controller) | -16.5 | 0.0% | 0.0% | — | — |
| weak_main | llm / 0 | 224.3 | 87.5% | 93.8% | +3.5 [-3.0, +10.2] | -2.5 |
| weak_main | scripted / 0 | 198.6 | 71.9% | 73.4% | -22.2 [-44.2, -0.9] | -20.7 |
| crosswind | unchanged | 81.6 | 25.0% | 45.3% | — | — |
| crosswind | Gym heuristic (external controller) | 67.7 | 40.6% | 42.2% | — | — |
| crosswind | llm-glm / 1 | 84.1 | 26.6% | 43.8% | +2.6 [-26.1, +32.0] | -1.2 |
| crosswind | llm / 0 | 56.3 | 26.6% | 43.8% | -25.3 [-60.7, +6.4] | -3.9 |
| crosswind | scripted / 0 | 70.9 | 17.2% | 40.6% | -10.7 [-47.5, +24.5] | -14.1 |
| compound | unchanged | -97.1 | 15.6% | 21.9% | — | — |
| compound | Gym heuristic (external controller) | -40.8 | 0.0% | 1.6% | — | — |
| compound | llm-glm / 1 | -11.0 | 12.5% | 25.0% | +86.1 [+34.4, +143.0] | -58.7 |
| compound | llm / 0 | -151.4 | 6.2% | 10.9% | -54.3 [-109.6, -1.3] | -3.9 |
| compound | scripted / 0 | -40.5 | 0.0% | 0.0% | +56.6 [+3.6, +110.1] | -191.1 |

## LLM versus scripted weight search

| Condition | Editor / run | Paired return advantage over scripted [95% CI] |
| --- | --- | ---: |
| compound | llm-glm / 1 | +29.5 [-7.2, +68.7] |
| compound | llm / 0 | -110.9 [-159.6, -63.4] |
| crosswind | llm-glm / 1 | +13.2 [-26.9, +54.4] |
| crosswind | llm / 0 | -14.6 [-59.8, +31.4] |
| mirrored_thrusters | llm / 0 | +0.0 [+0.0, +0.0] |
| weak_main | llm / 0 | +25.7 [+4.5, +47.2] |

## How to interpret this

The combined-physics GLM patch raises mean return from −97.1 to −11.0, but
reward≥200 success falls from 15.6% to 12.5%. Safe landings rise only from
21.9% to 25.0%, and nominal mean return drops by 58.7 points. The reward gain
is not evidence of a reliable repair. Wind and weak-thrust results do not
establish improvement over unchanged weights.

Mirrored thrusters is a sanity check: swapping two output rows exactly undoes
the action reversal. It is not the main research result. The harder conditions
change maximum available thrust, wind/torque, or multiple physical properties.
These conditions can require a different control strategy rather than a global
linear inverse. Weak thrust can reduce reward while still allowing safe landing.

Scripted search is an adaptive random search over input gains and action-logit
biases. It has the same proposal cap and selection protocol. The LLM additionally
has typed access to individual weight rows and arbitrary bounded scalar edits,
so this comparison does not isolate reasoning from differences in search space.
Actual proposal counts and all failed candidates are preserved in run manifests.
Three inference requests had unknown outcomes; their upper-bound costs stay
reserved. The weak-thrust GPT-OSS run stopped after one candidate. Two initial
GLM runs stopped before proposing any candidate; later attempts completed.
One scripted sanity run hit an integer-array casting bug after four proposals;
it is retained as failed, and the corrected run is reported separately.

Nominal regression measures a context-specific tradeoff, not mechanistic
proof of forgetting. In the reversal case it is expected: the same observation
requires opposite actions under opposite actuator mappings. Restoring the original
checkpoint restores the appropriate original-domain controller.
The GLM run index is not a provider sampling seed; provider seed support is
recorded in editor.json. Only one completed stochastic search per editor and
condition was measured, so editor-run variability remains unmeasured.

Intervals bootstrap paired initial seeds (2,000 resamples). They describe
variation across starts for one pretrained target, not across target networks
or editor training seeds. These are exploratory comparisons without correction
for multiple conditions; no general self-improvement claim follows.

## Provenance and cost

Target: [sb3/ppo-LunarLander-v2](https://huggingface.co/sb3/ppo-LunarLander-v2) at revision `0961e4e83c29ec7dd52be7dd08f7c045cb23bda3`. The actor was trained upstream
on LunarLander-v2 and is measured here on Gymnasium 1.4.0's LunarLander-v3.
The original Torch actor and imported NumPy actor agree on 1,024 sampled actions;
the exact measured numerical error and both tensor/source hashes are in protocol.json.
Upstream weights retain upstream terms; the task manifests and edit operations
are the new experiment artifacts. No upstream training-score reproduction is claimed.

Current durable inference ledger: {"cap_usd": 20.0, "estimated_committed_usd": 0.5948945, "pending_requests": 3, "requests": 76}.
Costs use conservative catalog token rates; unknown request outcomes retain
their reserved upper bounds. This campaign's inference allowance is $20, within
the user's $500 project ceiling. No GPU instance was rented: controller and
Box2D execution are local CPU work; the language-model editor uses Prime inference.

[Environment specification](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
