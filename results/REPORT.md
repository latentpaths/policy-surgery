# Self-Patch: task completion after frozen controller edits

Results prepared 3 November 2026 (Asia/Kolkata). All numbers below are derived
from the saved Box2D records; no new model calls or experiments were run.

## 1. Principal finding

A frozen language model can produce a direct weight patch that exactly repairs
reversed side thrusters in one pretrained LunarLander controller. Scripted search
produces the identical checkpoint. On 256 held-out starts, both recover helipad
completion from 0/256 to 226/256 (88.3%; paired gain +88.3 percentage points,
95% bootstrap interval +84.4 to +92.2). This is an exact-reversal sanity check.

**No patch was approved for weak thrust, crosswind, or combined physics.**
Seven of eight harder-condition runs failed selection; the remaining crosswind
GPT-OSS candidate failed the independent approval gate. Final deployment in those
conditions therefore uses the original weights. These results establish a limited
controller-editing demonstration and a negative result for reliable repair on
the harder conditions, with one target network and one completed search per
reported editor/condition.

## 2. Experimental design and outcome definitions

The target is a 4,996-parameter actor (`8 → tanh(64) → tanh(64) → 4 logits`,
argmax actions), imported from `sb3/ppo-LunarLander-v2` and evaluated in
Gymnasium 1.4.0 LunarLander-v3. Editors are off-the-shelf GPT-OSS-120B and GLM-5.3;
neither editor nor target receives gradient training in this experiment.

Each proposal starts from the original checkpoint. The editor can inspect bounded
tensor rows and submit bounded numeric edits. The scripted comparator searches
input gains and output-logit biases. Both have a twelve-proposal cap; their edit
spaces differ. Saved weights act alone during an episode. The reversal deployment
rule chooses the patch only for the known reversed context and the base checkpoint
for original physics; no observation/action wrapper is used within an episode.

| Condition | Physics change | Experimental role |
| --- | --- | --- |
| Weak main engine | Main impulse ×0.75 | Harder condition |
| Crosswind | Wind 18; turbulence 1.5 | Harder condition |
| Combined physics | Main impulse ×0.80; gravity −11.5; wind 12; turbulence 1 | Harder condition |
| Reversed thrusters | Left/right side-engine actions exchanged | Exact-repair sanity check |

**Primary completion outcome:** no crash, body asleep, both legs contacting the
ground and inside the helipad, within 1,000 steps. Reward ≥200 is a separate
benchmark outcome. Safe landing does not require helipad placement and must not
be interpreted as completion. Return is the undiscounted sum of episode rewards.

The original search used six development starts and twelve validation starts,
selecting by reward before a 64-start exploratory evaluation. The later completion
analysis reuses that existing candidate pool: it is a retrospective change of
objective, followed by separate selection, approval, and final episode sets.

| Completion phase | Starts | Seed range | Rule |
| --- | ---: | --- | --- |
| Selection | 64 | 20000–20063 | Filter nominal loss >5 pp; rank completion, nominal completion, then reward; require ≥5 pp task gain |
| Approval | 256 | 21000–21255 | Task gain ≥5 pp and lower bound >0; nominal lower bound ≥−5 pp |
| Final evaluation | 256 | 22000–22255 | Evaluate the approved checkpoint, or original checkpoint after rejection |

Approval uses 20,000 paired-seed bootstrap resamples and Bonferroni alpha
0.05/20 = 0.0025 per interval (99.75% marginal intervals). These percentile
bootstrap bounds are approximate. The nominal-retention rule is explicitly waived
for reversal; using that patch under original physics is harmful. Final intervals
are descriptive 95% paired-seed bootstrap intervals. Episode starts are the
sampling unit; intervals do not measure variation across target networks or editor runs.

## 3. Selection and independent approval

Table 1 includes every completed source search. Gains are percentage points.
A dash means no candidate reached approval, not a measured zero.

| Condition | Method | Candidate | Approval gain [99.75% interval], pp | Decision |
| --- | --- | --- | --- | --- |
| Combined physics | GLM-5.3 | — | — | No selection |
| Combined physics | GPT-OSS-120B | — | — | No selection |
| Combined physics | Scripted search | — | — | No selection |
| Crosswind | GLM-5.3 | — | — | No selection |
| Crosswind | GPT-OSS-120B | 3 | +6.2 [-1.2, +14.1] | Rejected at approval |
| Crosswind | Scripted search | — | — | No selection |
| Reversed thrusters | GPT-OSS-120B | 0 | +85.5 [+78.9, +91.8] | Approved (reversal only) |
| Reversed thrusters | Scripted search | 0 | +85.5 [+78.9, +91.8] | Approved (reversal only) |
| Weak main engine | GPT-OSS-120B | — | — | No selection |
| Weak main engine | Scripted search | — | — | No selection |

The crosswind candidate gained 6.25 pp at approval, but its adjusted interval
was [−1.17, +14.06] pp. Nominal gain was +4.69 pp [−2.34, +12.11].
It passed retention and the observed-gain threshold, but its task-gain lower
bound crossed zero. The rejected candidate was not evaluated as a patch in
the final phase; the final crosswind rows below describe fallback behavior.

## 4. Held-out completion and deployed behavior

Table 2. Each row uses 256 paired starts. Counts are helipad completions.
Repeated rows within a condition reuse the same base-policy outcomes and starts;
they are not independent replications. Zero effects after rejection arise from
deploying the identical original checkpoint, not from testing an ineffective patch.

| Condition | Method | Deployed weights | Completion count | Gain [95% interval], pp | Return |
| --- | --- | --- | --- | --- | --- |
| Combined physics | GLM-5.3 | Original | 13/256 → 13/256 | +0.0 [+0.0, +0.0] | -107.2 |
| Combined physics | GPT-OSS-120B | Original | 13/256 → 13/256 | +0.0 [+0.0, +0.0] | -107.2 |
| Combined physics | Scripted search | Original | 13/256 → 13/256 | +0.0 [+0.0, +0.0] | -107.2 |
| Crosswind | GLM-5.3 | Original | 23/256 → 23/256 | +0.0 [+0.0, +0.0] | 90.0 |
| Crosswind | GPT-OSS-120B | Original | 23/256 → 23/256 | +0.0 [+0.0, +0.0] | 90.0 |
| Crosswind | Scripted search | Original | 23/256 → 23/256 | +0.0 [+0.0, +0.0] | 90.0 |
| Reversed thrusters | GPT-OSS-120B | Patch | 0/256 → 226/256 | +88.3 [+84.4, +92.2] | 244.8 |
| Reversed thrusters | Scripted search | Patch | 0/256 → 226/256 | +88.3 [+84.4, +92.2] | 244.8 |
| Weak main engine | GPT-OSS-120B | Original | 204/256 → 204/256 | +0.0 [+0.0, +0.0] | 221.1 |
| Weak main engine | Scripted search | Original | 204/256 → 204/256 | +0.0 [+0.0, +0.0] | 221.1 |

Both reversal methods produce the same actor hash and identical episode outcomes.
Their agreement is one exact repair demonstrated twice, not two independent
successful strategies. Reversal gains require knowing which actuator mapping
is active; the patched checkpoint alone does not retain original-context ability.

## 5. Exploratory reward results

Table 3. Earlier reward-selected patches, 64 held-out starts per condition.
These are different selections and episodes from Table 2. All harder-condition
patches and controls are retained, including regressions. Reward≥200 and safe
landing are not the helipad completion metric. Nominal Δ is the same patch's
return change on original physics. Intervals use 2,000 paired-seed resamples
without multiplicity correction.

| Condition | Method | Return | Reward≥200 | Safe landing | Δ return [95% interval] | Nominal Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Weak main engine | Original | 220.8 | 89.1% | 95.3% | — | — |
| Weak main engine | Gym heuristic | -16.5 | 0.0% | 0.0% | — | — |
| Weak main engine | GPT-OSS-120B | 224.3 | 87.5% | 93.8% | +3.5 [-3.0, +10.2] | -2.5 |
| Weak main engine | Scripted search | 198.6 | 71.9% | 73.4% | -22.2 [-44.2, -0.9] | -20.7 |
| Crosswind | Original | 81.6 | 25.0% | 45.3% | — | — |
| Crosswind | Gym heuristic | 67.7 | 40.6% | 42.2% | — | — |
| Crosswind | GLM-5.3 | 84.1 | 26.6% | 43.8% | +2.6 [-26.1, +32.0] | -1.2 |
| Crosswind | GPT-OSS-120B | 56.3 | 26.6% | 43.8% | -25.3 [-60.7, +6.4] | -3.9 |
| Crosswind | Scripted search | 70.9 | 17.2% | 40.6% | -10.7 [-47.5, +24.5] | -14.1 |
| Combined physics | Original | -97.1 | 15.6% | 21.9% | — | — |
| Combined physics | Gym heuristic | -40.8 | 0.0% | 1.6% | — | — |
| Combined physics | GLM-5.3 | -11.0 | 12.5% | 25.0% | +86.1 [+34.4, +143.0] | -58.7 |
| Combined physics | GPT-OSS-120B | -151.4 | 6.2% | 10.9% | -54.3 [-109.6, -1.3] | -3.9 |
| Combined physics | Scripted search | -40.5 | 0.0% | 0.0% | +56.6 [+3.6, +110.1] | -191.1 |
| Reversed thrusters | Original | -685.3 | 0.0% | 0.0% | — | — |
| Reversed thrusters | GPT-OSS-120B | 247.5 | 98.4% | 98.4% | +932.7 [+896.2, +971.3] | -932.7 |
| Reversed thrusters | Scripted search | 247.5 | 98.4% | 98.4% | +932.7 [+896.2, +971.3] | -932.7 |

The GLM combined-physics patch raises return by +86.1 [34.4, 143.0] points
over unchanged weights, but reward≥200 falls from 15.6% to 12.5%, and nominal
return falls by 58.7. The same analysis does not establish better reward than
scripted search: +29.5 [−7.2, +68.7]. Reward gain alone therefore does not
establish successful repair. Weak-thrust GPT-OSS outperforms scripted search
on reward, but its reward gain over unchanged weights includes zero.

Table 4. Paired return advantage over the scripted patch; exploratory 95% intervals.

| Condition | Editor | Return advantage [95% interval] |
| --- | --- | --- |
| Combined physics | GLM-5.3 | +29.5 [-7.2, +68.7] |
| Combined physics | GPT-OSS-120B | -110.9 [-159.6, -63.4] |
| Crosswind | GLM-5.3 | +13.2 [-26.9, +54.4] |
| Crosswind | GPT-OSS-120B | -14.6 [-59.8, +31.4] |
| Reversed thrusters | GPT-OSS-120B | +0.0 [+0.0, +0.0] |
| Weak main engine | GPT-OSS-120B | +25.7 [+4.5, +47.2] |

## 6. Scope, failures, and reproducibility

Only one pretrained target and one completed stochastic search per available
editor/condition were measured. GLM was measured on crosswind and combined
physics only. The LLM has a broader edit space than the scripted comparator;
this experiment cannot isolate language-model reasoning from search-space
differences. Provider sampling reproducibility is not guaranteed by a run index.

The completion protocol was defined after the exploratory campaign and reuses
its candidate library. Independent episode sets protect the subsequent decisions
from reuse of those starts; they do not turn the study into a preregistered
experiment. There is no evidence here for general neural understanding, real-robot
transfer, modification of the LLM's own weights, or recursive self-improvement.

Three source searches failed before producing a frozen patch: two GLM attempts
produced no valid candidate, and one scripted reversal run stopped on an integer
casting error after four proposals. Corrected/later attempts are separately
identified. The weak-thrust GPT-OSS search stopped after one candidate. Failed
attempts, full candidate pools, transcripts, and frozen inventories remain in
`control-v2/search/`; none are removed based on score.

Recorded source-campaign inference ledger: 76 requests;
estimated committed cost $0.5948945, including
reserved upper bounds for 3 unknown outcomes.
This is an estimate from recorded token rates, not an invoice reconciliation.
The completion analysis made no new model calls. Simulation ran on local CPU.

Rebuild this report and the full-precision CSV/LaTeX tables from the repository root:

```sh
uv run python scripts/summarize_results.py
```

The report builder verifies protocol hashes, selection/approval locks, phase
separation, run coverage, deployed checkpoint identities, completion differences,
and final aggregate metrics against their recorded episodes. It does not rerun
the simulator. CSV rates are fractions (0–1); manuscript gains are percentage
points. CSV files retain exact run IDs and full numeric precision. LaTeX exports
use manuscript formatting; include them with `\input{}` and the `graphicx` package.

- [Completion protocol](control-completion-v1/protocol.json)
- [Selection](control-completion-v1/selection.json), [approval](control-completion-v1/approval.json), and [final episodes](control-completion-v1/evaluation.json)
- [Exploratory protocol](control-v2/protocol.json), [summary](control-v2/evaluation/summary.json), and [paired comparisons](control-v2/evaluation/head-to-head.json)
- [Full-precision tables](tables/completion.csv) and [artifact hashes](tables/provenance.json)
- [Experiment reproduction guide](../docs/CONTROL_EXPERIMENTS.md)

Upstream target revision: `0961e4e83c29ec7dd52be7dd08f7c045cb23bda3`.
Base actor SHA-256: `1ac3f4a0268e4ae96211a5c78e3f076afc08c1f412c0d4de521918fe9cc77df6`.
Completion protocol SHA-256: `32e51e6543a76204ab542132f16d7fa05a41c60b0f1c56cba59e33e51123e53d`.
