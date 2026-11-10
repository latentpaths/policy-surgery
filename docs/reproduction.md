# Protocol and additional commands

Run commands from the repository root after the [README setup](../README.md#reproduce).

## Protocol

The target has 4,996 parameters: `8 → tanh(64) → tanh(64) → 4 logits`, argmax
actions, no observation normalization. Its pinned upstream revision is recorded
in [NOTICE.md](../NOTICE.md). Evaluation uses Gymnasium 1.4.0 LunarLander-v3.
The editor sees disclosed dynamics changes and development feedback, can inspect
weights, and proposes numeric edits. Every candidate starts from the original
policy; each search allows 12 candidates and 28 tool turns.

The harder-condition campaign contains 18 LLM runs: reduced thrust, crosswind,
and combined perturbations × return/success objectives × three search seeds.
The success objective also targets original-dynamics retention. Auxiliary controls
bring the original campaign to 36 runs; all controls and failures are retained.

| Phase | Episodes | Seeds |
| --- | ---: | --- |
| Development | 64 | 50000–50063 |
| Selection | 64 | 60000–60063 |
| Approval | 256 | 61000–61255 |
| Final test | 256 | 62000–62255 |

Success requires both grounded legs inside the helipad, the body at rest, and no
crash within 1,000 steps. Selection requires ≥5 pp gain and ≤5 pp nominal loss.
Approval requires ≥5 pp gain, a positive task-gain lower bound, and nominal-gain
lower bound ≥−5 pp, using 20,000 paired resamples and Bonferroni alpha 0.05/72.
Rejection deploys the base policy. Final intervals are unadjusted; shared episode
seeds are not independent replications across policies.

Exact settings, prompts, edits, decisions, and episodes live in `results/`.
`results/provenance/` preserves source bytes authenticated by the acquisition
manifest. Historical reports describe earlier campaigns and retain their original names
and conclusions: [initial completion analysis](../results/REPORT.md),
[matched objective campaign](../results/control-objective-comparison-v1/REPORT.md),
and [independent approval and final test](../results/control-objective-comparison-gate-v1/REPORT.md).
Use the results above and `policy-surgery reproduce` for the current account.

## Weights, figures, and demos

```sh
uv sync --frozen --extra weights --extra figures
uv run policy-surgery weights
uv run policy-surgery verify --replay
uv run policy-surgery figures
SDL_VIDEODRIVER=dummy uv run policy-surgery render
```

`weights` downloads the pinned upstream actor and reconstructs the exact edits,
checking tensor hashes. `verify --replay` checks 384 candidate episodes and 36
deployments: 420 simulator replays. No inference calls are made. Figures and four
fixed-seed demos go to `output/`; rendering requires `ffmpeg`. Upstream weight
terms are separate, and reconstructed checkpoints stay out of Git.

### Retained article artifacts

[`article-assets/`](article-assets/) preserves the editable
[completion-effects](article-assets/completion-effects.svg) and
[paired-outcomes](article-assets/paired-outcomes.svg) figures, plus the
[seed-1000 reversal replay metadata](article-assets/reversal-llm.json) used by the
website's two-policy demo. These exact exports were moved from the website on
10 November 2026. The JSON records both actor hashes and episode traces; it is
an illustrative replay, separate from the 256-start evaluation. Regenerated
figures and videos still go to `output/`. The website retains only embedded
images, videos, and posters.

## New runs

```sh
uv sync --frozen --extra weights --extra inference
# Authenticate Prime, or set PRIME_API_KEY privately in your shell.
uv run policy-surgery run --output output/new-campaign --max-cost 5 --workers 1
```

This command buys inference under the explicit cumulative USD cap. It freezes
settings and current model prices before dispatch. Unknown outcomes retain budget
reservations. Existing results cannot be overwritten; provider sampling can make
fresh runs differ. Low-level operations: `policy-surgery experiment --help`.
