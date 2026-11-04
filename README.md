# Policy Surgery

A frozen GPT-OSS-120B edits a pretrained LunarLander PPO actor after its dynamics
change. The edited policy runs alone, with no gradient updates or LLM calls during
deployment. Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

[Project](https://www.latentpaths.com/projects/policy-surgery) · [Article and demos](https://www.latentpaths.com/writings/policy-surgery) · [Attribution and terms](NOTICE.md)

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/), from this checkout:

```sh
uv sync --frozen
uv run policy-surgery verify
uv run policy-surgery reproduce
```

After installation, both commands run offline without credentials or weight
downloads. Results go to `output/analysis/` as CSV, JSON, and Markdown; retained
measurements are unchanged.

## Results

All rows use 256 held-out episodes. Intervals are descriptive paired 95% bootstrap
bounds; gains are percentage points.

| Condition / LLM objective / search seed | Base → edited successes | Gain [95% interval] |
| --- | ---: | ---: |
| Reversed thrusters | 0 → 226 | +88.3 [+84.4, +92.2] |
| Crosswind / success / 0 | 25 → 40 | +5.9 [+0.8, +11.3] |
| Crosswind / return / 1 | 25 → 36 | +4.3 [−0.4, +9.0] |
| Crosswind / return / 2 | 25 → 36 | +4.3 [0.0, +9.0] |
| Reduced thrust / success / 1 | 207 → 216 | +3.5 [−0.4, +7.4] |

Reversal is an exact action permutation specific to reversed thrusters. Of the
four harder-condition policies that passed approval, only one final interval
excludes zero. No combined-dynamics repair qualified. Four provider-limited
searches remain in the records. One target and three search seeds do not establish
an advantage over PPO adaptation, other optimizers, or across-policy generalization.

## Protocol

The target has 4,996 parameters: `8 → tanh(64) → tanh(64) → 4 logits`, argmax
actions, no observation normalization. Its pinned upstream revision is recorded
in [NOTICE.md](NOTICE.md). Evaluation uses Gymnasium 1.4.0 LunarLander-v3.
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
and conclusions: [initial completion analysis](results/REPORT.md),
[matched objective campaign](results/control-objective-comparison-v1/REPORT.md),
and [independent approval and final test](results/control-objective-comparison-gate-v1/REPORT.md).
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

## Development

```sh
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check policy_surgery tests
uv run ruff format --check policy_surgery tests
```

`policy_surgery/` holds the code, `tests/` the checks, and `results/` the retained
evidence. Generated outputs, weights, new runs, and author tooling stay ignored.

Code: MIT. Author-created records and media: CC BY 4.0. See [NOTICE.md](NOTICE.md)
and [CITATION.cff](CITATION.cff).

[Citation guide](CITATION.md) · [BibTeX](CITATION.bib)
