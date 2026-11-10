# Policy Surgery

A frozen GPT-OSS-120B edits a pretrained LunarLander PPO actor after its dynamics
change. The edited policy runs alone, with no gradient updates or LLM calls during
deployment. Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

[Article and demos](https://www.latentpaths.com/writings/policy-surgery) · [Attribution and terms](NOTICE.md)

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

[Protocol, replay, figures, and new runs](docs/reproduction.md).

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

## Citation

<!-- BEGIN generated citations -->

```bibtex
@misc{bodadearticlepolicysurgery,
  author = {Bodade, Ayush},
  title = {{You Don’t Need Astra to Edit a Model’s Weights}},
  year = {2026},
  month = {11},
  day = {03},
  date = {2026-11-03},
  howpublished = {Latent Paths},
  url = {https://www.latentpaths.com/writings/policy-surgery},
  note = {Blog post}
}
```

<!-- END generated citations -->
