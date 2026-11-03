# Attribution and artifact terms

Policy Surgery code: MIT, Copyright 2026 Latent Paths.
Author-created experimental records, evaluation metadata, plots, and videos:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attribution
“Ayush Bodade / Latent Paths, Policy Surgery (2026)”.

## Pretrained policy

The target is the actor from [sb3/ppo-LunarLander-v2](https://huggingface.co/sb3/ppo-LunarLander-v2),
revision `0961e4e83c29ec7dd52be7dd08f7c045cb23bda3`, trained with Stable Baselines3
and RL Baselines3 Zoo. The source was trained on LunarLander-v2; this experiment
uses Gymnasium 1.4.0 LunarLander-v3, deterministic argmax actions, and no observation
normalization. The actor has 4,996 parameters.

The pinned upstream model card does not state an explicit model-weight license.
This repository therefore contains numeric edit operations, checkpoint tensor
hashes, and a reconstruction command, **not upstream or edited checkpoints**.
Do not assume the upstream library's code license licenses its model weights.
`policy-surgery weights` downloads the original checkpoint from its owner, loads only
tensors with `weights_only=True`, and rebuilds the exact recorded edits locally.
We do not relicense those locally reconstructed weights.

Base actor tensor SHA-256:
`1ac3f4a0268e4ae96211a5c78e3f076afc08c1f412c0d4de521918fe9cc77df6`.

## Related work

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/),
Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (2026), motivates the
experiment. Its supplied logs and code are not included in this repository.

## Frozen acquisition source

`results/provenance/acquisition-source.zip` contains only the files whose original hashes
were frozen in `results/control-objective-comparison-v1/experiment.json`.
Verification authenticates these bytes against that manifest. The maintained code
refactors paths, transport dependencies, and analysis entry points; the archive
preserves what actually generated the recorded measurements. No saved protocol,
episode record, statistical decision, or checkpoint identity is rewritten.
