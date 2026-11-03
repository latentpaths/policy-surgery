"""Pinned pretrained actor, safe tensor import, and explicit weight-edit operations."""

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = "sb3/ppo-LunarLander-v2"
REVISION = "0961e4e83c29ec7dd52be7dd08f7c045cb23bda3"
KEYS = {
    "w1": "mlp_extractor.policy_net.0.weight",
    "b1": "mlp_extractor.policy_net.0.bias",
    "w2": "mlp_extractor.policy_net.2.weight",
    "b2": "mlp_extractor.policy_net.2.bias",
    "w3": "action_net.weight",
    "b3": "action_net.bias",
}
SHAPES = {"w1": (64, 8), "b1": (64,), "w2": (64, 64), "b2": (64,), "w3": (4, 64), "b3": (4,)}


class Policy:
    def __init__(self, tensors):
        if set(tensors) != set(SHAPES):
            raise ValueError("Actor requires exactly w1, b1, w2, b2, w3, b3")
        self.tensors = {k: np.asarray(tensors[k], dtype=np.float32).copy() for k in SHAPES}
        if any(
            t.shape != SHAPES[k] or not np.isfinite(t).all() or np.abs(t).max() > 1000
            for k, t in self.tensors.items()
        ):
            raise ValueError("Invalid actor tensor")

    def forward(self, obs, hidden=False):
        t = self.tensors
        x = np.asarray(obs, dtype=np.float32)
        h1 = np.tanh(x @ t["w1"].T + t["b1"])
        h2 = np.tanh(h1 @ t["w2"].T + t["b2"])
        logits = h2 @ t["w3"].T + t["b3"]
        return (logits, h1, h2) if hidden else logits

    def act(self, obs):
        return int(np.argmax(self.forward(obs)))

    def digest(self):
        return hashlib.sha256(b"".join(t.tobytes() for t in self.tensors.values())).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.tensors)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls({key: data[key] for key in SHAPES})

    def changed(self, other):
        return sum(int(np.count_nonzero(t != other.tensors[k])) for k, t in self.tensors.items())


def fetch_actor(cache: Path) -> tuple[Policy, dict]:
    if (cache / "base.npz").exists() and (cache / "provenance.json").exists():
        actor = Policy.load(cache / "base.npz")
        metadata = json.loads((cache / "provenance.json").read_text())
        if (metadata["repo"], metadata["revision"], metadata["actor_sha256"]) != (
            REPO,
            REVISION,
            actor.digest(),
        ):
            raise ValueError("Cached pretrained actor provenance mismatch")
        return actor, metadata
    import torch
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(REPO, "ppo-LunarLander-v2/policy.pth", revision=REVISION))
    tensors = torch.load(source, map_location="cpu", weights_only=True)
    actor = Policy({key: tensors[original].numpy() for key, original in KEYS.items()})
    x = np.random.default_rng(99).normal(size=(1024, 8)).astype(np.float32)
    with torch.no_grad():
        y = torch.from_numpy(x)
        for weight, bias in (("w1", "b1"), ("w2", "b2")):
            y = torch.tanh(
                torch.nn.functional.linear(y, tensors[KEYS[weight]], tensors[KEYS[bias]])
            )
        expected = torch.nn.functional.linear(y, tensors[KEYS["w3"]], tensors[KEYS["b3"]]).numpy()
    actual = actor.forward(x)
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-4 or not np.array_equal(actual.argmax(1), expected.argmax(1)):
        raise RuntimeError(
            "Imported NumPy actor does not match the original Torch tensor computation"
        )
    metadata = {
        "repo": REPO,
        "revision": REVISION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "actor_sha256": actor.digest(),
        "parameters": 4996,
        "architecture": "8 -> tanh(64) -> tanh(64) -> 4 logits; argmax",
        "normalization": "none (upstream training config)",
        "torch_numpy_max_abs_error": error,
        "action_agreement_samples": len(x),
    }
    actor.save(cache / "base.npz")
    (cache / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return actor, metadata


def vector(value, size, default):
    result = np.asarray(value if value is not None else [default] * size, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all() or np.abs(result).max() > 20:
        raise ValueError(f"Expected {size} finite numbers with magnitude <= 20")
    return result


def apply_ops(base: Policy, ops: list[dict]) -> Policy:
    """Every candidate starts from the same base; no inference-time wrapper remains."""
    if not isinstance(ops, list) or len(ops) > 32:
        raise ValueError("At most 32 operations per candidate")
    result = Policy(base.tensors)
    t = result.tensors
    for op in ops:
        kind = op["op"]
        if kind == "input_affine":
            scale = vector(op.get("scale"), 8, 1)
            offset = vector(op.get("offset"), 8, 0)
            old = t["w1"].copy()
            t["w1"] = old * scale[None, :]
            t["b1"] += old @ offset
        elif kind == "output_affine":
            scale = vector(op.get("scale"), 4, 1)
            offset = vector(op.get("offset"), 4, 0)
            permutation = op.get("permutation", [0, 1, 2, 3])
            if sorted(permutation) != list(range(4)) or any(
                type(i) is not int for i in permutation
            ):
                raise ValueError("Invalid action permutation")
            t["w3"] = t["w3"][permutation] * scale[:, None]
            t["b3"] = t["b3"][permutation] * scale + offset
        elif kind == "hidden_scale":
            layer = op.get("layer")
            if layer not in (1, 2):
                raise ValueError("Hidden layer must be 1 or 2")
            scale = vector(op.get("scale"), 64, 1)
            t[f"w{layer}"] *= scale[:, None]
            t[f"b{layer}"] *= scale
        elif kind == "set":
            edits = op["edits"]
            if not isinstance(edits, list) or len(edits) > 4996:
                raise ValueError("Invalid scalar edit list")
            for edit in edits:
                key, index, value = edit["tensor"], edit["index"], edit["value"]
                if key not in SHAPES or len(index) != len(SHAPES[key]):
                    raise ValueError("Invalid tensor/index")
                if any(type(i) is not int or not 0 <= i < n for i, n in zip(index, SHAPES[key])):
                    raise ValueError("Invalid index")
                if type(value) not in (float, int) or not np.isfinite(value) or abs(value) > 1000:
                    raise ValueError("Invalid scalar weight")
                t[key][tuple(index)] = value
        else:
            raise ValueError("Unknown weight-edit operation")
    return Policy(t)
