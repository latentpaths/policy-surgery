"""Reconstruct pinned upstream weights and every recorded numeric edit; no inference."""

from . import campaign
from .analysis import read
from .io import ROOT, checkpoint_path
from .policy import Policy, apply_ops, fetch_actor


def reconstruct():
    base, metadata = fetch_actor(campaign.CACHE)
    count = 0
    for path in sorted((ROOT / "results").glob("*/search/*/candidates.json")):
        for candidate in read(path):
            if "actor_sha256" not in candidate:
                continue
            model = apply_ops(base, candidate["ops"])
            if model.digest() != candidate["actor_sha256"]:
                raise ValueError(f"Patch mismatch: {path}, {candidate['id']}")
            model.save(path.parent / "candidates" / f"{candidate['id']:02d}.npz")
            count += 1
        frozen = path.parent / "frozen.json"
        if frozen.exists():
            f = read(frozen)
            if "selected" in f:
                model = apply_ops(base, f["selected"]["ops"])
                if model.digest() != f["actor_sha256"]:
                    raise ValueError("Frozen checkpoint mismatch")
                model.save(path.parent / "frozen.npz")
    selected = 0
    for file in sorted((ROOT / "results").glob("*/selection.json")):
        for item in read(file)["runs"]:
            if item["candidate"] is None:
                continue
            model = Policy.load(checkpoint_path(item["candidate"]["path"]))
            if model.digest() != item["candidate"]["actor_sha256"]:
                raise ValueError("Selected checkpoint mismatch")
            model.save(file.parent / "selected" / f"{item['run']}.npz")
            selected += 1
    return {
        "base_sha256": metadata["actor_sha256"],
        "candidate_checkpoints": count,
        "selected_checkpoints": selected,
        "inference_calls": 0,
    }
