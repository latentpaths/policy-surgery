import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from policy_surgery import acceptance, world
from policy_surgery.acceptance import choose, decision
from policy_surgery.policy import SHAPES, Policy


def candidate(index, goal, nominal, reward):
    return {
        "candidate": {"id": index},
        "damaged": {"goal_success_rate": goal, "mean_return": reward},
        "nominal": {"goal_success_rate": nominal},
    }


def test_selection_prioritizes_completion_and_rejects_retention_regression():
    pool = [candidate(0, 0.4, 0.95, 300), candidate(1, 0.7, 0.94, -10), candidate(2, 1.0, 0.1, 500)]
    assert choose(pool, 0.3, 0.95)["candidate"]["id"] == 1
    assert choose(pool, 0.3, 0.95, context_specific=True)["candidate"]["id"] == 2
    assert choose([candidate(0, 0.31, 0.95, 1000)], 0.3, 0.95) is None


def test_approval_rejects_uncertain_gains_and_unsafe_improvements():
    assert not decision(
        {"gain": 0.2, "interval": [-0.01, 0.3]}, {"gain": 0, "interval": [0, 0]}, 0.05, 0.05
    )["accepted"]
    assert not decision(
        {"gain": 0.2, "interval": [0.1, 0.3]},
        {"gain": -0.1, "interval": [-0.15, -0.05]},
        0.05,
        0.05,
    )["accepted"]
    assert decision(
        {"gain": 0.2, "interval": [0.1, 0.3]}, {"gain": 0, "interval": [-0.04, 0.04]}, 0.05, 0.05
    )["accepted"]


@pytest.mark.parametrize("inside", [False, True])
def test_goal_requires_both_legs_inside_pad_not_just_asleep(monkeypatch, inside):
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.game_over = False
            self.lander = SimpleNamespace(awake=False)
            self.helipad_x1, self.helipad_x2 = -1, 1
            self.legs = [
                SimpleNamespace(ground_contact=True, position=SimpleNamespace(x=x))
                for x in ([-0.1, 0.1] if inside else [-2, 2])
            ]

        def reset(self, seed):
            return np.zeros(8), {}

        def step(self, action):
            return np.zeros(8), 250, True, False, {}

        def close(self):
            pass

    monkeypatch.setattr(world.gym, "make", lambda *args, **kwargs: Env())
    result = world.episode(SimpleNamespace(act=lambda obs: 0), world.CONDITIONS["nominal"], 0)
    assert result["safe_landing"]
    assert result["goal_success"] is inside


def test_independent_phases_freeze_approval_and_scope_deployment(tmp_path, monkeypatch):
    root = tmp_path / "approval"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    base = Policy({k: np.zeros(shape) for k, shape in SHAPES.items()})
    changed = Policy(base.tensors)
    changed.tensors["b3"][0] = 1
    base.save(cache / "base.npz")
    changed.save(tmp_path / "candidate.npz")
    monkeypatch.setattr(acceptance, "CACHE", cache)
    p = {
        "base_sha256": base.digest(),
        "runs": [
            {
                "run": "test-run",
                "condition": "crosswind",
                "method": "llm",
                "candidates": [
                    {
                        "id": 0,
                        "actor_sha256": changed.digest(),
                        "path": str(tmp_path / "candidate.npz"),
                    }
                ],
            }
        ],
        "selection_seeds": [1, 2, 3, 4],
        "approval_seeds": [10, 11, 12, 13],
        "test_seeds": [20, 21, 22, 23],
        "nominal_loss_tolerance": 0.05,
        "minimum_observed_gain": 0.05,
        "family_alpha": 0.05,
        "family_comparisons": 2,
        "bootstrap_resamples": 2000,
    }
    p["protocol_sha256"] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    (root / "protocol.json").write_text(json.dumps(p))
    seen = []

    def evaluate(actor, condition, seeds):
        seen.extend(seeds)
        good = condition.name == "nominal" or actor.digest() == changed.digest()
        return {
            "goal_success_rate": float(good),
            "mean_return": float(good),
            "per_episode": [{"goal_success": good} for _ in seeds],
        }

    monkeypatch.setattr(acceptance, "evaluate", evaluate)
    acceptance.select(root)
    assert set(seen) == {1, 2, 3, 4}
    seen.clear()
    result = acceptance.approve(root)
    assert result["runs"][0]["accepted"] and set(seen) == {10, 11, 12, 13}
    with pytest.raises(ValueError):
        acceptance.select(root)
    with pytest.raises(FileExistsError):
        acceptance.approve(root)
    assert acceptance.deployed_policy(root, "test-run", "nominal").digest() == base.digest()
    assert acceptance.deployed_policy(root, "test-run", "crosswind").digest() == changed.digest()
    seen.clear()
    acceptance.test(root)
    assert set(seen) == {20, 21, 22, 23}
