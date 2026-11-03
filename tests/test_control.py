import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("gymnasium")

import gymnasium.envs.box2d.lunar_lander as lunar

from policy_surgery import campaign
from policy_surgery.campaign import Budget, Search, scripted
from policy_surgery.policy import SHAPES, Policy, apply_ops
from policy_surgery.world import CONDITIONS, engine_power, episode


@pytest.fixture
def actor():
    rng = np.random.default_rng(5)
    return Policy({k: rng.normal(0, 0.2, shape).astype(np.float32) for k, shape in SHAPES.items()})


def test_affine_edits_are_compiled_into_weights_and_restore_exactly(actor, tmp_path):
    x = np.random.default_rng(7).normal(size=(12, 8)).astype(np.float32)
    scale, offset = np.linspace(0.7, 1.4, 8), np.linspace(-0.1, 0.1, 8)
    changed = apply_ops(
        actor, [{"op": "input_affine", "scale": scale.tolist(), "offset": offset.tolist()}]
    )
    np.testing.assert_allclose(changed.forward(x), actor.forward(x * scale + offset), atol=1e-5)
    mirrored = apply_ops(actor, [{"op": "output_affine", "permutation": [0, 3, 2, 1]}])
    np.testing.assert_allclose(mirrored.forward(x), actor.forward(x)[:, [0, 3, 2, 1]], atol=1e-6)
    before = actor.digest()
    actor.save(tmp_path / "original.npz")
    changed.save(tmp_path / "changed.npz")
    assert Policy.load(tmp_path / "original.npz").digest() == before
    assert Policy.load(tmp_path / "changed.npz").digest() == changed.digest()
    assert before != changed.digest()


def test_invalid_edit_is_atomic(actor):
    before = actor.digest()
    with pytest.raises(ValueError):
        apply_ops(actor, [{"op": "input_affine", "scale": [float("nan")] * 8}])
    with pytest.raises(ValueError):
        apply_ops(actor, [{"op": "set", "edits": [{"tensor": "b3", "index": [-1], "value": 1}]}])
    assert actor.digest() == before


def test_changed_engine_constants_are_always_restored():
    original = lunar.MAIN_ENGINE_POWER
    with pytest.raises(RuntimeError), engine_power(CONDITIONS["weak_main"]):
        assert lunar.MAIN_ENGINE_POWER == original * 0.75
        raise RuntimeError("interrupted episode")
    assert lunar.MAIN_ENGINE_POWER == original


def test_real_box2d_rollout_is_seeded_and_controller_only(actor):
    first = episode(actor, CONDITIONS["crosswind"], seed=501)
    second = episode(Policy(actor.tensors), CONDITIONS["crosswind"], seed=501)
    assert first == second


def test_durable_budget_prevents_concurrent_overspend(tmp_path):
    budget = Budget(tmp_path / "costs.sqlite", cap=1)

    def reserve():
        try:
            return budget.reserve("test", 0.6)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(2) as executor:
        keys = list(executor.map(lambda _: reserve(), range(2)))
    assert len([key for key in keys if key]) == 1
    assert budget.summary()["estimated_committed_usd"] == 0.6
    budget.settle(next(key for key in keys if key), 0.1)
    assert Budget(tmp_path / "costs.sqlite", cap=1).summary()["estimated_committed_usd"] == 0.1


def test_search_only_uses_dev_and_validation_and_freezes_hash(actor, tmp_path, monkeypatch):
    cache, root = tmp_path / "cache", tmp_path / "campaign"
    cache.mkdir()
    root.mkdir()
    actor.save(cache / "base.npz")
    monkeypatch.setattr(campaign, "CACHE", cache)
    protocol = {
        "source": {"actor_sha256": actor.digest()},
        "protocol_sha256": "test-protocol",
        "candidate_budget": 12,
        "development_seeds": [1, 2],
        "validation_seeds": [10, 11],
    }
    unsigned = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True).encode()
    ).hexdigest()
    (root / "protocol.json").write_text(json.dumps(protocol))
    seen = []

    def score(policy, condition, seeds, **kwargs):
        seen.extend(seeds)
        return {"mean_return": 1, "solution_rate": 0, "per_episode": []}

    monkeypatch.setattr(campaign, "evaluate", score)
    search = Search(root, "mirrored_thrusters", "scripted", 0)
    scripted(search)
    assert len(search.candidates) == 12
    assert set(seen) == {1, 2, 10, 11}
    frozen = json.loads((search.output / "frozen.json").read_text())
    assert Policy.load(search.output / "frozen.npz").digest() == frozen["actor_sha256"]
    (root / "evaluation").mkdir()
    with pytest.raises(ValueError, match="opened test data"):
        Search(root, "compound", "llm", 0)


def test_completion_search_default_and_no_unapproved_deployment(actor, tmp_path, monkeypatch):
    cache, root = tmp_path / "cache", tmp_path / "campaign"
    cache.mkdir()
    actor.save(cache / "base.npz")
    monkeypatch.setattr(campaign, "CACHE", cache)
    monkeypatch.setattr(
        campaign, "fetch_actor", lambda _: (actor, {"actor_sha256": actor.digest()})
    )
    seen = []

    def score(policy, condition, seeds, **kwargs):
        seen.extend(seeds)
        return {
            "mean_return": 10,
            "goal_success_rate": 0.5,
            "safe_landing_rate": 0.5,
            "solution_rate": 0.5,
            "per_episode": [],
        }

    monkeypatch.setattr(campaign, "evaluate", score)
    campaign.prepare(root)
    protocol = campaign.read_protocol(root)
    assert protocol["objective"] == "completion"
    search = Search(root, "crosswind", "scripted", 0)
    search.candidate({"ops": []})
    assert search.freeze()["status"] == "candidate_pool_frozen_requires_approval"
    assert not (search.output / "frozen.npz").exists()
    assert set(seen) == set(protocol["development_seeds"])
    better = {
        "development": {"goal_success_rate": 0.8, "mean_return": -100},
        "nominal": {"goal_success_rate": 0.5},
    }
    rewarding = {
        "development": {"goal_success_rate": 0.3, "mean_return": 1000},
        "nominal": {"goal_success_rate": 0.5},
    }
    regressing = {
        "development": {"goal_success_rate": 1.0, "mean_return": 1000},
        "nominal": {"goal_success_rate": 0.1},
    }
    assert max([better, rewarding, regressing], key=search.rank) is better
    with pytest.raises(ValueError, match="require gate"):
        campaign.final_evaluation(root)
    assert not (root / "evaluation").exists()

    from policy_surgery import acceptance

    approval = tmp_path / "gate"
    acceptance.initialize(root, approval)
    gate_protocol = acceptance.protocol(approval)
    phases = [
        protocol["development_seeds"],
        gate_protocol["selection_seeds"],
        gate_protocol["approval_seeds"],
        gate_protocol["test_seeds"],
    ]
    assert all(not set(a) & set(b) for i, a in enumerate(phases) for b in phases[i + 1 :])


def test_matched_objectives_share_feedback_and_freeze_empty_runs(actor, tmp_path, monkeypatch):
    cache, root = tmp_path / "cache", tmp_path / "comparison"
    cache.mkdir()
    actor.save(cache / "base.npz")
    monkeypatch.setattr(campaign, "CACHE", cache)
    monkeypatch.setattr(
        campaign, "fetch_actor", lambda _: (actor, {"actor_sha256": actor.digest()})
    )
    seen = []

    def score(policy, condition, seeds, **kwargs):
        seen.extend(seeds)
        return {
            "mean_return": 10,
            "goal_success_rate": 0.5,
            "safe_landing_rate": 0.5,
            "solution_rate": 0.5,
            "per_episode": [{"return": 10}],
            "diagnostics": [{"trace": [{"action": 2}]}],
        }

    monkeypatch.setattr(campaign, "evaluate", score)
    campaign.prepare(root, comparison=True)
    reward = Search(root, "compound", "llm-reward", 0)
    completion = Search(root, "compound", "llm-completion", 0)
    assert reward.protocol == completion.protocol
    assert reward.generation_objective == "reward"
    assert completion.generation_objective == "completion"
    assert "maximum mean undiscounted return" in campaign.editor_prompt(reward)
    assert "maximum goal_success_rate" in campaign.editor_prompt(completion)
    assert (
        campaign.editor_prompt(reward).split("Your search objective")[0]
        == campaign.editor_prompt(completion).split("Your search objective")[0]
    )
    feedback = score(actor, None, [])
    delivered = campaign.editor_feedback(feedback)
    assert "per_episode" not in delivered
    assert delivered["diagnostics"] == feedback["diagnostics"]
    assert "per_episode" in feedback
    assert set(seen) == set(range(50000, 50064))
    with pytest.raises(ValueError, match="outside the frozen"):
        Search(root, "compound", "llm-completion", 3)
    reward.freeze()
    assert json.loads((reward.output / "candidates.json").read_text()) == []
    assert (reward.output / "failed.json").exists()
    assert (reward.output / "frozen.json").exists()
    assert not (root / "evaluation").exists()

    from policy_surgery import acceptance

    approval = tmp_path / "gate"
    acceptance.initialize(root, approval)
    gate_protocol = acceptance.protocol(approval)
    phases = [
        reward.protocol["development_seeds"],
        gate_protocol["selection_seeds"],
        gate_protocol["approval_seeds"],
        gate_protocol["test_seeds"],
    ]
    assert all(not set(a) & set(b) for i, a in enumerate(phases) for b in phases[i + 1 :])


def test_exhausted_inference_budget_preserves_closed_candidate_pool(actor, tmp_path, monkeypatch):
    from types import SimpleNamespace

    cache, root = tmp_path / "cache", tmp_path / "campaign"
    cache.mkdir()
    actor.save(cache / "base.npz")
    monkeypatch.setattr(campaign, "CACHE", cache)
    monkeypatch.setattr(
        campaign, "fetch_actor", lambda _: (actor, {"actor_sha256": actor.digest()})
    )
    monkeypatch.setattr(
        campaign,
        "evaluate",
        lambda *a, **k: {
            "mean_return": 0,
            "goal_success_rate": 0,
            "safe_landing_rate": 0,
            "solution_rate": 0,
            "per_episode": [],
        },
    )
    campaign.prepare(root, comparison=True)

    def exhausted(*args):
        raise RuntimeError("Inference allowance exhausted")

    monkeypatch.setattr(
        campaign,
        "Budget",
        lambda: SimpleNamespace(reserve=exhausted, summary=lambda: {"cap_usd": 1}),
    )
    monkeypatch.setattr(
        campaign,
        "ChatEditor",
        lambda *a, **k: SimpleNamespace(
            request=lambda _: pytest.fail("No dispatch after budget exhaustion")
        ),
    )
    search = Search(root, "compound", "llm-completion", 0)
    result = campaign.llm(search)
    assert result["status"] == "candidate_pool_frozen_requires_approval"
    assert json.loads((search.output / "candidates.json").read_text()) == []
    assert (
        "Inference allowance exhausted"
        in json.loads((search.output / "inference-status.json").read_text())["failure"]
    )
