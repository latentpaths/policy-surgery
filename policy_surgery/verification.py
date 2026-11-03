"""Offline checks of objective-comparison checkpoints, feedback, phases and episodes."""

import hashlib
import json
import zipfile
from dataclasses import asdict

import numpy as np

from policy_surgery import acceptance, campaign
from policy_surgery.policy import Policy, apply_ops
from policy_surgery.world import CONDITIONS, episode

from .io import ROOT, acquisition_entry, write_json


def read(path):
    return json.loads(path.read_text())


def replay(model, condition, seed):
    return {
        k: v for k, v in episode(model, condition, seed).items() if k not in ("trace", "frames")
    }


def aggregates(score, n):
    episodes = score["per_episode"]
    assert len(episodes) == n
    for aggregate, key in (
        ("mean_return", "return"),
        ("goal_success_rate", "goal_success"),
        ("solution_rate", "benchmark_solved"),
        ("safe_landing_rate", "safe_landing"),
        ("crash_rate", "crashed"),
        ("mean_steps", "steps"),
    ):
        assert np.isclose(score[aggregate], np.mean([e[key] for e in episodes]), atol=1e-12)


def verify(root, gate, output=None):
    p = campaign.read_protocol(root)
    experiment = read(root / "experiment.json")
    assert experiment["protocol_sha256"] == p["protocol_sha256"]
    with zipfile.ZipFile(ROOT / "results/provenance/acquisition-source.zip") as acquisition:
        for name, checksum in experiment["source_hashes"].items():
            assert (
                hashlib.sha256(acquisition.read(acquisition_entry(name))).hexdigest() == checksum
            ), name
    gp = acceptance.protocol(gate)
    assert gp["source_protocol_sha256"] == p["protocol_sha256"]
    phases = [p["development_seeds"], gp["selection_seeds"], gp["approval_seeds"], gp["test_seeds"]]
    assert all(not set(a) & set(b) for i, a in enumerate(phases) for b in phases[i + 1 :])
    expected = {
        f"{condition}-{method}-seed{seed}"
        for condition in p["planned_conditions"]
        for method in p["generation_objectives"]
        for seed in p["planned_run_seeds"]
    }
    assert {r["run"] for r in gp["runs"]} == expected
    base = acceptance.base_policy(gp)
    for condition in p["planned_conditions"]:
        for seed in p["planned_run_seeds"]:
            folders = {
                goal: root / "search" / f"{condition}-llm-{goal}-seed{seed}"
                for goal in ("reward", "completion")
            }
            assert (
                read(folders["reward"] / "editor-input.json")["introduction"]
                == read(folders["completion"] / "editor-input.json")["introduction"]
            )
            configs = [read(folders[goal] / "editor.json") for goal in ("reward", "completion")]
            assert {k: v for k, v in configs[0].items() if k != "generation_objective"} == {
                k: v for k, v in configs[1].items() if k != "generation_objective"
            }
            scripts = [
                read(
                    root / "search" / f"{condition}-scripted-{goal}-seed{seed}" / "candidates.json"
                )
                for goal in ("reward", "completion")
            ]
            assert [c["actor_sha256"] for c in scripts[0][:4]] == [
                c["actor_sha256"] for c in scripts[1][:4]
            ]
    candidates = replays = turns = 0
    for name in sorted(expected):
        folder = root / "search" / name
        frozen = read(folder / "frozen.json")
        assert frozen["protocol_sha256"] == p["protocol_sha256"]
        condition = frozen["condition"]
        records = read(folder / "candidates.json")
        assert len(records) <= p["candidate_budget"]
        for record in records:
            if "development" not in record:
                continue
            model = Policy.load(folder / "candidates" / f"{record['id']:02d}.npz")
            assert (
                model.digest() == record["actor_sha256"] == apply_ops(base, record["ops"]).digest()
            )
            aggregates(record["development"], len(phases[0]))
            aggregates(record["nominal"], len(phases[0]))
            # Independent simulator replay of one prechosen development start per checkpoint.
            assert (
                replay(model, CONDITIONS[condition], phases[0][0])
                == record["development"]["per_episode"][0]
            )
            replays += 1
            candidates += 1
        transcript = folder / "transcript.json"
        if transcript.exists():
            records = read(transcript)
            assert len(records) <= p["tool_turn_budget"]
            for record in records:
                if "feedback" in record:
                    assert record["delivered_feedback"] == campaign.editor_feedback(
                        record["feedback"]
                    )
                turns += 1
    approval = read(gate / "approval.json")
    assert approval["selection_sha256"] == acceptance.digest(gate / "selection.json")

    def cached(model, condition, seeds, phase):
        key = hashlib.sha256(
            json.dumps(
                {
                    "actor": model.digest(),
                    "condition": asdict(condition),
                    "seeds": seeds,
                    "metric": "helipad-goal-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        score = read(gate / "measurements" / phase / f"{key}.json")
        assert score["measurement_actor_sha256"] == model.digest()
        aggregates(score, len(seeds))
        return score

    nominal_base = cached(base, CONDITIONS["nominal"], phases[2], "approval")
    alpha = gp["family_alpha"] / gp["family_comparisons"]
    for item in approval["runs"]:
        if item["candidate"] is None:
            assert not item["accepted"]
            continue
        model = Policy.load(gate / "selected" / f"{item['run']}.npz")
        before = cached(base, CONDITIONS[item["condition"]], phases[2], "approval")
        after = cached(model, CONDITIONS[item["condition"]], phases[2], "approval")
        nominal = cached(model, CONDITIONS["nominal"], phases[2], "approval")
        task_delta = acceptance.interval(before, after, alpha, gp["bootstrap_resamples"])
        nominal_delta = acceptance.interval(nominal_base, nominal, alpha, gp["bootstrap_resamples"])
        assert task_delta == item["task_approval"]
        assert nominal_delta == item["nominal_approval"]
        decision = acceptance.decision(
            task_delta, nominal_delta, gp["minimum_observed_gain"], gp["nominal_loss_tolerance"]
        )
        assert decision["accepted"] == item["accepted"]
        assert decision["reasons"] == item["reasons"]
    assert read(gate / "test-lock.json")["approval_sha256"] == acceptance.digest(
        gate / "approval.json"
    )
    final = read(gate / "evaluation.json")
    assert {r["run"] for r in final["runs"]} == expected
    for record in final["runs"]:
        assert record["accepted"] == next(
            r["accepted"] for r in approval["runs"] if r["run"] == record["run"]
        )
        for field in ("original", "deployed"):
            aggregates(record[field], len(phases[3]))
        model = acceptance.deployed_policy(gate, record["run"], record["condition"])
        assert model.digest() == record["deployed_sha256"]
        assert (
            replay(model, CONDITIONS[record["condition"]], phases[3][0])
            == record["deployed"]["per_episode"][0]
        )
        replays += 1
    result = {
        "protocol_sha256": p["protocol_sha256"],
        "gate_protocol_sha256": gp["protocol_sha256"],
        "planned_and_verified_runs": len(expected),
        "verified_candidates": candidates,
        "verified_transcript_turns": turns,
        "independent_simulator_replays": replays,
        "phase_separation": True,
        "checkpoint_edits_match": True,
        "delivered_feedback_matches": True,
        "matched_initial_observations_and_model_settings": True,
        "scripted_initial_proposals_match": True,
        "final_aggregates_and_deployment_match": True,
        "approval_intervals_and_decisions_recomputed": True,
    }
    write_json((output or root) / "verification.json", result)
    return result
