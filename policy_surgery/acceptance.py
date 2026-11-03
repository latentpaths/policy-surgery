"""Completion-first reranking and one-shot approval on independent episode sets.

The existing candidate library is exploratory, not newly generated under this
objective. No old test outcomes are inputs to ranking, approval, or deployment.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .campaign import CACHE, read_protocol
from .io import checkpoint_path, write_json
from .policy import Policy
from .world import CONDITIONS, evaluate

METRIC = "goal_success_rate"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(source: Path, root: Path):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new approval experiment directory")
    previous = read_protocol(source)
    runs = []
    for path in sorted((source / "search").glob("*/frozen.json")):
        frozen = json.loads(path.read_text())
        if frozen["protocol_sha256"] != previous["protocol_sha256"]:
            raise ValueError("Source run protocol mismatch")
        candidates_path = path.parent / "candidates.json"
        candidates = json.loads(candidates_path.read_text())
        pool = []
        for candidate in candidates:
            if "actor_sha256" not in candidate:
                continue
            checkpoint = path.parent / "candidates" / f"{candidate['id']:02d}.npz"
            if Policy.load(checkpoint).digest() != candidate["actor_sha256"]:
                raise ValueError("Source candidate hash mismatch")
            pool.append(
                {
                    "id": candidate["id"],
                    "actor_sha256": candidate["actor_sha256"],
                    "path": str(checkpoint.resolve()),
                }
            )
        runs.append(
            {
                "run": path.parent.name,
                "condition": frozen["condition"],
                "method": frozen["method"],
                "candidates": pool,
                "source_manifest_sha256": digest(path),
            }
        )
    if not runs:
        raise ValueError("No completed source candidate runs")
    seed_base = previous.get("gate_seed_base", 20000)
    protocol = {
        "version": "completion-approval-v1",
        "source_protocol_sha256": previous["protocol_sha256"],
        "base_sha256": previous["source"]["actor_sha256"],
        "runs": runs,
        "success": "No crash; body asleep; both legs in ground contact and within helipad bounds; horizon=1000",
        "selection_seeds": list(range(seed_base, seed_base + 64)),
        "approval_seeds": list(range(seed_base + 1000, seed_base + 1256)),
        "test_seeds": list(range(seed_base + 2000, seed_base + 2256)),
        "minimum_observed_gain": 0.05,
        "nominal_loss_tolerance": 0.05,
        "family_alpha": 0.05,
        "bootstrap_resamples": 20000,
        "family_comparisons": 2 * len(runs),
        "selection": "Filter nominal loss >5pp; highest goal success, then nominal success, then reward; require >=5pp goal gain",
        "approval": "One frozen candidate per run; task gain lower bound>0 and observed gain>=5pp; nominal lower bound>=-5pp",
        "reversal_exception": "Opposite motor mapping requires explicit context; normal operation reverts to original weights",
        "no_candidate_or_rejection": "Deploy original controller; no retry on approval episodes",
        "source_data_note": (
            "Matched reward/completion-guided candidate pools; independent selection/approval/test episodes."
            if previous.get("generation_objectives")
            else "Completion-guided candidate pool; independent selection/approval/test episodes."
            if previous.get("objective") == "completion"
            else "Retrospective candidate pool, prospective selection/approval/test episodes. No new model calls."
        ),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    write_json(root / "protocol.json", protocol)
    return {"runs": len(runs), "candidate_count": sum(len(r["candidates"]) for r in runs)}


def protocol(root):
    return read_protocol(root)


def base_policy(p):
    base = Policy.load(CACHE / "base.npz")
    if base.digest() != p["base_sha256"]:
        raise ValueError("Original controller hash mismatch")
    return base


def score(root, policy, condition, seeds, phase):
    key = hashlib.sha256(
        json.dumps(
            {
                "actor": policy.digest(),
                "condition": asdict(condition),
                "seeds": seeds,
                "metric": "helipad-goal-v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path = root / "measurements" / phase / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = evaluate(policy, condition, seeds)
    result["measurement_actor_sha256"] = policy.digest()
    write_json(path, result)
    return result


def choose(
    pool, original_goal, original_nominal, tolerance=0.05, minimum_gain=0.05, context_specific=False
):
    eligible = [
        c
        for c in pool
        if context_specific or c["nominal"][METRIC] >= original_nominal - tolerance - 1e-12
    ]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda c: (
            c["damaged"][METRIC],
            c["nominal"][METRIC],
            c["damaged"]["mean_return"],
            -c["candidate"]["id"],
        ),
    )
    if best["damaged"][METRIC] - original_goal < minimum_gain - 1e-12:
        return None
    return best


def select(root: Path):
    p = protocol(root)
    if (root / "approval-lock.json").exists() or (root / "test-lock.json").exists():
        raise ValueError("Approval/test data already opened; selection is frozen")
    if (root / "selection.json").exists():
        return json.loads((root / "selection.json").read_text())
    base = base_policy(p)
    seeds = p["selection_seeds"]
    original_nominal = score(root, base, CONDITIONS["nominal"], seeds, "selection")
    selections = []
    for run in p["runs"]:
        condition = CONDITIONS[run["condition"]]
        original = score(root, base, condition, seeds, "selection")
        pool = []
        for candidate in run["candidates"]:
            actor = Policy.load(checkpoint_path(candidate["path"]))
            if actor.digest() != candidate["actor_sha256"]:
                raise ValueError("Candidate changed since preregistration")
            pool.append(
                {
                    "candidate": candidate,
                    "damaged": score(root, actor, condition, seeds, "selection"),
                    "nominal": score(root, actor, CONDITIONS["nominal"], seeds, "selection"),
                }
            )
        best = choose(
            pool,
            original[METRIC],
            original_nominal[METRIC],
            p["nominal_loss_tolerance"],
            p["minimum_observed_gain"],
            condition.mirrored,
        )
        if best:
            actor = Policy.load(checkpoint_path(best["candidate"]["path"]))
            actor.save(root / "selected" / f"{run['run']}.npz")
        selections.append(
            {
                "run": run["run"],
                "condition": condition.name,
                "candidate": best["candidate"] if best else None,
                "goal_gain": best["damaged"][METRIC] - original[METRIC] if best else None,
                "nominal_gain": best["nominal"][METRIC] - original_nominal[METRIC]
                if best
                else None,
                "reason": "completion_first_selection" if best else "no_candidate_passed_selection",
            }
        )
        write_json(root / "selection-progress.json", selections)
    result = {"protocol_sha256": p["protocol_sha256"], "runs": selections}
    write_json(root / "selection.json", result)
    return result


def interval(before, after, alpha, resamples=20000):
    a = np.array([e["goal_success"] for e in before["per_episode"]], dtype=float)
    b = np.array([e["goal_success"] for e in after["per_episode"]], dtype=float)
    if len(a) != len(b) or not len(a):
        raise ValueError("Paired nonempty episodes required")
    delta = b - a
    rng = np.random.default_rng(20261103)
    estimates = delta[rng.integers(0, len(a), (resamples, len(a)))].mean(1)
    return {
        "gain": float(delta.mean()),
        "interval": np.quantile(estimates, [alpha / 2, 1 - alpha / 2]).tolist(),
        "confidence": 1 - alpha,
        "method": "paired-seed percentile bootstrap; approximate",
    }


def decision(task, nominal, minimum_gain, tolerance, context_specific=False):
    reasons = []
    if task["gain"] < minimum_gain - 1e-12 or task["interval"][0] <= 0:
        reasons.append("insufficient_completion_improvement")
    if not context_specific and nominal["interval"][0] < -tolerance - 1e-12:
        reasons.append("nominal_noninferiority_not_established")
    return {"accepted": not reasons, "reasons": reasons}


def approve(root: Path):
    p = protocol(root)
    if (root / "approval.json").exists() or (root / "evaluation.json").exists():
        raise FileExistsError("Approval may be run only once")
    selected_path = root / "selection.json"
    selections = json.loads(selected_path.read_text())
    if selections["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Selection protocol mismatch")
    lock = {"selection_sha256": digest(selected_path), "protocol_sha256": p["protocol_sha256"]}
    lock_path = root / "approval-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Approval inputs changed after the gate opened")
    if not lock_path.exists():
        write_json(lock_path, lock)
    base, seeds = base_policy(p), p["approval_seeds"]
    nominal_base = score(root, base, CONDITIONS["nominal"], seeds, "approval")
    alpha = p["family_alpha"] / p["family_comparisons"]
    decisions = []
    for item in selections["runs"]:
        row = {**item, "accepted": False}
        if item["candidate"] is None:
            row["reasons"] = ["no_candidate_passed_selection"]
        else:
            actor = Policy.load(root / "selected" / f"{item['run']}.npz")
            if actor.digest() != item["candidate"]["actor_sha256"]:
                raise ValueError("Selected weights changed")
            condition = CONDITIONS[item["condition"]]
            original = score(root, base, condition, seeds, "approval")
            tested = score(root, actor, condition, seeds, "approval")
            nominal = score(root, actor, CONDITIONS["nominal"], seeds, "approval")
            task_delta = interval(original, tested, alpha, p["bootstrap_resamples"])
            normal_delta = interval(nominal_base, nominal, alpha, p["bootstrap_resamples"])
            row.update(
                decision(
                    task_delta,
                    normal_delta,
                    p["minimum_observed_gain"],
                    p["nominal_loss_tolerance"],
                    condition.mirrored,
                )
            )
            row.update(
                task_approval=task_delta,
                nominal_approval=normal_delta,
                scope="changed context only; base checkpoint for original mapping"
                if condition.mirrored
                else "changed context with original-physics regression check",
            )
        decisions.append(row)
        write_json(root / "approval-progress.json", decisions)
    result = {
        "protocol_sha256": p["protocol_sha256"],
        "selection_sha256": digest(selected_path),
        "per_bound_alpha": alpha,
        "runs": decisions,
    }
    write_json(root / "approval.json", result)
    return result


def deployed_policy(root: Path, run: str, condition: str):
    p = protocol(root)
    approval = json.loads((root / "approval.json").read_text())
    if approval["protocol_sha256"] != p["protocol_sha256"] or approval[
        "selection_sha256"
    ] != digest(root / "selection.json"):
        raise ValueError("Deployment approval does not match the protocol")
    item = next((r for r in approval["runs"] if r["run"] == run), None)
    if item is None:
        raise ValueError("Run has no approval decision")
    if not item["accepted"] or condition != item["condition"]:
        return base_policy(p)
    actor = Policy.load(root / "selected" / f"{item['run']}.npz")
    if actor.digest() != item["candidate"]["actor_sha256"]:
        raise ValueError("Approved checkpoint changed")
    return actor


def test(root: Path):
    p = protocol(root)
    if (root / "evaluation.json").exists():
        raise FileExistsError("Final test already reported")
    approval = json.loads((root / "approval.json").read_text())
    if approval["protocol_sha256"] != p["protocol_sha256"] or approval[
        "selection_sha256"
    ] != digest(root / "selection.json"):
        raise ValueError("Approval inputs changed")
    lock = {"approval_sha256": digest(root / "approval.json")}
    lock_path = root / "test-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Approval decision changed after test opened")
    if not lock_path.exists():
        write_json(lock_path, lock)
    base, seeds, rows = base_policy(p), p["test_seeds"], []
    for item in approval["runs"]:
        condition = CONDITIONS[item["condition"]]
        original = score(root, base, condition, seeds, "test")
        deployed = deployed_policy(root, item["run"], item["condition"])
        result = score(root, deployed, condition, seeds, "test")
        rows.append(
            {
                "run": item["run"],
                "condition": item["condition"],
                "accepted": item["accepted"],
                "original": original,
                "deployed": result,
                "goal_difference": interval(original, result, 0.05),
                "deployed_sha256": deployed.digest(),
                "decision_reasons": item["reasons"],
            }
        )
    write_json(root / "evaluation.json", {"protocol_sha256": p["protocol_sha256"], "runs": rows})
    return rows


def report(root: Path):
    p = protocol(root)
    retrospective = p.get("source_data_note", "").startswith("Retrospective")
    selections = json.loads((root / "selection.json").read_text())
    approval = json.loads((root / "approval.json").read_text())
    final = json.loads((root / "evaluation.json").read_text())
    lines = [
        "# Completion-first approval experiment",
        "",
        p["source_data_note"],
        "The source candidate experiment and its evidence remain unchanged.",
        "",
        "## Frozen acceptance contract",
        "",
        "Success requires no crash, body asleep, and both grounded legs within the",
        "helipad bounds by 1,000 steps. This is a new goal metric; historical safe-landing",
        "and reward≥200 results have not been redefined. Reward is only a ranking tie-breaker.",
        "",
        f"Selection: {len(p['selection_seeds'])} new paired starts; reject candidates with >5pp nominal loss.",
        f"Approval: {len(p['approval_seeds'])} independent paired starts, >=5pp observed goal gain,",
        "positive task-gain lower bound, and nominal-gain lower bound >=−5pp.",
        (
            f"Bounds use {p['bootstrap_resamples']:,} paired-seed bootstrap resamples at confidence "
            f"{1 - approval['per_bound_alpha']:.2%} per comparison (Bonferroni-style adjustment across"
        ),
        "all registered task and retention comparisons). These are approximate statistical",
        "screens, not formal safety guarantees. One candidate per run reaches approval.",
        "No retries on approval data. Rejection deploys original weights.",
        "",
        "Reversed actuators require the opposite commands. Those edits are accepted only",
        "for that explicit context, and deployment on original physics restores the base",
        "checkpoint. They are a sanity check, not a universal skill-preserving repair.",
        "",
        "## Decisions before final test",
        "",
        "| Run | Selected candidate | Accepted | Reason |",
        "| --- | ---: | --- | --- |",
    ]
    for row in approval["runs"]:
        candidate = row["candidate"]["id"] if row["candidate"] else "—"
        lines.append(
            f"| {row['run']} | {candidate} | {row['accepted']} | {', '.join(row['reasons']) or row['scope']} |"
        )
    lines += [
        "",
        "## Fresh deployment test",
        "",
        f"{len(p['test_seeds'])} fresh paired starts per condition. Test outcomes never affect approval.",
        "",
        "| Run | Original goal completion | Deployed goal completion | Deployed patch |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in final["runs"]:
        lines.append(
            f"| {row['run']} | {row['original'][METRIC]:.1%} | "
            f"{row['deployed'][METRIC]:.1%} | {'approved weights' if row['accepted'] else 'base fallback'} |"
        )
    accepted = sum(r["accepted"] for r in approval["runs"])
    lines += [
        "",
        f"Accepted {accepted}/{len(approval['runs'])} proposed run-level repairs. Rejected edits",
        "are not described as improvements. A rejection can mean insufficient evidence,",
        "not proof that a candidate is harmful. The gate changes which edits are deployed;",
        "it does not create capabilities absent from the candidate pool.",
        "",
        (
            "This is exploratory reranking: candidate generation used the old reward-oriented prompts. "
            "It does not test a completion-trained model or completion-guided candidate generator."
            if retrospective
            else "Candidate generation uses matched reward/completion objectives; editors remain frozen, with no training."
            if p["source_data_note"].startswith("Matched")
            else "Candidate generation uses the completion objective; editors remain frozen, with no training."
        ),
        "The original target is still one pretrained actor, so cross-model transfer is untested.",
        "",
        f"Protocol hash: `{p['protocol_sha256']}`.",
        "Machine-readable selection, approval decisions, frozen weights, phase locks, and",
        "per-episode measurements accompany this report. No source measurement was overwritten.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    return {
        "report": str(root / "REPORT.md"),
        "accepted": accepted,
        "runs": len(selections["runs"]),
        "new_inference_cost_usd": 0,
    }
