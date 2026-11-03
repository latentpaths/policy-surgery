"""Offline numerical reproduction of the article, independent of credentials and ledgers."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import ROOT, acquisition_entry, write_json

SOURCE = ROOT / "results/control-objective-comparison-v1"
GATE = ROOT / "results/control-objective-comparison-gate-v1"


def read(path):
    return json.loads(Path(path).read_text())


def protocol(path):
    data = read(path)
    unsigned = {k: v for k, v in data.items() if k != "protocol_sha256"}
    if (
        hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        != data["protocol_sha256"]
    ):
        raise ValueError(f"Protocol checksum mismatch: {path}")
    return data


def score(data):
    episodes = data["per_episode"]
    if not episodes:
        raise ValueError("Empty episode set")
    for key, field in [
        ("goal_success_rate", "goal_success"),
        ("mean_return", "return"),
        ("solution_rate", "benchmark_solved"),
        ("safe_landing_rate", "safe_landing"),
        ("crash_rate", "crashed"),
        ("mean_steps", "steps"),
    ]:
        if field not in episodes[0]:
            if key == "goal_success_rate" and key not in data:
                continue  # Older reward-only records did not measure helipad completion.
            raise ValueError(f"Missing episode outcome {field}")
        if not np.isclose(data[key], np.mean([e[field] for e in episodes]), rtol=0, atol=1e-12):
            raise ValueError(f"Aggregate mismatch: {key}")


def interval(before, after):
    delta = np.array(
        [
            int(b["goal_success"]) - int(a["goal_success"])
            for a, b in zip(before["per_episode"], after["per_episode"], strict=True)
        ]
    )
    rng = np.random.default_rng(20261103)
    estimates = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1)
    return {
        "gain": float(delta.mean()),
        "interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
    }


def verify_records():
    """Verify numerical claims and acquisition source without downloading a model."""
    import zipfile

    p = protocol(SOURCE / "protocol.json")
    gp = protocol(GATE / "protocol.json")
    if gp["source_protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Source/gate protocol mismatch")
    with zipfile.ZipFile(ROOT / "results/provenance/acquisition-source.zip") as frozen:
        for name, digest in read(SOURCE / "experiment.json")["source_hashes"].items():
            if hashlib.sha256(frozen.read(acquisition_entry(name))).hexdigest() != digest:
                raise ValueError(f"Acquisition source mismatch: {name}")
    selection = read(GATE / "selection.json")
    approval = read(GATE / "approval.json")
    final = read(GATE / "evaluation.json")
    for item in [selection, approval, final]:
        if item["protocol_sha256"] != gp["protocol_sha256"]:
            raise ValueError("Phase protocol mismatch")
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if approval["selection_sha256"] != sha(GATE / "selection.json"):
        raise ValueError("Selection lock mismatch")
    if read(GATE / "test-lock.json")["approval_sha256"] != sha(GATE / "approval.json"):
        raise ValueError("Test lock mismatch")
    phases = [
        set(p["development_seeds"]),
        *[set(gp[k]) for k in ("selection_seeds", "approval_seeds", "test_seeds")],
    ]
    if any(a & b for i, a in enumerate(phases) for b in phases[i + 1 :]):
        raise ValueError("Overlapping phases")
    expected = {
        f"{c}-{m}-seed{s}"
        for c in p["planned_conditions"]
        for m in p["generation_objectives"]
        for s in p["planned_run_seeds"]
    }
    for phase in [gp, selection, approval, final]:
        if {r["run"] for r in phase["runs"]} != expected:
            raise ValueError("Incomplete run inventory")
    candidates = turns = 0
    for name in sorted(expected):
        folder = SOURCE / "search" / name
        for c in read(folder / "candidates.json"):
            if "development" in c:
                score(c["development"])
                score(c["nominal"])
                candidates += 1
        trace = folder / "transcript.json"
        if trace.exists():
            turns += len(read(trace))
    decisions = {r["run"]: r for r in approval["runs"]}
    for r in final["runs"]:
        score(r["original"])
        score(r["deployed"])
        if len(r["original"]["per_episode"]) != 256:
            raise ValueError("Wrong test sample")
        actual = interval(r["original"], r["deployed"])
        if (
            actual["gain"] != r["goal_difference"]["gain"]
            or actual["interval"] != r["goal_difference"]["interval"]
        ):
            raise ValueError("Final paired interval mismatch")
        if r["accepted"] != decisions[r["run"]]["accepted"]:
            raise ValueError("Deployment decision changed")
    # Exact-reversal result uses the earlier, separately frozen completion campaign.
    reversal = read(ROOT / "results/control-completion-v1/evaluation.json")
    for r in reversal["runs"]:
        if r["condition"] == "mirrored_thrusters":
            score(r["original"])
            score(r["deployed"])
            if (
                sum(e["goal_success"] for e in r["original"]["per_episode"]) != 0
                or sum(e["goal_success"] for e in r["deployed"]["per_episode"]) != 226
            ):
                raise ValueError("Reversal result mismatch")
    return {
        "runs": len(expected),
        "llm_runs": sum("-llm-" in x for x in expected),
        "valid_candidates": candidates,
        "transcript_turns": turns,
        "accepted_llm_policies": sum(r["accepted"] for r in final["runs"]),
        "acquisition_source_verified": True,
        "phase_separation_verified": True,
        "final_intervals_recomputed": True,
        "reversal_successes": 226,
        "reversal_episodes": 256,
    }


def reproduce(output):
    verification = verify_records()
    final = read(GATE / "evaluation.json")["runs"]
    rows = []
    for r in final:
        if not r["accepted"]:
            continue
        before = r["original"]["per_episode"]
        after = r["deployed"]["per_episode"]
        a = [e["goal_success"] for e in before]
        b = [e["goal_success"] for e in after]
        rows.append(
            {
                "run": r["run"],
                "condition": r["condition"],
                "episodes": len(a),
                "base_successes": sum(a),
                "edited_successes": sum(b),
                "gain": r["goal_difference"]["gain"],
                "ci_low": r["goal_difference"]["interval"][0],
                "ci_high": r["goal_difference"]["interval"][1],
                "gained": sum(not x and y for x, y in zip(a, b, strict=True)),
                "lost": sum(x and not y for x, y in zip(a, b, strict=True)),
                "both_success": sum(x and y for x, y in zip(a, b, strict=True)),
                "both_fail": sum(not x and not y for x, y in zip(a, b, strict=True)),
                "actor_sha256": r["deployed_sha256"],
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "verification.json", verification)
    write_json(output / "results.json", rows)
    lines = [
        "# Policy Surgery: reproduced article results",
        "",
        "All accepted LLM policies, 256 paired held-out episodes each. Intervals are descriptive, unadjusted 95% paired bootstrap intervals.",
        "",
        "| Policy | Base → edited successes | Gain, pp [95% interval] |",
        "| --- | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['base_successes']} → {r['edited_successes']} | {r['gain'] * 100:+.1f} [{r['ci_low'] * 100:+.1f}, {r['ci_high'] * 100:+.1f}] |"
        )
    lines += [
        "",
        "18 LLM editing runs belong to a 36-run frozen campaign. Controls and failures remain in the evidence. Approval adjusts across all 72 original comparisons; this analysis does not change that family. Four LLM policies passed approval, but only the completion-guided crosswind policy has a final interval wholly above zero.",
        "",
        "The separate exact-reversal demonstration recovers 226/256 successes from 0/256.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    return {"output": str(output), **verification}
