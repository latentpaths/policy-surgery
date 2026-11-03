"""Matched reward/completion generation experiment; never overwrites historical evidence."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from policy_surgery import acceptance, campaign

from .io import write_json


def code_hashes():
    from .io import ROOT

    paths = sorted((ROOT / "policy_surgery").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_one(root, condition, method, seed, editor):
    search = campaign.Search(root, condition, method, seed)
    if method.startswith("scripted"):
        result = campaign.scripted(search)
    else:
        result = campaign.llm(search, **editor)
    return {"condition": condition, "method": method, "seed": seed, **result}


def initialize(root):
    from prime_cli.api.inference import InferenceClient

    client = InferenceClient()
    try:
        metadata = client.retrieve_model("openai/gpt-oss-120b")
    finally:
        client._client.close()
    campaign.prepare(root, comparison=True)
    manifest = {
        "date": datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat(),
        "timezone": "Asia/Kolkata",
        "protocol_sha256": campaign.read_protocol(root)["protocol_sha256"],
        "source_hashes": code_hashes(),
        "editor": {
            "model": metadata["id"],
            "input_price": metadata["pricing"]["input_usd_per_mtok"],
            "output_price": metadata["pricing"]["output_usd_per_mtok"],
        },
        "budget_before": campaign.Budget().summary(),
        "primary_question": "Does completion-guided generation yield more approved harder-condition repairs?",
        "analysis": "Report all 36 runs and matched objective contrasts over three search seeds; exploratory, one target. Shared starts are not independent replications.",
        "fixed_limits": "12 proposals and 28 turns per run; existing cumulative $20 ledger cap; no new GPU",
        "selection": "Identical completion selection and approval for both generation objectives",
        "comparison_limits": "Changes objective instructions and adaptive ranking; completion arm additionally targets nominal retention. Not a pure metric-only ablation. Historical results are context only.",
    }
    write_json(root / "experiment.json", manifest)
    print("Frozen objective comparison and current model quote", flush=True)


def report(root, gate):
    p = campaign.read_protocol(root)
    manifest = json.loads((root / "experiment.json").read_text())
    selections = json.loads((gate / "selection.json").read_text())["runs"]
    approvals = json.loads((gate / "approval.json").read_text())["runs"]
    final = json.loads((gate / "evaluation.json").read_text())["runs"]
    selected = {r["run"]: r for r in selections}
    approved = {r["run"]: r for r in approvals}
    rows = []
    for item in final:
        name = item["run"]
        condition, rest = name.split("-", 1)
        method, seed = rest.rsplit("-seed", 1)
        folder = root / "search" / name
        candidates = json.loads((folder / "candidates.json").read_text())
        status_path = folder / "inference-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        rows.append(
            {
                "run": name,
                "condition": condition,
                "method": method,
                "seed": int(seed),
                "candidate_count": len(candidates),
                "valid_candidates": sum("development" in c for c in candidates),
                "failure": status.get("failure"),
                "selected": selected[name]["candidate"]["id"]
                if selected[name]["candidate"]
                else None,
                "selection_gain": selected[name]["goal_gain"],
                "accepted": approved[name]["accepted"],
                "approval_reasons": approved[name]["reasons"],
                "baseline_completion": item["original"]["goal_success_rate"],
                "deployed_completion": item["deployed"]["goal_success_rate"],
            }
        )
    assert len(rows) == 36, "Incomplete comparison"
    ledger_after = campaign.Budget().summary()
    inference_cost = (
        ledger_after["estimated_committed_usd"]
        - manifest["budget_before"]["estimated_committed_usd"]
    )
    lines = [
        "# Does completion-guided generation repair harder landing conditions?",
        "",
        "Matched experiment, 3 November 2026 (Asia/Kolkata). Frozen GPT-OSS-120B edits one pretrained 4,996-parameter actor; no training.",
        "",
        "Reward-guided and completion-guided runs share 64 development starts, edit tools, aggregate/trajectory feedback, 12-proposal and 28-turn caps, and three search seeds. Completion guidance also specifies nominal retention. Scripted search changes adaptive ranking under the same objective contrast.",
        "",
        "All 36 planned searches close before independent selection (64 starts), approval (256), and final testing (256). Both generation objectives use the same completion-based gate. Approval bounds adjust across 72 task/retention comparisons. No approval retries or outcome-driven extra runs.",
        "",
        "## All search and deployment outcomes",
        "",
        "| Condition | Generator / objective | Seed | Proposals | Selected | Approved | Final completions (base → deployed) |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['method']} | {r['seed']} | {r['candidate_count']} | {r['selected'] if r['selected'] is not None else '—'} | {r['accepted']} | {round(r['baseline_completion'] * 256)}/256 → {round(r['deployed_completion'] * 256)}/256 |"
        )
    lines += [
        "",
        "## Matched objective comparisons",
        "",
        "| Condition | Generator | Reward: selected / approved | Completion: selected / approved | Mean deployed gain from completion guidance, pp |",
        "| --- | --- | --- | --- | ---: |",
    ]
    contrasts = []
    for condition in p["planned_conditions"]:
        for generator in ("llm", "scripted"):
            arms = {
                goal: sorted(
                    [
                        r
                        for r in rows
                        if r["condition"] == condition and r["method"] == f"{generator}-{goal}"
                    ],
                    key=lambda r: r["seed"],
                )
                for goal in ("reward", "completion")
            }
            delta = (
                sum(
                    b["deployed_completion"] - a["deployed_completion"]
                    for a, b in zip(arms["reward"], arms["completion"], strict=True)
                )
                / 3
            )
            counts = {
                goal: (sum(r["selected"] is not None for r in arm), sum(r["accepted"] for r in arm))
                for goal, arm in arms.items()
            }
            contrasts.append(
                {
                    "condition": condition,
                    "generator": generator,
                    "mean_deployed_difference": delta,
                    "counts": counts,
                }
            )
            lines.append(
                f"| {condition} | {generator} | {counts['reward'][0]}/3 / {counts['reward'][1]}/3 | {counts['completion'][0]}/3 / {counts['completion'][1]}/3 | {delta * 100:+.2f} |"
            )
    accepted = sum(r["accepted"] for r in rows)
    lines += [
        "",
        f"Approved {accepted}/36 run-level repairs. A rejected run deploys the original checkpoint: a zero deployment effect is fallback behavior, not an estimate of the rejected candidate's effect.",
        "",
        "This tests whether changing generation guidance and ranking helps within the existing edit space and proposal budget. Three search seeds and one target cannot establish a broad model capability or an optimal search method. Provider seed support does not guarantee identical sampling. The historical six-start reward campaign is not a matched comparator.",
        "",
        f"Estimated ledger increase during this experiment: ${inference_cost:.6f}; token-rate estimates include conservative reservations for unknown outcomes, not invoice reconciliation. Cumulative cap remains $20.",
        "",
        "Full candidate scores, delivered prompts/feedback, usage, and failures are retained in search/. The independent gate report and per-episode measurements are in ../control-objective-comparison-gate-v1/. Earlier experiment reports and measurements are unchanged.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    write_json(
        root / "comparison.json",
        {
            "runs": rows,
            "contrasts": contrasts,
            "budget_after": ledger_after,
            "estimated_ledger_increase_usd": inference_cost,
        },
    )
    return {"report": str(root / "REPORT.md"), "accepted": accepted, "runs": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("results/control-objective-comparison-v1")
    )
    parser.add_argument(
        "--gate", type=Path, default=Path("results/control-objective-comparison-gate-v1")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if not (args.root / "experiment.json").exists():
        initialize(args.root)
    manifest = json.loads((args.root / "experiment.json").read_text())
    if manifest["source_hashes"] != code_hashes():
        raise ValueError("Experiment source changed; use a fresh campaign")
    p = campaign.read_protocol(args.root)
    if p["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Experiment protocol changed")
    planned = [
        (condition, method, seed)
        for seed in p["planned_run_seeds"]
        for condition in p["planned_conditions"]
        for method in p["generation_objectives"]
    ]
    missing = [
        r
        for r in planned
        if not (args.root / "search" / f"{r[0]}-{r[1]}-seed{r[2]}" / "frozen.json").exists()
    ]
    if missing and (args.gate / "protocol.json").exists():
        raise ValueError("Gate already registered; cannot add searches")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, args.root, *r, manifest["editor"]) for r in missing]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    if not (args.gate / "protocol.json").exists():
        print(json.dumps(acceptance.initialize(args.root, args.gate)), flush=True)
    for phase, filename in (
        (acceptance.select, "selection.json"),
        (acceptance.approve, "approval.json"),
        (acceptance.test, "evaluation.json"),
    ):
        if not (args.gate / filename).exists():
            print(f"Starting {phase.__name__}", flush=True)
            phase(args.gate)
            print(f"Completed {phase.__name__}", flush=True)
    acceptance.report(args.gate)
    print(json.dumps(report(args.root, args.gate)), flush=True)
