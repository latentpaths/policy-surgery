"""One entry point for the experiments and offline result reproduction."""

import argparse
import json
import os
import sys
from pathlib import Path

from .io import ROOT


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        protocol_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser(
        "verify", help="Verify frozen records offline; optional simulator replay"
    )
    verify.add_argument("--replay", action="store_true")
    verify.add_argument("--output", type=Path, default=ROOT / "output/verification")
    reproduce = sub.add_parser("reproduce", help="Recompute article results, without API calls")
    reproduce.add_argument("--output", type=Path, default=ROOT / "output/analysis")
    sub.add_parser(
        "weights", help="Fetch pinned source actor once; rebuild exact checkpoint tensors"
    )
    run = sub.add_parser("run", help="Run a fresh frozen 36-run campaign (paid inference)")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-cost", type=float, required=True)
    run.add_argument("--workers", type=int, default=1)
    render = sub.add_parser("render", help="Render all four fixed-seed policy demos")
    render.add_argument("--output", type=Path, default=ROOT / "output/demos")
    figures = sub.add_parser("figures", help="Rebuild the article figures from held-out episodes")
    figures.add_argument("--output", type=Path, default=ROOT / "output/figures")
    sub.add_parser("experiment", help="Low-level frozen protocol operations").add_argument(
        "args", nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    if args.command == "verify":
        from .analysis import verify_records

        result = verify_records()
        if args.replay:
            from .analysis import GATE, SOURCE
            from .verification import verify

            result["replays"] = verify(SOURCE, GATE, args.output)
    elif args.command == "reproduce":
        from .analysis import reproduce

        result = reproduce(args.output)
    elif args.command == "weights":
        from .weights import reconstruct

        result = reconstruct()
    elif args.command == "run":
        if not 0 < args.max_cost <= 20:
            parser.error("--max-cost must be >0 and <=20 USD")
        if not 1 <= args.workers <= 3:
            parser.error("--workers must be 1–3; one minimizes rate limiting")
        output = args.output.resolve()
        try:
            output.relative_to(ROOT / "results")
        except ValueError:
            pass
        else:
            parser.error(
                "Published results are immutable; choose output/ or another fresh directory"
            )
        os.environ["POLICY_SURGERY_MAX_COST"] = str(args.max_cost)
        ledger = (
            Path.home()
            / ".local/share/home/policy-surgery"
            / (
                "run-"
                + __import__("hashlib").sha256(str(output).encode()).hexdigest()[:16]
                + ".sqlite"
            )
        )
        os.environ["POLICY_SURGERY_BUDGET_DB"] = str(ledger)
        from .experiment import main as run

        run(
            [
                "--root",
                str(output / "search"),
                "--gate",
                str(output / "evaluation"),
                "--workers",
                str(args.workers),
            ]
        )
        return
    elif args.command == "render":
        from .analysis import GATE, SOURCE
        from .rendering import render, render_reversal

        result = [
            render(SOURCE, GATE, c, args.output) for c in ("weak_main", "crosswind", "compound")
        ]
        result.append(render_reversal(args.output))
    elif args.command == "figures":
        from .figures import figures

        result = figures(args.output)
    else:
        protocol_main(args.args)
        return
    print(json.dumps(result, indent=2))


def protocol_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "search",
            "evaluate",
            "gate-init",
            "gate-select",
            "gate-approve",
            "gate-test",
            "gate-report",
        ],
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path, default=Path("results/control-completion-search-v1"))
    parser.add_argument("--objective", choices=["completion", "reward"], default="completion")
    parser.add_argument("--condition", default="crosswind")
    parser.add_argument("--method", choices=["llm", "llm-glm", "scripted"], default="llm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    args = parser.parse_args(argv)
    if args.root is None:
        args.root = Path(
            "results/control-completion-v1"
            if args.action.startswith("gate-")
            else "results/control-completion-search-v1"
            if args.objective == "completion"
            else "results/control-v2"
        )
    try:
        from .campaign import Search, final_evaluation, llm, prepare, scripted

        if args.action.startswith("gate-"):
            from . import acceptance

            if args.action == "gate-init":
                result = acceptance.initialize(args.source, args.root)
            elif args.action == "gate-select":
                result = acceptance.select(args.root)
            elif args.action == "gate-approve":
                result = acceptance.approve(args.root)
            elif args.action == "gate-test":
                result = acceptance.test(args.root)
            else:
                result = acceptance.report(args.root)
            if isinstance(result, list):
                result = [
                    {k: r.get(k) for k in ("run", "accepted", "decision_reasons")} for r in result
                ]
            elif "runs" in result and isinstance(result["runs"], list):
                result = [
                    {k: r.get(k) for k in ("run", "accepted", "reason", "reasons")}
                    for r in result["runs"]
                ]
        elif args.action == "prepare":
            result = prepare(args.root, objective=args.objective)
        elif args.action == "search":
            model = args.model or (
                "z-ai/glm-5.3" if args.method == "llm-glm" else "openai/gpt-oss-120b"
            )
            if args.method != "scripted":
                from prime_cli.api.inference import InferenceClient

                client = InferenceClient()
                try:
                    pricing = client.retrieve_model(model)["pricing"]
                finally:
                    client._client.close()
                input_price = (
                    args.input_price
                    if args.input_price is not None
                    else pricing["input_usd_per_mtok"]
                )
                output_price = (
                    args.output_price
                    if args.output_price is not None
                    else pricing["output_usd_per_mtok"]
                )
                if (
                    input_price < pricing["input_usd_per_mtok"]
                    or output_price < pricing["output_usd_per_mtok"]
                ):
                    raise ValueError("Supplied prices understate the current Prime catalog quote")
            search = Search(args.root, args.condition, args.method, args.seed)
            result = (
                scripted(search)
                if args.method == "scripted"
                else llm(search, model, input_price, output_price)
            )
        elif args.action == "evaluate":
            result = final_evaluation(args.root)
        print(json.dumps(result, indent=2))
    except ImportError:
        parser.exit(
            1, "Install policy-surgery[inference,weights] for the controller experiments.\n"
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
