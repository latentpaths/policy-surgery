"""Budgeted frozen-editor searches; validation selection and test execution are separate."""

import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .inference import ChatEditor
from .io import ROOT, write_json
from .policy import SHAPES, Policy, apply_ops, fetch_actor
from .world import CONDITIONS, evaluate

CACHE = Path(os.environ.get("POLICY_SURGERY_WEIGHTS", ROOT / "output/weights"))
LEDGER = Path.home() / ".local/share/home/policy-surgery/control-budget.sqlite"
FEATURES = [
    "x relative to pad",
    "height above pad",
    "horizontal velocity",
    "vertical velocity",
    "body angle",
    "angular velocity",
    "left leg contact",
    "right leg contact",
]
EDITOR_PROMPT = """You operate a pretrained lunar-landing robot and can edit its neural controller.
No training, gradient descent, new modules, or inference-time Python wrapper is allowed.
The policy is float32: h1=tanh(w1@obs+b1), h2=tanh(w2@h1+b2),
logits=w3@h2+b3, action=argmax(logits). Shapes: w1[64,8], w2[64,64], w3[4,64].
obs=[x relative to pad,height,horizontal velocity,vertical velocity,body angle,
angular velocity,left leg contact,right leg contact]. Actions 0=off,1=left side
engine,2=main engine,3=right side engine. Target has no observation normalization.
Physics changes are described in the task. Examine failed trajectories and reason
about braking, attitude, and thrust timing. You are allowed 12 candidate evaluations
on development episodes and 28 total tool turns. Every candidate starts from ORIGINAL
weights, NOT the previous candidate. A hypothesis must accompany each candidate.
Reply with exactly one JSON object: {"tool": NAME, "args": {...}}.
Tools:
inspect: args {"tensor":"w1", "rows":[0,1]} (up to 8 rows; includes values/norms).
evaluate: args {"name":"short name", "hypothesis":"reason for the change", "ops":[...]}
  Supported weight operations:
  {"op":"input_affine","scale":[8 numbers],"offset":[8 numbers]} compiles a change
  of sensor coordinates into w1 and b1. Missing scale defaults to 1; offset to 0.
  {"op":"output_affine","scale":[4 numbers],"offset":[4 numbers],
   "permutation":[0,1,2,3]} changes action-logit weights/biases.
  {"op":"hidden_scale","layer":1 or 2,"scale":[64 numbers]} changes hidden units.
  {"op":"set","edits":[{"tensor":"b3","index":[2],"value":0.2}, ...]}
  does arbitrary scalar assignments. All values must be finite and magnitude<=1000.
finish: args {} ends search. The evaluator will select among the top three
development candidates using a separate validation set. You cannot see validation
or test outcomes. Do not overfit a single starting state. Strong original-domain
performance is desirable; nominal performance is reported separately.
"""


def read_protocol(root: Path):
    protocol = json.loads((root / "protocol.json").read_text())
    expected = protocol["protocol_sha256"]
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError("Protocol content does not match its frozen hash")
    return protocol


class Budget:
    """Durable reservations; unknown outcomes remain charged at their upper bound."""

    def __init__(self, path=None, cap=None):
        path = path or Path(os.environ.get("POLICY_SURGERY_BUDGET_DB", LEDGER))
        cap = float(os.environ.get("POLICY_SURGERY_MAX_COST", "20")) if cap is None else cap
        self.path, self.cap = path, cap
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, run TEXT, "
                "reserved REAL, actual REAL, created REAL)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, run, amount):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid reservation")
        key = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute(
                "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests"
            ).fetchone()[0]
            if used + amount > self.cap:
                raise RuntimeError(f"Inference allowance exhausted (USD {self.cap:g})")
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,NULL,?)", (key, run, amount, time.time())
            )
        return key

    def settle(self, key, actual):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,actual FROM requests WHERE id=?", (key,)).fetchone()
            if row is None or row[1] is not None or not math.isfinite(actual) or actual < 0:
                raise ValueError("Invalid settlement")
            db.execute("UPDATE requests SET actual=? WHERE id=?", (actual, key))
        if actual > row[0] + 1e-9:
            raise RuntimeError(
                "Actual token cost exceeded reservation; cost recorded, dispatch stopped"
            )

    def summary(self):
        with self.connect() as db:
            used, pending, calls = db.execute(
                "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0), "
                "COALESCE(SUM(actual IS NULL),0),COUNT(*) FROM requests"
            ).fetchone()
        return {
            "cap_usd": self.cap,
            "estimated_committed_usd": used,
            "pending_requests": pending,
            "requests": calls,
        }


def prepare(root: Path, objective="completion", *, comparison=False):
    if objective not in ("completion", "reward"):
        raise ValueError("Choose completion or legacy reward objective")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new campaign directory")
    actor, source = fetch_actor(CACHE)
    protocol = {
        "version": "self-patch-box2d-v1",
        "source": source,
        "gymnasium": "1.4.0",
        "environment": "LunarLander-v3",
        "conditions": {k: asdict(v) for k, v in CONDITIONS.items()},
        "development_seeds": list(range(6)),
        "validation_seeds": list(range(100, 112)),
        "test_seeds": list(range(1000, 1064)),
        "candidate_budget": 12,
        "tool_turn_budget": 28,
        "validation_selection": "top 3 development mean returns; best validation mean return",
        "primary_metric": "mean undiscounted return",
        "secondary_metrics": ["reward>=200", "safe landing", "crash", "nominal retention"],
        "training": "none: frozen off-the-shelf editor and pretrained target",
    }
    if objective == "completion":
        protocol.update(
            version="self-patch-completion-search-v1",
            objective="completion",
            development_seeds=list(range(30000, 30064)),
            validation_seeds=list(range(41000, 41256)),
            test_seeds=list(range(42000, 42256)),
            gate_seed_base=40000,
            primary_metric="goal_success_rate",
            minimum_observed_gain=0.05,
            nominal_loss_tolerance=0.05,
            validation_selection="Independent completion approval gate required; no direct deployment",
        )
    if comparison:
        if objective != "completion":
            raise ValueError("Objective comparison requires the independent completion gate")
        protocol.update(
            version="self-patch-objective-comparison-v1",
            development_seeds=list(range(50000, 50064)),
            validation_seeds=list(range(61000, 61256)),
            test_seeds=list(range(62000, 62256)),
            gate_seed_base=60000,
            generation_objectives={
                f"{method}-{goal}": goal
                for method in ("llm", "scripted")
                for goal in ("reward", "completion")
            },
            planned_conditions=["weak_main", "crosswind", "compound"],
            planned_run_seeds=[0, 1, 2],
            comparison="Change generation objective only; shared completion selection and approval",
            feedback="Aggregate task/nominal outcomes and two development trajectories; no episode lists",
        )
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    write_json(root / "protocol.json", protocol)
    baseline = {
        name: evaluate(actor, condition, protocol["development_seeds"], traces=True)
        for name, condition in CONDITIONS.items()
    }
    write_json(root / "development-baseline.json", baseline)
    return {
        k: {
            "return": v["mean_return"],
            "solution_rate": v["solution_rate"],
            "safe_landing_rate": v["safe_landing_rate"],
        }
        for k, v in baseline.items()
    }


class Search:
    def __init__(self, root: Path, condition: str, method: str, seed: int):
        if (root / "evaluation").exists():
            raise ValueError("This campaign has opened test data; start a new protocol")
        if condition not in CONDITIONS or condition == "nominal":
            raise ValueError("Choose a damaged condition")
        self.root, self.condition = root, CONDITIONS[condition]
        self.protocol = read_protocol(root)
        self.generation_objective = self.protocol.get("generation_objectives", {}).get(
            method, self.protocol.get("objective", "reward")
        )
        if self.protocol.get("generation_objectives") and (
            method not in self.protocol["generation_objectives"]
            or condition not in self.protocol["planned_conditions"]
            or seed not in self.protocol["planned_run_seeds"]
        ):
            raise ValueError("Run is outside the frozen objective-comparison plan")
        self.base = Policy.load(CACHE / "base.npz")
        if self.base.digest() != self.protocol["source"]["actor_sha256"]:
            raise ValueError("Base checkpoint hash mismatch")
        self.output = root / "search" / f"{condition}-{method}-seed{seed}"
        if self.output.exists():
            raise FileExistsError("Search runs are immutable; choose a fresh run/seed")
        self.output.mkdir(parents=True)
        self.method, self.seed, self.candidates = method, seed, []
        write_json(
            self.output / "condition.json",
            {
                "condition": condition,
                "method": method,
                "seed": seed,
                "protocol_sha256": self.protocol["protocol_sha256"],
                "generation_objective": self.generation_objective,
            },
        )

    def inspect(self, args):
        key, rows = args["tensor"], args.get("rows", [0])
        if (
            key not in SHAPES
            or len(rows) > 8
            or any(type(i) is not int or not 0 <= i < SHAPES[key][0] for i in rows)
        ):
            raise ValueError("Invalid tensor/row request")
        tensor = self.base.tensors[key]
        return {
            "tensor": key,
            "shape": list(tensor.shape),
            "values": tensor[rows].tolist(),
            "norm": float(np.linalg.norm(tensor)),
        }

    def candidate(self, args):
        if len(self.candidates) >= self.protocol["candidate_budget"]:
            raise ValueError("Candidate budget exhausted")
        index = len(self.candidates)
        # Invalid proposals consume a candidate slot, preserving a fixed search allowance.
        record = {
            "id": index,
            "name": str(args.get("name", f"candidate-{index}"))[:120],
            "hypothesis": str(args.get("hypothesis", ""))[:3000],
            "ops": args.get("ops", []),
        }
        self.candidates.append(record)
        try:
            model = apply_ops(self.base, record["ops"])
            started = time.perf_counter()
            score = evaluate(model, self.condition, self.protocol["development_seeds"], traces=True)
            record.update(
                development=score,
                changed_parameters=self.base.changed(model),
                actor_sha256=model.digest(),
                seconds=time.perf_counter() - started,
            )
            if self.protocol.get("objective") == "completion":
                record["nominal"] = evaluate(
                    model, CONDITIONS["nominal"], self.protocol["development_seeds"]
                )
            model.save(self.output / "candidates" / f"{index:02d}.npz")
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            record["error"] = str(exc)
        write_json(self.output / "candidates.json", self.candidates)
        return record

    def freeze(self):
        available = [c for c in self.candidates if "development" in c]
        if not available:
            write_json(
                self.output / "failed.json", {"reason": "No valid candidate", "method": self.method}
            )
            if self.protocol.get("objective") != "completion":
                return {"status": "no_valid_candidate"}
        if self.protocol.get("objective") == "completion":
            # Close the proposal pool without opening any approval or test episodes.
            write_json(self.output / "candidates.json", self.candidates)
            write_json(
                self.output / "frozen.json",
                {
                    "condition": self.condition.name,
                    "method": self.method,
                    "seed": self.seed,
                    "protocol_sha256": self.protocol["protocol_sha256"],
                    "requires_approval": True,
                    "base_sha256": self.base.digest(),
                    "candidate_count": len(self.candidates),
                    "training": "none",
                },
            )
            return {"status": "candidate_pool_frozen_requires_approval", "path": str(self.output)}
        top = sorted(available, key=lambda c: c["development"]["mean_return"], reverse=True)[:3]
        validations = []
        for candidate in top:
            model = Policy.load(self.output / "candidates" / f"{candidate['id']:02d}.npz")
            score = evaluate(model, self.condition, self.protocol["validation_seeds"])
            validations.append({"candidate_id": candidate["id"], "score": score})
        best = max(validations, key=lambda c: c["score"]["mean_return"])
        selected = next(c for c in available if c["id"] == best["candidate_id"])
        model = Policy.load(self.output / "candidates" / f"{selected['id']:02d}.npz")
        model.save(self.output / "frozen.npz")
        frozen = {
            "condition": self.condition.name,
            "method": self.method,
            "seed": self.seed,
            "protocol_sha256": self.protocol["protocol_sha256"],
            "selected": selected,
            "validation": validations,
            "actor_sha256": model.digest(),
            "base_sha256": self.base.digest(),
            "training": "none",
            "candidate_count": len(self.candidates),
        }
        write_json(self.output / "frozen.json", frozen)
        return {
            "path": str(self.output),
            "selected": selected["id"],
            "development_return": selected["development"]["mean_return"],
            "validation_return": best["score"]["mean_return"],
        }

    def rank(self, candidate):
        if self.generation_objective == "completion":
            baseline = json.loads((self.root / "development-baseline.json").read_text())
            retention = (
                self.condition.mirrored
                or candidate["nominal"]["goal_success_rate"]
                >= baseline["nominal"]["goal_success_rate"]
                - self.protocol["nominal_loss_tolerance"]
                - 1e-12
            )
            return (
                retention,
                candidate["development"]["goal_success_rate"],
                candidate["nominal"]["goal_success_rate"],
                candidate["development"]["mean_return"],
            )
        return (candidate["development"]["mean_return"],)


def scripted(search: Search):
    rng = np.random.default_rng(search.seed)
    count = search.protocol["candidate_budget"]
    # Includes a known physics correction for the explicit sanity case. Other
    # proposals explore observation scaling and action thresholds without gradients.
    for index in range(count):
        if search.condition.mirrored and index == 0:
            ops = [{"op": "output_affine", "permutation": [0, 3, 2, 1]}]
        else:
            if index >= 4:
                best = max(
                    (c for c in search.candidates if "development" in c),
                    key=search.rank,
                )
                previous = best["ops"]
                gains = (
                    np.array(previous[0]["scale"])
                    if previous[0]["op"] == "input_affine"
                    else np.ones(8)
                )
                bias = np.array(previous[-1].get("offset", [0] * 4), dtype=float)
            else:
                gains, bias = np.ones(8), np.zeros(4)
            gains *= np.exp(rng.normal(0, 0.35, 8))
            gains[6:] = 1
            bias += rng.normal(0, 0.45, 4)
            ops = [
                {"op": "input_affine", "scale": gains.tolist()},
                {"op": "output_affine", "offset": bias.tolist()},
            ]
        search.candidate(
            {
                "name": f"scripted-{index}",
                "ops": ops,
                "hypothesis": "Budget-matched derivative-free input-gain/logit-bias search",
            }
        )
    return search.freeze()


def editor_prompt(search):
    """Only objective instructions differ in the matched generation comparison."""
    if search.protocol.get("generation_objectives"):
        common = (
            EDITOR_PROMPT.split("finish: args")[0]
            + """finish: args {} closes the candidate pool.
Every candidate is evaluated on the same 64 development starts under changed and
original physics. You may use up to 12 proposals and 28 turns. Completion means
no crash, body asleep, both grounded legs inside the helipad by 1000 steps.
No selection, approval, or test outcomes are available. You cannot deploy edits.
"""
        )
        if search.generation_objective == "reward":
            return (
                common
                + "Your search objective is maximum mean undiscounted return under changed physics.\n"
            )
        return (
            common
            + """Your search objective is maximum goal_success_rate under changed physics,
while losing no more than 5 percentage points of original-physics completion.
Aim for at least 5 percentage points completion gain. Reward is only a tie-breaker.
"""
        )
    if search.generation_objective != "completion":
        return EDITOR_PROMPT
    return (
        EDITOR_PROMPT.split("finish: args")[0]
        + """finish: args {} closes the candidate pool.
Your objective is goal_success_rate: no crash, body asleep, and both grounded legs
inside the helipad by 1000 steps. Reward is only a tie-breaker. Aim for >=5 percentage
points completion gain while losing <=5 points nominal completion. Nominal scores
are included with every candidate. Reversed motors are a context-specific exception;
normal operation restores original weights. One selected candidate must independently
pass a completion/retention approval gate before deployment. No approval/test outcomes
are available to you. You cannot deploy a candidate directly.
"""
    )


def editor_feedback(record):
    """Bound context growth while preserving summaries and trajectory evidence on disk."""
    if isinstance(record, dict):
        return {k: editor_feedback(v) for k, v in record.items() if k != "per_episode"}
    if isinstance(record, list):
        return [editor_feedback(v) for v in record]
    return record


def llm(search: Search, model="openai/gpt-oss-120b", input_price=0.024, output_price=0.088):
    transport = ChatEditor(
        model,
        input_price=input_price,
        output_price=output_price,
        max_tokens=8192,
        max_cost_usd=20,
        request_timeout=180,
    )
    baseline = json.loads((search.root / "development-baseline.json").read_text())[
        search.condition.name
    ]
    prompt = editor_prompt(search)
    comparison = bool(search.protocol.get("generation_objectives"))
    introduction = {
        "physics": asdict(search.condition),
        "original_policy": baseline,
        "parameter_count": 4996,
        "feature_names": FEATURES,
    }
    if comparison:
        introduction["nominal_policy"] = json.loads(
            (search.root / "development-baseline.json").read_text()
        )["nominal"]
        introduction = editor_feedback(introduction)
    write_json(
        search.output / "editor-input.json", {"system": prompt, "introduction": introduction}
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(introduction),
        },
    ]
    ledger, transcript = Budget(), []
    write_json(
        search.output / "editor.json",
        {
            "model": model,
            "input_price_usd_per_mtok": input_price,
            "output_price_usd_per_mtok": output_price,
            "max_tokens": 8192,
            "reasoning_effort": "high",
            "temperature": 0.3,
            "seed": search.seed,
            "provider_seed_sent": model.startswith("openai/gpt-oss"),
            "generation_objective": search.generation_objective,
        },
    )
    failure = None
    for turn in range(search.protocol["tool_turn_budget"]):
        if len(search.candidates) >= search.protocol["candidate_budget"]:
            break
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.3,
            "seed": search.seed,
            "reasoning_effort": "high",
        }
        if model.startswith("openai/gpt-oss"):
            payload["response_format"] = {"type": "json_object"}
        else:
            payload.pop("seed")
        upper = (
            (len(json.dumps(messages).encode()) + 8192) * input_price + 8192 * output_price
        ) / 1e6
        try:
            reservation = ledger.reserve(str(search.output), upper)
            reply = transport.request(payload)
            usage = reply.get("usage", {})
            if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                raise RuntimeError("Missing usage; upper-bound cost remains reserved")
            cost = (
                usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price
            ) / 1e6
            ledger.settle(reservation, cost)
            content = reply["choices"][0]["message"]["content"]
            transcript.append({"turn": turn, "content": content, "usage": usage})
        except (RuntimeError, ValueError, KeyError, IndexError, OSError) as exc:
            failure = f"provider_or_budget_error:{type(exc).__name__}:{str(exc)[:200]}"
            break
        try:
            action = json.loads(content)
            json.dumps(action, allow_nan=False)
            if action["tool"] == "finish":
                break
            if action["tool"] == "inspect":
                feedback = search.inspect(action["args"])
            elif action["tool"] == "evaluate":
                feedback = search.candidate(action["args"])
            else:
                raise ValueError("Unknown tool")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            feedback = {"error": str(exc), "instruction": "Return a valid JSON tool action"}
        transcript[-1]["feedback"] = feedback
        if comparison:
            feedback = editor_feedback(feedback)
        transcript[-1]["delivered_feedback"] = feedback
        write_json(search.output / "transcript.json", transcript)
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {"role": "user", "content": json.dumps(feedback)},
            ]
        )
    write_json(search.output / "transcript.json", transcript)
    write_json(
        search.output / "inference-status.json",
        {"failure": failure, "turns": len(transcript), "budget": ledger.summary()},
    )
    return search.freeze()


def paired(before, after, key="return"):
    a = np.array([e[key] for e in before["per_episode"]], dtype=float)
    b = np.array([e[key] for e in after["per_episode"]], dtype=float)
    if len(a) != len(b):
        raise ValueError("Unpaired episodes")
    difference = b - a
    rng = np.random.default_rng(20261103)
    bootstrap = difference[rng.integers(0, len(a), (2000, len(a)))].mean(1)
    return {
        "mean_difference": float(difference.mean()),
        "paired_seed_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def final_evaluation(root: Path):
    destination = root / "evaluation"
    if destination.exists():
        raise FileExistsError("Final evaluations are immutable")
    protocol = read_protocol(root)
    if protocol.get("objective") == "completion":
        raise ValueError(
            "Completion searches require gate-init/select/approve/test before deployment"
        )
    unfinished = [
        p.name
        for p in (root / "search").iterdir()
        if p.is_dir() and not (p / "frozen.json").exists() and not (p / "failed.json").exists()
    ]
    if unfinished:
        raise RuntimeError(f"Freeze or record failures before opening test data: {unfinished}")
    frozen_paths = sorted((root / "search").glob("*/frozen.json"))
    write_json(
        destination / "frozen-inventory.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "runs": [
                {
                    "run": p.parent.name,
                    "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in frozen_paths
            ],
        },
    )
    actor = Policy.load(CACHE / "base.npz")
    if actor.digest() != protocol["source"]["actor_sha256"]:
        raise ValueError("Base hash mismatch")
    seeds = protocol["test_seeds"]
    baseline = {k: evaluate(actor, condition, seeds) for k, condition in CONDITIONS.items()}
    heuristic = {
        k: evaluate(actor, condition, seeds, heuristic=True)
        for k, condition in CONDITIONS.items()
        if k not in ("nominal", "mirrored_thrusters")
    }
    write_json(destination / "unchanged.json", baseline)
    write_json(destination / "heuristic.json", heuristic)
    rows = []
    for path in frozen_paths:
        frozen = json.loads(path.read_text())
        if frozen["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Protocol changed after patch selection")
        model = Policy.load(path.parent / "frozen.npz")
        if model.digest() != frozen["actor_sha256"]:
            raise ValueError("Frozen patch hash mismatch")
        condition = frozen["condition"]
        damaged = evaluate(model, CONDITIONS[condition], seeds)
        nominal = evaluate(model, CONDITIONS["nominal"], seeds)
        record = {
            "run": path.parent.name,
            "condition": condition,
            "method": frozen["method"],
            "editor_seed": frozen["seed"],
            "actor_sha256": model.digest(),
            "changed_parameters": actor.changed(model),
            "damaged": damaged,
            "nominal": nominal,
            "return_difference": paired(baseline[condition], damaged),
            "solution_difference": paired(baseline[condition], damaged, "benchmark_solved"),
            "nominal_difference": paired(baseline["nominal"], nominal),
            "deployment": "Only frozen NPZ weights loaded; no editor/context/wrapper",
            "candidate_count": frozen["candidate_count"],
        }
        write_json(destination / f"{path.parent.name}.json", record)
        rows.append(record)
    write_json(
        destination / "summary.json",
        {"protocol": protocol, "runs": rows, "training": "none", "budget": Budget().summary()},
    )
    return [
        {
            "run": r["run"],
            "return": r["damaged"]["mean_return"],
            "solution_rate": r["damaged"]["solution_rate"],
            "delta": r["return_difference"]["mean_difference"],
        }
        for r in rows
    ]
