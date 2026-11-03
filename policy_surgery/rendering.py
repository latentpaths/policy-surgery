"""Fixed-seed diagnostic videos: base versus reward/completion-guided proposals."""

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from policy_surgery import acceptance, campaign
from policy_surgery.policy import Policy
from policy_surgery.world import CONDITIONS, episode

from .io import write_json


def render(root, gate, condition, destination=None):
    protocol = campaign.read_protocol(root)
    gp = acceptance.protocol(gate)
    approvals = json.loads((gate / "approval.json").read_text())["runs"]
    base = acceptance.base_policy(gp)
    policies, details = [base], [{"label": "Original controller", "status": "No patch"}]
    for goal in ("reward", "completion"):
        search = root / "search" / f"{condition}-llm-{goal}-seed0"
        candidates = json.loads((search / "candidates.json").read_text())
        valid = [c for c in candidates if "development" in c]
        baseline = json.loads((root / "development-baseline.json").read_text())

        def rank(c, goal=goal, baseline=baseline):
            if goal == "reward":
                return (c["development"]["mean_return"],)
            return (
                c["nominal"]["goal_success_rate"]
                >= baseline["nominal"]["goal_success_rate"] - 0.05 - 1e-12,
                c["development"]["goal_success_rate"],
                c["nominal"]["goal_success_rate"],
                c["development"]["mean_return"],
                -c["id"],
            )

        chosen = max(valid, key=rank) if valid else None
        approval = next(r for r in approvals if r["run"] == search.name)
        accepted = bool(
            chosen
            and approval["accepted"]
            and approval["candidate"]["actor_sha256"] == chosen["actor_sha256"]
        )
        policies.append(
            Policy.load(search / "candidates" / f"{chosen['id']:02d}.npz") if chosen else base
        )
        details.append(
            {
                "label": f"{goal.title()}-guided proposal",
                "run": search.name,
                "candidate": chosen["id"] if chosen else None,
                "status": "APPROVED"
                if accepted
                else "UNAPPROVED / diagnostic only"
                if chosen
                else "No valid proposal; original",
                "rule": "Search seed 0; best development score under its generation objective",
            }
        )
    seed = gp["test_seeds"][0]
    episodes = [episode(model, CONDITIONS[condition], seed, render=True) for model in policies]
    output = (Path(destination) if destination else root / "videos") / condition
    if output.exists():
        raise FileExistsError("Video exists; preserve it")
    output.mkdir(parents=True)
    width, height = 1920, 800
    font = lambda size: ImageFont.load_default(size=size)
    colors = ["#c6cbd6", "#f0c371", "#76d9bb"]

    def frame(index):
        canvas = Image.new("RGB", (width, height), "#101725")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (32, 20), f"Weight editing: {condition.replace('_', ' ')}", font=font(40), fill="white"
        )
        draw.text(
            (32, 78),
            "Same starting state. Frozen GPT-OSS-120B. Direct controller-weight edits.",
            font=font(25),
            fill="#b7c2d0",
        )
        for i, (ep, detail) in enumerate(zip(episodes, details, strict=True)):
            x = 20 + 640 * i
            draw.text((x + 10, 137), detail["label"], font=font(28), fill=colors[i])
            picture = Image.fromarray(ep["frames"][min(index, len(ep["frames"]) - 1)])
            canvas.paste(picture.resize((600, 400)), (x, 185))
            draw.text((x + 10, 603), detail["status"], font=font(21), fill=colors[i])
            if index >= len(ep["frames"]) - 1:
                draw.text(
                    (x + 10, 645),
                    "HELIPAD COMPLETE" if ep["goal_success"] else "HELIPAD INCOMPLETE",
                    font=font(25),
                    fill=colors[i],
                )
                draw.text(
                    (x + 10, 683),
                    f"Episode reward: {ep['return']:.1f}",
                    font=font(23),
                    fill="white",
                )
        draw.text(
            (32, 740),
            f"Fixed seed {seed}; search seed 0; proposals selected on development data only.",
            font=font(23),
            fill="#b7c2d0",
        )
        draw.text(
            (32, 772),
            "Illustration only. Unapproved proposals are not deployed. Aggregate results: REPORT.md.",
            font=font(19),
            fill="#b7c2d0",
        )
        return canvas

    count = max(len(e["frames"]) for e in episodes)
    frame(count - 1).save(output / "poster.png")
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "replay.mp4"),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for index in [*range(count), *([count - 1] * 50)]:
            process.stdin.write(frame(index).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(error[:300])
    write_json(
        output / "replay.json",
        {
            "source_protocol_sha256": protocol["protocol_sha256"],
            "test_seed": seed,
            "condition": condition,
            "details": details,
            "actor_hashes": [p.digest() for p in policies],
            "episodes": [{k: v for k, v in e.items() if k != "frames"} for e in episodes],
        },
    )
    return str(output / "replay.mp4")


def render_reversal(destination):
    from .campaign import CACHE
    from .io import ROOT

    results = ROOT / "results"
    base = Policy.load(CACHE / "base.npz")
    policy = Policy.load(results / "control-v2/search/mirrored_thrusters-llm-seed0/frozen.npz")
    models = [base, policy]
    episodes = [episode(m, CONDITIONS["mirrored_thrusters"], 1000, render=True) for m in models]
    out = Path(destination) / "reversal"
    out.mkdir(parents=True, exist_ok=False)
    width, height = 1280, 720
    font = lambda n: ImageFont.load_default(size=n)

    def frame(index):
        canvas = Image.new("RGB", (width, height), "#101725")
        d = ImageDraw.Draw(canvas)
        d.text(
            (32, 22),
            "Reversed thrusters: original vs LLM-edited policy",
            font=font(35),
            fill="#f3f6f8",
        )
        d.text(
            (32, 77),
            "GPT-OSS-120B swaps the side-engine output rows and biases.",
            font=font(23),
            fill="#b7c2d0",
        )
        for i, (ep, label, color) in enumerate(
            zip(
                episodes,
                ["Original policy", "LLM-edited policy"],
                ["#abb8c4", "#71d2b5"],
                strict=True,
            )
        ):
            x = 24 + 640 * i
            d.text((x, 130), label, font=font(28), fill=color)
            img = Image.fromarray(ep["frames"][min(index, len(ep["frames"]) - 1)])
            canvas.paste(img.resize((600, 400)), (x, 180))
            if index >= len(ep["frames"]) - 1:
                d.text(
                    (x, 600),
                    "Successful landing" if ep["goal_success"] else "Landing failed",
                    font=font(25),
                    fill=color,
                )
        d.text(
            (32, 657),
            "Fixed replay seed 1000. Aggregate success uses 256 separate episodes.",
            font=font(21),
            fill="#b7c2d0",
        )
        return canvas

    count = max(len(e["frames"]) for e in episodes)
    frame(count - 1).save(out / "reversal-llm.png")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "reversal-llm.mp4"),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for i in [*range(count), *([count - 1] * 50)]:
            proc.stdin.write(frame(i).tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode()
        if proc.wait():
            raise RuntimeError(error[:300])
    (out / "reversal-llm.json").write_text(
        json.dumps(
            {
                "seed": 1000,
                "condition": "mirrored_thrusters",
                "actor_hashes": [m.digest() for m in models],
                "episodes": [{k: v for k, v in e.items() if k != "frames"} for e in episodes],
            },
            indent=2,
        )
        + "\n"
    )
    return str(out / "reversal-llm.mp4")
