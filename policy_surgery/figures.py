"""Rebuild compact article figures from the retained episode measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK = "#292d25"
MUTED = "#706d60"
GREEN = "#284d3c"
RED = "#864b46"
RULE = "#d5cebc"
BG = "#f7f3e9"


def save(fig, out, name):
    for ext in ("png", "svg"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=220, facecolor=BG)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def figures(output):
    from .io import ROOT

    rows = [
        r
        for r in json.loads(
            (ROOT / "results/control-objective-comparison-gate-v1/evaluation.json").read_text()
        )["runs"]
        if r["accepted"]
    ]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "svg.fonttype": "none",
        }
    )
    labels = [
        ("Crosswind", "Success / 0"),
        ("Crosswind", "Return / 1"),
        ("Crosswind", "Return / 2"),
        ("Reduced thrust", "Success / 1"),
    ]
    fig = plt.figure(figsize=(10, 3.65), facecolor=BG)
    fig.text(0.04, 0.90, "Four accepted edits. One clear test gain.", fontsize=19)
    fig.text(0.04, 0.82, "Landing success · 256 paired test episodes per edit", color=MUTED)
    for x, label in [(0.04, "DYNAMICS"), (0.235, "OBJECTIVE / SEED"), (0.45, "BEFORE → AFTER")]:
        fig.text(x, 0.71, label, fontsize=8, color=MUTED)
    fig.text(0.80, 0.71, "GAIN [95% INTERVAL]", ha="center", fontsize=8, color=MUTED)
    ax = fig.add_axes([0.68, 0.235, 0.28, 0.44], facecolor=BG)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-0.55, 3.55)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10], ["0", "+5", "+10 pp"])
    ax.tick_params(axis="x", labelsize=9, length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, (r, (condition, objective)) in enumerate(zip(rows, labels, strict=True)):
        y = 3 - i
        fy = 0.235 + (y + 0.55) / 4.1 * 0.44
        a = sum(e["goal_success"] for e in r["original"]["per_episode"])
        b = sum(e["goal_success"] for e in r["deployed"]["per_episode"])
        n = len(r["original"]["per_episode"])
        gain = r["goal_difference"]["gain"] * 100
        lo, hi = [v * 100 for v in r["goal_difference"]["interval"]]
        color = GREEN if lo > 0 else MUTED
        for x, value in [(0.04, condition), (0.235, objective)]:
            fig.text(x, fy, value, va="center", fontsize=10)
        fig.text(0.45, fy + 0.014, f"{a / n:.1%} → {b / n:.1%}", va="center", fontsize=11)
        fig.text(0.45, fy - 0.037, f"{a} → {b} / {n}", va="center", fontsize=8, color=MUTED)
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.7)
        ax.plot([lo, hi], [y, y], "|", color=color, markersize=7)
        ax.plot(gain, y, "o", color=color, markersize=5)
        ax.text(gain, y + 0.19, f"+{gain:.1f} [{lo:+.1f}, {hi:+.1f}]", ha="center", fontsize=8)
    fig.add_artist(
        Line2D([0.04, 0.96], [0.755, 0.755], transform=fig.transFigure, color=RULE, lw=0.7)
    )
    fig.text(
        0.04,
        0.105,
        "Points: paired gains. Lines: descriptive, unadjusted 95% bootstrap intervals.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.05,
        "All accepted edits shown; test seeds are shared across edits, not independent replications.",
        fontsize=8,
        color=MUTED,
    )
    save(fig, out, "completion-effects")

    r = rows[0]
    a = [e["goal_success"] for e in r["original"]["per_episode"]]
    b = [e["goal_success"] for e in r["deployed"]["per_episode"]]
    gained = sum(not u and v for u, v in zip(a, b, strict=True))
    lost = sum(u and not v for u, v in zip(a, b, strict=True))
    both = sum(u and v for u, v in zip(a, b, strict=True))
    neither = sum(not u and not v for u, v in zip(a, b, strict=True))
    assert gained + lost + both + neither == len(a)
    fig = plt.figure(figsize=(10, 2.35), facecolor=BG)
    fig.text(0.04, 0.83, "A net gain still hides lost landings.", fontsize=19)
    fig.text(0.96, 0.83, "CROSSWIND · SUCCESS / 0", fontsize=8, color=MUTED, ha="right")
    for x, number, label, color in [
        (0.12, gained, "gained", GREEN),
        (0.40, lost, "lost", RED),
        (0.68, gained - lost, "net gain", INK),
    ]:
        fig.text(x, 0.46, str(number), ha="center", fontsize=34, color=color)
        fig.text(x, 0.33, label, ha="center", fontsize=11, color=color)
    for x, symbol in [(0.26, "−"), (0.54, "=")]:
        fig.text(x, 0.47, symbol, fontsize=27, ha="center", color=MUTED)
    fig.text(0.96, 0.53, f"{sum(a)} → {sum(b)}", ha="right", fontsize=20, color=GREEN)
    fig.text(0.96, 0.37, f"landings / {len(a)}", ha="right", fontsize=10, color=MUTED)
    fig.add_artist(
        Line2D([0.04, 0.96], [0.25, 0.25], transform=fig.transFigure, color=RULE, lw=0.7)
    )
    fig.text(
        0.04,
        0.11,
        f"Unchanged: {both} land with both policies · {neither} fail with both.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(0.96, 0.11, "Same episodes, before and after.", ha="right", fontsize=9, color=MUTED)
    save(fig, out, "paired-outcomes")
    return str(out)
