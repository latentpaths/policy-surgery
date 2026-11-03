"""Real Box2D landings with changed physics, not a linear point-mass inverse."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar
import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    main_power: float = 1.0
    side_power: float = 1.0
    gravity: float = -10.0
    wind: float = 0.0
    turbulence: float = 0.0
    mirrored: bool = False


CONDITIONS = {
    c.name: c
    for c in [
        Condition("nominal"),
        Condition("mirrored_thrusters", mirrored=True),
        Condition("weak_main", main_power=0.75),
        Condition("crosswind", wind=18, turbulence=1.5),
        Condition("compound", main_power=0.8, gravity=-11.5, wind=12, turbulence=1.0),
    ]
}


@contextmanager
def engine_power(condition):
    """Gym reads module constants in step; restore them even on exceptions.

    Environments are evaluated serially in a process. This adapter is not thread-safe.
    """
    main, side = lunar.MAIN_ENGINE_POWER, lunar.SIDE_ENGINE_POWER
    lunar.MAIN_ENGINE_POWER = main * condition.main_power
    lunar.SIDE_ENGINE_POWER = side * condition.side_power
    try:
        yield
    finally:
        lunar.MAIN_ENGINE_POWER, lunar.SIDE_ENGINE_POWER = main, side


def episode(policy, condition: Condition, seed: int, render=False, horizon=1000, heuristic=False):
    env = gym.make(
        "LunarLander-v3",
        gravity=condition.gravity,
        enable_wind=condition.wind > 0,
        wind_power=condition.wind,
        turbulence_power=condition.turbulence,
        max_episode_steps=horizon,
        render_mode="rgb_array" if render else None,
    )
    frames, trace = [], []
    total, counts, terminated, truncated = 0.0, np.zeros(4, dtype=int), False, False
    try:
        obs, _ = env.reset(seed=int(seed))
        for step in range(horizon):
            if render and step % 2 == 0:
                frames.append(env.render())
            action = int(lunar.heuristic(env.unwrapped, obs)) if heuristic else policy.act(obs)
            counts[action] += 1
            if step % 40 == 0:
                trace.append({"step": step, "observation": obs.tolist(), "action": action})
            physical_action = {1: 3, 3: 1}.get(action, action) if condition.mirrored else action
            with engine_power(condition):
                obs, reward, terminated, truncated, _ = env.step(physical_action)
            total += reward
            if terminated or truncated:
                break
        safe = bool(terminated and not env.unwrapped.game_over and not env.unwrapped.lander.awake)
        goal_success = bool(
            safe
            and all(
                leg.ground_contact
                and env.unwrapped.helipad_x1 <= leg.position.x <= env.unwrapped.helipad_x2
                for leg in env.unwrapped.legs
            )
        )
        trace.append({"step": step + 1, "observation": obs.tolist(), "action": None})
        return {
            "return": float(total),
            "benchmark_solved": bool(total >= 200),
            "safe_landing": safe,
            "goal_success": goal_success,
            "steps": step + 1,
            "truncated": bool(truncated),
            "crashed": bool(env.unwrapped.game_over),
            "final_observation": obs.tolist(),
            "action_counts": counts.tolist(),
            "trace": trace,
            "frames": frames,
        }
    finally:
        env.close()


def evaluate(policy, condition: Condition, seeds: list[int], traces=False, heuristic=False):
    episodes = [episode(policy, condition, seed, heuristic=heuristic) for seed in seeds]
    output = {
        "condition": asdict(condition),
        "episodes": len(episodes),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "std_return": float(np.std([e["return"] for e in episodes])),
        "solution_rate": float(np.mean([e["benchmark_solved"] for e in episodes])),
        "safe_landing_rate": float(np.mean([e["safe_landing"] for e in episodes])),
        "goal_success_rate": float(np.mean([e["goal_success"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "crash_rate": float(np.mean([e["crashed"] for e in episodes])),
        "per_episode": [
            {k: v for k, v in e.items() if k not in ("frames", "trace")} for e in episodes
        ],
    }
    if traces:
        output["diagnostics"] = [{"return": e["return"], "trace": e["trace"]} for e in episodes[:2]]
    return output
