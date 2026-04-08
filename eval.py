from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from baselines.heuristics import (
    AggressiveFreezeFirstPolicy,
    FixedSequenceInvestigatorPolicy,
    SeedOnlyTriagePolicy,
)
from server.fraud_ring_investigator_arena_environment import (
    FraudRingInvestigatorArenaEnvironment,
)


TASK_NAMES = [
    "easy_single_ring_v1",
    "medium_confounded_ring_v1",
    "hard_reserve_ring_v1",
]

POLICIES = {
    "seed_only": SeedOnlyTriagePolicy,
    "fixed_sequence": FixedSequenceInvestigatorPolicy,
    "freeze_first": AggressiveFreezeFirstPolicy,
}


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _slice_runs(runs: list[dict[str, Any]], case_truth: str | None) -> list[dict[str, Any]]:
    if case_truth is None:
        return runs
    return [run for run in runs if run["case_truth"] == case_truth]


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not runs:
        return {
            "episode_count": 0,
            "episode_score": None,
            "prevented_loss_ratio": None,
            "benign_harm_ratio": None,
            "cost_ratio": None,
            "suspect_f1": None,
            "steps": None,
            "disposition_accuracy": None,
            "invalid_action_rate": None,
        }

    total_steps = sum(int(run["steps"]) for run in runs)
    total_invalid_actions = sum(int(run["invalid_action_count"]) for run in runs)

    return {
        "episode_count": len(runs),
        "episode_score": _round_metric(mean(run["episode_score"] for run in runs)),
        "prevented_loss_ratio": _round_metric(
            mean(run["prevented_loss_ratio"] for run in runs)
        ),
        "benign_harm_ratio": _round_metric(
            mean(run["benign_harm_ratio"] for run in runs)
        ),
        "cost_ratio": _round_metric(mean(run["cost_ratio"] for run in runs)),
        "suspect_f1": _round_metric(mean(run["suspect_f1"] for run in runs)),
        "steps": _round_metric(mean(run["steps"] for run in runs)),
        "disposition_accuracy": _round_metric(
            mean(run["disposition_accuracy"] for run in runs)
        ),
        "invalid_action_rate": _round_metric(
            total_invalid_actions / max(total_steps, 1)
        ),
    }


def run_episode(task_name: str, seed: int, policy_name: str) -> dict[str, Any]:
    env = FraudRingInvestigatorArenaEnvironment()
    observation = env.reset(seed=seed, task_name=task_name)
    policy = POLICIES[policy_name]()
    invalid_action_count = 0
    while not observation.done:
        action = policy.act(observation)
        observation = env.step(action)
        if (
            observation.last_action_result is not None
            and observation.last_action_result.status == "invalid"
        ):
            invalid_action_count += 1
    metadata = observation.metadata or {}
    terminal_metrics = metadata.get("terminal_metrics", {})
    world = env._world
    if world is None:
        raise RuntimeError("Evaluation world missing after rollout.")
    return {
        "task_name": task_name,
        "policy_name": policy_name,
        "seed": seed,
        "case_truth": world.case_truth,
        "episode_score": float(metadata.get("episode_score", observation.reward or 0.0)),
        "steps": int(observation.step_count),
        "prevented_loss_ratio": float(terminal_metrics.get("prevented_loss_ratio", 0.0)),
        "suspect_f1": float(terminal_metrics.get("suspect_f1", 0.0)),
        "benign_harm_ratio": float(terminal_metrics.get("benign_harm_ratio", 0.0)),
        "cost_ratio": float(terminal_metrics.get("cost_ratio", 0.0)),
        "disposition_accuracy": float(terminal_metrics.get("disposition_correct", 0.0)),
        "invalid_action_count": invalid_action_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic heuristic evaluation for Fraud Ring Investigator Arena. "
            "Outputs per-track, per-policy overall/fraud-only/benign-only summaries."
        )
    )
    parser.add_argument(
        "--policy",
        choices=["seed_only", "fixed_sequence", "freeze_first", "all"],
        default="all",
    )
    parser.add_argument(
        "--task-name",
        choices=[*TASK_NAMES, "all"],
        default="medium_confounded_ring_v1",
    )
    parser.add_argument("--seed-start", type=int, default=85_000)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON result payload.",
    )
    args = parser.parse_args()

    policy_names = list(POLICIES) if args.policy == "all" else [args.policy]
    task_names = TASK_NAMES if args.task_name == "all" else [args.task_name]
    runs: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}

    for task_index, task_name in enumerate(task_names):
        task_seed_start = args.seed_start + task_index * args.episodes
        task_results: dict[str, Any] = {}
        for policy_name in policy_names:
            policy_runs = [
                run_episode(task_name, task_seed_start + index, policy_name)
                for index in range(args.episodes)
            ]
            runs.extend(policy_runs)
            task_results[policy_name] = {
                "overall": _summarize_runs(_slice_runs(policy_runs, None)),
                "fraud_only": _summarize_runs(_slice_runs(policy_runs, "fraud")),
                "benign_only": _summarize_runs(_slice_runs(policy_runs, "benign")),
                "fraud_case_count": sum(
                    1 for run in policy_runs if run["case_truth"] == "fraud"
                ),
                "benign_case_count": sum(
                    1 for run in policy_runs if run["case_truth"] == "benign"
                ),
            }
        results[task_name] = task_results

    payload = {
        "config": {
            "episodes_per_policy_track": args.episodes,
            "policy_names": policy_names,
            "seed_start": args.seed_start,
            "task_names": task_names,
        },
        "runs": runs,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
