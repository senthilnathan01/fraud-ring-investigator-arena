from __future__ import annotations

import argparse
import json
from statistics import mean

from baselines.heuristics import (
    AggressiveFreezeFirstPolicy,
    FixedSequenceInvestigatorPolicy,
    SeedOnlyTriagePolicy,
)
from server.fraud_ring_investigator_arena_environment import (
    FraudRingInvestigatorArenaEnvironment,
)


POLICIES = {
    "seed_only": SeedOnlyTriagePolicy,
    "fixed_sequence": FixedSequenceInvestigatorPolicy,
    "freeze_first": AggressiveFreezeFirstPolicy,
}


def run_episode(task_name: str, seed: int, policy_name: str) -> dict[str, float]:
    env = FraudRingInvestigatorArenaEnvironment()
    observation = env.reset(seed=seed, task_name=task_name)
    policy = POLICIES[policy_name]()
    rewards: list[float] = []
    while not observation.done:
        action = policy.act(observation)
        observation = env.step(action)
        rewards.append(float(observation.reward or 0.0))
    metadata = observation.metadata or {}
    terminal_metrics = metadata.get("terminal_metrics", {})
    return {
        "episode_score": float(metadata.get("episode_score", observation.reward or 0.0)),
        "steps": float(observation.step_count),
        "prevented_loss_ratio": float(terminal_metrics.get("prevented_loss_ratio", 0.0)),
        "suspect_f1": float(terminal_metrics.get("suspect_f1", 0.0)),
        "benign_harm_ratio": float(terminal_metrics.get("benign_harm_ratio", 0.0)),
        "cost_ratio": float(terminal_metrics.get("cost_ratio", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic heuristic evaluation")
    parser.add_argument(
        "--policy",
        choices=["seed_only", "fixed_sequence", "freeze_first", "all"],
        default="all",
    )
    parser.add_argument(
        "--task-name",
        choices=[
            "easy_single_ring_v1",
            "medium_confounded_ring_v1",
            "hard_reserve_ring_v1",
        ],
        default="medium_confounded_ring_v1",
    )
    parser.add_argument("--seed-start", type=int, default=85_000)
    parser.add_argument("--episodes", type=int, default=25)
    args = parser.parse_args()

    policy_names = list(POLICIES) if args.policy == "all" else [args.policy]
    summaries: dict[str, dict[str, float]] = {}

    for policy_name in policy_names:
        runs = [
            run_episode(args.task_name, args.seed_start + index, policy_name)
            for index in range(args.episodes)
        ]
        summaries[policy_name] = {
            key: round(mean(run[key] for run in runs), 4)
            for key in runs[0]
        }

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
