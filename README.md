---
title: Fraud Ring Investigator Arena Environment Server
emoji: "🕵️"
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Fraud Ring Investigator Arena

Fraud Ring Investigator Arena is a compact OpenEnv benchmark for sequential fraud investigation. Each episode is a single alert-driven case inside a hidden local financial world. The agent must uncover linked structure, spend limited investigation budget, place risky interventions, decide when to wait for delayed outcomes, and finally clear or escalate the case.

## What The Environment Is

V1 is a single-case benchmark with three procedural task tracks:

- `easy_single_ring_v1`
- `medium_confounded_ring_v1`
- `hard_reserve_ring_v1`

Each case mixes a seed alert, a partially visible entity/link/payout slice, and a hidden fraud or benign lookalike structure.

## Why RL Is Justified Here

This is an RL-native task because the agent must make sequential, budgeted decisions under partial observability. Actions reveal different parts of the hidden world, interventions have explicit costs, and the most important consequences arrive later when held or unblocked payouts settle.

## Why This Is Not Just Fraud Classification

A one-shot classifier cannot solve the main tradeoff. The agent is not asked to label a static graph once. It must decide:

- what to inspect
- which links to expand
- when to trace funds
- whether to hold or freeze
- when to advance time
- when the case is strong enough to submit

The baseline results already show the difference: `seed_only` is near-perfect on benign cases but much weaker on fraud cases because it prevents no loss.

## Episode Structure

1. `reset()` returns a seed alert and a small visible slice of the case.
2. The agent investigates with typed tool-use actions.
3. Interventions can block fraud, hurt benign entities, or arm delayed downstream effects.
4. `advance_time` resolves pending payout waves.
5. The agent eventually `submit_case`s with `clear` or `escalate`.

## Hidden State And Delayed Consequences

The hidden state includes the latent entity graph, fraud motif roles, future payout schedule, and delayed blocked-versus-escaped value. Agents never see the full ring or the full benign lookalike cluster directly. The hard track adds reserve-route behavior, which makes reckless early freezing less reliable.

## Action Space

The typed V1 action surface is:

- `inspect_entity`
- `expand_links`
- `trace_funds`
- `inspect_payout`
- `hold_payout`
- `freeze_entity`
- `advance_time`
- `submit_case`

## Scoring Overview

Scoring is deterministic and reward-relevant metrics differ by case truth:

- Fraud cases emphasize prevented loss, suspect quality, intervention precision, and correct final disposition.
- Benign cases emphasize correct clearance, low benign harm, low false suspicion, and low investigation cost.

This makes the benchmark care about false positives, missed fraud, and action cost at the same time.

## Task Tracks

| Track | Design intent | Current baseline read |
| --- | --- | --- |
| `easy_single_ring_v1` | compact single-ring cases with clearer payout pressure | aggressive early intervention is very strong on fraud cases |
| `medium_confounded_ring_v1` | more confounders and noisier local structure | still favors fast intervention, but benign penalties remain real |
| `hard_reserve_ring_v1` | reserve-route and more balanced fraud/benign mix | punishes reckless intervention enough to change the ranking |

## Local Development

Start the environment server:

```bash
python3 -m server.app
```

Run the full deterministic baseline matrix and save JSON output:

```bash
python3 eval.py --policy all --task-name all --episodes 25 --output outputs/baseline_eval_raw.txt
```

Run the root inference script against a local server:

```bash
ENV_BASE_URL=http://localhost:8000 python3 inference.py
```

## Docker And Space Usage

Build the local image:

```bash
docker build -t fraud-ring-investigator-arena-local .
```

Run the inference script against the local Docker image:

```bash
IMAGE_NAME=fraud-ring-investigator-arena-local python3 inference.py
```

The benchmark has already been deployed as a Hugging Face Space and passed the provided submission prevalidation flow.

## Validation Status

The current repo state has already passed:

- local Docker image build
- root `inference.py` against the local Docker image
- `openenv validate`
- Hugging Face Space deployment
- submission prevalidation for live `/reset`, Docker build, and `openenv validate`

## Baseline Evaluation Snapshot

Overall episode score, 25 deterministic episodes per policy-track:

| Track | `seed_only` | `fixed_sequence` | `freeze_first` |
| --- | ---: | ---: | ---: |
| `easy_single_ring_v1` | 0.3307 | 0.6235 | 0.7483 |
| `medium_confounded_ring_v1` | 0.3896 | 0.5995 | 0.6798 |
| `hard_reserve_ring_v1` | 0.5444 | 0.4573 | 0.4403 |

That table is intentionally not the whole story. The truth-stratified view matters:

- `freeze_first` is strongest on fraud-only slices for easy and medium.
- `fixed_sequence` slightly beats `freeze_first` on the hard fraud-only slice.
- `seed_only` is almost perfect on benign-only slices but poor on fraud-only slices.

See [outputs/baseline_eval_summary.md](/Users/tsn/projects/fraud-ring-investigator-arena/outputs/baseline_eval_summary.md) and [outputs/benchmark_sanity_check.md](/Users/tsn/projects/fraud-ring-investigator-arena/outputs/benchmark_sanity_check.md) for the full benchmark readout.

## Planned Training Story

The intended first training pass is straightforward:

- train against the current benchmark without mechanic changes
- report overall, fraud-only, and benign-only slices together
- compare learned policies against `seed_only`, `fixed_sequence`, and `freeze_first`
- only consider tiny benchmark adjustments after training if the learned policy confirms the same easy/medium intervention dominance pattern

## OpenEnv Files

- `openenv.yaml`
- `pyproject.toml`
- `models.py`
- `client.py`
- `server/app.py`
- `inference.py`

## Locked V1 Spec

The locked V1 behavior contract lives in:

- `notes/16_env_spec_v1.md`
- `notes/17_action_space_v1.md`
- `notes/18_hidden_state_v1.md`
- `notes/19_reward_and_outcomes_v1.md`
- `notes/20_episode_generator_v1.md`
- `notes/21_baselines_and_training_v1.md`
- `notes/22_eval_plan_v1.md`
