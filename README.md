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

Fraud Ring Investigator Arena is a compact OpenEnv benchmark for sequential fraud investigation. Each episode is a single alert-driven case inside a hidden local financial world. The agent must uncover linked structure, choose costly investigation and intervention actions, decide when to wait, and eventually clear or escalate the case before steps run out.

## V1 Scope

- Single-case episodes only
- Three procedural task tracks:
  - `easy_single_ring_v1`
  - `medium_confounded_ring_v1`
  - `hard_reserve_ring_v1`
- Partial observability
- Hidden fraud or benign lookalike structure
- Multi-step tool use
- Action costs
- Delayed payout consequences
- Deterministic programmatic scoring

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

## Observation Space

Each observation includes:

- case metadata
- a seed alert
- visible entities
- visible links
- visible payouts
- active interventions
- realized blocked value
- realized escaped value
- last action result

The full hidden world is never exposed directly.

## Local Development

Start the environment server:

```bash
python3 -m server.app
```

Run a deterministic heuristic evaluation:

```bash
python3 eval.py --policy all --task-name medium_confounded_ring_v1 --episodes 25
```

Run the root inference script against a local server:

```bash
ENV_BASE_URL=http://localhost:8000 python3 inference.py
```

## OpenEnv Files

- `openenv.yaml`
- `pyproject.toml`
- `models.py`
- `client.py`
- `server/app.py`
- `inference.py`

## Notes

The locked V1 behavior contract lives in:

- `notes/16_env_spec_v1.md`
- `notes/17_action_space_v1.md`
- `notes/18_hidden_state_v1.md`
- `notes/19_reward_and_outcomes_v1.md`
- `notes/20_episode_generator_v1.md`
- `notes/21_baselines_and_training_v1.md`
- `notes/22_eval_plan_v1.md`
