---
title: Fraud Ring Investigator Arena
emoji: "🕵️"
sdk: docker
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - fraud-detection
---

# Fraud Ring Investigator Arena

Fraud Ring Investigator Arena is an OpenEnv benchmark for sequential fraud investigation. Each episode is a compact alert-driven AML case with hidden structure, partial observability, action costs, delayed payout consequences, and a final clear-or-escalate decision.

The benchmark is designed for RL-style tool use rather than one-shot classification. The agent must choose what to inspect, what to hold, when to wait, and when the evidence is strong enough to submit the case.

## Environment Summary

- Environment name: `fraud_ring_investigator_arena`
- Submission task IDs: `easy`, `medium`, `hard`
- Canonical tracks:
  - `easy` → `easy_single_ring_v1`
  - `medium` → `medium_confounded_ring_v1`
  - `hard` → `hard_reserve_ring_v1`
- Reward range: `[0.0, 1.0]`
- Runtime port: `7860`

## Why This Is RL-Native

This benchmark is not a wrapped fraud classifier.

- The agent starts from a partial visible slice of the case.
- Fraud structure and future payout consequences are hidden.
- Investigation actions have explicit costs.
- Some interventions help on fraud cases and hurt on benign cases.
- Important consequences are delayed until later payout waves settle.
- Final score depends on a sequence of actions, not a single label.

## Episode Flow

1. `reset()` returns a seed alert plus a small visible slice of the entity, link, and payout graph.
2. The agent explores with typed actions such as inspection, link expansion, payout inspection, and fund tracing.
3. The agent may intervene with `hold_payout` or `freeze_entity`.
4. The agent may `advance_time` to resolve delayed payout effects.
5. The agent ends the case with `submit_case(decision="clear" | "escalate")`.

## Action Space

The typed action surface is:

- `inspect_entity`
- `expand_links`
- `trace_funds`
- `inspect_payout`
- `hold_payout`
- `freeze_entity`
- `advance_time`
- `submit_case`

The full typed models live in:

- `models.py`
- `server/fraud_ring_investigator_arena_environment.py`
- `server/simulator.py`

## Observation Surface

Each observation contains:

- `task_id`, `task_name`, `case_id`
- `step_count`, `steps_remaining`
- `time_tick`, `time_ticks_remaining`
- `investigation_cost_used`
- `seed_alert`
- `visible_entities`
- `visible_links`
- `visible_payouts`
- `active_interventions`
- `blocked_value_realized`, `escaped_value_realized`
- `last_action_result`

## Reward and Scoring

Episode score is normalized to `[0, 1]`.

- Fraud cases reward prevented loss, good suspect identification, and correct escalation.
- Benign cases reward low intervention harm, low false suspicion, and correct clearance.
- Investigation actions apply step costs.
- Invalid actions are penalized.

This creates the intended tradeoff between false positives, missed fraud, and investigation cost.

## Deterministic Benchmark Baselines

The repo includes deterministic heuristic baselines in `baselines/heuristics.py` and a deterministic evaluator in `eval.py`.

Current deterministic benchmark snapshot, 25 episodes per policy-track:

| Track | `seed_only` | `fixed_sequence` | `freeze_first` |
| --- | ---: | ---: | ---: |
| `easy_single_ring_v1` | 0.3299 | 0.5758 | 0.7265 |
| `medium_confounded_ring_v1` | 0.3776 | 0.6144 | 0.6567 |
| `hard_reserve_ring_v1` | 0.5335 | 0.4201 | 0.4195 |

Run the deterministic evaluator:

```bash
python3 eval.py --policy all --task-name all --episodes 25 --seed-start 85000
```

## Reproducibility

The environment rollout is deterministic for a fixed task ID and seed.

- `inference.py` uses `FRAUD_RING_ARENA_SEED=123` by default.
- `eval.py` uses `--seed-start 85000` by default.
- `inference.py` uses `temperature=0.0` for model calls.

Important caveat: remote hosted LLM outputs can still vary slightly across runs even with the same prompt and temperature. The environment itself remains deterministic for the same seed.

## LLM Submission Inference

The root `inference.py` is the submission script.

- It uses the OpenAI client.
- It supports `API_KEY`, `OPENAI_API_KEY`, and `HF_TOKEN`.
- It uses `API_BASE_URL` with default `https://router.huggingface.co/v1`.
- It runs the three submission tasks `easy`, `medium`, and `hard` by default.

Recommended submission-style local command:

```bash
OPENAI_API_KEY="$HF_TOKEN" \
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
ENV_BASE_URL="http://127.0.0.1:7860" \
FRAUD_RING_ARENA_SEED=123 \
python3 inference.py
```

If you want to run a single task:

```bash
OPENAI_API_KEY="$HF_TOKEN" \
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
ENV_BASE_URL="http://127.0.0.1:7860" \
FRAUD_RING_ARENA_TASK_ID=easy \
FRAUD_RING_ARENA_SEED=123 \
python3 inference.py
```

## Local Development

Start the server:

```bash
python3 -m server.app
```

Validate the environment contract:

```bash
openenv validate
```

Build the Docker image:

```bash
docker build -t fraud-ring-investigator-arena-local .
```

## OpenEnv Files

- `openenv.yaml`
- `app.py`
- `models.py`
- `client.py`
- `server/app.py`
- `server/environment.py`
- `server/fraud_ring_investigator_arena_environment.py`
- `inference.py`

## Submission Checklist

Before final submission, do these in order:

1. Run `openenv validate`.
2. Confirm the Space responds to `POST /reset`.
3. Run `python3 eval.py --policy all --task-name all --episodes 25 --seed-start 85000`.
4. Run one real LLM inference pass with fixed seed `123` and record the three `[END]` scores.
5. Confirm the submitted `inference.py` uses the OpenAI client and the injected proxy env vars.
6. Rebuild Docker once after the last code change.
7. Submit only after the repo and deployed Space are in sync.
