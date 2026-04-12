"""
Root inference entrypoint for Fraud Ring Investigator Arena.

This script follows the hackathon logging format:

[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import Any

import requests
from openai import OpenAI

from baselines.heuristics import FixedSequenceInvestigatorPolicy
from models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or HF_TOKEN
EXPLICIT_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")
TASK_ID = (
    os.getenv("FRAUD_RING_ARENA_TASK_ID")
    or os.getenv("FRAUD_RING_ARENA_TASK")
    or os.getenv("TASK_ID")
)
BENCHMARK = "fraud_ring_investigator_arena"
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.50"))
ENV_SEED = int(os.getenv("FRAUD_RING_ARENA_SEED", "123"))
DEFAULT_TASK_IDS = ["easy", "medium", "hard"]
REQUEST_TIMEOUT = 30
LLM_REQUEST_ATTEMPT_COUNT = 0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving a structured fraud investigation task.
    Return exactly one JSON object matching this schema:
    {
      "action_type": "...",
      "entity_id": "... or null",
      "payout_id": "... or null",
      "relation_type": "... or null",
      "direction": "... or null",
      "depth": 1 or 2 or null,
      "decision": "clear" or "escalate" or null,
      "suspect_entity_ids": ["..."]
    }

    Valid action_type values:
    inspect_entity, expand_links, trace_funds, inspect_payout, hold_payout, freeze_entity, advance_time, submit_case
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _compact_action(action: FraudRingInvestigatorArenaAction) -> str:
    return json.dumps(action.model_dump(mode="json", exclude_none=True), separators=(",", ":"))


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _task_ids_to_run() -> list[str]:
    if TASK_ID:
        return [TASK_ID]
    return list(DEFAULT_TASK_IDS)


def find_env_url() -> str:
    if EXPLICIT_BASE_URL:
        return EXPLICIT_BASE_URL

    for port in (7860,):
        candidate = f"http://127.0.0.1:{port}"
        try:
            response = requests.get(f"{candidate}/health", timeout=3)
            if response.status_code == 200:
                print(f"[DEBUG] Found env at {candidate}", file=sys.stderr, flush=True)
                return candidate
        except Exception:
            pass

    raise RuntimeError(
        "Could not find environment server. Set ENV_BASE_URL or run the environment on localhost."
    )


def _parse_observation(payload: dict[str, Any]) -> FraudRingInvestigatorArenaObservation:
    obs_data = payload.get("observation", payload)
    return FraudRingInvestigatorArenaObservation(
        **obs_data,
        reward=payload.get("reward"),
        done=payload.get("done", False),
        metadata=obs_data.get("metadata", {}),
    )


def env_reset(env_base_url: str, task_id: str) -> FraudRingInvestigatorArenaObservation:
    response = requests.post(
        f"{env_base_url}/reset",
        json={"task_id": task_id, "seed": ENV_SEED},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_observation(response.json())


def env_step(
    env_base_url: str,
    action: FraudRingInvestigatorArenaAction,
) -> FraudRingInvestigatorArenaObservation:
    response = requests.post(
        f"{env_base_url}/step",
        json={"action": action.model_dump(mode="json", exclude_none=True)},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_observation(response.json())


def build_user_prompt(
    observation: FraudRingInvestigatorArenaObservation,
    history: list[str],
) -> str:
    observation_json = json.dumps(observation.model_dump(mode="json"), separators=(",", ":"))
    history_text = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        Current observation JSON:
        {observation_json}

        Previous actions:
        {history_text}

        Pick exactly one next action.
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    observation: FraudRingInvestigatorArenaObservation,
    history: list[str],
    fallback_policy: FixedSequenceInvestigatorPolicy,
) -> FraudRingInvestigatorArenaAction:
    global LLM_REQUEST_ATTEMPT_COUNT

    try:
        LLM_REQUEST_ATTEMPT_COUNT += 1
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation, history)},
            ],
            temperature=0.0,
            max_tokens=250,
            stream=False,
            timeout=120,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = _extract_json(text)
        if payload is None:
            print(
                f"[DEBUG] Model response was not valid JSON, using fallback action: {text[:200]}",
                file=sys.stderr,
                flush=True,
            )
            return fallback_policy.act(observation)
        return FraudRingInvestigatorArenaAction(**payload)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return fallback_policy.act(observation)


async def _run_task(task_id: str, llm_client: OpenAI, env_base_url: str) -> str | None:
    fallback_policy = FixedSequenceInvestigatorPolicy()

    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_message: str | None = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env_reset(env_base_url, task_id)

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            action = get_model_action(llm_client, observation, history, fallback_policy)
            observation = env_step(env_base_url, action)

            reward = float(observation.reward or 0.0)
            done = bool(observation.done)
            error = (
                observation.last_action_result.error
                if observation.last_action_result is not None
                else None
            )

            rewards.append(reward)
            steps_taken = step
            log_step(step, _compact_action(action), reward, done, error)
            history.append(_compact_action(action))

            if done:
                break

        score = float((observation.metadata or {}).get("episode_score", observation.reward or 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        error_message = str(exc)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return error_message


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or API_KEY or "dummy-key")
    error_messages: list[str] = []

    try:
        env_base_url = find_env_url()
    except Exception as exc:
        env_base_url = None
        error_messages.append(str(exc))

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[DEBUG] ENV_BASE_URL={env_base_url}", file=sys.stderr, flush=True)
    print(f"[DEBUG] HF_TOKEN set={bool(HF_TOKEN)}", file=sys.stderr, flush=True)

    for task_id in _task_ids_to_run():
        if env_base_url is None:
            log_start(task_id, BENCHMARK, MODEL_NAME)
            log_end(False, 0, 0.0, [])
            continue
        error_message = await _run_task(task_id, client, env_base_url)
        if error_message is not None:
            error_messages.append(f"{task_id}: {error_message}")

    if LLM_REQUEST_ATTEMPT_COUNT == 0:
        error_messages.append("No LLM proxy requests were attempted.")

    if error_messages:
        print("[DEBUG] " + "; ".join(error_messages), file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
