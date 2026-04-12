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

from openai import OpenAI

from client import FraudRingInvestigatorArenaEnv
from models import FraudRingInvestigatorArenaAction

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
EXPLICIT_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")
TASK_ID = (
    os.getenv("FRAUD_RING_ARENA_TASK_ID")
    or os.getenv("FRAUD_RING_ARENA_TASK")
    or os.getenv("TASK_ID")
)
BENCHMARK = "fraud_ring_investigator_arena"
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.50"))
SEED = os.getenv("FRAUD_RING_ARENA_SEED")
DEFAULT_TASK_IDS = ["easy", "medium", "hard"]
LLM_REQUEST_ATTEMPT_COUNT = 0
LLM_CALL_COUNT = 0

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


def build_user_prompt(observation, history: list[str]) -> str:
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
    observation,
    history: list[str],
) -> FraudRingInvestigatorArenaAction:
    global LLM_REQUEST_ATTEMPT_COUNT
    global LLM_CALL_COUNT

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
    LLM_CALL_COUNT += 1
    text = (completion.choices[0].message.content or "").strip()
    payload = _extract_json(text)
    if payload is None:
        raise RuntimeError(f"LLM response did not contain a valid JSON action: {text[:200]}")
    return FraudRingInvestigatorArenaAction(**payload)


def _require_proxy_client() -> OpenAI:
    missing = [
        name
        for name, value in (
            ("API_BASE_URL", API_BASE_URL),
            ("API_KEY", API_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required LLM proxy environment variables: " + ", ".join(missing)
        )
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


async def _connect_env() -> FraudRingInvestigatorArenaEnv:
    if IMAGE_NAME:
        return await FraudRingInvestigatorArenaEnv.from_docker_image(IMAGE_NAME)

    if EXPLICIT_BASE_URL:
        base_urls = [EXPLICIT_BASE_URL]
    else:
        base_urls = [
            "http://127.0.0.1:7860",
            "http://localhost:7860",
        ]

    errors: list[str] = []
    for base_url in base_urls:
        env = FraudRingInvestigatorArenaEnv(base_url=base_url)
        try:
            await env.connect()
            return env
        except Exception as exc:
            errors.append(f"{base_url}: {exc}")

    raise RuntimeError("Failed to connect to environment. Tried " + " | ".join(errors))


async def _run_task(task_id: str, llm_client: OpenAI) -> str | None:
    env: FraudRingInvestigatorArenaEnv | None = None

    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_message: str | None = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await _connect_env()
        reset_kwargs: dict[str, Any] = {"task_id": task_id}
        if SEED is not None:
            reset_kwargs["seed"] = int(SEED)
        result = await env.reset(**reset_kwargs)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(llm_client, result.observation, history)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = (
                result.observation.last_action_result.error
                if result.observation.last_action_result is not None
                else None
            )
            rewards.append(reward)
            steps_taken = step
            log_step(step, _compact_action(action), reward, done, error)
            history.append(_compact_action(action))
            if done:
                break

        score = float(
            (result.observation.metadata or {}).get("episode_score", result.reward or 0.0)
        )
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        error_message = str(exc)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as close_exc:
                error_message = error_message or str(close_exc)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return error_message


def _task_ids_to_run() -> list[str]:
    if TASK_ID:
        return [TASK_ID]
    return list(DEFAULT_TASK_IDS)


async def main() -> None:
    error_messages: list[str] = []
    llm_client: OpenAI | None = None

    try:
        llm_client = _require_proxy_client()
    except Exception as exc:
        error_messages.append(f"llm_proxy: {exc}")

    for task_id in _task_ids_to_run():
        if llm_client is None:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            continue
        error_message = await _run_task(task_id, llm_client)
        if error_message is not None:
            error_messages.append(f"{task_id}: {error_message}")

    if LLM_REQUEST_ATTEMPT_COUNT == 0:
        error_messages.append("No LLM proxy requests were attempted.")

    if error_messages:
        print("[DEBUG] " + "; ".join(error_messages), file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] fatal_inference_error: {exc}", file=sys.stderr, flush=True)
