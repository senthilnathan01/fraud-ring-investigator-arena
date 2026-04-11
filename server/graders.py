from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def _serialize(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "dict"):
        value = value.dict()
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _build_client() -> tuple[OpenAI | None, str | None]:
    model_name = os.getenv("GRADER_MODEL_NAME") or os.getenv("MODEL_NAME")
    api_key = (
        os.getenv("GRADER_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
    )
    api_base_url = (
        os.getenv("GRADER_API_BASE_URL")
        or os.getenv("API_BASE_URL")
        or "https://router.huggingface.co/v1"
    )
    if not api_key or not model_name:
        return None, None
    return OpenAI(base_url=api_base_url, api_key=api_key), model_name


def _grade_with_prompt(prompt_template: str, *args: Any, **kwargs: Any) -> float:
    client, model_name = _build_client()
    if client is None or model_name is None:
        return 0.0

    action = kwargs.get("action")
    observation = kwargs.get("observation")
    if action is None and args:
        action = args[0]
    if observation is None and len(args) > 1:
        observation = args[1]

    prompt = prompt_template.format(
        action=_serialize(action),
        observation=_serialize(observation),
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are grading an OpenEnv episode. Return only one numeric "
                        "score between 0.0 and 1.0."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=32,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
    except Exception:
        return 0.0

    match = re.search(r"(\d+\.?\d*)", content)
    if match is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return 0.0


def grade_easy(*args: Any, **kwargs: Any) -> float:
    return _grade_with_prompt(
        (
            "Score this easy fraud-investigation episode from 0.0 to 1.0.\n"
            "Reward prevented fraud loss, accurate suspect identification, and the "
            "correct final clear-or-escalate disposition.\n"
            "Penalize benign harm and unnecessary investigation cost.\n\n"
            "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
        ),
        *args,
        **kwargs,
    )


def grade_medium(*args: Any, **kwargs: Any) -> float:
    return _grade_with_prompt(
        (
            "Score this medium fraud-investigation episode from 0.0 to 1.0.\n"
            "Reward prevented fraud loss, accurate suspect identification, and the "
            "correct final clear-or-escalate disposition.\n"
            "Penalize benign harm, reckless intervention, and unnecessary investigation cost.\n\n"
            "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
        ),
        *args,
        **kwargs,
    )


def grade_hard(*args: Any, **kwargs: Any) -> float:
    return _grade_with_prompt(
        (
            "Score this hard fraud-investigation episode from 0.0 to 1.0.\n"
            "Reward prevented fraud loss, accurate suspect identification, and the "
            "correct final clear-or-escalate disposition.\n"
            "Penalize benign harm, premature intervention, reserve-route misses, and "
            "unnecessary investigation cost.\n\n"
            "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
        ),
        *args,
        **kwargs,
    )


__all__ = ["grade_easy", "grade_medium", "grade_hard"]
