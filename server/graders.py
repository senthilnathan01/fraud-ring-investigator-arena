from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI
from openenv.core.rubrics.base import Rubric


class BaseFraudArenaLLMGrader(Rubric):
    """OpenAI-compatible LLM grader for submission-manifest compatibility."""

    prompt_template: str = ""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        default_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name or os.getenv("GRADER_MODEL_NAME") or os.getenv("MODEL_NAME")
        self.api_base_url = (
            api_base_url
            or os.getenv("GRADER_API_BASE_URL")
            or os.getenv("API_BASE_URL")
            or "https://router.huggingface.co/v1"
        )
        self.api_key = (
            api_key
            or os.getenv("GRADER_API_KEY")
            or os.getenv("HF_TOKEN")
            or os.getenv("API_KEY")
        )
        self.default_score = default_score
        self._score_pattern = re.compile(r"(\d+\.?\d*)")
        self._client = (
            OpenAI(base_url=self.api_base_url, api_key=self.api_key)
            if self.api_key and self.model_name
            else None
        )

    def forward(self, action: Any, observation: Any) -> float:
        if self._client is None or not self.prompt_template:
            return self.default_score

        prompt = self.prompt_template.format(
            action=self._serialize(action),
            observation=self._serialize(observation),
        )
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
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
            return self.default_score

        match = self._score_pattern.search(content)
        if match is None:
            return self.default_score
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return self.default_score

    @staticmethod
    def _serialize(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        elif hasattr(value, "dict"):
            value = value.dict()
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


class EasyTaskGrader(BaseFraudArenaLLMGrader):
    prompt_template = (
        "Score this easy fraud-investigation episode from 0.0 to 1.0.\n"
        "Reward prevented fraud loss, accurate suspect identification, and the "
        "correct final clear-or-escalate disposition.\n"
        "Penalize benign harm and unnecessary investigation cost.\n\n"
        "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
    )


class MediumTaskGrader(BaseFraudArenaLLMGrader):
    prompt_template = (
        "Score this medium fraud-investigation episode from 0.0 to 1.0.\n"
        "Reward prevented fraud loss, accurate suspect identification, and the "
        "correct final clear-or-escalate disposition.\n"
        "Penalize benign harm, reckless intervention, and unnecessary investigation cost.\n\n"
        "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
    )


class HardTaskGrader(BaseFraudArenaLLMGrader):
    prompt_template = (
        "Score this hard fraud-investigation episode from 0.0 to 1.0.\n"
        "Reward prevented fraud loss, accurate suspect identification, and the "
        "correct final clear-or-escalate disposition.\n"
        "Penalize benign harm, premature intervention, reserve-route misses, and "
        "unnecessary investigation cost.\n\n"
        "Action:\n{action}\n\nObservation:\n{observation}\n\nScore:"
    )


__all__ = [
    "BaseFraudArenaLLMGrader",
    "EasyTaskGrader",
    "MediumTaskGrader",
    "HardTaskGrader",
]
