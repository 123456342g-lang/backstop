"""Jev-compatible decision backends. The stream talks only to this module."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from experiments.m0b.protocol import JEV_URL, MODEL_ID, QUESTION


class DecisionBackend(Protocol):
    """One typed-decision call. Keys are question names."""

    name: str

    def decide(self, state: dict, questions: dict) -> dict[str, dict[str, object]]:
        """Return choice, probabilities, and answer_confidence per question."""


class LayaBackend:
    """Local convaiinnovations/laya-typed-decisions on CPU."""

    name = "laya-typed-decisions"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya

        torch.set_num_threads(4)
        self._agent = laya.load(model_id, device="cpu")

    def decide(self, state: dict, questions: dict) -> dict[str, dict[str, object]]:
        raw = self._agent.predict(state, questions)
        parsed: dict[str, dict[str, object]] = {}
        for name, answer in raw["answers"].items():
            probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
            parsed[name] = {
                "choice": str(answer["choice"]),
                "probabilities": probabilities,
                "answer_confidence": float(answer["answer_confidence"]),
            }
        return parsed


class JevBackend:
    """Hosted TypeSafe Jev. Unused unless JEV_API_KEY is set."""

    name = "jev-latest"

    def __init__(self, api_key: str, url: str = JEV_URL, cap: int = 20) -> None:
        self._api_key = api_key
        self._url = url
        self._cap = cap
        self.calls = 0

    def decide(self, state: dict, questions: dict) -> dict[str, dict[str, object]]:
        if self.calls >= self._cap:
            raise RuntimeError("Jev cap reached")
        payload = {"model": "jev-latest", "state": state, "questions": questions}
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
        return _parse_jev(body)


def action_question(questions: dict) -> dict:
    if QUESTION not in questions:
        raise KeyError(f"missing {QUESTION} question")
    return {QUESTION: questions[QUESTION]}


def _parse_jev(body: dict) -> dict[str, dict[str, object]]:
    answers = body.get("answers") or body.get("result", {}).get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("Jev response has no answers object")
    parsed: dict[str, dict[str, object]] = {}
    for name, answer in answers.items():
        probabilities = {
            str(key): float(value)
            for key, value in dict(answer.get("probabilities") or {}).items()
        }
        if not probabilities:
            raise RuntimeError(f"Jev answer {name} has no probabilities")
        choice = str(answer.get("choice") or max(probabilities, key=probabilities.get))
        parsed[name] = {
            "choice": choice,
            "probabilities": probabilities,
            "answer_confidence": float(max(probabilities.values())),
        }
    return parsed
