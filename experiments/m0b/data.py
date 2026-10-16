"""Typed-decisions rows, the frozen split, and label/text drift."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from datasets import load_dataset

from experiments.m0b.protocol import (
    CALIBRATION_N,
    DATASET_CONFIG,
    DATASET_ID,
    PARAPHRASE,
    POLICY_CYCLE,
    RELABEL_FROM,
    RELABEL_TO,
    SLICE_ORDER,
    SLICE_QUOTA,
    STREAM_N,
)


def load_cases(split: str) -> tuple[dict[str, object], ...]:
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    cases: list[dict[str, object]] = []
    for row in dataset:
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        questions = (
            json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        )
        if "thread" not in state:
            raise RuntimeError(f"{row['id']} has no thread")
        cases.append(
            {
                "id": str(row["id"]),
                "state": state,
                "questions": questions,
                "gold": str(row["action__label"]),
            }
        )
    return tuple(cases)


def paraphrase_state(state: dict) -> dict:
    thread = []
    for turn in state["thread"]:
        text = str(turn["text"])
        for source, target in PARAPHRASE:
            text = text.replace(source, target)
        thread.append({**turn, "text": text})
    return {**state, "thread": thread}


def remap_gold(slice_name: str, gold: str) -> str:
    if slice_name == "policy":
        if gold not in POLICY_CYCLE:
            raise KeyError(gold)
        return POLICY_CYCLE[gold]
    if slice_name == "relabel":
        if gold == RELABEL_FROM:
            return RELABEL_TO
        return gold
    if slice_name in {"control", "paraphrase", "calibration"}:
        return gold
    raise ValueError(slice_name)


def assign_roles(count: int, seed: int) -> dict[str, tuple[int, ...]]:
    """Prefix split of a seeded permutation. Roles sum to `count`."""
    if count != CALIBRATION_N + STREAM_N:
        raise RuntimeError(f"expected {CALIBRATION_N + STREAM_N} rows, got {count}")
    order = np.random.default_rng(seed).permutation(count)
    cursor = CALIBRATION_N
    roles: dict[str, tuple[int, ...]] = {
        "calibration": tuple(int(index) for index in order[:CALIBRATION_N])
    }
    for name in SLICE_ORDER:
        quota = SLICE_QUOTA[name]
        roles[name] = tuple(int(index) for index in order[cursor : cursor + quota])
        cursor += quota
    if cursor != count:
        raise RuntimeError(f"split consumed {cursor} of {count}")
    return roles


def build_stream(
    cases: Sequence[dict[str, object]],
    roles: dict[str, tuple[int, ...]],
    seed: int,
) -> tuple[dict[str, object], ...]:
    tagged: list[dict[str, object]] = []
    for slice_name in SLICE_ORDER:
        for index in roles[slice_name]:
            case = cases[index]
            original = str(case["gold"])
            state = case["state"]
            if not isinstance(state, dict):
                raise RuntimeError(f"{case['id']} state is not an object")
            variant = "original"
            if slice_name == "paraphrase":
                state = paraphrase_state(state)
                variant = "paraphrase"
            tagged.append(
                {
                    "id": case["id"],
                    "slice": slice_name,
                    "variant": variant,
                    "state": state,
                    "questions": case["questions"],
                    "gold": remap_gold(slice_name, original),
                    "original_gold": original,
                }
            )
    if len(tagged) != STREAM_N:
        raise RuntimeError(f"stream {len(tagged)} != {STREAM_N}")
    order = np.random.default_rng(seed).permutation(len(tagged))
    return tuple(tagged[int(index)] for index in order)
