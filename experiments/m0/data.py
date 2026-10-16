"""MASSIVE loading and the frozen production stream."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from datasets import load_dataset

from experiments.m0.protocol import (
    ADVERSARIAL_PREFIX,
    BOILERPLATE,
    SEED_PRIMARY,
    SHARED_POOL_SLICES,
    SLICE_ORDER,
    SLICE_QUOTA,
    STREAM_SIZE,
)

COOKING = "cooking"
SPLIT_NAMES = ("train", "validation", "test")


def load_locale(locale: str) -> tuple[dict[str, str], ...]:
    """Return immutable utterance rows for one MASSIVE locale."""
    rows: list[dict[str, str]] = []
    for split in SPLIT_NAMES:
        dataset = load_dataset(
            "AmazonScience/massive",
            locale,
            split=split,
            trust_remote_code=True,
        )
        intents = dataset.features["intent"]
        scenarios = dataset.features["scenario"]
        for raw in dataset:
            rows.append(
                {
                    "uid": f"{locale}:{split}:{raw['id']}",
                    "split": split,
                    "locale": locale,
                    "scenario": scenarios.int2str(raw["scenario"]),
                    "intent": intents.int2str(raw["intent"]),
                    "text": raw["utt"],
                }
            )
    return tuple(rows)


def _not_cooking(rows: Sequence[dict[str, str]]) -> tuple[dict[str, str], ...]:
    return tuple(row for row in rows if row["scenario"] != COOKING)


def _cooking(rows: Sequence[dict[str, str]]) -> tuple[dict[str, str], ...]:
    return tuple(row for row in rows if row["scenario"] == COOKING)


def pools(
    english: Sequence[dict[str, str]],
    spanish: Sequence[dict[str, str]],
) -> dict[str, tuple[dict[str, str], ...]]:
    """Fit, calibration, and production source pools. Cooking never enters fit or calibration."""
    return {
        "fit": _not_cooking(tuple(row for row in english if row["split"] == "train")),
        "calibration": _not_cooking(
            tuple(row for row in english if row["split"] == "validation")
        ),
        "en_test_other": _not_cooking(
            tuple(row for row in english if row["split"] == "test")
        ),
        "cooking": _cooking(english),
        "es_test_other": _not_cooking(
            tuple(row for row in spanish if row["split"] == "test")
        ),
    }


def render_text(slice_name: str, text: str) -> str:
    if slice_name in {"id", "new_class", "language"}:
        return text
    if slice_name == "long":
        return f"{BOILERPLATE} {text}"
    if slice_name == "adversarial":
        return f"{ADVERSARIAL_PREFIX} {text}"
    raise ValueError(f"unknown slice: {slice_name}")


def _take_prefix(
    pool: Sequence[dict[str, str]],
    count: int,
    seed: int,
) -> tuple[dict[str, str], ...]:
    if len(pool) < count:
        raise RuntimeError(
            f"pool has {len(pool)} rows, need {count}; protocol forbids resizing"
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    return tuple(pool[int(index)] for index in order[:count])


def _shared_slices(
    pool: Sequence[dict[str, str]],
    seed: int,
) -> dict[str, tuple[dict[str, str], ...]]:
    needed = sum(SLICE_QUOTA[name] for name in SHARED_POOL_SLICES)
    if len(pool) < needed:
        raise RuntimeError(
            f"en-US test pool has {len(pool)} rows, need {needed} disjoint rows"
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    cursor = 0
    chosen: dict[str, tuple[dict[str, str], ...]] = {}
    for name in SHARED_POOL_SLICES:
        quota = SLICE_QUOTA[name]
        indexes = order[cursor : cursor + quota]
        chosen[name] = tuple(pool[int(index)] for index in indexes)
        cursor += quota
    return chosen


def build_stream(
    source: dict[str, tuple[dict[str, str], ...]],
    seed: int = SEED_PRIMARY,
) -> tuple[dict[str, str], ...]:
    """Sample the frozen quotas and shuffle the replay order."""
    chosen = _shared_slices(source["en_test_other"], seed)
    chosen["new_class"] = _take_prefix(source["cooking"], SLICE_QUOTA["new_class"], seed)
    chosen["language"] = _take_prefix(
        source["es_test_other"], SLICE_QUOTA["language"], seed
    )
    tagged: list[dict[str, str]] = []
    for slice_name in SLICE_ORDER:
        for row in chosen[slice_name]:
            tagged.append(
                {
                    "uid": row["uid"],
                    "slice": slice_name,
                    "intent": row["intent"],
                    "text": render_text(slice_name, row["text"]),
                    "raw_text": row["text"],
                }
            )
    if len(tagged) != STREAM_SIZE:
        raise RuntimeError(f"stream size {len(tagged)} != {STREAM_SIZE}")
    order = np.random.default_rng(seed).permutation(len(tagged))
    return tuple(tagged[int(index)] for index in order)
