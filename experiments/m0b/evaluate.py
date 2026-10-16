"""Gates, sampling policies, and the per-slice threshold refit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.m0b.protocol import (
    CONFIDENCE_BINS,
    FALLBACK_ACCURACY,
    FALLBACK_COVERAGE,
    MIDPOINT,
    P2_MAX_ACCEPTED_ACCURACY,
    P2_MIN_ACCEPTED,
    P3_MAX_CONFIDENCE_SHIFT,
    P3_MAX_COVERAGE_SHIFT,
    P4_MAX_CONFIDENCE_GAP,
    P4_MAX_COVERAGE_GAP,
    P5_MAX_WILSON,
    P5_MIN_SAMPLED,
    P6_CONTROL_COVERAGE_RATIO,
    P6_ESCALATED_COVERAGE,
    P6_MIN_GLOBAL_ACCURACY,
    P7_MIN_SENSITIVITY_PASSES,
    POLICY_SEED_OFFSET,
    POLICY_TIE_ORDER,
    SAMPLE_RATE,
    SLICE_ORDER,
    STRATIFIED_DRAWS,
    STREAM_N,
    TARGET_ACCURACY,
    TARGET_COVERAGE,
    TEACHER_USD,
    WILSON_Z,
)


def wilson_upper(successes: int, trials: int, z: float = WILSON_Z) -> float:
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * ((proportion * (1.0 - proportion) / trials) + z2 / (4.0 * trials * trials)) ** 0.5
        / denominator
    )
    return float(min(1.0, center + margin))


def lock_threshold(
    confidence: Sequence[float],
    correct: Sequence[bool],
) -> dict[str, float | str | int]:
    primary = _first_threshold(confidence, correct, TARGET_ACCURACY, TARGET_COVERAGE)
    if primary is not None:
        return _lock_record(primary, "primary")
    fallback = _first_threshold(confidence, correct, FALLBACK_ACCURACY, FALLBACK_COVERAGE)
    if fallback is not None:
        return _lock_record(fallback, "fallback_0.90")
    accuracy, coverage, accepted = _accepted_stats(confidence, correct, 0.20)
    return {
        "threshold": 0.20,
        "status": "failed",
        "calibration_accepted_accuracy": accuracy,
        "calibration_coverage": coverage,
        "calibration_accepted": accepted,
    }


def _lock_record(
    found: tuple[float, float, float, int],
    status: str,
) -> dict[str, float | str | int]:
    threshold, accuracy, coverage, accepted = found
    return {
        "threshold": threshold,
        "status": status,
        "calibration_accepted_accuracy": accuracy,
        "calibration_coverage": coverage,
        "calibration_accepted": accepted,
    }


def _first_threshold(
    confidence: Sequence[float],
    correct: Sequence[bool],
    min_accuracy: float,
    min_coverage: float,
) -> tuple[float, float, float, int] | None:
    for step in range(20, 100, 5):
        threshold = step / 100
        accuracy, coverage, accepted = _accepted_stats(confidence, correct, threshold)
        if accepted and accuracy >= min_accuracy and coverage >= min_coverage:
            return threshold, accuracy, coverage, accepted
    return None


def _accepted_stats(
    confidence: Sequence[float],
    correct: Sequence[bool],
    threshold: float,
) -> tuple[float, float, int]:
    chosen = [
        flag
        for score, flag in zip(confidence, correct, strict=True)
        if score >= threshold
    ]
    if not chosen:
        return 0.0, 0.0, 0
    return float(sum(chosen) / len(chosen)), len(chosen) / len(confidence), len(chosen)


def summarize(
    slices: Sequence[str],
    confidence: Sequence[float],
    correct: Sequence[bool],
    threshold: float | Sequence[float],
) -> dict[str, dict[str, float | int | None]]:
    thresholds = (
        [float(threshold)] * len(slices)
        if isinstance(threshold, float)
        else [float(value) for value in threshold]
    )
    accepted = [score >= limit for score, limit in zip(confidence, thresholds, strict=True)]
    names = ("global",) + tuple(dict.fromkeys(slices))
    return {
        name: _group(slices, confidence, correct, accepted, name)
        for name in names
    }


def _group(
    slices: Sequence[str],
    confidence: Sequence[float],
    correct: Sequence[bool],
    accepted: Sequence[bool],
    name: str,
) -> dict[str, float | int | None]:
    indexes = [
        index
        for index, slice_name in enumerate(slices)
        if name == "global" or slice_name == name
    ]
    if not indexes:
        return {
            "n": 0,
            "n_accepted": 0,
            "coverage": 0.0,
            "mean_confidence": None,
            "accuracy": None,
            "accepted_accuracy": None,
        }
    accepted_indexes = [index for index in indexes if accepted[index]]
    accepted_correct = [correct[index] for index in accepted_indexes]
    return {
        "n": len(indexes),
        "n_accepted": len(accepted_indexes),
        "coverage": len(accepted_indexes) / len(indexes),
        "mean_confidence": float(sum(confidence[index] for index in indexes) / len(indexes)),
        "accuracy": float(sum(correct[index] for index in indexes) / len(indexes)),
        "accepted_accuracy": (
            float(sum(accepted_correct) / len(accepted_correct)) if accepted_correct else None
        ),
    }


def uniform_mask(stream_seed: int, size: int = STREAM_N) -> tuple[int, ...]:
    generator = np.random.default_rng(stream_seed * 1000 + POLICY_SEED_OFFSET["uniform"])
    return tuple(index for index, draw in enumerate(generator.random(size)) if draw < SAMPLE_RATE)


def stratified_mask(confidence: Sequence[float], stream_seed: int) -> tuple[int, ...]:
    grouped = _bin_indexes(confidence, CONFIDENCE_BINS)
    generator = np.random.default_rng(
        stream_seed * 1000 + POLICY_SEED_OFFSET["confidence-stratified"]
    )
    return tuple(_draw_even(grouped, STRATIFIED_DRAWS, generator))


def slice_mask(slices: Sequence[str], stream_seed: int) -> tuple[int, ...]:
    grouped = [
        [index for index, name in enumerate(slices) if name == slice_name]
        for slice_name in SLICE_ORDER
    ]
    generator = np.random.default_rng(stream_seed * 1000 + POLICY_SEED_OFFSET["slice-targeted"])
    return tuple(_draw_even(grouped, STRATIFIED_DRAWS, generator))


def _bin_indexes(confidence: Sequence[float], edges: Sequence[float]) -> list[list[int]]:
    grouped = [[] for _ in range(len(edges) - 1)]
    for index, score in enumerate(confidence):
        for bin_index, (low, high) in enumerate(zip(edges, edges[1:], strict=True)):
            if low <= score < high:
                grouped[bin_index].append(index)
                break
    return grouped


def _draw_even(
    groups: Sequence[Sequence[int]],
    draws: int,
    generator: np.random.Generator,
) -> list[int]:
    nonempty = [list(group) for group in groups if group]
    if not nonempty or draws <= 0:
        return []
    base, remainder = divmod(draws, len(nonempty))
    quota = [base + (1 if index < remainder else 0) for index in range(len(nonempty))]
    for index, group in enumerate(nonempty):
        if quota[index] <= len(group):
            continue
        extra = quota[index] - len(group)
        quota[index] = len(group)
        for step in range(1, len(nonempty)):
            other = (index + step) % len(nonempty)
            room = len(nonempty[other]) - quota[other]
            take = min(room, extra)
            quota[other] += take
            extra -= take
            if extra == 0:
                break
    chosen: list[int] = []
    for group, count in zip(nonempty, quota, strict=True):
        picks = generator.permutation(len(group))[:count]
        chosen.extend(group[int(pick)] for pick in picks)
    return chosen


def detection(
    slices: Sequence[str],
    correct: Sequence[bool],
    sampled: Sequence[int],
    limit: int,
) -> dict[str, dict[str, float | int | bool | None]]:
    sampled_set = set(sampled)
    reports: dict[str, dict[str, float | int | bool | None]] = {}
    for name in SLICE_ORDER:
        indexes = [
            index
            for index in range(min(limit, len(slices)))
            if index in sampled_set and slices[index] == name
        ]
        hits = sum(1 for index in indexes if correct[index])
        trials = len(indexes)
        upper = wilson_upper(hits, trials)
        reports[name] = {
            "n_sampled": trials,
            "accuracy": float(hits / trials) if trials else None,
            "wilson_upper": upper,
            "flags": bool(trials >= P5_MIN_SAMPLED and upper < P5_MAX_WILSON),
        }
    return reports


def first_flag(
    slices: Sequence[str],
    correct: Sequence[bool],
    sampled: Sequence[int],
    slice_name: str,
) -> int | None:
    for limit in range(P5_MIN_SAMPLED, len(slices) + 1):
        if bool(detection(slices, correct, sampled, limit)[slice_name]["flags"]):
            return limit
    return None


def refit_thresholds(
    slices: Sequence[str],
    confidence: Sequence[float],
    correct: Sequence[bool],
    sampled: Sequence[int],
    flagged: Sequence[str],
    locked: float,
) -> dict[str, float]:
    sampled_set = {index for index in sampled if index < MIDPOINT}
    flagged_set = set(flagged)
    thresholds = {name: locked for name in SLICE_ORDER}
    for name in SLICE_ORDER:
        indexes = [
            index
            for index in range(MIDPOINT)
            if index in sampled_set and slices[index] == name
        ]
        if len(indexes) < P5_MIN_SAMPLED:
            continue
        chosen = _best_threshold(
            [confidence[index] for index in indexes],
            [correct[index] for index in indexes],
            escalate_if_empty=name in flagged_set,
        )
        if chosen is not None:
            thresholds[name] = chosen
    return thresholds


def _best_threshold(
    confidence: Sequence[float],
    correct: Sequence[bool],
    escalate_if_empty: bool,
) -> float | None:
    found: float | None = None
    for step in range(20, 100, 5):
        threshold = step / 100
        accepted = [
            flag
            for score, flag in zip(confidence, correct, strict=True)
            if score >= threshold
        ]
        if accepted and (sum(accepted) / len(accepted)) >= TARGET_ACCURACY:
            found = threshold
            break
    if found is not None:
        return found
    if escalate_if_empty:
        return 1.01
    return None


def silent_candidates(
    report: dict[str, dict[str, float | int | None]],
) -> tuple[str, ...]:
    found: list[str] = []
    control = report["control"]
    for name in SLICE_ORDER:
        if name == "control":
            continue
        row = report[name]
        accuracy = row["accepted_accuracy"]
        if (
            accuracy is None
            or int(row["n_accepted"]) < P2_MIN_ACCEPTED
            or float(accuracy) > P2_MAX_ACCEPTED_ACCURACY
        ):
            continue
        if row["mean_confidence"] is None or control["mean_confidence"] is None:
            continue
        confidence_gap = abs(float(row["mean_confidence"]) - float(control["mean_confidence"]))
        coverage_gap = abs(float(row["coverage"]) - float(control["coverage"]))
        if confidence_gap <= P4_MAX_CONFIDENCE_GAP and coverage_gap <= P4_MAX_COVERAGE_GAP:
            found.append(name)
    return tuple(found)


def global_flat(
    report: dict[str, dict[str, float | int | None]],
    calibration_confidence: float,
    calibration_coverage: float,
) -> tuple[bool, float, float]:
    mean_confidence = report["global"]["mean_confidence"]
    confidence_shift = (
        abs(float(mean_confidence) - calibration_confidence)
        if mean_confidence is not None
        else 1.0
    )
    coverage_shift = abs(float(report["global"]["coverage"]) - calibration_coverage)
    return (
        confidence_shift <= P3_MAX_CONFIDENCE_SHIFT
        and coverage_shift <= P3_MAX_COVERAGE_SHIFT,
        confidence_shift,
        coverage_shift,
    )


def second_half_ok(
    report: dict[str, dict[str, float | int | None]],
    first_control_coverage: float,
    slice_name: str,
) -> bool:
    accuracy = report["global"]["accepted_accuracy"]
    control_coverage = float(report["control"]["coverage"])
    restored = accuracy is not None and float(accuracy) >= P6_MIN_GLOBAL_ACCURACY
    control_held = (
        first_control_coverage <= 0
        or control_coverage >= P6_CONTROL_COVERAGE_RATIO * first_control_coverage
    )
    row = report[slice_name]
    slice_fixed = (
        row["accepted_accuracy"] is not None
        and float(row["accepted_accuracy"]) >= TARGET_ACCURACY
    ) or float(row["coverage"]) <= P6_ESCALATED_COVERAGE
    return bool(restored and control_held and slice_fixed)


def verdict(primary_pass: bool, sensitivity_passes: int) -> str:
    if primary_pass and sensitivity_passes >= P7_MIN_SENSITIVITY_PASSES:
        return "PROCEED"
    return "KILL"


def cost(calls: int) -> dict[str, float | int]:
    return {"teacher_calls": calls, "usd": calls * TEACHER_USD, "usd_per_call": TEACHER_USD}
