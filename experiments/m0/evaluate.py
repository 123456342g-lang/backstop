"""Slice metrics, shadow estimates, and the frozen proceed gates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.m0.protocol import (
    MIDPOINT,
    P2_MAX_ACCEPTED_ACCURACY,
    P2_MIN_ACCEPTED,
    P3_MIN_GLOBAL_ACCURACY,
    P4_MAX_CONFIDENCE_SHIFT,
    P4_MAX_COVERAGE_SHIFT,
    P5_MAX_CONFIDENCE_GAP,
    P5_MAX_COVERAGE_GAP,
    P6_MAX_WILSON_UPPER,
    P6_MIN_SHADOW_ACCEPTED,
    P7_MIN_SENSITIVITY_PASSES,
    SHADOW_RATES,
    SLICE_ORDER,
    TEACHER_USD_PER_CALL,
    WILSON_Z,
)


def wilson_upper(successes: int, trials: int, z: float = WILSON_Z) -> float:
    """One-sided 95% Wilson upper bound. Empty samples do not clear the gate."""
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * ((proportion * (1.0 - proportion) / trials) + z2 / (4.0 * trials * trials))
        ** 0.5
        / denominator
    )
    return float(min(1.0, center + margin))


def shadow_mask(stream_size: int, rate: float, stream_seed: int) -> np.ndarray:
    generator = np.random.default_rng(stream_seed * 1000 + int(rate * 100))
    return generator.random(stream_size) < rate


def slice_report(
    slices: Sequence[str],
    correct: Sequence[bool],
    confidence: Sequence[float],
    accepted: Sequence[bool],
) -> dict[str, dict[str, float | int | None]]:
    names = ("global",) + tuple(dict.fromkeys(slices))
    reports: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        indexes = [
            index
            for index, slice_name in enumerate(slices)
            if name == "global" or slice_name == name
        ]
        reports[name] = _summarize(indexes, correct, confidence, accepted)
    return reports


def _summarize(
    indexes: Sequence[int],
    correct: Sequence[bool],
    confidence: Sequence[float],
    accepted: Sequence[bool],
) -> dict[str, float | int | None]:
    count = len(indexes)
    if count == 0:
        return {
            "n": 0,
            "n_accepted": 0,
            "coverage": 0.0,
            "mean_confidence": None,
            "accuracy": None,
            "accepted_accuracy": None,
        }
    accepted_indexes = [index for index in indexes if accepted[index]]
    confidences = [confidence[index] for index in indexes]
    accuracy_values = [correct[index] for index in indexes]
    accepted_correct = [correct[index] for index in accepted_indexes]
    return {
        "n": count,
        "n_accepted": len(accepted_indexes),
        "coverage": len(accepted_indexes) / count,
        "mean_confidence": float(sum(confidences) / count),
        "accuracy": float(sum(accuracy_values) / count),
        "accepted_accuracy": (
            float(sum(accepted_correct) / len(accepted_correct))
            if accepted_correct
            else None
        ),
    }


def shadow_report(
    slices: Sequence[str],
    correct: Sequence[bool],
    accepted: Sequence[bool],
    mask: np.ndarray,
    limit: int,
) -> dict[str, dict[str, float | int | None]]:
    """Teacher-checked accepts inside the first `limit` rows."""
    names = ("global",) + SLICE_ORDER
    reports: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        checked = [
            index
            for index in range(min(limit, len(slices)))
            if mask[index]
            and accepted[index]
            and (name == "global" or slices[index] == name)
        ]
        hits = sum(1 for index in checked if correct[index])
        trials = len(checked)
        point = float(hits / trials) if trials else None
        reports[name] = {
            "n_shadow_accepted": trials,
            "accepted_accuracy": point,
            "wilson_upper": wilson_upper(hits, trials),
            "flags": bool(
                trials >= P6_MIN_SHADOW_ACCEPTED
                and wilson_upper(hits, trials) < P6_MAX_WILSON_UPPER
            ),
        }
    return reports


def failing_slices(
    report: dict[str, dict[str, float | int | None]],
) -> tuple[str, ...]:
    found: list[str] = []
    for name in SLICE_ORDER:
        row = report[name]
        accuracy = row["accepted_accuracy"]
        accepted_n = int(row["n_accepted"])
        if (
            accuracy is not None
            and accepted_n >= P2_MIN_ACCEPTED
            and float(accuracy) <= P2_MAX_ACCEPTED_ACCURACY
        ):
            found.append(name)
    return tuple(found)


def teacher_necessary(
    report: dict[str, dict[str, float | int | None]],
    slice_name: str,
) -> bool:
    candidate = report[slice_name]
    baseline = report["id"]
    if candidate["mean_confidence"] is None or baseline["mean_confidence"] is None:
        return False
    coverage_gap = abs(float(candidate["coverage"]) - float(baseline["coverage"]))
    confidence_gap = abs(
        float(candidate["mean_confidence"]) - float(baseline["mean_confidence"])
    )
    return (
        coverage_gap <= P5_MAX_COVERAGE_GAP
        and confidence_gap <= P5_MAX_CONFIDENCE_GAP
    )


def evaluate_gates(
    *,
    calibration_status: str,
    calibration_confidence: float,
    calibration_coverage: float,
    report: dict[str, dict[str, float | int | None]],
    shadow_at_midpoint: dict[str, dict[str, float | int | None]],
) -> dict[str, object]:
    """Existence reading: one slice must satisfy P2, P5, and P6 together."""
    candidates = failing_slices(report)
    hidden = [
        name
        for name in candidates
        if teacher_necessary(report, name)
    ]
    detected = [
        name
        for name in hidden
        if bool(shadow_at_midpoint[name]["flags"])
    ]
    global_accuracy = report["global"]["accepted_accuracy"]
    prod_confidence = report["global"]["mean_confidence"]
    prod_coverage = float(report["global"]["coverage"])
    confidence_shift = (
        abs(float(prod_confidence) - calibration_confidence)
        if prod_confidence is not None
        else 1.0
    )
    coverage_shift = abs(prod_coverage - calibration_coverage)
    p1 = calibration_status == "primary"
    p2 = len(candidates) > 0
    p3 = (
        global_accuracy is not None
        and float(global_accuracy) >= P3_MIN_GLOBAL_ACCURACY
    )
    p4 = (
        confidence_shift <= P4_MAX_CONFIDENCE_SHIFT
        and coverage_shift <= P4_MAX_COVERAGE_SHIFT
    )
    p5 = len(hidden) > 0
    p6 = len(detected) > 0
    return {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "P6": p6,
        "candidates": candidates,
        "teacher_necessary_slices": tuple(hidden),
        "shadow_detected_slices": tuple(detected),
        "confidence_shift": confidence_shift,
        "coverage_shift": coverage_shift,
        "core_pass": bool(p1 and p2 and p3 and p4 and p5 and p6),
    }


def sensitivity_core(
    report: dict[str, dict[str, float | int | None]],
    calibration_confidence: float,
    calibration_coverage: float,
) -> dict[str, object]:
    """P2–P5 on a resampled stream. The locked model and threshold stay fixed."""
    candidates = failing_slices(report)
    hidden = tuple(name for name in candidates if teacher_necessary(report, name))
    global_accuracy = report["global"]["accepted_accuracy"]
    prod_confidence = report["global"]["mean_confidence"]
    confidence_shift = (
        abs(float(prod_confidence) - calibration_confidence)
        if prod_confidence is not None
        else 1.0
    )
    coverage_shift = abs(float(report["global"]["coverage"]) - calibration_coverage)
    p2 = len(candidates) > 0
    p3 = (
        global_accuracy is not None
        and float(global_accuracy) >= P3_MIN_GLOBAL_ACCURACY
    )
    p4 = (
        confidence_shift <= P4_MAX_CONFIDENCE_SHIFT
        and coverage_shift <= P4_MAX_COVERAGE_SHIFT
    )
    p5 = len(hidden) > 0
    return {
        "P2": p2,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "pass": bool(p2 and p3 and p4 and p5),
        "candidates": candidates,
        "teacher_necessary_slices": hidden,
        "confidence_shift": confidence_shift,
        "coverage_shift": coverage_shift,
        "global_accepted_accuracy": global_accuracy,
    }


def cost_report(
    stream_size: int,
    n_escalated: int,
    rate: float,
) -> dict[str, float]:
    unit = TEACHER_USD_PER_CALL
    llm_only = stream_size * unit
    gate_only = n_escalated * unit
    shadow = rate * stream_size * unit
    return {
        "teacher_usd_per_call": unit,
        "llm_only_usd": llm_only,
        "gate_only_usd": gate_only,
        "shadow_usd": shadow,
        "gate_plus_shadow_usd": gate_only + shadow,
        "shadow_fraction_of_llm_only": rate,
    }


def full_shadow(
    slices: Sequence[str],
    correct: Sequence[bool],
    accepted: Sequence[bool],
    stream_seed: int,
) -> dict[str, dict[str, dict[str, dict[str, float | int | None]]]]:
    reports: dict[str, dict[str, dict[str, dict[str, float | int | None]]]] = {}
    for rate in SHADOW_RATES:
        mask = shadow_mask(len(slices), rate, stream_seed)
        key = f"{rate:.2f}"
        reports[key] = {
            "midpoint": shadow_report(slices, correct, accepted, mask, MIDPOINT),
            "end": shadow_report(slices, correct, accepted, mask, len(slices)),
        }
    return reports


def verdict_from_gates(
    primary: dict[str, object],
    sensitivity_passes: int,
) -> str:
    p7 = sensitivity_passes >= P7_MIN_SENSITIVITY_PASSES
    if bool(primary["core_pass"]) and p7:
        return "PROCEED"
    return "KILL"
