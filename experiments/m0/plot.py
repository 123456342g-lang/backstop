"""Figures for the frozen M0 stream. Gates are not decided from these pictures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.m0.protocol import SLICE_ORDER, TARGET_ACCURACY


def write_figures(
    destination: Path,
    report: Mapping[str, Mapping[str, float | int | None]],
    calibration_confidence: Sequence[float],
    calibration_correct: Sequence[bool],
    production_confidence: Sequence[float],
    production_correct: Sequence[bool],
    shadow_end: Mapping[str, Mapping[str, float | int | None]],
) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = (
        _accepted_accuracy(destination / "accepted-accuracy.png", report),
        _coverage_and_confidence(destination / "coverage-confidence.png", report),
        _reliability(
            destination / "reliability.png",
            calibration_confidence,
            calibration_correct,
            production_confidence,
            production_correct,
        ),
        _shadow(destination / "shadow-5pct.png", shadow_end),
    )
    return paths


def _accepted_accuracy(
    path: Path,
    report: Mapping[str, Mapping[str, float | int | None]],
) -> str:
    names = ("global",) + SLICE_ORDER
    values = [
        0.0 if report[name]["accepted_accuracy"] is None else float(report[name]["accepted_accuracy"])
        for name in names
    ]
    labels = [
        f"{name}\nn={report[name]['n_accepted']}"
        for name in names
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#243044" if name == "global" else "#3d7a6a" for name in names]
    axis.bar(labels, values, color=colors)
    axis.axhline(TARGET_ACCURACY, color="#9a3412", linestyle="--", label="target 0.95")
    axis.axhline(0.85, color="#b45309", linestyle=":", label="P2 fail line 0.85")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accepted accuracy")
    axis.set_title("Accepted accuracy by slice")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _coverage_and_confidence(
    path: Path,
    report: Mapping[str, Mapping[str, float | int | None]],
) -> str:
    names = ("global",) + SLICE_ORDER
    coverage = [float(report[name]["coverage"]) for name in names]
    confidence = [
        0.0 if report[name]["mean_confidence"] is None else float(report[name]["mean_confidence"])
        for name in names
    ]
    positions = range(len(names))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar([position - 0.18 for position in positions], coverage, width=0.36, label="coverage", color="#1f4e79")
    axis.bar(
        [position + 0.18 for position in positions],
        confidence,
        width=0.36,
        label="mean confidence",
        color="#c4a35a",
    )
    axis.set_xticks(list(positions), list(names))
    axis.set_ylim(0, 1.05)
    axis.set_title("Observable monitors: coverage and mean confidence")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _reliability(
    path: Path,
    calibration_confidence: Sequence[float],
    calibration_correct: Sequence[bool],
    production_confidence: Sequence[float],
    production_correct: Sequence[bool],
) -> str:
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for label, confidences, correct, color in (
        ("calibration", calibration_confidence, calibration_correct, "#1f4e79"),
        ("production", production_confidence, production_correct, "#9a3412"),
    ):
        centers, observed = _bins(confidences, correct)
        axis.plot(centers, observed, marker="o", label=label, color=color)
    axis.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="#94a3b8", label="perfect")
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Mean confidence in bin")
    axis.set_ylabel("Accuracy in bin")
    axis.set_title("Reliability of max-probability")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> tuple[list[float], list[float]]:
    centers: list[float] = []
    observed: list[float] = []
    for step in range(5, 10):
        low = step / 10
        high = low + 0.1
        chosen = [
            flag
            for score, flag in zip(confidences, correct, strict=True)
            if (score >= low and score < high) or (step == 9 and score == 1.0)
        ]
        if not chosen:
            continue
        centers.append((low + high) / 2)
        observed.append(sum(chosen) / len(chosen))
    return centers, observed


def _shadow(
    path: Path,
    shadow_end: Mapping[str, Mapping[str, float | int | None]],
) -> str:
    names = SLICE_ORDER
    point = [
        float("nan")
        if int(shadow_end[name]["n_shadow_accepted"]) == 0
        else float(shadow_end[name]["accepted_accuracy"])
        for name in names
    ]
    upper = [
        float("nan")
        if int(shadow_end[name]["n_shadow_accepted"]) == 0
        else float(shadow_end[name]["wilson_upper"])
        for name in names
    ]
    labels = [f"{name}\nn={shadow_end[name]['n_shadow_accepted']}" for name in names]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(labels, point, color="#3d7a6a", label="shadow point estimate")
    axis.plot(labels, upper, color="#9a3412", marker="o", linestyle="none", label="Wilson upper 95%")
    axis.axhline(0.90, color="#b45309", linestyle=":", label="P6 line 0.90")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accepted accuracy")
    axis.set_title("5% teacher sample at end of stream")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)
