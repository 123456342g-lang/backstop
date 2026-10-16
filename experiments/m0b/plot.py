"""Figures for the M0b replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.m0b.protocol import SLICE_ORDER, TARGET_ACCURACY


def write_figures(
    destination: Path,
    report: Mapping[str, Mapping[str, float | int | None]],
    policies: Mapping[str, Mapping[str, Mapping[str, float | int | bool | None]]],
    before: Mapping[str, Mapping[str, float | int | None]],
    after: Mapping[str, Mapping[str, float | int | None]],
) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    return (
        _accuracy(destination / "accepted-accuracy.png", report),
        _monitors(destination / "coverage-confidence.png", report),
        _sampling(destination / "sampling-midpoint.png", policies),
        _refit(destination / "refit-second-half.png", before, after),
    )


def _accuracy(path: Path, report: Mapping[str, Mapping[str, float | int | None]]) -> str:
    names = ("global",) + SLICE_ORDER
    values = [_metric(report, name, "accepted_accuracy") for name in names]
    labels = [f"{name}\nn={report[name]['n_accepted']}" for name in names]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(labels, values, color=["#243044", "#3d7a6a", "#9a3412", "#b45309", "#1f4e79"])
    axis.axhline(TARGET_ACCURACY, color="#9a3412", linestyle="--", label="target 0.95")
    axis.axhline(0.85, color="#c4a35a", linestyle=":", label="P2 line 0.85")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accepted accuracy")
    axis.set_title("Accepted accuracy by slice")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _monitors(path: Path, report: Mapping[str, Mapping[str, float | int | None]]) -> str:
    names = ("global",) + SLICE_ORDER
    positions = range(len(names))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(
        [position - 0.18 for position in positions],
        [_metric(report, name, "coverage") for name in names],
        width=0.36,
        label="coverage",
        color="#1f4e79",
    )
    axis.bar(
        [position + 0.18 for position in positions],
        [_metric(report, name, "mean_confidence") for name in names],
        width=0.36,
        label="mean answer_confidence",
        color="#c4a35a",
    )
    axis.set_xticks(list(positions), list(names))
    axis.set_ylim(0, 1.05)
    axis.set_title("Confidence and coverage")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _sampling(
    path: Path,
    policies: Mapping[str, Mapping[str, Mapping[str, float | int | bool | None]]],
) -> str:
    names = SLICE_ORDER
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    width = 0.25
    positions = list(range(len(names)))
    for offset, (policy, color) in enumerate(
        (("uniform", "#1f4e79"), ("confidence-stratified", "#3d7a6a"), ("slice-targeted", "#9a3412"))
    ):
        heights = [int(policies[policy][name]["n_sampled"]) for name in names]
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            heights,
            width=width,
            label=policy,
            color=color,
        )
    axis.axhline(4, color="#b45309", linestyle=":", label="n=4 rule")
    axis.set_xticks(positions, list(names))
    axis.set_ylabel("Teacher labels in first half")
    axis.set_title("5% sample count by slice at midpoint")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _refit(
    path: Path,
    before: Mapping[str, Mapping[str, float | int | None]],
    after: Mapping[str, Mapping[str, float | int | None]],
) -> str:
    names = ("global",) + SLICE_ORDER
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(
        [position - 0.18 for position in positions],
        [_metric(before, name, "accepted_accuracy") for name in names],
        width=0.36,
        label="second half, locked threshold",
        color="#94a3b8",
    )
    axis.bar(
        [position + 0.18 for position in positions],
        [_metric(after, name, "accepted_accuracy") for name in names],
        width=0.36,
        label="second half, after refit",
        color="#3d7a6a",
    )
    axis.axhline(TARGET_ACCURACY, color="#9a3412", linestyle="--", label="target 0.95")
    axis.set_xticks(positions, list(names))
    axis.set_ylim(0, 1.05)
    axis.set_title("Auto-refit on the second half")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _metric(
    report: Mapping[str, Mapping[str, float | int | None]],
    name: str,
    key: str,
) -> float:
    value = report[name][key]
    if value is None:
        return 0.0
    return float(value)
