"""Replay the frozen M0 stream. The proceed rule lives in docs/M0-predictions.md."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.m0.data import build_stream, load_locale, pools
from experiments.m0.evaluate import (
    cost_report,
    evaluate_gates,
    full_shadow,
    sensitivity_core,
    slice_report,
    verdict_from_gates,
)
from experiments.m0.model import fit_decision_model, lock_threshold, predict
from experiments.m0.plot import write_figures
from experiments.m0.protocol import (
    PREDICTION_COMMIT,
    SEED_PRIMARY,
    SENSITIVITY_SEEDS,
    STREAM_SIZE,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "m0"
FIGURE_DIR = OUT_DIR / "figures"


def main() -> None:
    english = load_locale("en-US")
    spanish = load_locale("es-ES")
    source = pools(english, spanish)
    _check_pool_sizes(source)

    fit_rows = source["fit"]
    vectorizer, classifier = fit_decision_model(
        [row["text"] for row in fit_rows],
        [row["intent"] for row in fit_rows],
    )
    calibration_rows = source["calibration"]
    calibration_scored = predict(
        vectorizer,
        classifier,
        [row["text"] for row in calibration_rows],
    )
    locked = lock_threshold(
        calibration_scored,
        [row["intent"] for row in calibration_rows],
    )
    threshold = float(locked["threshold"])
    calibration_correct = [
        prediction == row["intent"]
        for (prediction, _confidence), row in zip(
            calibration_scored, calibration_rows, strict=True
        )
    ]
    calibration_confidence = [score for _prediction, score in calibration_scored]
    calibration_mean = sum(calibration_confidence) / len(calibration_confidence)

    primary = _score_stream(
        build_stream(source, SEED_PRIMARY),
        vectorizer,
        classifier,
        threshold,
        SEED_PRIMARY,
    )
    gates = evaluate_gates(
        calibration_status=str(locked["status"]),
        calibration_confidence=calibration_mean,
        calibration_coverage=float(locked["calibration_coverage"]),
        report=primary["report"],
        shadow_at_midpoint=primary["shadow"]["0.05"]["midpoint"],
    )
    sensitivity = tuple(
        _sensitivity_seed(
            source,
            vectorizer,
            classifier,
            threshold,
            seed,
            calibration_mean,
            float(locked["calibration_coverage"]),
        )
        for seed in SENSITIVITY_SEEDS
    )
    sensitivity_passes = sum(1 for row in sensitivity if bool(row["pass"]))
    verdict = verdict_from_gates(gates, sensitivity_passes)

    figure_paths = write_figures(
        FIGURE_DIR,
        primary["report"],
        calibration_confidence,
        calibration_correct,
        primary["confidence"],
        primary["correct"],
        primary["shadow"]["0.05"]["end"],
    )
    payload = {
        "prediction_commit": PREDICTION_COMMIT,
        "teacher": "massive_gold_labels",
        "teacher_note": "No LLM API key was set. Gold MASSIVE intents stand in for the teacher.",
        "model": "tfidf_word_1_2_logistic_regression",
        "locked_threshold": locked,
        "calibration_mean_confidence": calibration_mean,
        "calibration_n": len(calibration_rows),
        "fit_n": len(fit_rows),
        "label_set_size": int(len(classifier.classes_)),
        "primary_seed": SEED_PRIMARY,
        "primary_report": primary["report"],
        "primary_shadow": primary["shadow"],
        "primary_gates": gates,
        "sensitivity": sensitivity,
        "sensitivity_passes": sensitivity_passes,
        "cost": {
            "0.02": cost_report(STREAM_SIZE, primary["n_escalated"], 0.02),
            "0.05": cost_report(STREAM_SIZE, primary["n_escalated"], 0.05),
        },
        "mean_characters": primary["mean_characters"],
        "figures": [str(Path(path).relative_to(ROOT)) for path in figure_paths],
        "verdict": verdict,
        "p7": sensitivity_passes >= 3,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "gates": _jsonable(gates), "sensitivity_passes": sensitivity_passes}, indent=2))


def _check_pool_sizes(source: dict[str, tuple[dict[str, str], ...]]) -> None:
    required = {
        "fit": 1,
        "calibration": 1,
        "en_test_other": 1600 + 400 + 300,
        "cooking": 300,
        "es_test_other": 400,
    }
    for name, minimum in required.items():
        if len(source[name]) < minimum:
            raise RuntimeError(f"{name} has {len(source[name])} rows, need {minimum}")


def _score_stream(stream, vectorizer, classifier, threshold: float, seed: int) -> dict[str, object]:
    scored = predict(vectorizer, classifier, [row["text"] for row in stream])
    slices = [row["slice"] for row in stream]
    gold = [row["intent"] for row in stream]
    confidence = [score for _prediction, score in scored]
    correct = [
        prediction == label
        for (prediction, _score), label in zip(scored, gold, strict=True)
    ]
    accepted = [score >= threshold for score in confidence]
    return {
        "report": slice_report(slices, correct, confidence, accepted),
        "shadow": full_shadow(slices, correct, accepted, seed),
        "confidence": confidence,
        "correct": correct,
        "n_escalated": sum(1 for flag in accepted if not flag),
        "mean_characters": {
            name: _mean_characters(stream, name) for name in ("global",) + tuple(dict.fromkeys(slices))
        },
    }


def _mean_characters(stream, slice_name: str) -> float:
    texts = [
        row["text"]
        for row in stream
        if slice_name == "global" or row["slice"] == slice_name
    ]
    return float(sum(len(text) for text in texts) / len(texts))


def _sensitivity_seed(source, vectorizer, classifier, threshold, seed, calibration_mean, calibration_coverage):
    scored = _score_stream(build_stream(source, seed), vectorizer, classifier, threshold, seed)
    summary = sensitivity_core(scored["report"], calibration_mean, calibration_coverage)
    summary["seed"] = seed
    summary["shadow_0.05_midpoint_flags"] = {
        name: bool(row["flags"])
        for name, row in scored["shadow"]["0.05"]["midpoint"].items()
    }
    summary["report"] = scored["report"]
    return summary


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value
    return value


if __name__ == "__main__":
    main()
