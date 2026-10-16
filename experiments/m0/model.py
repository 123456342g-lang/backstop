"""Task-specific CPU decision model and the jevcal-style threshold lock."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiments.m0.protocol import (
    FALLBACK_ACCURACY,
    FALLBACK_COVERAGE,
    TARGET_ACCURACY,
    TARGET_COVERAGE,
)


def fit_decision_model(
    texts: Sequence[str],
    labels: Sequence[str],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=40_000,
        sublinear_tf=True,
        lowercase=True,
    )
    features = vectorizer.fit_transform(list(texts))
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    classifier.fit(features, list(labels))
    return vectorizer, classifier


def predict(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    texts: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    """Return (predicted intent, max probability) for each text."""
    probabilities = classifier.predict_proba(vectorizer.transform(list(texts)))
    classes = classifier.classes_
    chosen = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(chosen)), chosen]
    return tuple(
        (str(classes[int(index)]), float(score))
        for index, score in zip(chosen, confidence, strict=True)
    )


def _accepted_stats(
    scored: Sequence[tuple[str, float]],
    gold: Sequence[str],
    threshold: float,
) -> tuple[float, float, int]:
    accepted = [
        prediction == label
        for (prediction, confidence), label in zip(scored, gold, strict=True)
        if confidence >= threshold
    ]
    if not accepted:
        return 0.0, 0.0, 0
    accuracy = float(sum(accepted) / len(accepted))
    coverage = len(accepted) / len(scored)
    return accuracy, coverage, len(accepted)


def lock_threshold(
    scored: Sequence[tuple[str, float]],
    gold: Sequence[str],
) -> dict[str, float | str | int]:
    """Lowest grid threshold meeting the frozen accuracy and coverage floors."""
    primary = _first_threshold(scored, gold, TARGET_ACCURACY, TARGET_COVERAGE)
    if primary is not None:
        accuracy, coverage, accepted = primary[1], primary[2], primary[3]
        return {
            "threshold": primary[0],
            "status": "primary",
            "calibration_accepted_accuracy": accuracy,
            "calibration_coverage": coverage,
            "calibration_accepted": accepted,
        }
    fallback = _first_threshold(scored, gold, FALLBACK_ACCURACY, FALLBACK_COVERAGE)
    if fallback is not None:
        accuracy, coverage, accepted = fallback[1], fallback[2], fallback[3]
        return {
            "threshold": fallback[0],
            "status": "fallback_0.90",
            "calibration_accepted_accuracy": accuracy,
            "calibration_coverage": coverage,
            "calibration_accepted": accepted,
        }
    accuracy, coverage, accepted = _accepted_stats(scored, gold, 0.50)
    return {
        "threshold": 0.50,
        "status": "failed",
        "calibration_accepted_accuracy": accuracy,
        "calibration_coverage": coverage,
        "calibration_accepted": accepted,
    }


def _first_threshold(
    scored: Sequence[tuple[str, float]],
    gold: Sequence[str],
    min_accuracy: float,
    min_coverage: float,
) -> tuple[float, float, float, int] | None:
    for step in range(50, 100):
        threshold = step / 100
        accuracy, coverage, accepted = _accepted_stats(scored, gold, threshold)
        if accepted and accuracy >= min_accuracy and coverage >= min_coverage:
            return threshold, accuracy, coverage, accepted
    return None
