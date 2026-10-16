"""Replay the frozen M0b stream through a Laya typed-decision backend."""

from __future__ import annotations

import json
import os
from pathlib import Path

from experiments.m0b.backend import JevBackend, LayaBackend, action_question
from experiments.m0b.data import assign_roles, build_stream, load_cases, paraphrase_state
from experiments.m0b.evaluate import (
    cost,
    detection,
    first_flag,
    global_flat,
    lock_threshold,
    refit_thresholds,
    second_half_ok,
    silent_candidates,
    slice_mask,
    stratified_mask,
    summarize,
    uniform_mask,
    verdict,
)
from experiments.m0b.plot import write_figures
from experiments.m0b.protocol import (
    JEV_CAP,
    MIDPOINT,
    MODEL_ID,
    POLICY_TIE_ORDER,
    PREDICTION_COMMIT,
    QUESTION,
    SEED_PRIMARY,
    SENSITIVITY_SEEDS,
    SLICE_ORDER,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "m0b"
CACHE_PATH = OUT_DIR / "prediction-cache.json"


def main() -> None:
    train = load_cases("train")
    test = load_cases("test")
    if len(train) != 300:
        raise RuntimeError(f"train split is {len(train)}, protocol expects 300")
    needed = _needed_keys(train)
    cache = _load_cache()
    missing = [key for key in needed if key not in cache]
    if missing or os.environ.get("M0B_RECOMPUTE") == "1":
        cache = _fill_cache(train, test, cache, missing)
        _save_cache(cache)

    primary_roles = assign_roles(len(train), SEED_PRIMARY)
    calibration = _score_indexes(train, primary_roles["calibration"], cache)
    locked = lock_threshold(calibration["confidence"], calibration["correct"])
    threshold = float(locked["threshold"])
    stream = _score_stream(build_stream(train, primary_roles, SEED_PRIMARY), cache)
    report = summarize(stream["slices"], stream["confidence"], stream["correct"], threshold)
    flat, confidence_shift, coverage_shift = global_flat(
        report,
        calibration["mean_confidence"],
        float(locked["calibration_coverage"]),
    )
    candidates = silent_candidates(report)
    policies = _policies(stream, SEED_PRIMARY)
    choice = _choose_policy(stream, policies, candidates if flat else ())
    refit = _apply_refit(stream, policies, choice, threshold)
    passed = bool(
        locked["status"] == "primary"
        and flat
        and choice["slice"] is not None
        and refit["restored"]
    )
    sensitivity = tuple(
        _sensitivity(train, cache, seed) for seed in SENSITIVITY_SEEDS
    )
    sensitivity_passes = sum(1 for row in sensitivity if row["pass"])
    outcome = verdict(passed, sensitivity_passes)
    test_report = _score_test(test, cache, threshold)
    jev = _maybe_jev(train, primary_roles["calibration"])
    figure_paths = write_figures(
        OUT_DIR / "figures",
        report,
        {name: policies[name]["midpoint"] for name in POLICY_TIE_ORDER},
        refit["before"],
        refit["after"],
    )
    payload = {
        "prediction_commit": PREDICTION_COMMIT,
        "model": MODEL_ID,
        "teacher": "typed_decisions_action_label",
        "teacher_note": "No LLM API key was set. Gold action labels stand in for the teacher.",
        "locked_threshold": locked,
        "calibration_mean_confidence": calibration["mean_confidence"],
        "primary_report": report,
        "confidence_shift": confidence_shift,
        "coverage_shift": coverage_shift,
        "global_flat": flat,
        "silent_candidates": candidates,
        "policies": policies,
        "chosen_policy": choice,
        "refit": refit,
        "primary_pass": passed,
        "sensitivity": sensitivity,
        "sensitivity_passes": sensitivity_passes,
        "test_report": test_report,
        "jev": jev,
        "cost": {
            name: cost(len(policies[name]["sampled"])) for name in POLICY_TIE_ORDER
        },
        "figures": [str(Path(path).relative_to(ROOT)) for path in figure_paths],
        "verdict": outcome,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": outcome, "primary_pass": passed, "candidates": list(candidates), "choice": _jsonable(choice), "sensitivity_passes": sensitivity_passes}, indent=2))


def _needed_keys(train: tuple[dict[str, object], ...]) -> set[str]:
    keys = {f"{case['id']}|original" for case in train}
    for seed in (SEED_PRIMARY,) + SENSITIVITY_SEEDS:
        roles = assign_roles(len(train), seed)
        for index in roles["paraphrase"]:
            keys.add(f"{train[index]['id']}|paraphrase")
    return keys


def _fill_cache(train, test, cache: dict, missing: list[str]) -> dict:
    backend = LayaBackend()
    jobs: list[tuple[str, dict, dict]] = []
    by_id = {str(case["id"]): case for case in train}
    for case in test:
        by_id[str(case["id"])] = case
        missing.append(f"{case['id']}|original")
    seen: set[str] = set()
    for key in missing:
        if key in seen or key in cache:
            continue
        seen.add(key)
        case_id, variant = key.split("|", 1)
        case = by_id[case_id]
        state = case["state"]
        if not isinstance(state, dict):
            raise RuntimeError(f"{case_id} state is not an object")
        if variant == "paraphrase":
            state = paraphrase_state(state)
        questions = case["questions"]
        if not isinstance(questions, dict):
            raise RuntimeError(f"{case_id} questions are not an object")
        jobs.append((key, state, action_question(questions)))
    filled = dict(cache)
    for number, (key, state, questions) in enumerate(jobs, start=1):
        if number == 1 or number % 25 == 0:
            print(f"laya {number}/{len(jobs)}", flush=True)
        answer = backend.decide(state, questions)[QUESTION]
        filled[key] = answer
    return filled


def _score_indexes(cases, indexes: tuple[int, ...], cache: dict) -> dict[str, object]:
    confidence: list[float] = []
    correct: list[bool] = []
    for index in indexes:
        case = cases[index]
        answer = cache[f"{case['id']}|original"]
        confidence.append(float(answer["answer_confidence"]))
        correct.append(str(answer["choice"]) == str(case["gold"]))
    return {
        "confidence": confidence,
        "correct": correct,
        "mean_confidence": float(sum(confidence) / len(confidence)),
    }


def _score_stream(stream: tuple[dict[str, object], ...], cache: dict) -> dict[str, list]:
    slices: list[str] = []
    confidence: list[float] = []
    correct: list[bool] = []
    for row in stream:
        answer = cache[f"{row['id']}|{row['variant']}"]
        slices.append(str(row["slice"]))
        confidence.append(float(answer["answer_confidence"]))
        correct.append(str(answer["choice"]) == str(row["gold"]))
    return {"slices": slices, "confidence": confidence, "correct": correct}


def _policies(stream: dict[str, list], seed: int) -> dict[str, dict]:
    sampled = {
        "uniform": uniform_mask(seed),
        "confidence-stratified": stratified_mask(stream["confidence"], seed),
        "slice-targeted": slice_mask(stream["slices"], seed),
    }
    reports = {}
    for name, indexes in sampled.items():
        reports[name] = {
            "sampled": indexes,
            "midpoint": detection(stream["slices"], stream["correct"], indexes, MIDPOINT),
            "end": detection(stream["slices"], stream["correct"], indexes, len(stream["slices"])),
            "first_flag": {
                slice_name: first_flag(stream["slices"], stream["correct"], indexes, slice_name)
                for slice_name in SLICE_ORDER
            },
        }
    return reports


def _choose_policy(stream, policies, candidates: tuple[str, ...]) -> dict[str, object]:
    best: tuple[int, int, str, str] | None = None
    for slice_name in candidates:
        for order, policy in enumerate(POLICY_TIE_ORDER):
            flag_at = policies[policy]["first_flag"][slice_name]
            if flag_at is None or flag_at > MIDPOINT:
                continue
            ranked = (flag_at, order, policy, slice_name)
            if best is None or ranked < best:
                best = ranked
    if best is None:
        return {"policy": None, "slice": None, "flag_at": None}
    return {"policy": best[2], "slice": best[3], "flag_at": best[0]}


def _apply_refit(stream, policies, choice: dict[str, object], locked: float) -> dict[str, object]:
    before = summarize(
        stream["slices"][MIDPOINT:],
        stream["confidence"][MIDPOINT:],
        stream["correct"][MIDPOINT:],
        locked,
    )
    if choice["policy"] is None or choice["slice"] is None:
        return {"thresholds": {}, "before": before, "after": before, "restored": False, "flagged": []}
    policy = str(choice["policy"])
    flagged = [
        name
        for name in SLICE_ORDER
        if bool(policies[policy]["midpoint"][name]["flags"])
    ]
    thresholds = refit_thresholds(
        stream["slices"],
        stream["confidence"],
        stream["correct"],
        policies[policy]["sampled"],
        flagged,
        locked,
    )
    per_row = [thresholds[name] for name in stream["slices"][MIDPOINT:]]
    after = summarize(
        stream["slices"][MIDPOINT:],
        stream["confidence"][MIDPOINT:],
        stream["correct"][MIDPOINT:],
        per_row,
    )
    first = summarize(
        stream["slices"][:MIDPOINT],
        stream["confidence"][:MIDPOINT],
        stream["correct"][:MIDPOINT],
        locked,
    )
    return {
        "thresholds": thresholds,
        "flagged": flagged,
        "before": before,
        "after": after,
        "restored": second_half_ok(after, float(first["control"]["coverage"]), str(choice["slice"])),
    }


def _sensitivity(train, cache: dict, seed: int) -> dict[str, object]:
    roles = assign_roles(len(train), seed)
    calibration = _score_indexes(train, roles["calibration"], cache)
    locked = lock_threshold(calibration["confidence"], calibration["correct"])
    stream = _score_stream(build_stream(train, roles, seed), cache)
    report = summarize(
        stream["slices"],
        stream["confidence"],
        stream["correct"],
        float(locked["threshold"]),
    )
    flat, confidence_shift, coverage_shift = global_flat(
        report,
        float(calibration["mean_confidence"]),
        float(locked["calibration_coverage"]),
    )
    candidates = silent_candidates(report) if flat else ()
    return {
        "seed": seed,
        "status": locked["status"],
        "pass": bool(candidates),
        "candidates": candidates,
        "confidence_shift": confidence_shift,
        "coverage_shift": coverage_shift,
        "global_accepted_accuracy": report["global"]["accepted_accuracy"],
        "report": report,
    }


def _score_test(test, cache: dict, threshold: float) -> dict[str, object]:
    confidence = []
    correct = []
    for case in test:
        key = f"{case['id']}|original"
        if key not in cache:
            continue
        answer = cache[key]
        confidence.append(float(answer["answer_confidence"]))
        correct.append(str(answer["choice"]) == str(case["gold"]))
    report = summarize(["test"] * len(confidence), confidence, correct, threshold)
    return report["global"]


def _maybe_jev(train, calibration_indexes: tuple[int, ...]) -> dict[str, object]:
    api_key = os.environ.get("JEV_API_KEY", "")
    if not api_key:
        return {"ran": False, "reason": "JEV_API_KEY unset"}
    backend = JevBackend(api_key, cap=JEV_CAP)
    rows = []
    for index in calibration_indexes[:JEV_CAP]:
        case = train[index]
        state = case["state"]
        if not isinstance(state, dict):
            continue
        questions = case["questions"]
        if not isinstance(questions, dict):
            continue
        answer = backend.decide(state, action_question(questions))[QUESTION]
        rows.append(
            {
                "id": case["id"],
                "choice": answer["choice"],
                "answer_confidence": answer["answer_confidence"],
                "gold": case["gold"],
            }
        )
    return {"ran": True, "calls": backend.calls, "rows": rows}


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
