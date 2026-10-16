"""Constants locked by docs/M0b-predictions.md."""

from __future__ import annotations

PREDICTION_COMMIT = "097663bff83fb3c7b03568c9be8887b8ddc9bced"
MODEL_ID = "convaiinnovations/laya-typed-decisions"
DATASET_ID = "LocalLLaMA/typed-decisions"
DATASET_CONFIG = "customer_service"
QUESTION = "action"

SEED_PRIMARY = 42
SENSITIVITY_SEEDS = (0, 1, 2, 3, 4)

CALIBRATION_N = 60
STREAM_N = 240
MIDPOINT = 120
SLICE_QUOTA = {
    "control": 80,
    "policy": 80,
    "relabel": 40,
    "paraphrase": 40,
}
SLICE_ORDER = ("control", "policy", "relabel", "paraphrase")

TARGET_ACCURACY = 0.95
TARGET_COVERAGE = 0.20
FALLBACK_ACCURACY = 0.90
FALLBACK_COVERAGE = 0.10

P2_MAX_ACCEPTED_ACCURACY = 0.85
P2_MIN_ACCEPTED = 20
P3_MAX_CONFIDENCE_SHIFT = 0.05
P3_MAX_COVERAGE_SHIFT = 0.10
P4_MAX_CONFIDENCE_GAP = 0.08
P4_MAX_COVERAGE_GAP = 0.12
P5_MIN_SAMPLED = 4
P5_MAX_WILSON = 0.80
P6_MIN_GLOBAL_ACCURACY = 0.95
P6_CONTROL_COVERAGE_RATIO = 0.70
P6_ESCALATED_COVERAGE = 0.05
P7_MIN_SENSITIVITY_PASSES = 3

SAMPLE_RATE = 0.05
STRATIFIED_DRAWS = 12
WILSON_Z = 1.6448536269514722
TEACHER_USD = 0.002
JEV_CAP = 20
JEV_URL = "https://api.typesafe.ai/v1/systemone"

POLICY_CYCLE = {
    "answer_directly": "request_information",
    "request_information": "close_no_action",
    "close_no_action": "execute_refund",
    "execute_refund": "escalate_to_human",
    "escalate_to_human": "answer_directly",
}
RELABEL_FROM = "escalate_to_human"
RELABEL_TO = "request_information"

PARAPHRASE = (
    ("refund", "reimbursement"),
    ("Refund", "Reimbursement"),
    ("order", "purchase"),
    ("Order", "Purchase"),
    ("broken", "not working"),
    ("cancel", "end"),
    ("Cancel", "End"),
    ("charged", "billed"),
    ("package", "parcel"),
    ("login", "sign-in"),
    ("password", "passphrase"),
    ("invoice", "bill"),
    ("subscription", "plan"),
)

POLICY_SEED_OFFSET = {"uniform": 5, "confidence-stratified": 6, "slice-targeted": 7}
POLICY_TIE_ORDER = ("uniform", "confidence-stratified", "slice-targeted")
CONFIDENCE_BINS = (0.0, 0.4, 0.6, 0.8, 1.01)
