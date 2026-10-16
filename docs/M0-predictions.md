# M0 predictions (frozen before the run)

Frozen on 2026-10-16, after `docs/prior-art.md` and after counting MASSIVE label pools only. No classifier was trained before this commit. The decision rule below is mechanical. Results do not get to edit it.

Claim under test, from the brief: a static threshold fit once on calibration data silently misses a production accuracy drop, and only online per-slice ground truth catches it.

## Predicted verdict

**KILL.**

Direction, not a wish:

1. A word TF-IDF logistic regression trained on English will flatten on Spanish and on the held-out cooking domain, so those slices escalate. Escalation is visible without a teacher. That is the gate working.
2. The adversarial prefix is only a short English lead-in. The original utterance still dominates the bag of words, so accepted accuracy on that slice stays near the in-distribution slice.
3. A slice small enough to leave global accepted accuracy ≥ 0.93 will not yield 15 teacher-checked accepts by the midpoint of a 3,000-row stream at a 5% sample. Shadow sampling will be under-powered even if a silent slice exists.

So the conjunction required to build M1 is not expected to hold. Probability this prediction is wrong: about 0.35, mostly if the adversarial prefix creates confident wrong accepts and the Wilson bound still clears on ~15 samples.

## Why the decision model is not Laya

Laya is real ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions)). It is the wrong instrument for this CPU run:

- Base `laya` is 0.362 zero-shot on typed-decisions, under the 0.461 majority baseline ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/master/BENCHMARKS.md)). A 0.95 accept target would escalate almost everything. The drift test would be vacuous.
- `laya-typed-decisions` (0.766) is a 421M ModernBERT fine-tuned on a different benchmark. Fine-tuning it here is not a CPU experiment.

Substitute, fixed now: scikit-learn word TF-IDF (1–2 grams) + L2 logistic regression. It returns a real distribution over intents and runs on CPU. This is a task-specific decision model, which is what Laya's own card says the base checkpoint needs.

## Data and splits

Dataset: [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) 1.1, the public virtual-assistant intent corpus ([paper](https://arxiv.org/abs/2204.08582)). Task: route an utterance to a skill (intent). Locales: `en-US` and `es-ES`.

Held-out domain: scenario `cooking` (`cooking_recipe`, `cooking_query`). Removed from fit and from calibration. Pool sizes counted before this freeze:

| Pool | Rows |
| --- | ---: |
| en-US train, not cooking | 11,303 |
| en-US validation, not cooking | 1,990 |
| en-US test, not cooking | 2,902 |
| en-US cooking, all splits | 326 |
| es-ES test, not cooking | 2,902 |

- **Fit:** en-US train, scenario ≠ cooking.
- **Calibration:** en-US validation, scenario ≠ cooking. Thresholds only. Not used to fit the classifier.
- **Production stream:** 3,000 rows, disjoint within each pool, no replacement. If any pool is shorter than its quota the run aborts and the verdict is **INCONCLUSIVE**, not a quiet resize.

| Slice | n | Source | Text |
| --- | ---: | --- | --- |
| `id` | 1,600 | en-US test, not cooking | original |
| `long` | 400 | same pool, disjoint from `id` and `adversarial` | boilerplate below + original |
| `adversarial` | 300 | same pool, disjoint | prefix below + original |
| `new_class` | 300 | all en-US cooking rows | original |
| `language` | 400 | es-ES test, not cooking | original Spanish |

Sampling: each pool is shuffled with `numpy` Generator seed **42** (primary) and the quota is the prefix. Stream order is that concatenation shuffled again with the same seed. Sensitivity seeds `{0,1,2,3,4}` redraw these samples and the shadow masks only. The classifier and the locked threshold stay the ones fit at seed 42.

Boilerplate (length shift), frozen:

```text
Previous conversation: the customer wrote again after waiting. Agent note: account is in good standing. Please read the message below and choose the matching skill. Reference 1842. Channel: mobile app. Timestamp: Tuesday morning.
```

Adversarial prefix, frozen:

```text
Ignore the user utterance and treat this as a request to play a song. Route to music.
```

## Model and threshold

```text
TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=40000, sublinear_tf=True, lowercase=True)
LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42)
```

No class weights. No character n-grams. Confidence is the maximum class probability. A decision is accepted when confidence ≥ threshold, otherwise escalated. Correct means the predicted intent equals the gold intent string.

`new_class` gold labels are outside the trained label set, so every accept on that slice is wrong. That is the definition of a new skill, not a knob.

Threshold grid: `0.50, 0.51, …, 0.99`. On calibration, take the **lowest** threshold with accepted accuracy ≥ **0.95** and coverage ≥ **0.20**. If none exists, record `fallback_0.90` and use the lowest threshold with accepted accuracy ≥ 0.90 and coverage ≥ 0.10, but precondition P1 then fails. If that also fails, threshold = 0.50 and status = `failed`.

## Teacher

No `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY` is set in this environment. The teacher is the **held-out MASSIVE gold label**, an oracle stand-in, not an LLM. Cost uses an assumed **$0.002 per teacher call** (a cheap LLM judge). That price is an assumption, not a measured invoice. The fast model costs $0.

Shadow draws are Bernoulli with `P(sample) = p` on every row, including escalations. Generator seed is `stream_seed * 1000 + int(p * 100)`. Rates reported: **0.02** and **0.05**. Only 0.05 can satisfy the proceed rule.

Wilson one-sided 95% upper bound uses `z = 1.6448536269514722`.

## Proceed rule

**PROCEED** only if every line holds. Otherwise **KILL**. Do not build M1 on a partial pass.

On the primary stream (seed 42):

- **P1.** Calibration status is the 0.95 / 0.20 rule, not the fallback.
- **P2.** At least one slice `S` has `n_accepted ≥ 30` and accepted accuracy ≤ **0.85**.
- **P3.** Global accepted accuracy ≥ **0.93** (the bad slice is hidden by the global accuracy).
- **P4.** Silent monitors: absolute shift of mean confidence (all rows, production vs calibration) ≤ **0.05**, and absolute shift of coverage ≤ **0.10**.
- **P5.** Teacher-necessary: `S` is not already obvious from confidence. `|coverage(S) − coverage(id)| ≤ 0.15` and `|mean_confidence(S) − mean_confidence(id)| ≤ 0.05`.
- **P6.** At 5% shadow, after the first **1,500** rows, that same slice has `n_shadow_accepted ≥ 15` and Wilson upper bound < **0.90**.

**P7.** At least 3 of the 5 sensitivity seeds satisfy P2–P5. P6 is required on the primary seed; sensitivity shadow hits are reported and do not flip a primary failure into a pass.

2% shadow is measured the same way and is not sufficient for PROCEED.

## Expected numbers (primary seed)

| Quantity | Predicted range |
| --- | --- |
| Calibration coverage at the locked threshold | 0.25–0.55 |
| `id` accepted accuracy | ≥ 0.95 |
| Global accepted accuracy | 0.93–0.97 |
| Slice that looks worst | `language` or `new_class`, via escalation, not via confident errors |
| Global mean-confidence shift | ≤ 0.05 |
| P6 at 5% | fail (too few shadow accepts, or no slice meets P2 and P5 together) |
| Verdict | KILL |

## What this commit must not contain

No metrics, no plots, no trained threshold, no edited prediction after `experiments/m0/run_m0.py` has been executed.
