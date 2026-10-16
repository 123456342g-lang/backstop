# M0b predictions (frozen before the run)

Frozen on 2026-10-16 after M0's kill (`docs/M0-results.md`). No drifted stream has been scored. One CPU smoke call on a single `customer_service` test row returned `answer_confidence` 0.38, which is why the threshold grid starts at 0.20. That call is not a result and was not used to pick slices, rates, or the verdict.

Question: does a slice exist whose accepted accuracy falls while confidence and coverage stay flat, does a small teacher sample see it, and does a threshold refit on that sample put accepted accuracy back on target?

## Predicted verdict

**KILL.**

1. A full relabel of `action` on unchanged text will drop accepted accuracy and will not move the model's probabilities. That slice should clear the silent-failure gates. This part is nearly guaranteed by using the same forward pass and is not, by itself, a reason to build the product.
2. A 5% sample of a 240-row stream will usually not collect 4 teacher labels from the poisoned slice in the first half. Detection at the midpoint is the fragile gate.
3. A second, smaller poisoned slice (neighbor relabel) will also be wrong and will be too small to flag. Escalating only the slice that was detected leaves that residue in the accept set, so post-refit accepted accuracy stays under 0.95.

Chance this is wrong and the rule says PROCEED: about 0.25. The ways it flips are a lucky 5% draw that flags the big slice by row 120, plus a relabel slice whose accepted accuracy stays high enough not to drag the second half under 0.95, plus a calibration threshold that actually accepts 20% of rows at 0.95 accuracy.

## Model

Backend is an adapter with one method: `decide(state, questions) -> per-question probabilities and answer_confidence`.

- Primary backend: `LayaBackend` on `convaiinnovations/laya-typed-decisions` (the fine-tune, Laya 0.4.8, torch 2.14.1+cpu, transformers 5.20.0), `device="cpu"`. One choice question, `action`.
- `JevBackend` posts to `https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $JEV_API_KEY` and model `jev-latest`. If `JEV_API_KEY` is unset, skip it and say so. If it is set, score at most 20 calibration rows, log the call count, and do not let those calls affect the verdict.
- Teacher: gold `action__label` from [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), config `customer_service`. No LLM key is set. Gold is an oracle, not a model judge.
- Gate number: Laya's `answer_confidence` (max probability), not the entropy `confidence`.
- Correct: predicted choice string equals the (possibly remapped) gold string.

The checkpoint was fine-tuned on this config's 300 training rows. Calibration and the stream are a split of those same 300 rows. The official 100-row test split is scored at the locked threshold and reported only. It is not a gate. This run measures a policy change on text the model has already seen. It does not measure generalization to new tickets.

## Split (seed 42)

Permute the 300 training rows with `numpy` Generator(seed).

| Role | Rows | Text | Gold |
| --- | ---: | --- | --- |
| calibration | 60 | original | original `action__label` |
| control | 40 | original | original |
| policy | 160 | original | cycle below, every row |
| relabel | 40 | original | `escalate_to_human` becomes `request_information`; other labels stay |
| paraphrase | 40 | frozen substitutions below | original |

Stream = those 240 production rows, concatenated in the order control, policy, relabel, paraphrase, then shuffled with a new Generator(seed). Midpoint is row 120 (half of 240).

Sensitivity seeds `{0,1,2,3,4}` redraw this split, the shuffle, and the sample masks. They reuse cached forward passes. The locked threshold is refit on each seed's own calibration rows.

Policy cycle, applied to gold only:

```text
answer_directly -> request_information
request_information -> close_no_action
close_no_action -> execute_refund
execute_refund -> escalate_to_human
escalate_to_human -> answer_directly
```

Paraphrase substitutions, applied in this order, only to `state["thread"][*]["text"]`:

```text
refund -> reimbursement
Refund -> Reimbursement
order -> purchase
Order -> Purchase
broken -> not working
cancel -> end
Cancel -> End
charged -> billed
package -> parcel
login -> sign-in
password -> passphrase
invoice -> bill
subscription -> plan
```

## Threshold lock

Grid `0.20, 0.25, …, 0.95`. On calibration, take the lowest threshold with accepted accuracy ≥ **0.95** and coverage ≥ **0.20**. If none exists, status `fallback_0.90` and use the lowest threshold with accepted accuracy ≥ 0.90 and coverage ≥ 0.10. P1 then fails. If that also fails, threshold 0.20 and status `failed`.

## Sampling (budget 5% of the 240-row stream)

Teacher labels are free in this run. Cost is reported at the M0 assumption of **$0.002 per call**. Laya calls are local and cost $0. All three policies have the same expected spend. "Faster" means fewer rows of the replay before the flag, not a cheaper unit price.

Detection, for a named slice, on a prefix of the replay: `n_sampled >= 4` and the one-sided 95% Wilson upper bound on top-1 accuracy (z = 1.6448536269514722) is **< 0.80**. Top-1 accuracy uses every sampled row, escalated or not.

1. **uniform.** Each row is included with probability 0.05. Generator seed `stream_seed * 1000 + 5`.
2. **confidence-stratified.** Draw `round(0.05 * 240) = 12` rows. Bins of `answer_confidence`: `[0, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.01]`. Split the 12 draws as evenly as possible across non-empty bins, remainder to the earlier bins. Sample without replacement inside a bin. If a bin is smaller than its quota, give the leftover to the next non-empty bins. Generator seed `stream_seed * 1000 + 6`. This is a replay stratification over the whole stream; a row counts as "seen" at the midpoint only if its shuffled index is `< 120`.
3. **slice-targeted.** Same 12 draws, split as evenly as possible across the four slice names, sampled without replacement inside the slice. Generator seed `stream_seed * 1000 + 7`.

## Auto-refit

Uses only the detecting policy's samples from the first 120 rows. For each slice with at least 4 of those samples, search thresholds `0.20, 0.25, …, 0.95` and `1.01` (accept nothing). On the sampled rows, a candidate accepts when `answer_confidence >= threshold`. Take the lowest candidate whose accepted accuracy is ≥ 0.95. If the slice was flagged and no candidate that accepts at least one sampled row meets 0.95, use `1.01`. Slices with fewer than 4 samples keep the locked threshold.

Apply those per-slice thresholds to rows 120–239. Judge restoration with full gold. That oracle is only for the scoreboard. The refit itself does not see it.

## Proceed rule

**PROCEED** only if every line holds on seed 42. Otherwise **KILL**.

- **P1.** Calibration status is `primary` (0.95 accuracy at coverage ≥ 0.20).
- **P2.** Some slice `S` has `n_accepted >= 20` and accepted accuracy ≤ **0.85**.
- **P3.** Global monitors stay flat versus calibration: absolute mean-`answer_confidence` shift ≤ **0.05** and absolute coverage shift ≤ **0.10**.
- **P4.** `S` is flat versus control: absolute mean-confidence gap ≤ **0.08** and absolute coverage gap ≤ **0.12**.
- **P5.** At least one of the three 5% policies flags that same `S` on the first 120 rows.
- **P6.** After the refit from the earliest such policy (tie order: uniform, then confidence-stratified, then slice-targeted), the second half has global accepted accuracy ≥ **0.95**, control coverage at least **0.70** times control coverage on the first half, and `S` either has accepted accuracy ≥ 0.95 or coverage ≤ 0.05.
- **P7.** At least 3 of the 5 sensitivity seeds satisfy P2–P4. They do not have to detect or refit.

## Expected numbers (seed 42)

| Quantity | Predicted |
| --- | --- |
| Verdict | KILL |
| P1 | uncertain; the smoke row was not confident. If coverage at 0.95 accuracy is under 0.20, stop at KILL |
| Silent slice | `policy`, accepted accuracy well below 0.85, confidence and coverage near control |
| `relabel` | also wrong on a majority of its rows, too small to flag at 5% |
| `paraphrase` | accuracy stays near control, or confidence drops. Not the silent slice |
| P5 | fail. Uniform expects about 2 policy labels in the first half, and the rule needs 4 |
| P6 | fail even if P5 is lucky, because `relabel` remains accepted and wrong |
| Test-split accepted accuracy | reported, not gated. Expect it below the training-split control number |

## What this commit must not contain

No M0b metrics, no M0b plots, no edited prediction after `experiments/m0b/run_m0b.py` has scored the stream.
