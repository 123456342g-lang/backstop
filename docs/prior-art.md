# Prior art (verified 2026-10-16)

Skeptical check of the backstop brief. Star counts are from the GitHub API on this date. Functional search covered confidence-gated routing, shadow evaluation, and per-slice drift, not only the names in the brief.

## Named decision-model stack

| Name | What it actually is | Stars | Verdict |
| --- | --- | --- | --- |
| Jev | Hosted System One model from TypeSafe AI. Typed `choice` / `score` / `noul` answers with probabilities. Trained with unpublished RLCD. No public weights. | n/a (closed) | **Real.** [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [typesafe.ai](https://typesafe.ai/) |
| Laya | Open non-autoregressive decision model. `convaiinnovations/laya` is ModernBERT-large (421M). Code: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). | 33,060 | **Real.** Base zero-shot accuracy on the typed-decisions benchmark is **0.362**, below the 0.461 majority baseline. The **0.766** figure is the separate fine-tune [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions), not the base checkpoint. Sources: [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/master/BENCHMARKS.md), [model card](https://huggingface.co/convaiinnovations/laya-typed-decisions). |
| kev | Train-your-own Jev-like models on Qwen. [jaredpalmer/kev](https://github.com/jaredpalmer/kev). | 9,190 | **Real.** Brief's "9k" matches. |
| ollaya | Local pull/serve runtime, TypeSafe-compatible API. [ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya), [ollaya.dev](https://ollaya.dev/docs/quickstart). | 1,322 | **Real.** Brief's "1.3k" matches. |
| AnyJev | Nokia training-free adapter that reads probabilities out of an existing LLM. [nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev), [PyPI anyjev 0.3.0](https://pypi.org/project/anyjev/). | 1,283 | **Real.** Brief's "1.3k" matches. |
| verdict | The brief lists "verdict" next to ollaya as a local server. No ~1.3k-star project of that kind exists. | — | **Name is ambiguous. Do not treat the brief's pairing as fact.** Substitutes below. |
| laya-coreml | Apple Core ML port. [mizorewww/laya-coreml](https://github.com/mizorewww/laya-coreml). | 1,588 | **Real**, and a closer star-peer to ollaya than any verdict repo. |
| von | Open System One model. [wfzyx/von](https://github.com/wfzyx/von). | 910 | **Real.** |
| receptron/laya | Node/ONNX runtime. [receptron/laya](https://github.com/receptron/laya). | 938 | **Real.** |

### What "verdict" actually points at

Three different repos use the name. None is the 1.3k-star Ollama-style server the brief implies.

- [sgyaqing/verdict](https://github.com/sgyaqing/verdict) (13★) — serving engine for Laya checkpoints. This is the functional match for "local pull/serve".
- [Manavarya09/verdict](https://github.com/Manavarya09/verdict) (3★) — small CPU decision model (`verdict-small`, 118M) with a conformal abstain. Not a server. PyPI name is `verdictml`.
- [Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) (110★) — 151M ModernBERT decision model.
- [haizelabs/verdict](https://github.com/haizelabs/verdict) (349★) — inference-time scaling for LLM-as-judge. Different product. Not a substitute.

## jevcal is two projects

| Repo | Stars | Role |
| --- | --- | --- |
| [abhixhek/jevcal](https://github.com/abhixhek/jevcal) | 10 | **The offline tool the brief means.** Fits a per-question confidence threshold to a target accuracy, verifies on a held-out split, reports escalation share and cascade cost, writes `decisions.lock.json`, and fails CI when a model update breaks the lock. Teacher labelling via `jevcal label`. README: "Not on PyPI yet." MIT. |
| [aakgna/jevcal](https://github.com/aakgna/jevcal) | 0 | **Different project**, and the one on PyPI ([jevcal 0.2.0](https://pypi.org/project/jevcal/), homepage `aakgna/jevcal`). Typed decision schemas plus ECE / Brier / reliability diagrams from logged outcomes. It does not fit production thresholds or fail CI. |
| [Adilmp/jevcal](https://github.com/Adilmp/jevcal) | 0 | Third, smaller "what does this threshold buy you" script. Not the CI tool. |

abhixhek/jevcal is offline, one-shot, and global-per-question. It does not sample a live stream, does not slice by traffic segment, and does not refit thresholds from online teacher labels.

## Closest online neighbors (checked, not copied)

| Repo | Stars | What it does | What it does not do |
| --- | --- | --- | --- |
| [AntonioCoppe/jev-harness](https://github.com/AntonioCoppe/jev-harness) | 15 | Confidence gate, policy, eval CLI. `mode: "shadow"` logs the intended action and does not execute it. | Shadow is not a teacher comparison. No per-slice error estimate, no threshold refit. |
| [openlayer-ai/jevals](https://github.com/openlayer-ai/jevals) | 105 | Per-trace agent evals packed into one Jev/Kev/Laya request. The same eval class can sit offline, on traces, or as an in-loop gate. | Not a calibrated accept/escalate policy, not a rolling per-slice accuracy estimate against a teacher, not a closed-loop threshold update. |
| [priyansh19/reflex](https://github.com/priyansh19/reflex) | 0 | Description says "System 1 decision gateway… System 2 LLM escalation". | **Empty repository** (GitHub `size` 0, no README). Not an implementation. |
| [reaatech/confidence-router](https://github.com/reaatech/confidence-router) | 0 | Thresholds for route / clarify / fallback, pluggable classifiers, offline threshold grid search. | No teacher, no live per-slice accuracy, no drift loop. Last push 2026-06-08, before this decision-model wave. |

No single public repo found here implements teacher-labeled, per-slice accepted-accuracy on a live decision stream plus a closed-loop threshold refit.

## Functional neighbors outside the Jev namespace

These already cover pieces of the proposed control plane. The gap, if it exists, is the combination, not the individual mechanisms.

- **Confidence-gated model routing.** [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) (5,606★) serves and evaluates routers that send easy prompts to a cheap model. [Not Diamond](https://github.com/Not-Diamond/notdiamond-python) (SDK archived 2026-08-03) routed prompts across LLMs. Neither speaks the typed decision contract, and neither estimates per-slice decision error against a teacher.
- **Shadow evaluation of a router.** [LiteLLM shadow evaluations](https://docs.litellm.ai/blog/auto-router-shadow-evaluations) sample live traffic, duplicate it through a router, and blind-judge quality, including per complexity tier. That is online teacher sampling. It judges generated answers, not typed-decision accepted-accuracy, and it does not refit a decision threshold.
- **Drift and slices.** [evidentlyai/evidently](https://github.com/evidentlyai/evidently) (7,991★) reports data drift, embedding drift, and model performance, including segmenting by a column. [Arize Phoenix](https://github.com/Arize-ai/phoenix) (11,862★) is tracing and evaluation; the hosted Arize product adds drift and performance slicing. These see a regression when labels or predictions are already logged. They do not fit a jevcal-style accept threshold or close the loop back into a decision gate.
- **Trace evals and scores.** [langfuse/langfuse](https://github.com/langfuse/langfuse) (35,882★) stores traces, scores, and sampled evals. Sampling rate and per-trace scores exist. A typed-decision policy, locked thresholds, and per-slice accepted-accuracy are application-level, not the product.

## Implication for M0

The brief's premise that "no single tool" does fallback policy, per-slice risk, audit, and task-distribution monitoring is **directionally true and overstated**. Gates, shadow judging, and slice/drift monitors all exist. They are split across jev-harness, LiteLLM, Evidently/Arize, and Langfuse, and none of them is the online counterpart of abhixhek/jevcal.

M0 therefore does not need to prove that the category is empty. It needs to prove the pain: a locked threshold's accepted accuracy falls below target on a slice, global confidence and global accuracy still look acceptable, and a 2–5% teacher sample is what reveals the slice. If that conjunction fails, the control plane is not justified.
