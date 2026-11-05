# Muse Spark Open — Synthetic volatility regime experiments

**Status:** EXP-001/002/003 completed as planned (November 5, 2026). This is an exploratory, inference-only prompt-comparison study in `baseten-learning`, **not** the separate frozen financial-time-series foundation-model EXP-001.

## Research question

Can clearer instructions about the meaning of a probability estimate improve a language model's distinction between a sustained volatility regime change and a temporary large return?

## Setup and controls

- Model: `meta-models/muse-spark-open-1026`, served via Baseten; no weight updates or fine-tuning.
- Each case comprises 60 synthetic daily returns in percent, split into two 30-day windows.
- Positive cases: Gaussian returns with standard deviation 1 before the boundary and 2.5 afterward (persistent higher volatility).
- Negative cases: standard deviation 1 throughout, with one added positive or negative 5-percentage-point isolated shock in the second window.
- Balanced labels: 12 positive and 12 negative cases per dataset. Ground truth comes from the generator, not from statistical certainty about a specific finite sample.
- Fixed comparison baseline: second-half divided by first-half median absolute return; classify as change if ratio >= 1.75. The baseline is illustrative, not optimized.
- `reasoning_effort="low"` and `max_tokens=2500` throughout.
- Original prompt: `muse_regime_smoke.make_prompt`; revised prompt: `muse_regime_exp002.make_prompt`. The revised version specifies that `probability` means P(persistent volatility regime change | observations), explicitly distinguishes outliers from regime shifts, directs `decision` to follow the 0.5 threshold, and discourages invented statistics. These changes are one bundled intervention, not a factorial attribution test.
- The original dataset uses seed 20261105. The holdout set uses seed 20261106 with the **same data-generating process**, but new random draws. EXP-003 imports both prompt implementations unchanged.

## Observed results

All reported scores below are taken from the user's November 5 terminal outputs. Raw local JSONL files are intentionally gitignored and not part of this report. The published dashboard data are aggregates only. Accuracy is the share of `probability >= 0.5` predictions matching generator truth; Brier is mean squared error of the reported `probability` versus truth. The probabilities were not calibrated.

### Original dataset (EXP-001 versus EXP-002)

**Paired comparison: same 17 successfully answered cases**

| Measure | Original prompt | Revised prompt |
|---|---:|---:|
| Correct | 13/17 | 16/17 |
| Accuracy | 76.5% | 94.1% |
| Brier | 0.152 | 0.066 |
| Decision / probability conflicts | 3 | 0 |

The original run produced 17 usable responses before stopping at Case 18 with `finish_reason=length` and no visible answer. EXP-002 completed all 24 planned cases: **22/24 correct (91.7%), Brier 0.073, and zero decision/probability conflicts**. The 24-case revised result must not be compared directly to the 17-case original result as though both had the same evaluated cases.

Examples from the original run: Cases 4, 8, and 14 assigned 72%, 88%, and 68% regime-change probability respectively, despite no-change ground truth and explanations focused on isolated shocks. The revised prompt assigned 30%, 28%, and 32% on those same cases.

### New dataset (EXP-003)

**Paired comparison: 22 cases where both prompts returned usable responses**

| Measure | Original prompt | Revised prompt |
|---|---:|---:|
| Correct | 16/22 | 21/22 |
| Accuracy | 72.7% | 95.5% |
| Brier | 0.205 | 0.052 |
| Decision / probability conflicts | 5 | 0 |

**Response completion (all 24 planned cases per prompt):** original 22/24 successful, 2/24 failed (Cases 8 and 20); revised 24/24 successful, no failures. The original-prompt failures exhausted the 2,500-token completion budget without usable visible output. The revised prompt correctly classified 22/24 overall; it misclassified the no-change Cases 8 and 14. The separate all-24 revised score is not the paired 21/22 score.

The EXP-003 original prompt yielded Case 2: `probability=0.90`, `decision=no_change`, and an explanation describing a temporary outlier. The revised prompt gave `probability=0.15`, `decision=no_change`. Other original-prompt contradictions appeared in Cases 4, 16, 18, and 22. Notably, *the original prompt can generate the correct narrative and binary decision yet express an inconsistent probability*.

## Interpretation

The revised instructions improved paired accuracy, subjective-probability Brier score, and decision/probability agreement on both the development dataset and the **new random-draw evaluation dataset**. This supports repeatability within the chosen synthetic generator, not broad out-of-distribution generalization.

The consistency result is partly expected: the revised prompt **explicitly orders** the binary decision to match its probability. We cannot isolate that instruction from the change in task wording, nor infer from a better Brier score alone that the model's statistical reasoning improved. Some outcomes are sensitive to natural variation in generated responses; the number of cases is small. Failed responses are reported separately and excluded from paired predictive metrics, so usability and prediction quality are distinct outcomes. The reported Brier scores do not establish calibrated probabilities.

**Limits:** no real financial observations; no market microstructure, fat tails, clustering, or unknown change point; synthetic scenario families are highly structured and unchanged across seeds; only one model and one fixed inference setting; no repeated-response stability trials; no claim of causal reasoning, representation learning, or investment utility.

## Reproduction and evidence boundaries

- Source scripts: `muse_regime_smoke.py`, `muse_regime_exp002.py`, `muse_regime_exp003.py`.
- Original local files: `muse_regime_results.jsonl`, `muse_regime_exp002.jsonl`, and `muse_regime_exp003.jsonl` (gitignored).
- Published aggregates: `dashboard/findings.json` (manually transcribed from observed terminal summaries, not derived from committed raw JSONL).
- The dashboard's `dashboard/snapshot.json` remains a separate, partial 17-case EXP-001 transcript snapshot. Neither aggregate file nor this report substitutes for an independent rerun or complete raw record audit.

**Stopping rule met:** both frozen prompts were evaluated on the planned 24 new cases, with response failures retained. No further prompt revisions are justified by this holdout set. A separate future study would need a materially different, precommitted scientific question.
