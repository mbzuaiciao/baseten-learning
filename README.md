# Baseten Learning

Minimal experiments using the special Meta hackathon Muse Spark Open inference key.

## Setup

From the repository root:

```bash
uv sync
```

Store the hackathon API key locally in `.env`:

```text
BASETEN_API_KEY=your_key_here
```

Never commit the key. The hackathon key is separate from ordinary Baseten Console credentials.

## Existing API sanity check

```bash
uv run python hello_baseten.py
```

## Synthetic volatility regime experiment

`muse_regime_smoke.py` compares Muse Spark Open against a fixed, robust median-absolute-return ratio rule. It uses 24 deterministic synthetic 60-day return series: 12 with a persistent volatility increase, 12 with unchanged volatility plus one isolated shock. The model only receives the observations, not the generator labels or parameters. This is **inference only** (not GPU training), and is separate from the frozen EXP-001 financial time-series foundation-model protocol.

First run just two API calls:

```bash
uv run python muse_regime_smoke.py --limit 2
```

If the output looks sensible, run the remaining cases:

```bash
uv run python muse_regime_smoke.py --limit 24
```

The output file `muse_regime_results.jsonl` is local and ignored by Git. Completed cases are skipped on rerun, and each failed request stops the run without retries. The script prints classification accuracy, baseline accuracy, and the Brier score of Muse's **uncalibrated subjective** probability estimates. The sample is small and exploratory; the baseline threshold is fixed at 1.75, not tuned. Do not interpret strong performance as proof of time-series representation learning or causality.

**Stopping rule:** stop after 24 cases and inspect the results before extending the study. No changes to EXP-001 and no hidden future financial data are involved.
