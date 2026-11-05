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

## Experiment results dashboard

A static, Apple-inspired report is available in [`dashboard/index.html`](dashboard/index.html).
It opens with a **17-case observed transcript snapshot**, not a completed study or
a simulated score. The snapshot records the visible terminal observations from
November 5, 2026, and explicitly marks the empty Case 18 response.

To open the site locally, from the repository root:

```bash
python3 -m http.server 8000
```

Then visit [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/).
The dashboard needs no npm installation, build step, GPU, or Baseten API calls.

To display current results, click **Load local results** and select
`muse_regime_results.jsonl` from the repository root. The file is read in your
browser only; it is not uploaded to a server. This also works if the JSONL file
has additional completed cases beyond the 17-case snapshot.

Metrics are recomputed from the selected records. The local JSONL contains
Muse's explicit `decision`, so the page can flag a disagreement between the
decision and its reported probability. The original terminal log lacks that
field; the included snapshot flags only *possible qualitative inconsistencies*
in the written explanations, and should not be treated as a measured decision
conflict.

The dashboard does not publish the local JSONL or API key. To publish a later
results snapshot, review the contents and intentionally commit a sanitized
snapshot (do not automatically commit the local output).
