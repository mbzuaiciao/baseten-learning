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

## EXP-002: Clarified probability prompt

EXP-002 asks whether **unambiguous probability instructions** improve Muse's
classification and internal consistency. It reuses exactly the same 24 generated
synthetic cases, the same Muse model, `reasoning_effort=low`, and the same
2,500-token budget as the original pilot. **Only the text prompt changes.**
The primary comparison is **paired by case ID**; incomplete API responses are not
counted as incorrect classifications. The recorded probability is still
uncalibrated, and requested decision consistency is part of the intervention.

The original results stay untouched at `muse_regime_results.jsonl`.
The new results are saved separately at `muse_regime_exp002.jsonl`.

From the repository root, run the two-case smoke test first:

```bash
uv run python muse_regime_exp002.py --limit 2
```

Review the two answers for valid JSON and decision/probability consistency.
If successful, complete the planned dataset:

```bash
uv run python muse_regime_exp002.py --limit 24
```

The script skips completed EXP-002 cases, never retries automatically, and stops
on an invalid/empty response. Its terminal report compares both runs only on
case IDs present in both local files. A failure can legitimately end the run;
do not alter the sample or secretly change the token budget to improve scores.
Inspect any claimed statistical numbers in explanations separately.

Open the dashboard with `uv run python -m http.server 8001`, then visit
`http://localhost:8001/dashboard/`. Keep the included EXP-001 snapshot or
load the original local JSONL with **Load local results**. Load
`muse_regime_exp002.jsonl` with **Load Experiment 002** to display
matched-case accuracy and Brier scores. Both files stay in your browser.

**Stopping rule:** evaluate at most 24 planned cases, compare on paired cases,
and record whether the revised prompt improved reliability. This remains an
exploratory prompt comparison, not foundation-model pretraining.

## EXP-003: Fresh-case generalization check

This is a **new, frozen-seed evaluation** of the two existing prompts on 24
different synthetic 60-day return series (seed `20261106` instead of
`20261105`). The data-generating procedure, label balance, model,
`reasoning_effort="low"`, and 2,500-token budget remain unchanged.
The script imports the EXP-001 and EXP-002 prompt functions unchanged.
The aim is to test whether the apparent EXP-002 improvement extends to unseen
randomly generated cases, not to tune another prompt.

From the repository root, with your existing local `.env`:

```bash
uv run python muse_regime_exp003.py --limit 2
```

This sends up to **four** API requests (original and revised for each case).
If the smoke test works, complete the fixed evaluation once:

```bash
uv run python muse_regime_exp003.py --limit 24
```

This makes up to 48 calls in total, including already completed calls.
The script saves every successful response in the separate, ignored
`muse_regime_exp003.jsonl`. Re-running skips existing case/prompt pairs.
On any API/parse failure the run stops without automatic retries; the terminal
summary includes only **paired completed cases** so neither prompt benefits
from extra unpaired observations. Empty responses are not counted as wrong
classifications. A model can still have a lower completion rate.

The dashboard includes an **Experiment 003** loader for this single JSONL file
and shows matched-case accuracy, Brier scores, and decision/probability conflicts.
As with the previous experiments, the local file is read in-browser and not
automatically committed.

**Stopping rule:** After the planned 24 new cases (48 inference calls), report
whether the revised prompt generalizes, including negative/inconclusive
results, without changing prompts based on this test.
