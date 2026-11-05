"""Tiny exploratory Muse regime-change test. No GPU training required."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import median

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "meta-models/muse-spark-open-1026"
SEED = 20261105
N = 24
RESULTS = Path("muse_regime_results.jsonl")


def generate_cases():
    rng = random.Random(SEED)
    rows = []
    for index in range(N):
        changed = index % 2 == 0
        before = [rng.gauss(0, 1) for _ in range(30)]
        after = [rng.gauss(0, 2.5 if changed else 1) for _ in range(30)]
        if not changed:
            after[rng.randrange(30)] += rng.choice((-1, 1)) * 5
        first = median(abs(x) for x in before)
        second = median(abs(x) for x in after)
        ratio = second / max(first, 1e-12)
        rows.append({
            "id": index + 1, "truth": int(changed),
            "before": [round(x, 3) for x in before],
            "after": [round(x, 3) for x in after],
            "baseline_ratio": round(ratio, 5),
            "baseline_label": int(ratio >= 1.75),
        })
    return rows


def make_prompt(row):
    return (
        "These are 60 consecutive synthetic daily returns, in percent. "
        "Judge whether the underlying volatility changed persistently between "
        "days 1-30 and 31-60, rather than showing only random variation or a "
        "temporary shock. Do not invent economic causes. Return only a JSON "
        "object with keys: probability (number 0 to 1), decision ('change' or "
        "'no_change'), reason (one short sentence). The probability is a "
        "subjective confidence estimate, not a calibrated probability.\n"
        f"Days 1-30: {row['before']}\nDays 31-60: {row['after']}"
    )


def parse_answer(content):
    if not content:
        raise ValueError("Model returned empty content")
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON")
    result = json.loads(match.group())
    probability = float(result["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    if result["decision"] not in ("change", "no_change"):
        raise ValueError("Invalid decision")
    return {"probability": probability, "decision": result["decision"],
            "reason": str(result["reason"])}


def load_results(path):
    saved = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["id"]] = row
    return saved


def report(saved):
    rows = [row for row in saved.values() if "model" in row]
    if not rows:
        print("No completed cases.")
        return
    n = len(rows)
    accuracy = sum((r["model"]["probability"] >= 0.5) == bool(r["truth"])
                   for r in rows) / n
    baseline = sum(r["baseline_label"] == r["truth"] for r in rows) / n
    brier = sum((r["model"]["probability"] - r["truth"]) ** 2
                for r in rows) / n
    print(f"Scored {n}/{N}; Muse accuracy={accuracy:.3f}; "
          f"baseline accuracy={baseline:.3f}; Brier={brier:.3f}")
    print("Small exploratory sample; confidences are not calibrated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not 1 <= args.limit <= N:
        parser.error("--limit must be in 1..24")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.getenv("BASETEN_API_KEY"):
        raise SystemExit("BASETEN_API_KEY missing; put it in local .env")
    client = OpenAI(api_key=os.environ["BASETEN_API_KEY"],
                    base_url="https://inference.baseten.co/v1",
                    timeout=60.0, max_retries=0)
    saved = load_results(args.output)
    for row in generate_cases()[:args.limit]:
        if row["id"] in saved:
            print(f"Skipping completed case {row['id']}")
            continue
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": make_prompt(row)}],
                reasoning_effort="low", max_tokens=1000,
            )
            answer = parse_answer(response.choices[0].message.content)
        except Exception as exc:
            print(f"Case {row['id']} failed: {type(exc).__name__}: {exc}")
            print("Stopped without retrying or using further credits.")
            break
        record = {
            "id": row["id"], "truth": row["truth"],
            "baseline_label": row["baseline_label"],
            "baseline_ratio": row["baseline_ratio"],
            "model": answer, "model_id": MODEL, "seed": SEED,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        with args.output.open("a") as out:
            out.write(json.dumps(record) + "\n")
        saved[row["id"]] = record
        print(f"Case {row['id']:02d} truth={row['truth']} "
              f"baseline={row['baseline_label']} "
              f"Muse_p={answer['probability']:.2f} reason={answer['reason']}")
    report(saved)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
