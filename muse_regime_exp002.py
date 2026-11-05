"""EXP-002: clarity-only prompt revision, same synthetic cases/model/budget as pilot."""
import argparse
import json
import math
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI

from muse_regime_smoke import generate_cases, MODEL, N, SEED

RESULTS = Path("muse_regime_exp002.jsonl")
PILOT = Path("muse_regime_results.jsonl")
MAX_TOKENS = 2500  # Keep the EXP-001 inference budget unchanged for comparison.


def make_prompt(case):
    return (
        "You are evaluating whether a PERSISTENT CHANGE IN THE UNDERLYING "
        "VOLATILITY of daily returns occurred between days 1-30 and 31-60. "
        "A single unusual return or a temporary shock alone is NOT a regime change. "
        "You are given only 60 synthetic daily returns (percent) and must assess "
        "the evidence, not claim certainty.\n"
        "Return exactly one JSON object with:\n"
        '- "probability": number from 0 to 1 representing P(persistent volatility '
        'regime change | these 60 observations). 0 means certainly NO change; '
        '1 means certainly a persistent change. This is a subjective model '
        'estimate, not a statistically calibrated posterior.\n'
        '- "decision": "change" if your estimated probability is >= 0.5, '
        'otherwise "no_change".\n'
        '- "reason": one short sentence explaining the observed pattern '
        'and uncertainty, without inventing causes or test statistics.\n'
        "Do not confuse the probability of NO change with the probability "
        "of change. Do not report a numerical test statistic unless computed.\n"
        f"Days 1-30: {case['before']}\n"
        f"Days 31-60: {case['after']}"
    )


def parse(content):
    if not content:
        raise ValueError("Model returned empty content")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match is None:
        raise ValueError("No JSON object in response")
    obj = json.loads(match.group())
    p = float(obj["probability"])
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid probability")
    decision = obj["decision"]
    if decision not in {"change", "no_change"}:
        raise ValueError("Invalid decision")
    return {"probability": p, "decision": decision, "reason": str(obj["reason"])}


def load(path):
    saved = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                saved[entry["id"]] = entry
    return saved


def summarize(name, records):
    n = len(records)
    if not n:
        print(f"{name}: 0 scored")
        return
    correct = sum(int(r["model"]["probability"] >= 0.5) == r["truth"] for r in records)
    brier = sum((r["model"]["probability"] - r["truth"]) ** 2 for r in records) / n
    inconsistent = sum((r["model"]["decision"] == "change") !=
                       (r["model"]["probability"] >= .5) for r in records
                       if r["model"].get("decision") in ("change", "no_change"))
    print(f"{name}: {correct}/{n} correct ({correct/n:.3f}); "
          f"Brier={brier:.3f}; decision/probability conflicts={inconsistent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    args = parser.parse_args()
    if not 1 <= args.limit <= N:
        parser.error(f"--limit must be from 1 to {N}")
    if args.output.resolve() == args.pilot.resolve():
        parser.error("EXP-002 output must differ from EXP-001 pilot results")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("BASETEN_API_KEY"):
        parser.error("BASETEN_API_KEY missing from .env")
    saved = load(args.output)
    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url="https://inference.baseten.co/v1",
        timeout=60.0, max_retries=0,
    )
    for case in generate_cases()[:args.limit]:
        if case["id"] in saved:
            print(f"Skipping completed EXP-002 case {case['id']}")
            continue
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": make_prompt(case)}],
                reasoning_effort="low",
                max_tokens=MAX_TOKENS,
            )
            choice = response.choices[0]
            if not choice.message.content:
                usage = response.usage.model_dump() if response.usage else None
                print(f"Case {case['id']} empty: finish_reason={choice.finish_reason}, "
                      f"usage={usage}")
            answer = parse(choice.message.content)
        except Exception as exc:
            print(f"Case {case['id']} failed: {type(exc).__name__}: {exc}")
            print("Stopped; no automatic retries.")
            break
        entry = {
            "experiment": "EXP-002",
            "id": case["id"],
            "truth": case["truth"],
            "baseline_label": case["baseline_label"],
            "baseline_ratio": case["baseline_ratio"],
            "model": answer,
            "model_id": MODEL,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        with args.output.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        saved[case["id"]] = entry
        conflict = (answer["decision"] == "change") != (answer["probability"] >= .5)
        print(f"Case {case['id']:02d}: truth={case['truth']}, "
              f"p={answer['probability']:.2f}, decision={answer['decision']}, "
              f"conflict={conflict} — {answer['reason']}")
    current = [saved[c["id"]] for c in generate_cases()[:args.limit] if c["id"] in saved]
    summarize("EXP-002", current)
    previous = load(args.pilot)
    common = sorted(set(previous) & {r["id"] for r in current})
    if common:
        summarize("EXP-001 matched cases", [previous[i] for i in common])
        summarize("EXP-002 matched cases", [saved[i] for i in common])
        print("Paired cases:", common)
    else:
        print("No matching EXP-001 file found for paired comparison.")
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
