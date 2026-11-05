"""EXP-003: evaluate both frozen prompts on a fresh paired synthetic dataset."""
import argparse
import json
import os
from pathlib import Path
import random
from statistics import median

from dotenv import load_dotenv
from openai import OpenAI

from muse_regime_smoke import MODEL, N, make_prompt as original_prompt, parse_answer
from muse_regime_exp002 import make_prompt as revised_prompt, MAX_TOKENS

SEED = 20261106  # Different from EXP-001/002; frozen before outcome inspection.
RESULTS = Path("muse_regime_exp003.jsonl")
PROMPTS = {"original": original_prompt, "revised": revised_prompt}


def generate_cases():
    """Same DGP and case ordering as EXP-001/002, with a new random seed."""
    rng = random.Random(SEED)
    cases = []
    for i in range(N):
        change = i % 2 == 0
        before = [rng.gauss(0, 1) for _ in range(30)]
        after = [rng.gauss(0, 2.5 if change else 1) for _ in range(30)]
        if not change:
            after[rng.randrange(30)] += rng.choice((-1, 1)) * 5
        ratio = median(abs(x) for x in after) / max(
            median(abs(x) for x in before), 1e-12
        )
        cases.append({
            "id": i + 1, "truth": int(change),
            "before": [round(x, 3) for x in before],
            "after": [round(x, 3) for x in after],
            "baseline_ratio": round(ratio, 5),
            "baseline_label": int(ratio >= 1.75),
        })
    return cases


def load_results(path):
    saved = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["id"], row["prompt"])
            if key in saved:
                raise ValueError(f"Duplicate result: {key}")
            if row.get("experiment") != "EXP-003" or row.get("seed") != SEED:
                raise ValueError(f"Unexpected experiment/seed: {key}")
            saved[key] = row
    return saved


def failure_record(case, variant, kind, detail, usage=None):
    return {
        "experiment": "EXP-003", "seed": SEED, "id": case["id"],
        "prompt": variant, "truth": case["truth"],
        "baseline_label": case["baseline_label"],
        "baseline_ratio": case["baseline_ratio"],
        "model_id": MODEL, "max_tokens": MAX_TOKENS,
        "failure": {"kind": kind, "detail": detail}, "usage": usage,
    }


def save(path, saved, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    saved[(row["id"], row["prompt"])] = row


def score(rows):
    n = len(rows)
    correct = sum((r["model"]["probability"] >= .5) == bool(r["truth"]) for r in rows)
    brier = sum((r["model"]["probability"] - r["truth"]) ** 2 for r in rows) / n
    conflicts = sum(
        (r["model"]["decision"] == "change") !=
        (r["model"]["probability"] >= .5) for r in rows
    )
    return f"{correct}/{n} correct ({correct/n:.3f}); Brier={brier:.3f}; conflicts={conflicts}"


def report(saved):
    paired = [
        (saved[(i, "original")], saved[(i, "revised")])
        for i in range(1, N + 1)
        if (i, "original") in saved and (i, "revised") in saved
        and "model" in saved[(i, "original")] and "model" in saved[(i, "revised")]
    ]
    failures = sum("failure" in r for r in saved.values())
    print(f"Attempted calls: {len(saved)}/{2*N}; failures: {failures}; "
          f"fully paired successful cases: {len(paired)}/{N}")
    for variant in PROMPTS:
        success = sum(r["prompt"] == variant and "model" in r for r in saved.values())
        failed = sum(r["prompt"] == variant and "failure" in r for r in saved.values())
        print(f"{variant}: {success} successful, {failed} failed, "
              f"{N-success-failed} not attempted")
    if not paired:
        print("No paired cases available yet.")
        return
    print("Original prompt:", score([a for a, _ in paired]))
    print("Revised prompt: ", score([b for _, b in paired]))
    print("Matched case IDs:", [a["id"] for a, _ in paired])
    print("Inconclusive or negative results are legitimate; do not revise prompts on this set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2,
                        help="Number of new cases to evaluate (2 calls per case)")
    parser.add_argument("--mark-failed", metavar="CASE:PROMPT",
                        help="Record a previously observed failed attempt without API calls")
    args = parser.parse_args()
    if not 1 <= args.limit <= N:
        parser.error(f"--limit must be between 1 and {N}")
    root = Path(__file__).resolve().parent
    saved = load_results(root / RESULTS)
    cases = generate_cases()
    if args.mark_failed:
        try:
            number, variant = args.mark_failed.split(":")
            number = int(number)
        except ValueError:
            parser.error("--mark-failed must look like 8:original")
        if not 1 <= number <= N or variant not in PROMPTS:
            parser.error("Invalid case ID or prompt")
        if (number, variant) in saved:
            parser.error("This case/prompt already has a recorded outcome")
        row = failure_record(
            cases[number - 1], variant, "previous_empty_response",
            "Previously observed token-limit failure; finish_reason=length; "
            "completion_tokens=2500; reasoning_tokens=2497",
        )
        save(root / RESULTS, saved, row)
        print(f"Recorded prior failure: case {number:02d} / {variant}; no API calls")
        report(saved)
        return
    load_dotenv(root / ".env")
    if not os.environ.get("BASETEN_API_KEY"):
        parser.error("BASETEN_API_KEY missing from local .env")
    client = OpenAI(api_key=os.environ["BASETEN_API_KEY"],
                    base_url="https://inference.baseten.co/v1",
                    timeout=60.0, max_retries=0)
    for case in cases[:args.limit]:
        for variant, build_prompt in PROMPTS.items():
            key = (case["id"], variant)
            if key in saved:
                print(f"Skipping case {case['id']:02d} / {variant}")
                continue
            response = None
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": build_prompt(case)}],
                    reasoning_effort="low", max_tokens=MAX_TOKENS,
                )
                choice = response.choices[0]
                answer = parse_answer(choice.message.content)
            except Exception as exc:
                usage = (response.usage.model_dump()
                         if response is not None and response.usage else None)
                finish = (response.choices[0].finish_reason
                          if response is not None and response.choices else None)
                row = failure_record(case, variant, type(exc).__name__,
                                     f"{exc}; finish_reason={finish}", usage)
                save(root / RESULTS, saved, row)
                print(f"Case {case['id']:02d} / {variant} FAILED: "
                      f"{type(exc).__name__}: {exc}; finish_reason={finish}")
                continue
            row = {
                "experiment": "EXP-003", "seed": SEED, "id": case["id"],
                "prompt": variant, "truth": case["truth"],
                "baseline_label": case["baseline_label"],
                "baseline_ratio": case["baseline_ratio"],
                "model": answer, "model_id": MODEL,
                "max_tokens": MAX_TOKENS,
                "usage": response.usage.model_dump() if response.usage else None,
            }
            save(root / RESULTS, saved, row)
            conflict = (answer["decision"] == "change") != (
                answer["probability"] >= .5
            )
            print(f"Case {case['id']:02d} / {variant}: truth={case['truth']} "
                  f"p={answer['probability']:.2f} decision={answer['decision']} "
                  f"conflict={conflict} — {answer['reason']}")
    report(saved)
    print("Saved:", root / RESULTS)


if __name__ == "__main__":
    main()
