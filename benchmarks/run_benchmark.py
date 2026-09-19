"""Benchmarks QuantGuard against a naive baseline on `dataset.json`.

The naive baseline mirrors the failure mode the design doc calls out
explicitly: it asks only "does this number's exact text occur
somewhere in the source?" -- no entity binding, no unit awareness, no
temporal awareness, no arithmetic. This is meant to make QuantGuard's
actual value legible with real measured numbers, not just examples.

No LLM-judge comparison is included here: this environment has no
access to a hosted model to call as a judge, and the design doc's own
rule ("only publish actual measured numbers") means that comparison
is left out rather than estimated.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from quantguard import QuantitativeGuard

DATASET_PATH = Path(__file__).parent / "dataset.json"

_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?%?")


def naive_is_valid(generated_text: str, source_chunks: list[str]) -> bool:
    """Baseline: every numeric token in the generated text must appear
    verbatim somewhere in the concatenated source text. No entity,
    unit, or temporal awareness at all.
    """
    source_text = " ".join(source_chunks)
    tokens = _NUMBER_TOKEN.findall(generated_text)
    return all(token in source_text for token in tokens)


def run_benchmark() -> None:
    cases = json.loads(DATASET_PATH.read_text())
    guard = QuantitativeGuard(tolerance=0.05)

    quantguard_correct = 0
    naive_correct = 0
    quantguard_latencies = []
    per_category: dict[str, dict[str, int]] = {}

    rows = []
    for case in cases:
        expected = case["expected_valid"]
        category = case["category"]
        per_category.setdefault(category, {"total": 0, "quantguard_correct": 0, "naive_correct": 0})
        per_category[category]["total"] += 1

        start = time.perf_counter()
        result = guard.verify(case["generated_text"], case["source_chunks"])
        quantguard_latencies.append(time.perf_counter() - start)

        qg_valid = result.is_valid
        naive_valid = naive_is_valid(case["generated_text"], case["source_chunks"])

        qg_ok = qg_valid == expected
        naive_ok = naive_valid == expected
        quantguard_correct += qg_ok
        naive_correct += naive_ok
        per_category[category]["quantguard_correct"] += qg_ok
        per_category[category]["naive_correct"] += naive_ok

        rows.append(
            {
                "id": case["id"],
                "category": category,
                "expected_valid": expected,
                "quantguard_valid": qg_valid,
                "quantguard_correct": qg_ok,
                "naive_valid": naive_valid,
                "naive_correct": naive_ok,
            }
        )

    n = len(cases)
    print("=" * 78)
    print(f"QUANTGUARD BENCHMARK -- {n} cases from {DATASET_PATH.name}")
    print("=" * 78)
    print(f"{'Case':30} {'Expected':10} {'QuantGuard':12} {'Naive':10}")
    for row in rows:
        mark_qg = "OK" if row["quantguard_correct"] else "WRONG"
        mark_naive = "OK" if row["naive_correct"] else "WRONG"
        print(
            f"{row['id']:30} {str(row['expected_valid']):10} "
            f"{str(row['quantguard_valid']) + ' (' + mark_qg + ')':22} "
            f"{str(row['naive_valid']) + ' (' + mark_naive + ')':10}"
        )

    print()
    print("=" * 78)
    print("PER-CATEGORY ACCURACY")
    print("=" * 78)
    print(f"{'Category':22} {'N':4} {'QuantGuard':12} {'Naive':10}")
    for category, counts in sorted(per_category.items()):
        qg_acc = counts["quantguard_correct"] / counts["total"]
        naive_acc = counts["naive_correct"] / counts["total"]
        print(f"{category:22} {counts['total']:<4} {qg_acc:<12.0%} {naive_acc:<10.0%}")

    print()
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(f"QuantGuard accuracy : {quantguard_correct}/{n} = {quantguard_correct/n:.1%}")
    print(f"Naive accuracy      : {naive_correct}/{n} = {naive_correct/n:.1%}")
    print(f"QuantGuard p50 latency : {sorted(quantguard_latencies)[n // 2] * 1000:.2f} ms")
    print(f"QuantGuard mean latency: {sum(quantguard_latencies) / n * 1000:.2f} ms")
    print("API cost            : $0 (no LLM call in the deterministic path)")


if __name__ == "__main__":
    run_benchmark()
