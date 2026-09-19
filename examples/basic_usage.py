"""Basic QuantGuard usage: verify a generated RAG answer against its
retrieved source chunks.

Run: python examples/basic_usage.py
"""

from quantguard import QuantitativeGuard


def main() -> None:
    guard = QuantitativeGuard(tolerance=0.05)

    generated_answer = (
        "Latency was reduced to 18ms per 3GPP TS 38.331 Section 5.3.5.4, "
        "and 71% of tasks were completed."
    )
    source_chunks = [
        "Latency achieved was 12.4ms (3GPP TS 38.331 Section 5.3.5.4).",
        "262 completed out of 484 total tasks.",
    ]

    result = guard.verify(generated_answer, source_chunks)

    print("Generated answer:", generated_answer)
    print("Overall valid:", result.is_valid)
    print("Status counts:", result.status_counts)
    print()

    for claim_result in result.results:
        print(f"Claim: {claim_result.claim.raw_text!r}")
        print(f"  Status: {claim_result.status.value}")
        print(f"  Reason: {claim_result.reason}")
        if claim_result.evidence:
            print(f"  Matched evidence: {claim_result.evidence.raw_text!r}")
        print("  Trace:")
        for line in claim_result.trace:
            print(f"    - {line}")
        print()


if __name__ == "__main__":
    main()
