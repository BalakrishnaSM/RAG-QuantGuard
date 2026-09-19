"""Verify claims incrementally as a generated answer streams in,
rather than waiting for the full text.

A claim is only finalized (verified and yielded) once it looks
STABLE: the same span/text was seen on the previous chunk update too,
AND there's a settle margin of subsequent characters after its end.
Both conditions matter -- a number token can still grow mid-stream
("12" -> "12ms"), so finalizing the instant a claim first appears
would risk verifying a claim against text that hasn't finished being
generated yet.

This is a heuristic, not a guarantee: a genuinely pathological stream
(e.g. a unit word arriving many tokens after its number, separated by
other content) could in principle still be finalized too early. The
settle margin trades a little verification latency for a large
reduction in that risk; it does not eliminate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from quantguard.claims import extract_claims
from quantguard.evidence import extract_evidence
from quantguard.guard import QuantitativeGuard, VerificationPolicy
from quantguard.models import Claim, VerificationResult


@dataclass
class StreamEvent:
    result: VerificationResult
    text_so_far: str


def verify_stream(
    guard: QuantitativeGuard,
    tokens: Iterable[str],
    source_chunks: list[str],
    settle_chars: int = 8,
    policy: VerificationPolicy | None = None,
) -> Iterator[StreamEvent]:
    """Yield a StreamEvent for each claim as soon as it settles.

    `tokens` is any iterable of text fragments (words, subword tokens,
    whatever granularity the caller's generation loop produces) --
    they're just concatenated to grow the buffer; this makes no
    assumption about tokenization scheme.
    """
    all_evidence = extract_evidence(source_chunks)

    buffer = ""
    previous_claims_by_start: dict[int, tuple[int, str]] = {}
    yielded_starts: set[int] = set()

    def _settled_claims(claims: list[Claim], buffer_len: int) -> list[Claim]:
        settled = []
        for claim in claims:
            if claim.span.start in yielded_starts:
                continue
            previously_seen = previous_claims_by_start.get(claim.span.start)
            is_stable = previously_seen == (claim.span.end, claim.raw_text)
            has_settle_margin = (buffer_len - claim.span.end) >= settle_chars
            if is_stable and has_settle_margin:
                settled.append(claim)
        return settled

    for token in tokens:
        buffer += token
        current_claims = extract_claims(buffer)

        for claim in _settled_claims(current_claims, len(buffer)):
            yielded_starts.add(claim.span.start)
            result = guard.verify_claims([claim], source_chunks, policy=policy, all_evidence=all_evidence)[0]
            yield StreamEvent(result=result, text_so_far=buffer)

        previous_claims_by_start = {c.span.start: (c.span.end, c.raw_text) for c in current_claims}

    # Flush: the stream has ended, so every remaining not-yet-yielded
    # claim is as final as it will ever be, regardless of settle margin.
    final_claims = extract_claims(buffer)
    for claim in final_claims:
        if claim.span.start in yielded_starts:
            continue
        yielded_starts.add(claim.span.start)
        result = guard.verify_claims([claim], source_chunks, policy=policy, all_evidence=all_evidence)[0]
        yield StreamEvent(result=result, text_so_far=buffer)
