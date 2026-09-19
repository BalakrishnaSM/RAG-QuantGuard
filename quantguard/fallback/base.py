"""The optional NLI/LLM fallback stage.

QuantGuard's core is deterministic: numbers, units, dates, bounds,
ranges, arithmetic, and technical references are computed, not judged.
But some assertions in a generated answer aren't computable at all --
"the new architecture improves reliability," "the system is
significantly faster." These are genuinely semantic claims, and this
module is the seam the design doc describes for handling them: a
pluggable `FallbackHandler` that `QuantitativeGuard` calls only for
sentences that produced no structured (numeric/reference) claim, and
only when a handler is actually configured.

This module ships one real, working, dependency-free default
(`LexicalOverlapFallback`) so the fallback path is testable and usable
out of the box -- but it is a lexical heuristic, not a real entailment
model, and is documented as such. For production use, plug in an
actual NLI model or LLM call via the same `FallbackHandler` protocol
(see `quantguard.fallback.nli` for adapter examples).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from quantguard.models import Status

_NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "can't", "won't", "doesn't", "isn't", "aren't"}
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")


@dataclass
class FallbackResult:
    status: Status
    confidence: float  # 0-1, this handler's confidence in `status`
    explanation: str


class FallbackHandler(Protocol):
    """Anything callable as `handler(claim_text, source_chunks) ->
    FallbackResult` satisfies this -- a plain function works fine, no
    need to subclass anything.
    """

    def __call__(self, claim_text: str, source_chunks: list[str]) -> FallbackResult: ...


class LexicalOverlapFallback:
    """A real, working, dependency-free default fallback.

    Scores each source chunk by word overlap with the claim sentence,
    and flags a likely negation mismatch (the claim asserts something
    the best-matching chunk denies, or vice versa) as a specific,
    lower-confidence CONTRADICTED signal.

    This is intentionally NOT a substitute for a real NLI/entailment
    model -- it cannot detect paraphrase, implication, or
    contradictions that don't share vocabulary. It exists so the
    fallback seam has a working, honest, zero-dependency default
    rather than silently doing nothing until someone wires up a real
    model. Confidence scores are capped well below 1.0 for exactly
    this reason.
    """

    def __init__(self, support_threshold: float = 0.25, min_overlap_words: int = 2):
        self.support_threshold = support_threshold
        self.min_overlap_words = min_overlap_words

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w.lower() for w in _WORD.findall(text)}

    @staticmethod
    def _has_negation(text: str) -> bool:
        lowered = text.lower()
        return any(neg in lowered for neg in _NEGATION_WORDS)

    def __call__(self, claim_text: str, source_chunks: list[str]) -> FallbackResult:
        claim_words = self._words(claim_text)
        if not claim_words:
            return FallbackResult(Status.NOT_APPLICABLE, 0.0, "Claim has no comparable content.")

        best_score = 0.0
        best_chunk = None
        best_overlap_count = 0
        for chunk in source_chunks:
            chunk_words = self._words(chunk)
            if not chunk_words:
                continue
            overlap = claim_words & chunk_words
            score = len(overlap) / len(claim_words)
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_overlap_count = len(overlap)

        if best_chunk is None or best_overlap_count < self.min_overlap_words:
            return FallbackResult(
                Status.UNSUPPORTED, 0.3,
                "No source chunk shares enough vocabulary with this claim to assess it.",
            )

        claim_negated = self._has_negation(claim_text)
        chunk_negated = self._has_negation(best_chunk)
        if claim_negated != chunk_negated and best_score >= self.support_threshold:
            return FallbackResult(
                Status.CONTRADICTED, 0.4,
                "The claim and its best-matching source disagree on negation "
                "(one asserts something the other denies) despite shared vocabulary. "
                "This is a lexical heuristic, not a real entailment check -- treat "
                "this as a signal to review, not a confirmed contradiction.",
            )

        if best_score >= self.support_threshold:
            return FallbackResult(
                Status.APPROXIMATE, min(0.6, 0.3 + best_score),
                f"Claim shares {best_score:.0%} of its vocabulary with a source chunk. "
                "This is lexical overlap, not verified entailment -- treat as weak support.",
            )

        return FallbackResult(
            Status.UNSUPPORTED, 0.3,
            f"Best matching source chunk shares only {best_score:.0%} vocabulary overlap.",
        )
