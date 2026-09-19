"""Core data models shared across QuantGuard's extraction, evidence,
and verification stages.

Kept as plain dataclasses (not pydantic) so the library has zero
required dependencies beyond the standard library for its core path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimKind(str, Enum):
    NUMBER = "number"
    PERCENTAGE = "percentage"
    RANGE = "range"
    BOUND = "bound"
    REFERENCE = "reference"
    SEMANTIC = "semantic"  # a non-quantitative assertion, routed to a fallback handler


class Comparator(str, Enum):
    """For BOUND claims: the direction of the inequality."""

    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"


class Status(str, Enum):
    VERIFIED = "VERIFIED"
    APPROXIMATE = "APPROXIMATE"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Span:
    """A character offset range into the text a claim/evidence came from."""

    start: int
    end: int

    def text_from(self, source: str) -> str:
        return source[self.start : self.end]

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass
class Claim:
    """A single structured, checkable assertion extracted from generated text."""

    kind: ClaimKind
    raw_text: str
    span: Span

    # NUMBER / PERCENTAGE / one side of a RANGE
    value: float | None = None
    unit: str | None = None  # normalized unit symbol, e.g. "m", "ms", "%", None

    # RANGE
    low: float | None = None
    high: float | None = None

    # BOUND
    comparator: Comparator | None = None

    # REFERENCE (e.g. "3GPP TS 38.331 Section 5.3.5.4")
    organization: str | None = None
    document: str | None = None
    section: str | None = None

    # Heuristic entity/context binding (see claims.py)
    subject: str = ""
    context_tokens: frozenset[str] = field(default_factory=frozenset)
    sentence: str = ""
    time_context: str | None = None  # a year or "Q1 2024"-style mention found in the same sentence

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "raw_text": self.raw_text,
            "span": self.span.to_dict(),
            "value": self.value,
            "unit": self.unit,
            "low": self.low,
            "high": self.high,
            "comparator": self.comparator.value if self.comparator else None,
            "organization": self.organization,
            "document": self.document,
            "section": self.section,
            "subject": self.subject,
            "time_context": self.time_context,
        }


@dataclass
class Evidence:
    """A structured value found in a source chunk, comparable to a Claim."""

    kind: ClaimKind
    raw_text: str
    span: Span
    source_index: int  # which source_chunks[i] this came from

    value: float | None = None
    unit: str | None = None
    low: float | None = None
    high: float | None = None
    organization: str | None = None
    document: str | None = None
    section: str | None = None

    subject: str = ""
    context_tokens: frozenset[str] = field(default_factory=frozenset)
    sentence: str = ""
    time_context: str | None = None
    match_score: float = 0.0  # populated by the evidence ranker

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "raw_text": self.raw_text,
            "span": self.span.to_dict(),
            "source_index": self.source_index,
            "value": self.value,
            "unit": self.unit,
            "low": self.low,
            "high": self.high,
            "organization": self.organization,
            "document": self.document,
            "section": self.section,
            "subject": self.subject,
            "time_context": self.time_context,
            "match_score": round(self.match_score, 4),
        }


@dataclass
class VerificationResult:
    """The outcome of checking one Claim against the best matching Evidence."""

    claim: Claim
    status: Status
    evidence: Evidence | None = None
    candidate_evidence: list[Evidence] = field(default_factory=list)
    relative_error: float | None = None
    tolerance: float | None = None
    reason: str = ""
    trace: list[str] = field(default_factory=list)
    candidate_count: int = 0  # how many evidence candidates were considered

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "candidate_evidence": [e.to_dict() for e in self.candidate_evidence],
            "relative_error": self.relative_error,
            "tolerance": self.tolerance,
            "reason": self.reason,
            "trace": self.trace,
            "candidate_count": self.candidate_count,
        }


@dataclass
class GuardResult:
    """The full result of verifying a generated answer against source chunks."""

    results: list[VerificationResult]
    generated_text: str = ""

    @property
    def is_valid(self) -> bool:
        """True iff no claim was CONTRADICTED or AMBIGUOUS.

        UNSUPPORTED claims (no evidence found at all) are treated as
        invalid too -- an unverifiable quantitative claim in a
        grounded RAG answer is a failure mode QuantGuard exists to
        catch, not to wave through.

        NOT_APPLICABLE is deliberately excluded from `bad`: it's
        reserved for claims that aren't meaningfully computable/checkable
        (not currently produced by `extract_claims`, but kept available
        for future claim kinds) and should never fail validation on its
        own -- it means "nothing to check here," not "this is wrong."
        """
        bad = {Status.CONTRADICTED, Status.AMBIGUOUS, Status.UNSUPPORTED}
        return all(r.status not in bad for r in self.results)

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "is_valid": self.is_valid,
            "status_counts": self.status_counts,
            "results": [r.to_dict() for r in self.results],
        }

    def auto_correct(self) -> tuple[str, list]:
        """Apply every proposable patch and return (patched_text, patches).

        Only ever replaces a claim when there's exactly one clear
        evidence-backed replacement -- an AMBIGUOUS result never gets
        patched, since guessing between equally-plausible values would
        defeat the purpose of a conservative correction tool.
        """
        from quantguard.correction.patches import build_patched_text

        return build_patched_text(self.generated_text, self)

    @property
    def patches(self) -> list:
        from quantguard.correction.patches import propose_patch

        return [p for r in self.results if (p := propose_patch(r)) is not None]

    @property
    def patched_text(self) -> str:
        text, _ = self.auto_correct()
        return text
