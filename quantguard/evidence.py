"""Extracts structured Evidence from retrieved source chunks, and
ranks candidate evidence against a given Claim.

The ranking step is the anti-bug the design doc calls out explicitly:
QuantGuard must never conclude a claim is supported merely because
the same number appears *somewhere* in the source. Evidence is
scored on how well its subject, unit dimension, and temporal context
match the claim's -- not on numeric proximity alone.
"""

from __future__ import annotations

from quantguard.claims import extract_claims
from quantguard.models import Claim, ClaimKind, Evidence
from quantguard.units import dimension_of

# Weights for the evidence-ranking score. Kept as named constants
# (not magic numbers inline) so the ranking policy is legible and
# tunable in one place.
_WEIGHT_SUBJECT_OVERLAP = 2.0
_WEIGHT_CONTEXT_OVERLAP = 1.0
_WEIGHT_UNIT_DIMENSION_MATCH = 1.5
_WEIGHT_TIME_CONTEXT_MATCH = 2.0
_WEIGHT_TIME_CONTEXT_MISMATCH_PENALTY = -3.0
_WEIGHT_REFERENCE_MATCH = 3.0


def extract_evidence(source_chunks: list[str]) -> list[Evidence]:
    """Extract structured evidence from every source chunk, tagging
    each with which chunk (source_index) it came from.
    """
    evidence: list[Evidence] = []
    for index, chunk in enumerate(source_chunks):
        for claim in extract_claims(chunk):
            evidence.append(
                Evidence(
                    kind=claim.kind,
                    raw_text=claim.raw_text,
                    span=claim.span,
                    source_index=index,
                    value=claim.value,
                    unit=claim.unit,
                    low=claim.low,
                    high=claim.high,
                    organization=claim.organization,
                    document=claim.document,
                    section=claim.section,
                    subject=claim.subject,
                    context_tokens=claim.context_tokens,
                    sentence=claim.sentence,
                    time_context=claim.time_context,
                )
            )
    return evidence


def _subject_overlap(a: str, b: str) -> float:
    a_words, b_words = set(a.split()), set(b.split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def score_evidence(claim: Claim, candidate: Evidence) -> float:
    """Score how well `candidate` supports `claim`. Higher is better."""
    score = 0.0

    score += _WEIGHT_SUBJECT_OVERLAP * _subject_overlap(claim.subject, candidate.subject)

    if claim.context_tokens and candidate.context_tokens:
        overlap = len(claim.context_tokens & candidate.context_tokens)
        union = len(claim.context_tokens | candidate.context_tokens)
        score += _WEIGHT_CONTEXT_OVERLAP * (overlap / union if union else 0.0)

    claim_dim = dimension_of(claim.unit)
    candidate_dim = dimension_of(candidate.unit)
    if claim_dim is not None and claim_dim == candidate_dim:
        score += _WEIGHT_UNIT_DIMENSION_MATCH
    elif claim.unit is None and candidate.unit is None:
        # Both bare numbers (e.g. plain counts) -- a weak positive signal.
        score += _WEIGHT_UNIT_DIMENSION_MATCH * 0.5

    if claim.time_context and candidate.time_context:
        if claim.time_context == candidate.time_context:
            score += _WEIGHT_TIME_CONTEXT_MATCH
        else:
            # Both claim and evidence mention a time period, but a
            # DIFFERENT one -- this is the exact "2024 vs 2025" trap
            # from the design doc, and should actively count against
            # the match, not just fail to help it.
            score += _WEIGHT_TIME_CONTEXT_MISMATCH_PENALTY

    if claim.kind == ClaimKind.REFERENCE and candidate.kind == ClaimKind.REFERENCE:
        if claim.organization == candidate.organization and claim.document == candidate.document:
            score += _WEIGHT_REFERENCE_MATCH

    return score


def find_candidate_evidence(
    claim: Claim, all_evidence: list[Evidence], max_candidates: int = 5
) -> list[Evidence]:
    """Return the best-matching evidence candidates for `claim`, scored
    and sorted highest-first. Only evidence of a compatible kind is
    considered (a NUMBER claim isn't matched against a REFERENCE, etc.),
    except that RANGE/BOUND claims may be checked against NUMBER evidence.
    """
    compatible_kinds = {claim.kind}
    if claim.kind in (ClaimKind.RANGE, ClaimKind.BOUND):
        compatible_kinds.add(ClaimKind.NUMBER)
        compatible_kinds.add(ClaimKind.PERCENTAGE)

    candidates = [e for e in all_evidence if e.kind in compatible_kinds]
    scored = [(score_evidence(claim, c), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    ranked: list[Evidence] = []
    for score, evidence in scored[:max_candidates]:
        evidence.match_score = score
        ranked.append(evidence)
    return ranked
