"""The deterministic verification engine.

Two kinds of tolerance are kept deliberately separate (design doc
section 11):

- **Representation tolerance**: the ambiguity introduced by rounding
  in *either* the generated text or the source. "12" could be a
  rounded display of anything in [11.5, 12.5); "12.4" could be
  anything in [12.35, 12.45). Two values are treated as the same
  underlying number if either one's rounding interval contains the
  other's value.
- **Verification tolerance**: a configurable, domain-chosen acceptable
  relative deviation (e.g. 5%), applied only after representation
  tolerance has already been ruled out.

These are not conflated: a value can fail representation-equality
but still pass verification-tolerance (APPROXIMATE), and a value
outside both is CONTRADICTED.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantguard.models import Claim, ClaimKind, Comparator, Evidence, Status, VerificationResult
from quantguard.units import dimension_of, normalize


@dataclass
class TolerancePolicy:
    verification_tolerance: float = 0.05  # relative error allowed for APPROXIMATE
    ambiguity_score_gap: float = 0.75  # min score gap needed to prefer one candidate over another
    arithmetic_tolerance: float = 0.005  # relative error allowed for sum/derived-value identities

    # Arithmetic identities (a claimed total, a claimed part/whole
    # percentage) are exact by construction -- the only slack needed
    # is for floating-point/display rounding of the individual
    # components, not the several-percent slack appropriate for noisy
    # real-world measurements. Reusing `verification_tolerance` here
    # would let a claim like "500 tasks" pass against a true total of
    # 484 (a 3.3% gap, comfortably inside a 5% measurement tolerance)
    # even though 500 is simply wrong, not a rounding of 484.


def rounding_interval(raw_text: str, value: float) -> tuple[float, float]:
    """The half-open interval of true values `value` could represent,
    given how many decimal places it was written with in `raw_text`.
    "12" -> [11.5, 12.5); "12.4" -> [12.35, 12.45); "12.40" -> same
    precision as "12.4" for this purpose (trailing zero still counts
    as a decimal place written).
    """
    digits_after_point = 0
    if "." in raw_text:
        # Count digits after the decimal point, stopping at the first
        # non-digit (e.g. a trailing unit or punctuation).
        after = raw_text.split(".", 1)[1]
        for ch in after:
            if ch.isdigit():
                digits_after_point += 1
            else:
                break

    half_step = 0.5 * (10 ** -digits_after_point)
    return (value - half_step, value + half_step)


def _representation_match(
    claim: Claim, claim_norm: float, evidence: Evidence, evid_norm: float
) -> bool:
    """Whether claim_norm/evid_norm (already normalized to a shared base
    unit) are the same value up to rounding.

    The rounding interval must be computed in each value's OWN
    original unit/text (e.g. "12ms" -> +/-0.5 ms) and THEN normalized
    the same way the point value was -- computing the interval on the
    already-normalized value would apply an original-unit tolerance
    (e.g. 0.5) to a value on a completely different numeric scale
    (e.g. 0.012 seconds), making the tolerance meaninglessly huge.
    """
    claim_low, claim_high = rounding_interval(claim.raw_text, claim.value)
    evid_low, evid_high = rounding_interval(evidence.raw_text, evidence.value)

    claim_low_norm, _ = normalize(claim_low, claim.unit)
    claim_high_norm, _ = normalize(claim_high, claim.unit)
    evid_low_norm, _ = normalize(evid_low, evidence.unit)
    evid_high_norm, _ = normalize(evid_high, evidence.unit)

    return (claim_low_norm <= evid_norm < claim_high_norm) or (evid_low_norm <= claim_norm < evid_high_norm)


def relative_error(generated: float, source: float) -> float | None:
    if source == 0:
        return None if generated == 0 else float("inf")
    return abs(generated - source) / abs(source)


def _normalized_pair(claim_value: float, claim_unit: str | None, evid_value: float, evid_unit: str | None):
    """Normalize claim and evidence values to a shared base unit if
    their dimensions match. Returns (claim_norm, evid_norm, compatible).
    """
    claim_dim = dimension_of(claim_unit)
    evid_dim = dimension_of(evid_unit)

    if claim_dim is None and evid_dim is None:
        return claim_value, evid_value, True
    if claim_dim != evid_dim:
        return claim_value, evid_value, False

    claim_norm, _ = normalize(claim_value, claim_unit)
    evid_norm, _ = normalize(evid_value, evid_unit)
    return claim_norm, evid_norm, True


def _time_context_mismatch(claim: Claim, evidence: Evidence) -> VerificationResult | None:
    """Shared guard against the "2024 vs 2025" trap: if both the claim
    and the evidence mention a specific time period and they disagree,
    that's a contradiction regardless of how well the numbers/bounds
    themselves line up -- a value that's correct for the wrong period
    is not a correct claim. Returns None when there's no mismatch to
    report (including when one or both sides don't mention a period at
    all, in which case there's nothing to contradict).

    Used by every comparator that can receive a claim with a bound
    time_context (NUMBER, PERCENTAGE, BOUND, RANGE) so the guard can't
    silently apply to only some claim kinds.
    """
    if claim.time_context and evidence.time_context and claim.time_context != evidence.time_context:
        return VerificationResult(
            claim=claim, status=Status.CONTRADICTED, evidence=evidence,
            reason=f"Claim refers to {claim.time_context} but the matched evidence is for {evidence.time_context}.",
            trace=[f"Time context mismatch: claim={claim.time_context} vs evidence={evidence.time_context}"],
        )
    return None


def compare_numeric(
    claim: Claim, evidence: Evidence, policy: TolerancePolicy
) -> VerificationResult:
    trace = [f"Comparing claim value={claim.value}{claim.unit or ''} against evidence value={evidence.value}{evidence.unit or ''}"]

    claim_norm, evid_norm, compatible = _normalized_pair(claim.value, claim.unit, evidence.value, evidence.unit)
    if not compatible:
        trace.append(f"Unit dimensions incompatible: {dimension_of(claim.unit)} vs {dimension_of(evidence.unit)}")
        return VerificationResult(
            claim=claim, status=Status.UNSUPPORTED, candidate_evidence=[evidence],
            reason="Evidence found but its unit is a different dimension; not comparable.", trace=trace,
        )

    time_mismatch = _time_context_mismatch(claim, evidence)
    if time_mismatch is not None:
        trace.extend(time_mismatch.trace)
        time_mismatch.trace = trace
        return time_mismatch

    if _representation_match(claim, claim_norm, evidence, evid_norm):
        trace.append("Values match within rounding/representation tolerance.")
        return VerificationResult(
            claim=claim, status=Status.VERIFIED, evidence=evidence,
            relative_error=0.0, reason="Exact match (within representation precision).", trace=trace,
        )

    err = relative_error(claim_norm, evid_norm)
    trace.append(f"Relative error = {err:.4f}" if err is not None else "Relative error undefined (evidence is zero).")

    if err is not None and err <= policy.verification_tolerance:
        trace.append(f"Within configured verification tolerance ({policy.verification_tolerance:.0%}).")
        return VerificationResult(
            claim=claim, status=Status.APPROXIMATE, evidence=evidence,
            relative_error=err, tolerance=policy.verification_tolerance,
            reason="Value is close to evidence, within tolerance.", trace=trace,
        )

    trace.append(f"Exceeds verification tolerance ({policy.verification_tolerance:.0%}).")
    return VerificationResult(
        claim=claim, status=Status.CONTRADICTED, evidence=evidence,
        relative_error=err, tolerance=policy.verification_tolerance,
        reason="Value contradicts the best matching evidence.", trace=trace,
    )


def compare_bound(claim: Claim, evidence: Evidence) -> VerificationResult:
    trace = [f"Checking bound '{claim.comparator.value if claim.comparator else '?'} {claim.value}' against evidence value={evidence.value}"]

    claim_norm, evid_norm, compatible = _normalized_pair(claim.value, claim.unit, evidence.value, evidence.unit)
    if not compatible:
        return VerificationResult(
            claim=claim, status=Status.UNSUPPORTED, candidate_evidence=[evidence],
            reason="Evidence unit dimension doesn't match the bound's unit.", trace=trace,
        )

    time_mismatch = _time_context_mismatch(claim, evidence)
    if time_mismatch is not None:
        trace.extend(time_mismatch.trace)
        time_mismatch.trace = trace
        return time_mismatch

    ops = {
        Comparator.LESS_THAN: lambda e, c: e < c,
        Comparator.LESS_THAN_OR_EQUAL: lambda e, c: e <= c,
        Comparator.GREATER_THAN: lambda e, c: e > c,
        Comparator.GREATER_THAN_OR_EQUAL: lambda e, c: e >= c,
    }
    satisfied = ops[claim.comparator](evid_norm, claim_norm)
    trace.append(f"Evidence {'satisfies' if satisfied else 'violates'} the bound.")

    return VerificationResult(
        claim=claim,
        status=Status.VERIFIED if satisfied else Status.CONTRADICTED,
        evidence=evidence,
        reason="Evidence satisfies the stated bound." if satisfied else "Evidence violates the stated bound.",
        trace=trace,
    )


def compare_range(claim: Claim, evidence: Evidence) -> VerificationResult:
    trace = [f"Checking whether evidence value={evidence.value} falls in claimed range [{claim.low}, {claim.high}]"]

    claim_low_norm, evid_norm, compatible = _normalized_pair(claim.low, claim.unit, evidence.value, evidence.unit)
    claim_high_norm, _, _ = _normalized_pair(claim.high, claim.unit, evidence.value, evidence.unit)
    if not compatible:
        return VerificationResult(
            claim=claim, status=Status.UNSUPPORTED, candidate_evidence=[evidence],
            reason="Evidence unit dimension doesn't match the range's unit.", trace=trace,
        )

    time_mismatch = _time_context_mismatch(claim, evidence)
    if time_mismatch is not None:
        trace.extend(time_mismatch.trace)
        time_mismatch.trace = trace
        return time_mismatch

    in_range = claim_low_norm <= evid_norm <= claim_high_norm
    trace.append(f"Evidence is {'inside' if in_range else 'outside'} the range.")

    return VerificationResult(
        claim=claim,
        status=Status.VERIFIED if in_range else Status.CONTRADICTED,
        evidence=evidence,
        reason="Evidence falls within the claimed range." if in_range else "Evidence falls outside the claimed range.",
        trace=trace,
    )


def compare_reference(claim: Claim, evidence: Evidence) -> VerificationResult:
    trace = [f"Comparing reference claim {claim.organization} {claim.document} §{claim.section} against evidence {evidence.organization} {evidence.document} §{evidence.section}"]

    if claim.organization != evidence.organization:
        trace.append("Organization does not match.")
        return VerificationResult(
            claim=claim, status=Status.CONTRADICTED, evidence=evidence,
            reason="Cited organization does not match the evidence.", trace=trace,
        )

    # Document identifiers only conflict when BOTH sides actually specify
    # one and they disagree. A citation that omits the document number
    # (e.g. "3GPP Section 5.3.5.4" with no "TS 38.331") isn't wrong about
    # the document -- it just doesn't say, which is a gap in specificity,
    # not a disagreement. Treating "unspecified" the same as "specified
    # and different" was a real bug: it marked correct section citations
    # as CONTRADICTED just because they were less precise than the
    # evidence. So a one-sided None downgrades confidence (below) instead
    # of forcing an immediate contradiction.
    document_confirmed = True
    if claim.document is not None and evidence.document is not None:
        if claim.document != evidence.document:
            trace.append("Document identifier does not match.")
            return VerificationResult(
                claim=claim, status=Status.CONTRADICTED, evidence=evidence,
                reason="Cited document does not match the evidence.", trace=trace,
            )
    elif claim.document is not None or evidence.document is not None:
        document_confirmed = False
        trace.append("Only one side specifies a document identifier; document match unconfirmed.")

    if claim.section is None:
        status = Status.VERIFIED if document_confirmed else Status.APPROXIMATE
        trace.append(
            "Claim doesn't cite a specific section; document match is sufficient."
            if document_confirmed else
            "Claim doesn't cite a specific section, and the document identifier is unconfirmed."
        )
        return VerificationResult(
            claim=claim, status=status, evidence=evidence,
            reason=(
                "Document reference matches." if document_confirmed else
                "Organization matches, but the document identifier could not be confirmed."
            ),
            trace=trace,
        )

    if evidence.section is None:
        trace.append("Evidence doesn't specify a section; document matches but the clause is unconfirmed.")
        return VerificationResult(
            claim=claim, status=Status.APPROXIMATE, evidence=evidence,
            reason="Document matches, but the cited section could not be confirmed against evidence.",
            trace=trace,
        )

    if claim.section == evidence.section:
        status = Status.VERIFIED if document_confirmed else Status.APPROXIMATE
        trace.append("Section matches exactly." + ("" if document_confirmed else " (document identifier unconfirmed)"))
        return VerificationResult(
            claim=claim, status=status, evidence=evidence,
            reason=(
                "Document and section both match." if document_confirmed else
                "Section matches; document identifier could not be confirmed."
            ),
            trace=trace,
        )

    trace.append(f"Section mismatch: claimed §{claim.section} vs evidence §{evidence.section}.")
    return VerificationResult(
        claim=claim, status=Status.CONTRADICTED, evidence=evidence,
        reason=f"Cited section {claim.section} does not match evidence section {evidence.section}.",
        trace=trace,
    )


def is_kind_ambiguous(claim: Claim, ranked_candidates: list[Evidence], policy: TolerancePolicy) -> bool:
    """True if the top two evidence candidates are close enough in
    match score that picking between them would be a guess, AND they
    would lead to different verdicts.
    """
    if len(ranked_candidates) < 2:
        return False

    top, second = ranked_candidates[0], ranked_candidates[1]
    if top.match_score - second.match_score > policy.ambiguity_score_gap:
        return False  # top candidate is clearly preferred

    if claim.kind in (ClaimKind.NUMBER, ClaimKind.PERCENTAGE):
        return top.value != second.value
    if claim.kind == ClaimKind.REFERENCE:
        return top.section != second.section
    return False
