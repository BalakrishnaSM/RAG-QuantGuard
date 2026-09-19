"""Arithmetic and derived-value verification.

These are fallback strategies tried when a claim doesn't directly
match any single piece of evidence, but might still be correct as a
*computation over* several pieces of evidence -- e.g. a claimed total
that equals the sum of several component values, or a claimed
percentage that equals part/whole from two other evidence values.

Bounded by design: only evidence from the SAME source chunk as the
claim's best candidates is considered, and subset-sum search is
capped at a small number of components to keep this deterministic
and fast rather than combinatorially expensive.
"""

from __future__ import annotations

from itertools import combinations

from quantguard.models import Claim, ClaimKind, Evidence, Status, VerificationResult
from quantguard.verification import TolerancePolicy, relative_error

_MAX_COMPONENTS_CONSIDERED = 10  # cap to keep subset search bounded
_MAX_SUBSET_SIZE = 6


def _numeric_evidence_in_chunk(all_evidence: list[Evidence], source_index: int) -> list[Evidence]:
    return [
        e
        for e in all_evidence
        if e.source_index == source_index and e.kind == ClaimKind.NUMBER and e.value is not None
    ]


def try_arithmetic_sum(
    claim: Claim, all_evidence: list[Evidence], source_index: int, policy: TolerancePolicy
) -> VerificationResult | None:
    """Check whether `claim`'s value equals the sum of some subset of
    the numeric evidence found in the same source chunk.

    Returns None only when there isn't enough numeric evidence in the
    chunk to compute a sum at all. If a sum CAN be computed but
    doesn't match the claim, returns an explicit CONTRADICTED result
    citing the computed total -- silently returning None in that case
    would understate what we actually know (real evidence exists and
    disagrees with the claim, which is a stronger, more specific
    finding than "no evidence found").
    """
    if claim.kind != ClaimKind.NUMBER or claim.value is None:
        return None

    # NOTE: earlier versions filtered out any evidence value equal to the
    # claimed total (on the theory that it was the claim being trivially
    # restated back as "evidence"). That's wrong when a real component
    # legitimately happens to equal the total -- e.g. "20 done, 0
    # pending, total 20" -- since it silently dropped a genuine
    # component instead of just skipping a degenerate match. It's also
    # unnecessary: subset search only considers subsets of size >= 2
    # (below), so a single component equal to the claim can never
    # produce a trivial size-1 "match" on its own.
    pool = _numeric_evidence_in_chunk(all_evidence, source_index)[:_MAX_COMPONENTS_CONSIDERED]
    if len(pool) < 2:
        return None

    trace = [f"Attempting arithmetic-sum verification for claimed total {claim.value}"]

    # Try the full set first (the common case: every component is listed).
    candidate_sets = [pool] if len(pool) <= _MAX_SUBSET_SIZE else []
    # Then smaller subsets, largest first, so we prefer explaining the
    # claim with as many of the found components as possible.
    for size in range(min(len(pool), _MAX_SUBSET_SIZE), 1, -1):
        candidate_sets.extend(combinations(pool, size))

    for subset in candidate_sets:
        total = sum(e.value for e in subset)
        err = relative_error(claim.value, total) if total != 0 else (0.0 if claim.value == 0 else None)
        if err is not None and err <= policy.arithmetic_tolerance:
            component_desc = " + ".join(str(e.value) for e in subset)
            trace.append(f"{component_desc} = {total} (relative error {err:.4f}) -- matches claimed total.")
            return VerificationResult(
                claim=claim,
                status=Status.VERIFIED if err == 0.0 else Status.APPROXIMATE,
                candidate_evidence=list(subset),
                relative_error=err,
                tolerance=policy.arithmetic_tolerance,
                reason=f"Claimed total is the sum of {len(subset)} evidence components ({component_desc}).",
                trace=trace,
            )

    # No subset matched -- report the full-set sum as a concrete
    # contradiction rather than returning None (which the caller would
    # read as "no relevant evidence exists," understating what we know).
    full_total = sum(e.value for e in pool)
    component_desc = " + ".join(str(e.value) for e in pool)
    err = relative_error(claim.value, full_total)
    trace.append(f"No subset matched. Full component sum: {component_desc} = {full_total} (relative error {err:.4f}).")
    return VerificationResult(
        claim=claim,
        status=Status.CONTRADICTED,
        candidate_evidence=list(pool),
        relative_error=err,
        tolerance=policy.arithmetic_tolerance,
        reason=f"Evidence components sum to {full_total}, which contradicts the claimed total of {claim.value}.",
        trace=trace,
    )


def try_derived_percentage(
    claim: Claim, all_evidence: list[Evidence], source_index: int, policy: TolerancePolicy
) -> VerificationResult | None:
    """Check whether a claimed percentage equals part/whole*100 for
    some pair of numeric evidence values found in the same chunk.

    Like `try_arithmetic_sum`, returns None only when there isn't
    enough evidence to compute a derived percentage at all; if a
    part/whole pair exists but implies a different percentage than
    claimed, returns an explicit CONTRADICTED result citing the
    derived value instead of silently giving up.
    """
    if claim.kind != ClaimKind.PERCENTAGE or claim.value is None:
        return None

    pool = _numeric_evidence_in_chunk(all_evidence, source_index)[:_MAX_COMPONENTS_CONSIDERED]
    if len(pool) < 2:
        return None

    trace = [f"Attempting derived-percentage verification for claimed {claim.value}%"]
    valid_pairs: list[tuple[Evidence, Evidence, float]] = []

    for part_evidence, whole_evidence in combinations(pool, 2):
        for part, whole in ((part_evidence.value, whole_evidence.value), (whole_evidence.value, part_evidence.value)):
            if whole == 0 or part > whole:
                continue
            derived_pct = (part / whole) * 100
            err = relative_error(claim.value, derived_pct)
            if err is not None and err <= policy.arithmetic_tolerance:
                trace.append(f"{part} / {whole} * 100 = {derived_pct:.2f}% (relative error {err:.4f}) -- matches claimed percentage.")
                return VerificationResult(
                    claim=claim,
                    status=Status.VERIFIED if err == 0.0 else Status.APPROXIMATE,
                    candidate_evidence=[part_evidence, whole_evidence],
                    relative_error=err,
                    tolerance=policy.arithmetic_tolerance,
                    reason=f"Claimed percentage matches {part}/{whole} from evidence.",
                    trace=trace,
                )
            valid_pairs.append((part_evidence, whole_evidence, derived_pct))

    if not valid_pairs:
        trace.append("No valid part/whole pair (part <= whole) found in this chunk's evidence.")
        return None

    # No pair matched the claim -- report the most specific
    # (largest-whole) derived value as an explicit contradiction
    # rather than returning None.
    part_evidence, whole_evidence, derived_pct = max(valid_pairs, key=lambda triple: triple[1].value)
    err = relative_error(claim.value, derived_pct)
    trace.append(f"Best available derived percentage: {derived_pct:.2f}% (relative error {err:.4f}), which contradicts the claim.")
    return VerificationResult(
        claim=claim,
        status=Status.CONTRADICTED,
        candidate_evidence=[part_evidence, whole_evidence],
        relative_error=err,
        tolerance=policy.arithmetic_tolerance,
        reason=f"Evidence implies {derived_pct:.1f}%, which contradicts the claimed {claim.value}%.",
        trace=trace,
    )
