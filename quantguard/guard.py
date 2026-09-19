"""Public entry point: `QuantitativeGuard.verify(...)`.

Pipeline: extract claims from the generated text -> extract evidence
from the source chunks -> for each claim, find and rank candidate
evidence -> deterministically verify (with arithmetic/derived-value
fallbacks) -> return a GuardResult with per-claim traces.

A `fallback_handler` hook is called for genuinely semantic claims
(e.g. "the system is significantly faster") that extraction can't
turn into a structured Claim at all -- this is the seam described in
the design doc for an optional NLI/LLM stage. It's only invoked when
configured: without one, sentences with no structured claim are
simply outside what QuantGuard checks, exactly as in v0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from quantguard.arithmetic import try_arithmetic_sum, try_derived_percentage
from quantguard.claims import extract_claims, extract_semantic_claims
from quantguard.evidence import extract_evidence, find_candidate_evidence
from quantguard.fallback.base import FallbackResult
from quantguard.models import Claim, ClaimKind, GuardResult, Status, VerificationResult
from quantguard.verification import (
    TolerancePolicy,
    compare_bound,
    compare_numeric,
    compare_range,
    compare_reference,
    is_kind_ambiguous,
)


@dataclass
class VerificationPolicy:
    numeric_tolerance: float = 0.05
    arithmetic_tolerance: float = 0.005
    ambiguity_score_gap: float = 0.75
    max_candidates: int = 5

    def to_tolerance_policy(self) -> TolerancePolicy:
        return TolerancePolicy(
            verification_tolerance=self.numeric_tolerance,
            arithmetic_tolerance=self.arithmetic_tolerance,
            ambiguity_score_gap=self.ambiguity_score_gap,
        )


class QuantitativeGuard:
    def __init__(
        self,
        tolerance: float = 0.05,
        max_candidates: int = 5,
        fallback_handler: Callable[[str, list[str]], FallbackResult] | None = None,
    ):
        self.policy = VerificationPolicy(numeric_tolerance=tolerance, max_candidates=max_candidates)
        self.fallback_handler = fallback_handler

    def verify(
        self,
        generated_text: str,
        source_chunks: list[str],
        policy: VerificationPolicy | None = None,
    ) -> GuardResult:
        policy = policy or self.policy
        tol_policy = policy.to_tolerance_policy()

        claims = extract_claims(generated_text)
        all_evidence = extract_evidence(source_chunks)

        results = [
            self._verify_claim(claim, all_evidence, source_chunks, tol_policy, policy)
            for claim in claims
        ]

        if self.fallback_handler is not None:
            semantic_claims = extract_semantic_claims(generated_text, claims)
            results.extend(self._run_fallback(claim, source_chunks) for claim in semantic_claims)
            results.sort(key=lambda r: r.claim.span.start)

        return GuardResult(generated_text=generated_text, results=results)

    def verify_claims(
        self,
        claims: list[Claim],
        source_chunks: list[str],
        policy: VerificationPolicy | None = None,
        all_evidence=None,
    ) -> list[VerificationResult]:
        """Verify a pre-extracted list of Claims against source_chunks.

        Exposed as a public seam (used by `quantguard.streaming`) so a
        caller that already has claims -- e.g. a subset finalized from
        a growing text stream -- doesn't need to go through
        `extract_claims` again or duplicate the per-claim dispatch
        logic in `_verify_claim`.
        """
        policy = policy or self.policy
        tol_policy = policy.to_tolerance_policy()
        evidence = all_evidence if all_evidence is not None else extract_evidence(source_chunks)
        return [self._verify_claim(claim, evidence, source_chunks, tol_policy, policy) for claim in claims]

    def _run_fallback(self, claim: Claim, source_chunks: list[str]) -> VerificationResult:
        fallback_result = self.fallback_handler(claim.raw_text, source_chunks)
        return VerificationResult(
            claim=claim,
            status=fallback_result.status,
            reason=fallback_result.explanation,
            trace=[
                f"Routed to fallback_handler (no structured claim extractable): {fallback_result.status.value} "
                f"at confidence {fallback_result.confidence:.2f}"
            ],
        )

    def _verify_claim(
        self,
        claim: Claim,
        all_evidence,
        source_chunks: list[str],
        tol_policy: TolerancePolicy,
        policy: VerificationPolicy,
    ) -> VerificationResult:
        candidates = find_candidate_evidence(claim, all_evidence, max_candidates=policy.max_candidates)

        if not candidates:
            # No relevant chunk is known at all here, so search every
            # chunk as a last resort -- but only accept a POSITIVE
            # rescue (a real computed match); an arithmetic contradiction
            # found in an unrelated chunk isn't more informative than
            # UNSUPPORTED, since we have no reason to think that chunk
            # is even about the same thing as the claim.
            rescued = self._try_arithmetic_fallback(claim, all_evidence, source_chunks, tol_policy)
            if rescued is not None and rescued.status in (Status.VERIFIED, Status.APPROXIMATE):
                return rescued
            return VerificationResult(
                claim=claim,
                status=Status.UNSUPPORTED,
                reason="No matching evidence found for this claim.",
                trace=["No candidate evidence of a compatible kind/unit was found in the source chunks."],
            )

        ambiguous_result = None
        if is_kind_ambiguous(claim, candidates, tol_policy):
            ambiguous_result = VerificationResult(
                claim=claim,
                status=Status.AMBIGUOUS,
                candidate_evidence=candidates[:2],
                reason="Multiple equally-plausible pieces of evidence disagree; cannot resolve without guessing.",
                trace=[
                    f"Top candidates scored {candidates[0].match_score:.2f} and {candidates[1].match_score:.2f} "
                    f"(gap below {tol_policy.ambiguity_score_gap}) but have different values "
                    f"({candidates[0].value} vs {candidates[1].value})."
                ],
            )
            # A claim can look ambiguous among single-evidence candidates
            # and STILL be explainable as a sum/derived value (e.g. a
            # claimed total isn't close to any one component, so no
            # single candidate stands out, but the components DO sum to
            # it). Try that rescue, restricted to the ambiguous
            # candidates' own chunks -- unlike the fully-unanchored
            # "no candidates at all" case below, this search is already
            # anchored to chunks direct matching identified as relevant,
            # so a CONTRADICTED conclusion here is trustworthy too, not
            # just a VERIFIED/APPROXIMATE one.
            candidate_chunks = {c.source_index for c in candidates[:2]}
            rescued = self._try_arithmetic_fallback(claim, all_evidence, source_chunks, tol_policy, source_indices=list(candidate_chunks))
            if rescued is not None:
                return rescued
            return ambiguous_result

        top = candidates[0]

        if claim.kind in (ClaimKind.NUMBER, ClaimKind.PERCENTAGE):
            result = compare_numeric(claim, top, tol_policy)
            if result.status == Status.CONTRADICTED:
                # Only let the rescue REPLACE this result if it actually
                # explains the claim (VERIFIED/APPROXIMATE) -- and only
                # search the chunk direct matching already identified as
                # relevant, not every chunk, so we never swap a correct,
                # specific contradiction for an irrelevant one found
                # elsewhere.
                rescued = self._try_arithmetic_fallback(claim, all_evidence, source_chunks, tol_policy, source_indices=[top.source_index])
                if rescued is not None and rescued.status in (Status.VERIFIED, Status.APPROXIMATE):
                    return rescued
            return result

        if claim.kind == ClaimKind.BOUND:
            return compare_bound(claim, top)

        if claim.kind == ClaimKind.RANGE:
            return compare_range(claim, top)

        if claim.kind == ClaimKind.REFERENCE:
            return compare_reference(claim, top)

        return VerificationResult(
            claim=claim, status=Status.NOT_APPLICABLE,
            reason="This claim kind has no deterministic verification strategy.",
        )

    def _try_arithmetic_fallback(
        self,
        claim: Claim,
        all_evidence,
        source_chunks: list[str],
        tol_policy: TolerancePolicy,
        source_indices: list[int] | None = None,
    ) -> VerificationResult | None:
        """Try the arithmetic-sum/derived-percentage rescue, restricted
        to `source_indices` when given (the chunk(s) direct matching
        already identified as relevant). Falls back to searching every
        chunk only when no relevant chunk is known at all (the
        no-candidate-evidence case) -- searching blindly otherwise risks
        finding an unrelated "contradiction" in a chunk that has nothing
        to do with the claim and reporting a confusing, wrong reason.
        """
        indices = source_indices if source_indices is not None else range(len(source_chunks))
        for source_index in indices:
            if claim.kind == ClaimKind.NUMBER:
                result = try_arithmetic_sum(claim, all_evidence, source_index, tol_policy)
                if result is not None:
                    return result
            elif claim.kind == ClaimKind.PERCENTAGE:
                result = try_derived_percentage(claim, all_evidence, source_index, tol_policy)
                if result is not None:
                    return result
        return None
