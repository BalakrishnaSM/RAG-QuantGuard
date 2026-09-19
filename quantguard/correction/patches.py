"""Conservative auto-correction: turn a CONTRADICTED result into a
suggested text patch, without ever guessing when evidence is unclear.

Per the design doc's rule: if two pieces of evidence could equally
explain a correction, return AMBIGUOUS-style non-action rather than
picking one arbitrarily. A patch is only ever proposed when there is
exactly one evidence value to correct towards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantguard.models import ClaimKind, GuardResult, Span, Status, VerificationResult


@dataclass
class Patch:
    span: Span
    original: str
    replacement: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": self.span.to_dict(),
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


def _format_replacement(claim_kind: ClaimKind, evidence_value: float | None, unit: str | None) -> str | None:
    if evidence_value is None:
        return None
    # Preserve integer-looking values as integers (e.g. "484", not "484.0").
    text = f"{evidence_value:g}"
    if claim_kind == ClaimKind.PERCENTAGE:
        return f"{text}%"
    if unit:
        return f"{text}{unit}"
    return text


def propose_patch(result: VerificationResult) -> Patch | None:
    """Propose a single-value replacement patch for a CONTRADICTED
    result, or None if the result doesn't have exactly one clear
    replacement value to suggest.
    """
    if result.status != Status.CONTRADICTED:
        return None

    claim = result.claim

    # Numeric/percentage claims backed by exactly one piece of direct
    # evidence: replace with that evidence's value.
    if claim.kind in (ClaimKind.NUMBER, ClaimKind.PERCENTAGE) and result.evidence is not None:
        replacement = _format_replacement(claim.kind, result.evidence.value, result.evidence.unit or claim.unit)
        if replacement is None:
            return None
        return Patch(
            span=claim.span,
            original=claim.raw_text,
            replacement=replacement,
            reason="contradiction",
            confidence=0.98 if result.relative_error not in (None,) and result.relative_error > 0.5 else 0.9,
        )

    # Arithmetic/derived-value rescues carry the corrected value in the
    # reason text's computed total, not a single evidence.value -- but
    # candidate_evidence has the components, so recompute it here for
    # a clean patch rather than re-parsing the reason string.
    if claim.kind == ClaimKind.NUMBER and result.candidate_evidence:
        total = sum(e.value for e in result.candidate_evidence if e.value is not None)
        replacement = _format_replacement(claim.kind, total, claim.unit)
        if replacement is None:
            return None
        return Patch(
            span=claim.span,
            original=claim.raw_text,
            replacement=replacement,
            reason="arithmetic_contradiction",
            confidence=0.85,
        )

    if claim.kind == ClaimKind.PERCENTAGE and len(result.candidate_evidence) == 2:
        part, whole = sorted((e.value for e in result.candidate_evidence), )
        if whole:
            derived = (part / whole) * 100
            return Patch(
                span=claim.span,
                original=claim.raw_text,
                replacement=f"{derived:.1f}%",
                reason="derived_percentage_contradiction",
                confidence=0.85,
            )

    # Reference claims: only propose a patch if the evidence has a
    # concrete, differing section to correct towards.
    if claim.kind == ClaimKind.REFERENCE and result.evidence is not None and result.evidence.section:
        original_ref = claim.raw_text
        replacement_ref = original_ref.replace(claim.section or "", result.evidence.section) if claim.section else None
        if replacement_ref and replacement_ref != original_ref:
            return Patch(
                span=claim.span,
                original=original_ref,
                replacement=replacement_ref,
                reason="reference_contradiction",
                confidence=0.9,
            )

    return None


def build_patched_text(generated_text: str, guard_result: GuardResult) -> tuple[str, list[Patch]]:
    """Apply every proposable patch to `generated_text`, right-to-left
    by span so earlier offsets stay valid as later ones are replaced.
    """
    patches = [p for r in guard_result.results if (p := propose_patch(r)) is not None]
    patches_sorted = sorted(patches, key=lambda p: p.span.start, reverse=True)

    text = generated_text
    for patch in patches_sorted:
        text = text[: patch.span.start] + patch.replacement + text[patch.span.end :]

    # Return patches in reading order for display purposes.
    return text, sorted(patches, key=lambda p: p.span.start)
