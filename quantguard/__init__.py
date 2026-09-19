"""QuantGuard: deterministic, unit-aware structured-claim verification for RAG.

    from quantguard import QuantitativeGuard

    guard = QuantitativeGuard(tolerance=0.05)
    result = guard.verify(generated_text, source_chunks)

    result.is_valid          # bool
    result.status_counts     # {"VERIFIED": 2, "CONTRADICTED": 1, ...}
    result.to_dict()         # JSON-serializable
"""

from quantguard.guard import QuantitativeGuard, VerificationPolicy
from quantguard.models import (
    Claim,
    ClaimKind,
    Comparator,
    Evidence,
    GuardResult,
    Span,
    Status,
    VerificationResult,
)

__all__ = [
    "QuantitativeGuard",
    "VerificationPolicy",
    "Claim",
    "ClaimKind",
    "Comparator",
    "Evidence",
    "GuardResult",
    "Span",
    "Status",
    "VerificationResult",
]

__version__ = "0.2.3"
