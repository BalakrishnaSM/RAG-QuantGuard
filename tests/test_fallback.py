from quantguard import QuantitativeGuard
from quantguard.claims import extract_claims, extract_semantic_claims
from quantguard.fallback.base import FallbackResult, LexicalOverlapFallback
from quantguard.models import ClaimKind, Status


def test_no_fallback_configured_means_semantic_sentences_are_not_checked():
    guard = QuantitativeGuard()  # no fallback_handler
    result = guard.verify(
        "The system is significantly more reliable now.",
        ["The new architecture improves reliability."],
    )
    assert result.results == []


def test_fallback_is_invoked_for_sentences_with_no_structured_claim():
    calls = []

    def fake_handler(claim_text, source_chunks):
        calls.append(claim_text)
        return FallbackResult(Status.VERIFIED, 0.9, "mock support")

    guard = QuantitativeGuard(fallback_handler=fake_handler)
    result = guard.verify("The system is significantly more reliable now.", ["some source"])

    assert len(calls) == 1
    assert result.results[0].status == Status.VERIFIED
    assert result.results[0].claim.kind == ClaimKind.SEMANTIC


def test_structured_and_semantic_claims_coexist_in_one_answer():
    def fake_handler(claim_text, source_chunks):
        return FallbackResult(Status.CONTRADICTED, 0.5, "mock contradiction")

    guard = QuantitativeGuard(fallback_handler=fake_handler)
    result = guard.verify(
        "Latency was 12.4ms, and this proves the architecture is fundamentally broken.",
        ["Latency achieved was 12.4ms."],
    )
    kinds = {r.claim.kind for r in result.results}
    assert ClaimKind.NUMBER in kinds
    assert ClaimKind.SEMANTIC in kinds
    assert not result.is_valid  # the mock semantic contradiction should make it invalid


def test_extract_semantic_claims_skips_sentences_fully_covered_by_structured_claims():
    text = "Latency was 12.4ms."
    structured = extract_claims(text)
    semantic = extract_semantic_claims(text, structured)
    assert semantic == []


def test_extract_semantic_claims_finds_uncovered_sentences():
    text = "Latency was 12.4ms. The system feels much faster overall."
    structured = extract_claims(text)
    semantic = extract_semantic_claims(text, structured)
    assert len(semantic) == 1
    assert "feels much faster" in semantic[0].raw_text


def test_lexical_overlap_fallback_flags_high_overlap_as_approximate():
    handler = LexicalOverlapFallback()
    result = handler(
        "The new architecture improves reliability substantially.",
        ["The new architecture improves reliability substantially for all users."],
    )
    assert result.status == Status.APPROXIMATE
    assert result.confidence < 1.0  # never fully confident -- it's a lexical heuristic


def test_lexical_overlap_fallback_flags_negation_mismatch():
    handler = LexicalOverlapFallback()
    result = handler(
        "The architecture does not improve reliability at all.",
        ["The new architecture improves reliability substantially for all users."],
    )
    assert result.status == Status.CONTRADICTED


def test_lexical_overlap_fallback_unsupported_when_no_overlap():
    handler = LexicalOverlapFallback()
    result = handler("Completely unrelated statement about weather patterns.", ["Latency was 12ms."])
    assert result.status == Status.UNSUPPORTED
