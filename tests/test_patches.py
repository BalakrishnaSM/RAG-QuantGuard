from quantguard import QuantitativeGuard
from quantguard.correction.patches import propose_patch


def test_patch_proposed_for_direct_contradiction():
    guard = QuantitativeGuard()
    result = guard.verify("Latency was 18ms.", ["Latency achieved was 12.4ms."])
    patched, patches = result.auto_correct()
    assert patched == "Latency was 12.4ms."
    assert len(patches) == 1
    assert patches[0].replacement == "12.4ms"


def test_patch_proposed_for_arithmetic_contradiction():
    guard = QuantitativeGuard()
    result = guard.verify(
        "There were 500 tasks in total.",
        ["262 completed, 197 not started, 22 in progress, 3 on hold."],
    )
    patched, patches = result.auto_correct()
    assert "484" in patched
    assert patches[0].reason == "arithmetic_contradiction"


def test_no_patch_for_verified_claims():
    guard = QuantitativeGuard()
    result = guard.verify("Latency was 12ms.", ["Latency achieved was 12.4ms."])
    assert result.patches == []
    assert result.patched_text == "Latency was 12ms."


def test_no_patch_for_ambiguous_claims():
    """An ambiguous result must never be auto-patched -- picking one
    of two equally-plausible values would be a guess, exactly what
    this tool exists to avoid.
    """
    guard = QuantitativeGuard()
    result = guard.verify(
        "User A paid $15.",
        ["User A paid $10.", "User B paid $20."],
    )
    for r in result.results:
        if r.status.value == "AMBIGUOUS":
            assert propose_patch(r) is None


def test_multiple_patches_apply_correctly_with_shifting_offsets():
    """Two patches with different-length replacements in the same text
    must not corrupt each other's spans. Also a regression test for a
    real bug: two numeric claims joined by a comma in one sentence
    used to share the same sentence-wide context window, diluting
    each one's entity-disambiguation signal with the other's ("first"
    bleeding into "second"'s context) enough to fall into AMBIGUOUS
    instead of resolving correctly.
    """
    guard = QuantitativeGuard()
    result = guard.verify(
        "First value was 999ms, second value was 888ms.",
        ["First measured 12ms.", "Second measured 100ms."],
    )
    patched, patches = result.auto_correct()
    assert "12ms" in patched
    assert "100ms" in patched
