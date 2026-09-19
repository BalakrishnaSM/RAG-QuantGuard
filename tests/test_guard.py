from quantguard import QuantitativeGuard, Status


def _status_of(result, expected_substring):
    for r in result.results:
        if expected_substring in r.claim.raw_text:
            return r.status
    raise AssertionError(f"No claim containing {expected_substring!r} found among {[r.claim.raw_text for r in result.results]}")


def test_correct_percentage_is_verified():
    guard = QuantitativeGuard()
    result = guard.verify("The completion rate was 33%.", ["Completion rate: 33%."])
    assert result.is_valid
    assert _status_of(result, "33%") == Status.VERIFIED


def test_wrong_percentage_is_contradicted():
    guard = QuantitativeGuard()
    result = guard.verify("The completion rate was 37%.", ["Completion rate: 33%."])
    assert not result.is_valid
    assert _status_of(result, "37%") == Status.CONTRADICTED


def test_rounding_is_verified_not_contradicted():
    guard = QuantitativeGuard()
    result = guard.verify("Latency was reduced to 12ms.", ["Latency achieved was 12.4ms."])
    assert _status_of(result, "12ms") == Status.VERIFIED


def test_entity_swap_is_caught():
    """The design doc's flagship anti-bug example: naive number
    matching would see 20 exists in the source and pass; QuantGuard
    must bind '20' to the correct entity (User A actually paid 10).
    """
    guard = QuantitativeGuard()
    result = guard.verify(
        "User A paid $20.",
        ["User A paid $10.", "User B paid $20."],
    )
    assert not result.is_valid
    assert _status_of(result, "20") == Status.CONTRADICTED


def test_temporal_mismatch_is_caught():
    guard = QuantitativeGuard()
    result = guard.verify(
        "Latency in 2024 was 12ms.",
        ["2024 value = 25ms.", "2025 value = 12ms."],
    )
    assert _status_of(result, "12ms") == Status.CONTRADICTED


def test_temporal_mismatch_is_caught_for_bound_claims():
    """End-to-end regression test: the "2024 vs 2025" trap must be
    caught for BOUND claims too, not just plain NUMBER claims. A
    correct-sounding bound ("under 50ms") for the wrong year is still
    a wrong claim.
    """
    guard = QuantitativeGuard()
    result = guard.verify(
        "In 2024, latency was under 50ms.",
        ["In 2025, latency was 12ms."],
    )
    assert not result.is_valid
    assert _status_of(result, "under 50ms") == Status.CONTRADICTED


def test_temporal_mismatch_is_caught_for_range_claims():
    guard = QuantitativeGuard()
    result = guard.verify(
        "In 2024, latency was between 10 and 20ms.",
        ["In 2025, latency was 15ms."],
    )
    assert not result.is_valid
    assert _status_of(result, "between 10 and 20ms") == Status.CONTRADICTED


def test_unit_conversion_across_dimension_is_verified():
    guard = QuantitativeGuard()
    result = guard.verify("The distance was 5 km.", ["Measured distance: 5000 m."])
    assert result.is_valid


def test_spec_section_mismatch_is_contradicted():
    guard = QuantitativeGuard()
    result = guard.verify(
        "This is defined in 3GPP TS 38.331 Section 5.3.5.4.",
        ["See 3GPP TS 38.331 Section 5.3.5.5 for details."],
    )
    assert not result.is_valid


def test_arithmetic_sum_verifies_claimed_total():
    guard = QuantitativeGuard()
    result = guard.verify(
        "There were 484 tasks in total.",
        ["262 completed, 197 not started, 22 in progress, 3 on hold."],
    )
    assert _status_of(result, "484") == Status.VERIFIED


def test_derived_percentage_verifies_part_over_whole():
    guard = QuantitativeGuard()
    result = guard.verify(
        "54.1% of tasks were completed.",
        ["262 completed out of 484 total tasks."],
    )
    assert _status_of(result, "54.1%") == Status.APPROXIMATE


def test_no_evidence_is_unsupported():
    guard = QuantitativeGuard()
    result = guard.verify("Latency was 12ms.", ["This document says nothing about timing."])
    assert not result.is_valid
    assert _status_of(result, "12ms") == Status.UNSUPPORTED


def test_result_is_json_serializable():
    import json

    guard = QuantitativeGuard()
    result = guard.verify("Latency was 12ms.", ["Latency achieved was 12.4ms."])
    serialized = json.dumps(result.to_dict())
    assert "VERIFIED" in serialized
