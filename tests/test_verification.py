from quantguard.models import Claim, ClaimKind, Comparator, Evidence, Span, Status
from quantguard.verification import TolerancePolicy, compare_bound, compare_range, compare_reference, compare_numeric


def _number_claim(raw_text, value, unit=None, time_context=None):
    return Claim(kind=ClaimKind.NUMBER, raw_text=raw_text, span=Span(0, len(raw_text)), value=value, unit=unit, time_context=time_context)


def _number_evidence(raw_text, value, unit=None, time_context=None, source_index=0):
    return Evidence(kind=ClaimKind.NUMBER, raw_text=raw_text, span=Span(0, len(raw_text)), source_index=source_index, value=value, unit=unit, time_context=time_context)


def test_rounding_tolerance_does_not_explode_after_unit_normalization():
    """Regression test: computing the rounding interval on the
    ORIGINAL text ("12ms" -> +/-0.5) but applying it to the
    NORMALIZED value (0.012 seconds) used to make 12ms and 25ms
    falsely "match" -- the +/-0.5 tolerance, in seconds, dwarfed both
    values. This must be CONTRADICTED, not VERIFIED.
    """
    claim = _number_claim("12ms", 12.0, unit="ms")
    evidence = _number_evidence("25ms", 25.0, unit="ms")
    result = compare_numeric(claim, evidence, TolerancePolicy())
    assert result.status == Status.CONTRADICTED


def test_exact_rounding_match_across_precision():
    claim = _number_claim("12ms", 12.0, unit="ms")
    evidence = _number_evidence("12.4ms", 12.4, unit="ms")
    result = compare_numeric(claim, evidence, TolerancePolicy())
    assert result.status == Status.VERIFIED


def test_unit_conversion_verifies_equal_quantities():
    claim = _number_claim("5 km", 5.0, unit="km")
    evidence = _number_evidence("5000 m", 5000.0, unit="m")
    result = compare_numeric(claim, evidence, TolerancePolicy())
    assert result.status == Status.VERIFIED


def test_incompatible_unit_dimension_is_unsupported():
    claim = _number_claim("5 km", 5.0, unit="km")
    evidence = _number_evidence("5 GB", 5.0, unit="GB")
    result = compare_numeric(claim, evidence, TolerancePolicy())
    assert result.status == Status.UNSUPPORTED


def test_mismatched_time_context_is_contradicted_even_if_number_matches():
    """The '2024 vs 2025' trap: naive number matching would see 12
    exists in the source and pass; time-context awareness must catch
    that the 12 belongs to a different year than claimed.
    """
    claim = _number_claim("12ms", 12.0, unit="ms", time_context="2024")
    evidence = _number_evidence("12ms", 12.0, unit="ms", time_context="2025")
    result = compare_numeric(claim, evidence, TolerancePolicy())
    assert result.status == Status.CONTRADICTED


def test_tolerance_approximate_within_configured_band():
    claim = _number_claim("100", 100.0)
    evidence = _number_evidence("103", 103.0)
    result = compare_numeric(claim, evidence, TolerancePolicy(verification_tolerance=0.05))
    assert result.status == Status.APPROXIMATE


def test_bound_satisfied():
    claim = Claim(kind=ClaimKind.BOUND, raw_text="under 50ms", span=Span(0, 10), value=50.0, unit="ms", comparator=Comparator.LESS_THAN)
    evidence = _number_evidence("12.4ms", 12.4, unit="ms")
    result = compare_bound(claim, evidence)
    assert result.status == Status.VERIFIED


def test_bound_violated():
    claim = Claim(kind=ClaimKind.BOUND, raw_text="under 50ms", span=Span(0, 10), value=50.0, unit="ms", comparator=Comparator.LESS_THAN)
    evidence = _number_evidence("62ms", 62.0, unit="ms")
    result = compare_bound(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_bound_mismatched_time_context_is_contradicted_even_if_bound_is_satisfied():
    """Regression test: compare_bound previously didn't apply the
    "2024 vs 2025" temporal guard at all, unlike compare_numeric --
    so a bound claim scoped to one period could be silently verified
    against evidence for a completely different period, as long as
    the number happened to satisfy the bound. A value that's correct
    for the wrong period is not a correct claim.
    """
    claim = Claim(kind=ClaimKind.BOUND, raw_text="under 50ms", span=Span(0, 10), value=50.0, unit="ms", comparator=Comparator.LESS_THAN, time_context="2024")
    evidence = _number_evidence("12ms", 12.0, unit="ms", time_context="2025")
    result = compare_bound(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_bound_matching_time_context_is_unaffected():
    claim = Claim(kind=ClaimKind.BOUND, raw_text="under 50ms", span=Span(0, 10), value=50.0, unit="ms", comparator=Comparator.LESS_THAN, time_context="2024")
    evidence = _number_evidence("12ms", 12.0, unit="ms", time_context="2024")
    result = compare_bound(claim, evidence)
    assert result.status == Status.VERIFIED


def test_range_inside():
    claim = Claim(kind=ClaimKind.RANGE, raw_text="10-20ms", span=Span(0, 7), low=10.0, high=20.0, unit="ms")
    evidence = _number_evidence("15ms", 15.0, unit="ms")
    result = compare_range(claim, evidence)
    assert result.status == Status.VERIFIED


def test_range_outside():
    claim = Claim(kind=ClaimKind.RANGE, raw_text="10-20ms", span=Span(0, 7), low=10.0, high=20.0, unit="ms")
    evidence = _number_evidence("25ms", 25.0, unit="ms")
    result = compare_range(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_range_mismatched_time_context_is_contradicted_even_if_value_is_inside():
    """Mirror of the BOUND regression test, for RANGE claims."""
    claim = Claim(kind=ClaimKind.RANGE, raw_text="10-20ms", span=Span(0, 7), low=10.0, high=20.0, unit="ms", time_context="2024")
    evidence = _number_evidence("15ms", 15.0, unit="ms", time_context="2025")
    result = compare_range(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_range_matching_time_context_is_unaffected():
    claim = Claim(kind=ClaimKind.RANGE, raw_text="10-20ms", span=Span(0, 7), low=10.0, high=20.0, unit="ms", time_context="2024")
    evidence = _number_evidence("15ms", 15.0, unit="ms", time_context="2024")
    result = compare_range(claim, evidence)
    assert result.status == Status.VERIFIED


def test_reference_section_mismatch_is_contradicted():
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="TS 38.331 5.3.5.4", span=Span(0, 10), organization="3GPP", document="TS 38.331", section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="TS 38.331 5.3.5.5", span=Span(0, 10), source_index=0, organization="3GPP", document="TS 38.331", section="5.3.5.5")
    result = compare_reference(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_reference_exact_match_is_verified():
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="TS 38.331 5.3.5.4", span=Span(0, 10), organization="3GPP", document="TS 38.331", section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="TS 38.331 5.3.5.4", span=Span(0, 10), source_index=0, organization="3GPP", document="TS 38.331", section="5.3.5.4")
    result = compare_reference(claim, evidence)
    assert result.status == Status.VERIFIED


def test_reference_missing_document_on_claim_is_not_contradicted():
    """Regression test: a citation that omits the document number
    (e.g. "3GPP Section 5.3.5.4" with no "TS 38.331") must not be
    treated the same as a citation to the WRONG document. The section
    still matches, so this should be a (possibly downgraded) match,
    not CONTRADICTED.
    """
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="3GPP Section 5.3.5.4", span=Span(0, 10), organization="3GPP", document=None, section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="3GPP TS 38.331 Section 5.3.5.4", span=Span(0, 10), source_index=0, organization="3GPP", document="TS 38.331", section="5.3.5.4")
    result = compare_reference(claim, evidence)
    assert result.status in (Status.VERIFIED, Status.APPROXIMATE)


def test_reference_missing_document_on_evidence_is_not_contradicted():
    """Mirror of the above: the evidence chunk doesn't happen to
    restate the document number, but organization and section agree.
    """
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="3GPP TS 38.331 Section 5.3.5.4", span=Span(0, 10), organization="3GPP", document="TS 38.331", section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="3GPP Section 5.3.5.4", span=Span(0, 10), source_index=0, organization="3GPP", document=None, section="5.3.5.4")
    result = compare_reference(claim, evidence)
    assert result.status in (Status.VERIFIED, Status.APPROXIMATE)


def test_reference_wrong_document_is_still_contradicted():
    """The fix must not weaken the case that actually matters: both
    sides specify a document and they genuinely disagree.
    """
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="3GPP TS 38.331 Section 5.3.5.4", span=Span(0, 10), organization="3GPP", document="TS 38.331", section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="3GPP TS 38.212 Section 5.3.5.4", span=Span(0, 10), source_index=0, organization="3GPP", document="TS 38.212", section="5.3.5.4")
    result = compare_reference(claim, evidence)
    assert result.status == Status.CONTRADICTED


def test_reference_wrong_organization_is_still_contradicted():
    claim = Claim(kind=ClaimKind.REFERENCE, raw_text="3GPP Section 5.3.5.4", span=Span(0, 10), organization="3GPP", document=None, section="5.3.5.4")
    evidence = Evidence(kind=ClaimKind.REFERENCE, raw_text="IEEE Section 5.3.5.4", span=Span(0, 10), source_index=0, organization="IEEE", document=None, section="5.3.5.4")
    result = compare_reference(claim, evidence)
    assert result.status == Status.CONTRADICTED
