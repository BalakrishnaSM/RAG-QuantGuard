from quantguard.claims import extract_claims
from quantguard.models import ClaimKind


def test_number_regex_does_not_truncate_multidigit_numbers():
    """Regression test: the number regex used to short-match "2024" as
    "202" + "4" because its thousands-separator branch matched greedily
    before falling through to the plain-number branch. Checked at the
    extract_numbers level (not extract_claims), since a bare year is
    deliberately excluded from standalone Claims -- see
    test_bare_year_is_context_not_a_standalone_claim.
    """
    from quantguard.extraction import extract_numbers

    numbers = extract_numbers("The 2024 value was 25ms.")
    values = [n.value for n in numbers]
    assert 2024.0 in values
    assert 202.0 not in values


def test_bare_year_is_context_not_a_standalone_claim():
    """A bare 4-digit year with no unit is temporal context for other
    claims in its sentence, not an independently verifiable claim --
    extracting it standalone would ask a meaningless question ("does
    the year 2024 match evidence value 2024?") and introduce spurious
    ambiguity between unrelated year mentions.
    """
    claims = extract_claims("The 2024 value was 25ms.")
    assert not any(c.value == 2024.0 for c in claims)

    number_claim = next(c for c in claims if c.kind == ClaimKind.NUMBER)
    assert number_claim.value == 25.0
    assert number_claim.time_context == "2024"


def test_extracts_thousands_separator_number():
    claims = extract_claims("The system processed 12,345 requests.")
    assert any(c.value == 12345.0 for c in claims)


def test_percentage_extraction():
    claims = extract_claims("Accuracy was 94.3%.")
    pct = [c for c in claims if c.kind == ClaimKind.PERCENTAGE]
    assert len(pct) == 1
    assert pct[0].value == 94.3
    assert pct[0].unit == "%"


def test_unit_qualified_number_extraction():
    claims = extract_claims("Latency was 12.4ms.")
    number_claims = [c for c in claims if c.kind == ClaimKind.NUMBER]
    assert len(number_claims) == 1
    assert number_claims[0].value == 12.4
    assert number_claims[0].unit == "ms"


def test_range_extraction():
    claims = extract_claims("Latency was between 10 and 20ms.")
    ranges = [c for c in claims if c.kind == ClaimKind.RANGE]
    assert len(ranges) == 1
    assert ranges[0].low == 10.0
    assert ranges[0].high == 20.0


def test_bound_extraction():
    claims = extract_claims("Latency was under 50ms.")
    bounds = [c for c in claims if c.kind == ClaimKind.BOUND]
    assert len(bounds) == 1
    assert bounds[0].value == 50.0


def test_reference_extraction_no_trailing_period():
    """Regression test: the section pattern used to greedily consume a
    sentence-final period as part of the section number.
    """
    claims = extract_claims("This is defined in 3GPP TS 38.331 Section 5.3.5.4. More text follows.")
    refs = [c for c in claims if c.kind == ClaimKind.REFERENCE]
    assert len(refs) == 1
    assert refs[0].section == "5.3.5.4"
    assert not refs[0].section.endswith(".")


def test_rfc_reference_extraction():
    claims = extract_claims("RFC 8446 defines TLS 1.3.")
    refs = [c for c in claims if c.kind == ClaimKind.REFERENCE]
    assert any(r.organization == "RFC" and r.document == "8446" for r in refs)


def test_range_and_bound_take_priority_over_bare_numbers():
    """A number inside an already-extracted range/bound span should not
    ALSO be extracted as a separate bare-number claim.
    """
    claims = extract_claims("Latency was between 10 and 20ms.")
    bare_numbers = [c for c in claims if c.kind == ClaimKind.NUMBER and c.value in (10.0, 20.0)]
    assert bare_numbers == []
