from quantguard.evidence import extract_evidence
from quantguard.arithmetic import try_arithmetic_sum, try_derived_percentage
from quantguard.claims import extract_claims
from quantguard.models import Status
from quantguard.verification import TolerancePolicy


def test_arithmetic_sum_finds_matching_subset():
    claim = extract_claims("There were 484 tasks in total.")[0]
    evidence = extract_evidence(["262 completed, 197 not started, 22 in progress, 3 on hold."])
    result = try_arithmetic_sum(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is not None
    assert result.status == Status.VERIFIED


def test_arithmetic_sum_reports_contradiction_when_no_subset_matches():
    """When components exist but none sum to the claim, this should be
    a specific CONTRADICTED (citing the real computed total) rather
    than a vague None/UNSUPPORTED -- real evidence exists and disagrees.
    """
    claim = extract_claims("There were 999 tasks in total.")[0]
    evidence = extract_evidence(["262 completed, 197 not started, 22 in progress, 3 on hold."])
    result = try_arithmetic_sum(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is not None
    assert result.status == Status.CONTRADICTED
    assert result.reason is not None and "484" in result.reason


def test_arithmetic_sum_accepts_component_equal_to_the_total():
    """Regression test: a genuine component can coincidentally equal
    the claimed total (e.g. "20 done, 0 pending, total 20"). An
    earlier version filtered out any evidence value equal to the
    claim's value before searching for a matching subset, which
    silently dropped a real component instead of just ignoring a
    degenerate single-item match -- leaving too few components to
    verify a total that was, in fact, correct.
    """
    claim = extract_claims("There were 20 tasks in total.")[0]
    evidence = extract_evidence(["20 completed, 0 pending."])
    result = try_arithmetic_sum(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is not None
    assert result.status == Status.VERIFIED


def test_arithmetic_sum_returns_none_without_enough_components():
    claim = extract_claims("There were 999 tasks in total.")[0]
    evidence = extract_evidence(["Only 262 completed."])
    result = try_arithmetic_sum(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is None


def test_derived_percentage_finds_matching_pair():
    claim = extract_claims("54.1% of tasks were completed.")[0]
    evidence = extract_evidence(["262 completed out of 484 total tasks."])
    result = try_derived_percentage(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is not None
    assert result.status in (Status.VERIFIED, Status.APPROXIMATE)


def test_derived_percentage_reports_contradiction_when_no_pair_matches():
    """When a part/whole pair exists but implies a different
    percentage than claimed, this should be a specific CONTRADICTED
    citing the derived value, not a vague None/UNSUPPORTED.
    """
    claim = extract_claims("99.9% of tasks were completed.")[0]
    evidence = extract_evidence(["262 completed out of 484 total tasks."])
    result = try_derived_percentage(claim, evidence, source_index=0, policy=TolerancePolicy())
    assert result is not None
    assert result.status == Status.CONTRADICTED
    assert result.reason is not None and "54." in result.reason
