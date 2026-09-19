from quantguard.units import dimension_of, normalize


def test_time_units_normalize_to_seconds():
    assert normalize(1000, "ms")[0] == 1.0
    assert normalize(1, "s")[0] == 1.0
    assert normalize(1, "min")[0] == 60.0


def test_length_units_normalize_to_meters():
    assert normalize(5, "km")[0] == 5000.0
    assert normalize(5000, "m")[0] == 5000.0


def test_decimal_and_binary_data_units_are_separate_dimensions():
    """2.5 GB (decimal) and 2560 MiB (binary) describe the same number
    of bytes in different conventions -- QuantGuard treats them as
    different dimensions rather than silently reconciling them, per
    the design doc's rule against conflating GB and GiB.
    """
    assert dimension_of("GB") == "data_decimal"
    assert dimension_of("MiB") == "data_binary"
    assert dimension_of("GB") != dimension_of("MiB")


def test_percentage_is_its_own_dimension():
    assert dimension_of("%") == "percentage"
    assert normalize(50, "%")[0] == 50.0


def test_unknown_unit_passes_through_unchanged():
    value, dimension = normalize(42, "furlongs")
    assert value == 42
    assert dimension is None


def test_no_unit_passes_through_unchanged():
    value, dimension = normalize(42, None)
    assert value == 42
    assert dimension is None
