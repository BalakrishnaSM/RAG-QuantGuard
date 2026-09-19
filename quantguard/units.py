"""Unit definitions and normalization.

Each unit belongs to a `dimension` (time, length, data_decimal,
data_binary, frequency, percentage, mass). Two values can only be
compared if they share a dimension -- comparing a `data_decimal`
value against a `data_binary` value is deliberately NOT done
automatically (see the design note on GB vs GiB below), matching the
project's rule against silently conflating them.

`to_base` converts a value in this unit to the dimension's canonical
base unit (seconds, meters, bytes, hertz, grams, or the identity for
percentage), so any two same-dimension values can be compared after
multiplying by `to_base`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    symbol: str  # canonical short symbol, e.g. "ms", "m", "GB"
    dimension: str
    to_base: float  # multiply a value in this unit by this to get the base unit


# GB/MB/KB/TB are treated as decimal (1000-based), per SI/JEDEC-storage
# convention, and are a SEPARATE dimension from the explicit binary
# units (KiB/MiB/GiB/TiB, 1024-based). This is a deliberate design
# choice, not an oversight: "2.5 GB" and "2560 MiB" describe the same
# number of bytes, but treating "GB" as ambiguous between the two
# conventions would make QuantGuard's own verification non-deterministic.
# If a source means binary, it should say so (MiB/GiB/KiB/TiB).
_UNIT_TABLE: dict[str, Unit] = {
    # --- time (base: seconds) ---
    "ns": Unit("ns", "time", 1e-9),
    "nanosecond": Unit("ns", "time", 1e-9),
    "nanoseconds": Unit("ns", "time", 1e-9),
    "us": Unit("us", "time", 1e-6),
    "microsecond": Unit("us", "time", 1e-6),
    "microseconds": Unit("us", "time", 1e-6),
    "ms": Unit("ms", "time", 1e-3),
    "millisecond": Unit("ms", "time", 1e-3),
    "milliseconds": Unit("ms", "time", 1e-3),
    "s": Unit("s", "time", 1.0),
    "sec": Unit("s", "time", 1.0),
    "secs": Unit("s", "time", 1.0),
    "second": Unit("s", "time", 1.0),
    "seconds": Unit("s", "time", 1.0),
    "min": Unit("min", "time", 60.0),
    "mins": Unit("min", "time", 60.0),
    "minute": Unit("min", "time", 60.0),
    "minutes": Unit("min", "time", 60.0),
    "hr": Unit("hr", "time", 3600.0),
    "hrs": Unit("hr", "time", 3600.0),
    "hour": Unit("hr", "time", 3600.0),
    "hours": Unit("hr", "time", 3600.0),
    "day": Unit("day", "time", 86400.0),
    "days": Unit("day", "time", 86400.0),
    "week": Unit("week", "time", 604800.0),
    "weeks": Unit("week", "time", 604800.0),
    # --- length (base: meters) ---
    "mm": Unit("mm", "length", 1e-3),
    "millimeter": Unit("mm", "length", 1e-3),
    "millimeters": Unit("mm", "length", 1e-3),
    "cm": Unit("cm", "length", 1e-2),
    "centimeter": Unit("cm", "length", 1e-2),
    "centimeters": Unit("cm", "length", 1e-2),
    "m": Unit("m", "length", 1.0),
    "meter": Unit("m", "length", 1.0),
    "meters": Unit("m", "length", 1.0),
    "km": Unit("km", "length", 1000.0),
    "kilometer": Unit("km", "length", 1000.0),
    "kilometers": Unit("km", "length", 1000.0),
    # --- data, decimal (base: bytes) ---
    "b": Unit("B", "data_decimal", 1.0),
    "byte": Unit("B", "data_decimal", 1.0),
    "bytes": Unit("B", "data_decimal", 1.0),
    "kb": Unit("KB", "data_decimal", 1e3),
    "mb": Unit("MB", "data_decimal", 1e6),
    "gb": Unit("GB", "data_decimal", 1e9),
    "tb": Unit("TB", "data_decimal", 1e12),
    # --- data, binary (base: bytes) ---
    "kib": Unit("KiB", "data_binary", 1024.0),
    "mib": Unit("MiB", "data_binary", 1024.0**2),
    "gib": Unit("GiB", "data_binary", 1024.0**3),
    "tib": Unit("TiB", "data_binary", 1024.0**4),
    # --- frequency (base: hertz) ---
    "hz": Unit("Hz", "frequency", 1.0),
    "khz": Unit("kHz", "frequency", 1e3),
    "mhz": Unit("MHz", "frequency", 1e6),
    "ghz": Unit("GHz", "frequency", 1e9),
    # --- mass (base: grams) ---
    "mg": Unit("mg", "mass", 1e-3),
    "g": Unit("g", "mass", 1.0),
    "gram": Unit("g", "mass", 1.0),
    "grams": Unit("g", "mass", 1.0),
    "kg": Unit("kg", "mass", 1000.0),
    "kilogram": Unit("kg", "mass", 1000.0),
    "kilograms": Unit("kg", "mass", 1000.0),
    # --- percentage (base: itself, 0-100 scale) ---
    "%": Unit("%", "percentage", 1.0),
    "percent": Unit("%", "percentage", 1.0),
}

UNIT_ALIASES: dict[str, Unit] = _UNIT_TABLE


def sorted_unit_aliases() -> list[str]:
    """Aliases sorted longest-first, so a regex alternation of these
    (e.g. "ms|m") never matches the shorter "m" inside a token that
    should match the longer "ms".
    """
    return sorted(_UNIT_TABLE.keys(), key=len, reverse=True)


def normalize(value: float, unit_symbol: str | None) -> tuple[float, str | None]:
    """Convert `value` in `unit_symbol` to its dimension's base unit.

    Returns (normalized_value, dimension). If `unit_symbol` is None or
    unrecognized, returns (value, None) unchanged -- callers treat a
    None dimension as "compare raw values, no unit conversion".
    """
    if unit_symbol is None:
        return value, None

    unit = _UNIT_TABLE.get(unit_symbol.lower())
    if unit is None:
        return value, None

    return value * unit.to_base, unit.dimension


def dimension_of(unit_symbol: str | None) -> str | None:
    if unit_symbol is None:
        return None
    unit = _UNIT_TABLE.get(unit_symbol.lower())
    return unit.dimension if unit else None
