"""Turns free text into structured Claim/Evidence-shaped dicts.

This is deliberately regex-based, not a full parser. It is honest
about that limitation: extraction handles the numeric surface forms
listed in the README well; it does not do general information
extraction. See claims.py for how extracted spans are then bound to
subjects/context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quantguard.models import ClaimKind, Comparator, Span
from quantguard.units import UNIT_ALIASES, sorted_unit_aliases

# A number: optional sign, digits, optional decimal part, optional
# scientific notation, optional thousands separators.
#
# The thousands-separator branch requires at least one ",ddd" group
# (`+`, not `*`) so a plain multi-digit number with no comma (e.g. a
# bare year like "2024") can't be short-matched as just its first 1-3
# digits by this branch before falling through -- regex alternation
# takes the first branch that matches AT ALL, not the longest overall
# match, so a `*` here would silently truncate "2024" to "202".
_NUMBER = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

_UNIT_PATTERN = "|".join(re.escape(alias) for alias in sorted_unit_aliases())

BOUND_PHRASES: dict[str, Comparator] = {
    "under": Comparator.LESS_THAN,
    "below": Comparator.LESS_THAN,
    "less than": Comparator.LESS_THAN,
    "fewer than": Comparator.LESS_THAN,
    "at most": Comparator.LESS_THAN_OR_EQUAL,
    "no more than": Comparator.LESS_THAN_OR_EQUAL,
    "up to": Comparator.LESS_THAN_OR_EQUAL,
    "over": Comparator.GREATER_THAN,
    "above": Comparator.GREATER_THAN,
    "more than": Comparator.GREATER_THAN,
    "greater than": Comparator.GREATER_THAN,
    "at least": Comparator.GREATER_THAN_OR_EQUAL,
    "no less than": Comparator.GREATER_THAN_OR_EQUAL,
    "minimum of": Comparator.GREATER_THAN_OR_EQUAL,
}
# Longest phrase first so "no more than" matches before "more than".
_BOUND_PHRASE_PATTERN = "|".join(
    re.escape(p) for p in sorted(BOUND_PHRASES, key=len, reverse=True)
)

_RANGE_CONNECTORS = r"(?:-|–|—|to|and)"

_REFERENCE_PATTERN = re.compile(
    r"""
    (?P<org>3GPP|IEEE|IETF|RFC|ITU-T|ITU)\s*
    (?P<doc>(?:TS|TR)?\s*\d+(?:\.\d+)*[A-Za-z]?)?
    \s*(?:,?\s*(?:Section|Sec\.?|§|Clause)\s*(?P<section>\d+(?:\.\d+)*[A-Za-z]?))?
    """,
    re.IGNORECASE | re.VERBOSE,
)
# RFC XXXX shorthand, and IEEE 802.11ax-style identifiers.
_RFC_PATTERN = re.compile(r"\bRFC\s*(?P<num>\d+)\b", re.IGNORECASE)
_IEEE_PATTERN = re.compile(r"\bIEEE\s*(?P<num>\d{3}(?:\.\d+)*[a-zA-Z]*)\b", re.IGNORECASE)


@dataclass
class RawNumber:
    value: float
    span: Span
    unit: str | None  # normalized unit symbol, or None
    is_percentage: bool


def _parse_number_token(token: str) -> float:
    return float(token.replace(",", ""))


def extract_numbers(text: str) -> list[RawNumber]:
    """Extract bare numbers, unit-qualified numbers, and percentages.

    Ranges and bounds are extracted separately (extract_ranges,
    extract_bounds) and take priority over the plain numbers they
    contain -- see extract_all, which de-duplicates by span overlap.
    """
    results: list[RawNumber] = []

    # Percentages: "54.1%" or "54.1 percent"
    pct_pattern = re.compile(rf"({_NUMBER})\s*(?:%|percent\b)", re.IGNORECASE)
    for m in pct_pattern.finditer(text):
        results.append(
            RawNumber(
                value=_parse_number_token(m.group(1)),
                span=Span(m.start(), m.end()),
                unit="%",
                is_percentage=True,
            )
        )

    # Unit-qualified numbers: "12.4 ms", "5km", "2.5 GB"
    if _UNIT_PATTERN:
        unit_pattern = re.compile(
            rf"({_NUMBER})\s*({_UNIT_PATTERN})\b", re.IGNORECASE
        )
        for m in unit_pattern.finditer(text):
            alias = m.group(2)
            normalized = UNIT_ALIASES.get(alias.lower())
            if normalized is None:
                continue
            results.append(
                RawNumber(
                    value=_parse_number_token(m.group(1)),
                    span=Span(m.start(), m.end()),
                    unit=normalized.symbol,
                    is_percentage=False,
                )
            )

    # Bare numbers not already covered above (e.g. plain counts: "484 tasks").
    bare_pattern = re.compile(_NUMBER)
    covered = [(r.span.start, r.span.end) for r in results]
    for m in bare_pattern.finditer(text):
        start, end = m.start(), m.end()
        if any(start >= c_start and end <= c_end for c_start, c_end in covered):
            continue
        results.append(
            RawNumber(
                value=_parse_number_token(m.group(0)),
                span=Span(start, end),
                unit=None,
                is_percentage=False,
            )
        )

    return results


@dataclass
class RawRange:
    low: float
    high: float
    span: Span
    unit: str | None


def extract_ranges(text: str) -> list[RawRange]:
    """Extract numeric ranges: '10-20ms', 'between 10 and 20 days', '5 to 10 GHz'."""
    results: list[RawRange] = []

    unit_group = rf"(?:\s*({_UNIT_PATTERN}))?" if _UNIT_PATTERN else ""

    pattern = re.compile(
        rf"(?:between\s+)?({_NUMBER}){unit_group}?\s*{_RANGE_CONNECTORS}\s*({_NUMBER})\s*({_UNIT_PATTERN})?\b",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        low = _parse_number_token(m.group(1))
        high = _parse_number_token(m.group(3))
        if high < low:
            continue  # not actually a range (e.g. "10 and 3 users" from unrelated text)
        unit_alias = m.group(2) or m.group(4)
        unit = UNIT_ALIASES.get(unit_alias.lower()).symbol if unit_alias and unit_alias.lower() in UNIT_ALIASES else None
        results.append(RawRange(low=low, high=high, span=Span(m.start(), m.end()), unit=unit))

    return results


@dataclass
class RawBound:
    value: float
    comparator: Comparator
    span: Span
    unit: str | None


def extract_bounds(text: str) -> list[RawBound]:
    """Extract inequality-style claims: 'under 50ms', 'at least 95%'."""
    results: list[RawBound] = []
    if not _BOUND_PHRASE_PATTERN:
        return results

    unit_group = rf"\s*({_UNIT_PATTERN})?" if _UNIT_PATTERN else ""
    pattern = re.compile(
        rf"\b({_BOUND_PHRASE_PATTERN})\s+({_NUMBER})\s*(%|percent)?{unit_group}",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        phrase = m.group(1).lower()
        comparator = BOUND_PHRASES.get(phrase)
        if comparator is None:
            continue
        value = _parse_number_token(m.group(2))
        if m.group(3):
            unit = "%"
        elif m.group(4):
            unit = UNIT_ALIASES.get(m.group(4).lower(), None)
            unit = unit.symbol if unit else None
        else:
            unit = None
        results.append(
            RawBound(value=value, comparator=comparator, span=Span(m.start(), m.end()), unit=unit)
        )

    return results


@dataclass
class RawReference:
    organization: str
    document: str | None
    section: str | None
    span: Span


def extract_references(text: str) -> list[RawReference]:
    """Extract technical spec references: '3GPP TS 38.331 Section 5.3.5.4', 'RFC 8446'."""
    results: list[RawReference] = []
    seen_spans: list[tuple[int, int]] = []

    for m in _REFERENCE_PATTERN.finditer(text):
        if not m.group("doc") and not m.group("section"):
            continue
        span = (m.start(), m.end())
        results.append(
            RawReference(
                organization=m.group("org").upper(),
                document=(m.group("doc") or "").strip() or None,
                section=m.group("section"),
                span=Span(*span),
            )
        )
        seen_spans.append(span)

    for m in _RFC_PATTERN.finditer(text):
        span = (m.start(), m.end())
        if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
            continue
        results.append(
            RawReference(organization="RFC", document=m.group("num"), section=None, span=Span(*span))
        )

    for m in _IEEE_PATTERN.finditer(text):
        span = (m.start(), m.end())
        if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
            continue
        results.append(
            RawReference(organization="IEEE", document=m.group("num"), section=None, span=Span(*span))
        )

    return results
