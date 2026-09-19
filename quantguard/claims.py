"""Turns the raw extraction primitives (extraction.py) into `Claim`
objects, binding each one to a lightweight local subject and a set of
context tokens.

This is deliberately a heuristic, not a dependency parser: it looks
at the words immediately preceding a number within its sentence,
skipping copulas and articles, and uses what's left as the `subject`.
This is intentionally the "lightweight contextual heuristics" path
described in the design doc -- a stronger spaCy-based resolver is a
plausible v0.2 addition, kept optional so the core library has no
required dependency beyond the standard library.
"""

from __future__ import annotations

import re

from quantguard.extraction import (
    extract_bounds,
    extract_numbers,
    extract_ranges,
    extract_references,
)
from quantguard.models import Claim, ClaimKind, Span

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A clause boundary is a comma followed by a letter (not a digit) --
# this splits "First was 999ms, second was 888ms" into two clauses for
# subject/context binding, without splitting inside a thousands-
# separated number like "12,345".
_CLAUSE_SPLIT = re.compile(r",\s+(?=[A-Za-z])")

_STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
    "of", "to", "at", "in", "on", "for", "with", "and", "or", "but",
    "this", "that", "it", "its", "has", "have", "had", "will", "would",
    "achieved", "reduced", "increased", "reported", "measured", "recorded",
}

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER_PATTERN = re.compile(r"\bQ[1-4]\s*(?:19|20)\d{2}\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def _sentence_spans(text: str) -> list[Span]:
    """Split text into sentences, returning each sentence's character span."""
    spans: list[Span] = []
    start = 0
    for part in _SENTENCE_SPLIT.split(text):
        end = start + len(part)
        spans.append(Span(start, end))
        start = end + 1  # account for the split whitespace
    if not spans:
        spans = [Span(0, len(text))]
    return spans


def _containing_sentence(span: Span, sentences: list[Span], text: str) -> Span:
    for sentence in sentences:
        if sentence.start <= span.start and span.end <= sentence.end + 1:
            return sentence
    return Span(0, len(text))


def _clause_spans(sentence: Span, text: str) -> list[Span]:
    """Split a sentence into comma-delimited clauses, for a TIGHTER
    subject/context window than the whole sentence.

    Without this, "First was 999ms, second was 888ms." binds BOTH
    numbers to a context containing both "first" and "second",
    diluting exactly the signal entity disambiguation depends on. A
    sentence enumerating components for arithmetic ("262 completed,
    197 not started, ...") is unaffected, since arithmetic pooling
    scopes by source chunk, not by sentence or clause.
    """
    spans: list[Span] = []
    start = sentence.start
    for part in _CLAUSE_SPLIT.split(text[sentence.start : sentence.end]):
        end = start + len(part)
        spans.append(Span(start, end))
        start = end + 2  # account for ", " consumed by the split
    if not spans:
        spans = [sentence]
    return spans


def _containing_clause(span: Span, clauses: list[Span], sentence: Span) -> Span:
    for clause in clauses:
        if clause.start <= span.start and span.end <= clause.end + 1:
            return clause
    return sentence


def _is_stopword(word: str) -> bool:
    """A word is a stopword unless it's a bare capitalized single letter.

    A capitalized single letter ("A", "B", "X") is almost always an
    entity/option label ("User A", "Node B") rather than the article
    "a" or similar -- lowercasing before this check would conflate
    "User A" with the word "a", stripping exactly the identifier that
    distinguishes it from "User B". A lowercase single letter ("a")
    is still treated as a stopword.
    """
    if len(word) == 1 and word.isupper():
        return False
    return word.lower() in _STOPWORDS


def _subject_before(span: Span, sentence: Span, text: str, max_words: int = 4) -> str:
    """Heuristic subject: up to `max_words` non-stopword tokens immediately
    preceding the claim within its sentence. Case is preserved for
    single-letter tokens (see `_is_stopword`) so entity labels like
    "A"/"B" survive; everything else is lowercased for matching.
    """
    preceding = text[sentence.start : span.start]
    words = _WORD_RE.findall(preceding)
    kept = [w if (len(w) == 1 and w.isupper()) else w.lower() for w in words if not _is_stopword(w)]
    return " ".join(kept[-max_words:])


def _context_tokens(sentence: Span, text: str) -> frozenset[str]:
    """Word-level context for evidence ranking. See `_is_stopword` for
    why single capitalized letters are preserved rather than folded
    into the stopword "a".
    """
    words = _WORD_RE.findall(text[sentence.start : sentence.end])
    return frozenset(
        (w if (len(w) == 1 and w.isupper()) else w.lower())
        for w in words
        if not _is_stopword(w)
    )


def _time_context(sentence: Span, text: str) -> str | None:
    window = text[sentence.start : sentence.end]
    m = _QUARTER_PATTERN.search(window)
    if m:
        return m.group(0)
    m = _YEAR_PATTERN.search(window)
    if m:
        return m.group(0)
    return None


def extract_claims(text: str) -> list[Claim]:
    """Extract every checkable Claim from generated text, in span order.

    Ranges and bounds take priority over the plain numbers they
    contain (a number inside an already-extracted range/bound span is
    dropped, not double-counted).
    """
    sentences = _sentence_spans(text)
    claims: list[Claim] = []
    consumed: list[tuple[int, int]] = []

    def _bind(span: Span) -> tuple[str, frozenset[str], str | None]:
        sentence = _containing_sentence(span, sentences, text)
        clauses = _clause_spans(sentence, text)
        clause = _containing_clause(span, clauses, sentence)
        subject = _subject_before(span, clause, text)
        tokens = _context_tokens(clause, text)
        # Time context uses the WIDER sentence window deliberately: "In
        # 2024, the first value was X and the second was Y" should
        # attach 2024 to both clauses' claims, not just the one nearest
        # the year mention.
        time_ctx = _time_context(sentence, text)
        return subject, tokens, time_ctx

    for r in extract_ranges(text):
        subject, tokens, time_ctx = _bind(r.span)
        sentence = _containing_sentence(r.span, sentences, text)
        claims.append(
            Claim(
                kind=ClaimKind.RANGE,
                raw_text=r.span.text_from(text),
                span=r.span,
                low=r.low,
                high=r.high,
                unit=r.unit,
                subject=subject,
                context_tokens=tokens,
                sentence=text[sentence.start : sentence.end],
                time_context=time_ctx,
            )
        )
        consumed.append((r.span.start, r.span.end))

    for b in extract_bounds(text):
        if any(b.span.start >= s and b.span.end <= e for s, e in consumed):
            continue
        subject, tokens, time_ctx = _bind(b.span)
        sentence = _containing_sentence(b.span, sentences, text)
        claims.append(
            Claim(
                kind=ClaimKind.BOUND,
                raw_text=b.span.text_from(text),
                span=b.span,
                value=b.value,
                unit=b.unit,
                comparator=b.comparator,
                subject=subject,
                context_tokens=tokens,
                sentence=text[sentence.start : sentence.end],
                time_context=time_ctx,
            )
        )
        consumed.append((b.span.start, b.span.end))

    for ref in extract_references(text):
        subject, tokens, time_ctx = _bind(ref.span)
        sentence = _containing_sentence(ref.span, sentences, text)
        claims.append(
            Claim(
                kind=ClaimKind.REFERENCE,
                raw_text=ref.span.text_from(text),
                span=ref.span,
                organization=ref.organization,
                document=ref.document,
                section=ref.section,
                subject=subject,
                context_tokens=tokens,
                sentence=text[sentence.start : sentence.end],
                time_context=time_ctx,
            )
        )
        consumed.append((ref.span.start, ref.span.end))

    for n in extract_numbers(text):
        if any(n.span.start >= s and n.span.end <= e for s, e in consumed):
            continue
        # A bare 4-digit year-looking number with no unit is temporal
        # CONTEXT for other claims in its sentence (see _time_context),
        # not an independently verifiable claim in its own right --
        # extracting it as a standalone NUMBER claim would ask "does
        # the year 2025 match some evidence value 2025?", which isn't
        # a meaningful question and only introduces spurious ambiguity
        # noise between unrelated year mentions.
        if n.unit is None and _YEAR_PATTERN.fullmatch(n.span.text_from(text)):
            continue
        subject, tokens, time_ctx = _bind(n.span)
        sentence = _containing_sentence(n.span, sentences, text)
        kind = ClaimKind.PERCENTAGE if n.is_percentage else ClaimKind.NUMBER
        claims.append(
            Claim(
                kind=kind,
                raw_text=n.span.text_from(text),
                span=n.span,
                value=n.value,
                unit=n.unit,
                subject=subject,
                context_tokens=tokens,
                sentence=text[sentence.start : sentence.end],
                time_context=time_ctx,
            )
        )

    claims.sort(key=lambda c: c.span.start)
    return claims


def extract_semantic_claims(text: str, structured_claims: list[Claim]) -> list[Claim]:
    """Wrap every sentence NOT already fully explained by structured
    claims as a SEMANTIC claim, for routing to an optional fallback
    handler.

    A sentence is "fully explained" when, after removing every
    structured claim's span from it, fewer than two meaningful
    (non-stopword) words remain -- e.g. "Latency was 12.4ms." reduces
    to just "Latency" once "12.4ms" is removed, so nothing is left to
    check semantically. A sentence like "Latency was 12.4ms, which
    proves the system is unreliable." still has a real remaining
    assertion after removing "12.4ms", and gets a semantic claim for
    the sentence as a whole.

    Only called by the guard when a `fallback_handler` is configured
    (see guard.py) -- without one, these sentences are simply outside
    what QuantGuard checks, exactly as documented for v0.1.
    """
    sentences = _sentence_spans(text)
    covered: list[tuple[int, int]] = [(c.span.start, c.span.end) for c in structured_claims]

    semantic_claims: list[Claim] = []
    for sentence in sentences:
        sentence_text = text[sentence.start : sentence.end].strip()
        if not sentence_text:
            continue

        remaining_chars = []
        pos = sentence.start
        for c_start, c_end in sorted(s for s in covered if sentence.start <= s[0] < sentence.end):
            if c_start > pos:
                remaining_chars.append(text[pos:c_start])
            pos = max(pos, c_end)
        remaining_chars.append(text[pos : sentence.end])
        remaining_text = "".join(remaining_chars)

        remaining_words = [w for w in _WORD_RE.findall(remaining_text) if not _is_stopword(w)]
        if len(remaining_words) < 2:
            continue

        semantic_claims.append(
            Claim(
                kind=ClaimKind.SEMANTIC,
                raw_text=sentence_text,
                span=sentence,
                sentence=sentence_text,
            )
        )
    return semantic_claims
