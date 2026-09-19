from quantguard import QuantitativeGuard
from quantguard.streaming.verifier import verify_stream


def test_streaming_verifies_simple_claim_word_by_word():
    guard = QuantitativeGuard()
    tokens = ["Latency ", "was ", "12ms", "."]
    events = list(verify_stream(guard, tokens, ["Latency achieved was 12.4ms."]))
    assert len(events) == 1
    assert events[0].result.claim.raw_text == "12ms"
    assert events[0].result.status.value == "VERIFIED"


def test_streaming_handles_number_growing_across_multiple_tokens():
    """The exact risk the module's docstring warns about: a number
    built up digit-by-digit (as a subword tokenizer might produce)
    must resolve to ONE final claim, not several premature partial
    ones ("1", then "18", each separately finalized and verified).
    """
    guard = QuantitativeGuard()
    tokens = ["Latency ", "was ", "1", "8", "ms", ", ", "a ", "real ", "number."]
    events = list(verify_stream(guard, tokens, ["Latency achieved was 12.4ms."]))

    claim_texts = [e.result.claim.raw_text for e in events]
    assert claim_texts == ["18ms"]  # exactly one claim, the final settled form


def test_streaming_flushes_claim_at_end_of_stream_with_no_settle_margin():
    """A claim right at the very end of the stream, with no trailing
    text to provide a settle margin, must still be finalized once the
    stream ends (not silently dropped).
    """
    guard = QuantitativeGuard()
    tokens = ["Latency ", "was ", "12ms"]  # stream ends immediately after the claim
    events = list(verify_stream(guard, tokens, ["Latency achieved was 12.4ms."]))
    assert len(events) == 1
    assert events[0].result.claim.raw_text == "12ms"


def test_streaming_yields_multiple_claims_in_order():
    guard = QuantitativeGuard()
    tokens = ["Latency was 12ms, and ", "uptime ", "was ", "99%", ", all good."]
    events = list(
        verify_stream(guard, tokens, ["Latency achieved was 12.4ms.", "Uptime recorded: 99%."])
    )
    claim_texts = [e.result.claim.raw_text for e in events]
    assert claim_texts == ["12ms", "99%"]


def test_streaming_result_matches_non_streaming_result():
    """Sanity check: streaming a whole answer at once should agree with
    the equivalent non-streaming .verify() call on the same text.
    """
    guard = QuantitativeGuard()
    text = "Latency was 18ms."
    sources = ["Latency achieved was 12.4ms."]

    batch_result = guard.verify(text, sources)
    stream_events = list(verify_stream(guard, [text], sources))

    assert len(stream_events) == len(batch_result.results)
    assert stream_events[0].result.status == batch_result.results[0].status
