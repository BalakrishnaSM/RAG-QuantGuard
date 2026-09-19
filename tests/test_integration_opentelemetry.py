import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from quantguard import QuantitativeGuard  # noqa: E402
from quantguard.integrations.opentelemetry import InstrumentedGuard  # noqa: E402


@pytest.fixture(scope="module")
def _otel_provider():
    """OpenTelemetry only allows the global TracerProvider to be set
    ONCE per process -- a second `set_tracer_provider` call is a no-op
    (with a warning), so setting a fresh provider per test silently
    leaves every test after the first still using the first test's
    provider/exporter. Set it once per module instead, and reset the
    exporter's captured spans between tests via `.clear()`.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def span_exporter(_otel_provider):
    _otel_provider.clear()
    return _otel_provider


def test_verify_emits_one_span_with_correct_attributes(span_exporter):
    guard = InstrumentedGuard(QuantitativeGuard())
    result = guard.verify("Latency was 18ms.", ["Latency achieved was 12.4ms."])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "quantguard.verify"
    assert span.attributes["quantguard.claim_count"] == 1
    assert span.attributes["quantguard.is_valid"] is False
    assert span.attributes["quantguard.status.CONTRADICTED"] == 1
    assert result.is_valid is False  # the wrapped call still returns the real result


def test_contradicted_claim_emits_a_span_event(span_exporter):
    guard = InstrumentedGuard(QuantitativeGuard())
    guard.verify("Latency was 18ms.", ["Latency achieved was 12.4ms."])

    span = span_exporter.get_finished_spans()[0]
    assert len(span.events) == 1
    assert span.events[0].name == "quantguard.contradicted"
    assert span.events[0].attributes["claim"] == "18ms"


def test_verified_claim_does_not_emit_a_span_event(span_exporter):
    guard = InstrumentedGuard(QuantitativeGuard())
    guard.verify("Latency was 12ms.", ["Latency achieved was 12.4ms."])

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["quantguard.is_valid"] is True
    assert len(span.events) == 0


def test_instrumented_guard_passes_through_other_attributes():
    guard = InstrumentedGuard(QuantitativeGuard(tolerance=0.1))
    assert guard.policy.numeric_tolerance == 0.1
