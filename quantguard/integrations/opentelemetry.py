"""OpenTelemetry integration: instrument QuantGuard calls with a trace
span and per-status counters, for observability in production.

Lazily imports `opentelemetry` -- has no effect on importing
`quantguard` itself without it installed.

    from quantguard.integrations.opentelemetry import InstrumentedGuard

    guard = InstrumentedGuard(QuantitativeGuard(tolerance=0.05))
    result = guard.verify(answer, source_chunks)  # same call, now traced

Each `verify()` call produces one span ("quantguard.verify") with
attributes for claim count and overall validity, plus child events for
each CONTRADICTED/AMBIGUOUS/UNSUPPORTED claim (kept off VERIFIED/
APPROXIMATE claims to avoid span event volume scaling with every
correct claim in a large answer). A counter metric
("quantguard.claims.verified") is incremented once per claim, with a
`status` attribute for filtering/aggregation in a metrics backend.
"""

from __future__ import annotations

from typing import Any


class InstrumentedGuard:
    """Wraps a QuantitativeGuard, adding an OpenTelemetry span and
    counter around every `.verify()` call. Delegates every other
    attribute/method to the wrapped guard unchanged.
    """

    def __init__(self, guard: "QuantitativeGuard", tracer_name: str = "quantguard", meter_name: str = "quantguard"):  # noqa: F821
        from opentelemetry import metrics, trace  # noqa: PLC0415

        self._guard = guard
        self._tracer = trace.get_tracer(tracer_name)
        self._meter = metrics.get_meter(meter_name)
        self._claims_counter = self._meter.create_counter(
            "quantguard.claims.verified",
            description="Count of claims verified, labeled by status.",
        )

    def verify(self, generated_text: str, source_chunks: list[str], **kwargs) -> Any:
        with self._tracer.start_as_current_span("quantguard.verify") as span:
            result = self._guard.verify(generated_text, source_chunks, **kwargs)

            span.set_attribute("quantguard.claim_count", len(result.results))
            span.set_attribute("quantguard.is_valid", result.is_valid)
            for status, count in result.status_counts.items():
                span.set_attribute(f"quantguard.status.{status}", count)

            for r in result.results:
                self._claims_counter.add(1, {"status": r.status.value})
                if r.status.value in ("CONTRADICTED", "AMBIGUOUS", "UNSUPPORTED"):
                    span.add_event(
                        f"quantguard.{r.status.value.lower()}",
                        {"claim": r.claim.raw_text, "reason": r.reason},
                    )

            return result

    def __getattr__(self, name: str):
        return getattr(self._guard, name)
