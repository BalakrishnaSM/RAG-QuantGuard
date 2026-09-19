"""Adapters showing how to wire a REAL NLI model or LLM into the
`FallbackHandler` protocol, in place of the lexical-overlap default.

These are documented, structurally-correct reference implementations
following each library's public API as of the versions noted below.
They are lazy-imported (no hard dependency on transformers/an LLM SDK
just to import quantguard), and this sandbox has no access to
download hosted models or call external LLM APIs, so they're
exercised here only against a fake/mock model in tests -- treat them
as a documented starting point to adapt, not a guarantee against a
live model's exact output format, which can drift between versions.
"""

from __future__ import annotations

from typing import Any, Callable

from quantguard.fallback.base import FallbackResult
from quantguard.models import Status


def make_transformers_nli_fallback(
    pipeline_or_factory: Any = None,
    model_name: str = "facebook/bart-large-mnli",
    entailment_threshold: float = 0.7,
    contradiction_threshold: float = 0.7,
):
    """Build a FallbackHandler backed by a HuggingFace `transformers`
    zero-shot-classification / NLI pipeline.

        from transformers import pipeline
        nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        handler = make_transformers_nli_fallback(nli)
        guard = QuantitativeGuard(fallback_handler=handler)

    If `pipeline_or_factory` is None, this lazily constructs one with
    `model_name` on first call (deferred so importing this module
    never requires `transformers` to be installed).
    """

    def handler(claim_text: str, source_chunks: list[str]) -> FallbackResult:
        nonlocal pipeline_or_factory
        if pipeline_or_factory is None:
            from transformers import pipeline  # noqa: PLC0415

            pipeline_or_factory = pipeline("zero-shot-classification", model=model_name)

        if not source_chunks:
            return FallbackResult(Status.UNSUPPORTED, 0.0, "No source chunks provided.")

        # Use the single best-supporting chunk as the NLI premise.
        # zero-shot-classification scores `claim_text` against the
        # candidate labels ["supported", "contradicted"], with the
        # chunk as context prepended to the sequence.
        best_result = None
        for chunk in source_chunks:
            sequence = f"{chunk}\n\nClaim: {claim_text}"
            output = pipeline_or_factory(sequence, candidate_labels=["supported", "contradicted", "unrelated"])
            top_label, top_score = output["labels"][0], output["scores"][0]
            if best_result is None or top_score > best_result[1]:
                best_result = (top_label, top_score, chunk)

        label, score, chunk = best_result
        if label == "supported" and score >= entailment_threshold:
            return FallbackResult(Status.VERIFIED, score, f"NLI model scored 'supported' at {score:.2f} against: {chunk[:100]!r}")
        if label == "contradicted" and score >= contradiction_threshold:
            return FallbackResult(Status.CONTRADICTED, score, f"NLI model scored 'contradicted' at {score:.2f} against: {chunk[:100]!r}")
        return FallbackResult(Status.AMBIGUOUS, score, f"NLI model was not confident ({label} at {score:.2f}).")

    return handler


def make_llm_fallback(
    call_llm: Callable[[str], str],
    parse_response: Callable[[str], tuple[Status, float, str]] | None = None,
):
    """Build a FallbackHandler backed by an arbitrary LLM call.

    `call_llm` is any `str -> str` function -- wrap your provider's
    SDK call in a lambda/function of that shape:

        from anthropic import Anthropic
        client = Anthropic()

        def call_llm(prompt: str) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        handler = make_llm_fallback(call_llm)
        guard = QuantitativeGuard(fallback_handler=handler)

    The default prompt asks for a single-word verdict on the first
    line ("SUPPORTED"/"CONTRADICTED"/"UNSUPPORTED") followed by a
    one-line explanation, and parses that. Pass `parse_response` to
    use your own prompt/response format instead.
    """

    def default_parse(response: str) -> tuple[Status, float, str]:
        lines = response.strip().splitlines()
        verdict = lines[0].strip().upper() if lines else ""
        explanation = " ".join(lines[1:]).strip() or response.strip()

        mapping = {
            "SUPPORTED": Status.VERIFIED,
            "CONTRADICTED": Status.CONTRADICTED,
            "UNSUPPORTED": Status.UNSUPPORTED,
            "AMBIGUOUS": Status.AMBIGUOUS,
        }
        status = mapping.get(verdict, Status.AMBIGUOUS)
        confidence = 0.7 if verdict in mapping else 0.3
        return status, confidence, explanation

    parser = parse_response or default_parse

    def handler(claim_text: str, source_chunks: list[str]) -> FallbackResult:
        context = "\n\n".join(f"Source {i + 1}: {chunk}" for i, chunk in enumerate(source_chunks))
        prompt = (
            "You are checking whether a claim is supported by the given sources.\n\n"
            f"{context}\n\nClaim: {claim_text}\n\n"
            "Respond with exactly one word on the first line -- SUPPORTED, CONTRADICTED, "
            "or UNSUPPORTED -- then a one-line explanation on the next line."
        )
        response = call_llm(prompt)
        status, confidence, explanation = parser(response)
        return FallbackResult(status, confidence, explanation)

    return handler
