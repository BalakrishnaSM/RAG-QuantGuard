"""LlamaIndex integration.

Lazily compatible with `llama_index.core` -- this module has no import
of llama_index at module load time (only inside functions that need
it for type checking convenience, which we avoid), so it never
requires llama_index to be installed just to import quantguard.

A LlamaIndex query engine returns a `Response` object with `.response`
(the answer text) and `.source_nodes` (a list of `NodeWithScore`,
each wrapping a `TextNode`/`Document` with `.get_content()`):

    from quantguard.integrations.llamaindex import verify_response

    response = query_engine.query("What was the latency?")
    result = verify_response(guard, response)
    result.is_valid
"""

from __future__ import annotations

from typing import Any


def _extract_node_text(node_with_score: Any) -> str:
    """A LlamaIndex NodeWithScore wraps a node with `.get_content()`;
    fall back to `str()` for anything else passed in directly.
    """
    node = getattr(node_with_score, "node", node_with_score)
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        return get_content()
    return str(node_with_score)


def verify_response(guard: "QuantitativeGuard", response: Any):  # noqa: F821
    """Verify a LlamaIndex query engine `Response` against its own
    `source_nodes`. Works with any object exposing `.response` (str)
    and `.source_nodes` (iterable), not just the exact Response class,
    so it also accepts a StreamingResponse's final aggregated form.
    """
    answer = str(response.response)
    source_chunks = [_extract_node_text(node) for node in getattr(response, "source_nodes", [])]
    return guard.verify(answer, source_chunks)


def make_verifying_query_engine(query_engine: Any, guard: "QuantitativeGuard"):  # noqa: F821
    """Wrap a LlamaIndex query engine so every `.query(...)` call also
    attaches a `.quantguard_result` attribute to the returned Response,
    without needing to subclass LlamaIndex's query engine base class.
    """

    class _VerifyingQueryEngine:
        def __init__(self, inner):
            self._inner = inner

        def query(self, *args, **kwargs):
            response = self._inner.query(*args, **kwargs)
            response.quantguard_result = verify_response(guard, response)
            return response

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _VerifyingQueryEngine(query_engine)
