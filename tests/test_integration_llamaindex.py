import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.base.response.schema import Response  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

from quantguard import QuantitativeGuard  # noqa: E402
from quantguard.integrations.llamaindex import make_verifying_query_engine, verify_response  # noqa: E402


def _fake_response(answer: str) -> Response:
    return Response(
        response=answer,
        source_nodes=[NodeWithScore(node=TextNode(text="Latency achieved was 12.4ms."), score=0.95)],
    )


def test_verify_response_extracts_node_content():
    guard = QuantitativeGuard()
    result = verify_response(guard, _fake_response("Latency was 18ms."))
    assert result.is_valid is False
    assert result.status_counts.get("CONTRADICTED") == 1


def test_verify_response_verified_case():
    guard = QuantitativeGuard()
    result = verify_response(guard, _fake_response("Latency was 12ms."))
    assert result.is_valid is True


class _FakeQueryEngine:
    def __init__(self, answer: str):
        self._answer = answer

    def query(self, question: str) -> Response:
        return _fake_response(self._answer)

    def some_other_method(self) -> str:
        return "passthrough works"


def test_make_verifying_query_engine_attaches_result():
    guard = QuantitativeGuard()
    wrapped = make_verifying_query_engine(_FakeQueryEngine("Latency was 18ms."), guard)

    response = wrapped.query("What was the latency?")
    assert hasattr(response, "quantguard_result")
    assert response.quantguard_result.is_valid is False


def test_make_verifying_query_engine_passes_through_other_attributes():
    guard = QuantitativeGuard()
    wrapped = make_verifying_query_engine(_FakeQueryEngine("Latency was 12ms."), guard)
    assert wrapped.some_other_method() == "passthrough works"
