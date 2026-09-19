import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from quantguard import QuantitativeGuard  # noqa: E402
from quantguard.integrations.langchain import QuantGuardRunnable, verify_langchain_output  # noqa: E402


def _fake_chain_output(answer: str) -> dict:
    return {
        "input": "What was the latency?",
        "answer": answer,
        "context": [Document(page_content="Latency achieved was 12.4ms.")],
    }


def test_verify_langchain_output_extracts_document_page_content():
    guard = QuantitativeGuard()
    result = verify_langchain_output(guard, _fake_chain_output("Latency was 18ms."))
    assert result.is_valid is False
    assert result.status_counts.get("CONTRADICTED") == 1


def test_verify_langchain_output_verified_case():
    guard = QuantitativeGuard()
    result = verify_langchain_output(guard, _fake_chain_output("Latency was 12ms."))
    assert result.is_valid is True


def test_quantguard_runnable_composes_with_pipe_operator():
    guard = QuantitativeGuard()
    fake_chain = RunnableLambda(lambda _: _fake_chain_output("Latency was 18ms."))

    verified_chain = fake_chain | QuantGuardRunnable(guard)
    output = verified_chain.invoke({"input": "anything"})

    assert "quantguard_result" in output
    assert output["quantguard_result"].is_valid is False
    assert output["answer"] == "Latency was 18ms."  # passthrough preserved


def test_quantguard_runnable_custom_keys():
    guard = QuantitativeGuard()
    fake_chain = RunnableLambda(
        lambda _: {"my_answer": "Latency was 12ms.", "my_context": ["Latency achieved was 12.4ms."]}
    )

    verified_chain = fake_chain | QuantGuardRunnable(
        guard, answer_key="my_answer", context_key="my_context", result_key="qg"
    )
    output = verified_chain.invoke({})

    assert output["qg"].is_valid is True
