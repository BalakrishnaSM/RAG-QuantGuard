"""LangChain integration.

Lazily imports `langchain_core` -- importing `quantguard.integrations.langchain`
does nothing until you actually construct `QuantGuardRunnable`, so the
core library never requires langchain to be installed.

Typical LangChain RAG chains (e.g. built with
`create_retrieval_chain`) produce a dict shaped like:

    {"input": "...", "answer": "...", "context": [Document(...), ...]}

`QuantGuardRunnable` consumes exactly that shape and adds a
`quantguard_result: GuardResult` key, so it composes directly onto
the end of an existing chain with `|`:

    from quantguard.integrations.langchain import QuantGuardRunnable

    verified_chain = rag_chain | QuantGuardRunnable(guard)
    output = verified_chain.invoke({"input": "What was the latency?"})
    output["quantguard_result"].is_valid
"""

from __future__ import annotations

from typing import Any


def _extract_text(item: Any) -> str:
    """A LangChain Document has `.page_content`; anything else is
    treated as already being plain text (str(item)).
    """
    page_content = getattr(item, "page_content", None)
    return page_content if page_content is not None else str(item)


def verify_langchain_output(
    guard: "QuantitativeGuard",  # noqa: F821 -- documented via docstring, avoids a hard import at module load
    chain_output: dict,
    answer_key: str = "answer",
    context_key: str = "context",
):
    """Verify a LangChain chain's dict output directly, without needing
    a Runnable in the chain. Useful for a one-off check rather than
    permanently composing verification into the chain itself.
    """
    answer = chain_output[answer_key]
    context_items = chain_output.get(context_key, [])
    source_chunks = [_extract_text(item) for item in context_items]
    return guard.verify(answer, source_chunks)


def QuantGuardRunnable(guard: "QuantitativeGuard", answer_key: str = "answer", context_key: str = "context", result_key: str = "quantguard_result"):  # noqa: N802
    """Build a LangChain Runnable that verifies `chain_output[answer_key]`
    against `chain_output[context_key]` and adds the GuardResult under
    `result_key`, passing every other key through unchanged.

    Returns a `RunnableLambda` rather than requiring a custom Runnable
    subclass -- this is the standard, idiomatic way to plug arbitrary
    Python logic into a LangChain pipeline (LCEL).
    """
    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415

    def _verify(chain_output: dict) -> dict:
        result = verify_langchain_output(guard, chain_output, answer_key, context_key)
        return {**chain_output, result_key: result}

    return RunnableLambda(_verify, name="QuantGuardRunnable")
