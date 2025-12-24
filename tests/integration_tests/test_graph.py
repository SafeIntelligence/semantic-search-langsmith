from contextlib import asynccontextmanager

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langsmith import expect, unit

from retrieval_graph import graph, retrieval

pytestmark = pytest.mark.anyio


class _StubRetriever:
    """Simple in-memory retriever to avoid external services during tests."""

    async def ainvoke(self, _query: str, _config: RunnableConfig | None = None):
        return [
            Document(page_content="runner up", metadata={"score": 0.2}),
            Document(page_content="winner", metadata={"score": 0.9}),
        ]

    async def aadd_documents(self, _docs):
        return None


@asynccontextmanager
async def _stub_make_retriever(_config: RunnableConfig):
    yield _StubRetriever()


@unit
async def test_retrieval_graph_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval, "make_retriever", _stub_make_retriever)

    res = await graph.ainvoke(
        {"messages": [("user", "Where do cats perform synchronized swimming routes?")]},
        RunnableConfig(configurable={"user_id": "test-user"}),
    )

    ranked = res["reranked_docs"]
    expect(ranked[0].page_content).to_equal("winner")
    expect(ranked[1].metadata["score"]).to_be_less_than(ranked[0].metadata["score"])
