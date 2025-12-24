"""Main entrypoint for semantic search + reranking.

The graph now performs three steps:
1. Extract the user's query text
2. Retrieve documents from MongoDB using vector search
3. Rerank the retrieved documents and return the ranked list
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain.retrievers.document_compressors import CohereRerank
from langgraph.graph import StateGraph
from pydantic import BaseModel

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import get_message_text

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query from the latest user message.

    We no longer call an LLM here—semantic search should remain lightweight and
    deterministic. The newest human message text becomes the query.
    """
    messages = state.messages
    if not messages:
        raise ValueError("At least one user message is required to build a query.")

    human_input = get_message_text(messages[-1])
    return {"queries": [human_input]}


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    async with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def rerank(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Rerank retrieved documents and return the ordered list.

    Cohere's reranker is used when available; otherwise we fall back to sorting
    by the `score` metadata produced by the retriever.
    """

    configuration = Configuration.from_runnable_config(config)
    query = state.queries[-1]
    docs = state.retrieved_docs

    # Try cross-encoder reranking first
    try:
        # CohereRerank defaults to returning top 3 results; respect the caller's
        # requested `k` (if provided) or fall back to all retrieved docs.
        desired_k = configuration.search_kwargs.get("k", len(docs))
        top_n = min(desired_k, len(docs)) if docs else 0

        reranker = CohereRerank(
            model=configuration.reranker_model.split("/", 1)[1],
            top_n=top_n or 1,  # cohere requires a positive top_n
        )
        reranked = await reranker.acompress_documents(docs, query)
        return {"reranked_docs": list(reranked)}
    except Exception:
        # Fallback: sort by similarity score if present
        sorted_docs = sorted(
            docs,
            key=lambda d: d.metadata.get("score", 0),
            reverse=True,
        )
        return {"reranked_docs": sorted_docs}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_node(rerank)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "rerank")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
