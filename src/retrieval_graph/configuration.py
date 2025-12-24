"""Define the configurable parameters for the semantic search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="mongodb",
        metadata={
            "description": "The vector store provider to use for retrieval. This template now uses MongoDB exclusively."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    mongodb_namespace: str = field(
        default="langgraph_retrieval_agent.default",
        metadata={
            "description": "MongoDB Atlas namespace to use (database.collection)."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: RunnableConfig | None = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """The configuration for the semantic search agent."""

    reranker_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "reranker"}},
    ] = field(
        default="cohere/rerank-english-v3.0",
        metadata={
            "description": "Cross-encoder used to rerank retrieved documents. Format: provider/model-name."
        },
    )
