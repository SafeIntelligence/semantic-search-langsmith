"""CLI utility to ingest in-memory documents into MongoDB Atlas vector store.

This version expects a list of dictionaries (one per document) with keys:
``text``, ``title`` and ``doc_id``. It embeds each document as-is (no
chunking/splitting) and writes them to MongoDB Atlas using the same schema
expected by :func:`retrieval_graph.retrieval.make_mongodb_retriever`.
"""

from __future__ import annotations

import json
import argparse
import logging
import sys
import os
from typing import Sequence

from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

LOGGER = logging.getLogger("retrieval_graph.scripts.mongodb_ingest")


def build_documents_from_dicts(
    document_dicts: Sequence[dict[str, str]], user_id: str
) -> tuple[list[Document], list[str]]:
    """Create LangChain `Document`s from raw dictionaries without splitting."""

    documents: list[Document] = []
    doc_ids: list[str] = []

    for entry in document_dicts:
        missing_keys = {k for k in ("text", "title", "doc_id") if k not in entry}
        if missing_keys:
            LOGGER.warning("Skipping entry missing keys %s", ", ".join(sorted(missing_keys)))
            continue

        text = entry.get("text", "")
        if not text:
            LOGGER.warning("Skipping empty document for doc_id=%s", entry.get("doc_id"))
            continue

        doc_id = entry["doc_id"]
        metadata = {
            "title": entry.get("title", ""),
            "doc_id": doc_id,
            "user_id": user_id,
        }

        documents.append(Document(page_content=text, metadata=metadata))
        doc_ids.append(f"{user_id}::{doc_id}")

    return documents, doc_ids


def ingest_documents(args: argparse.Namespace, document_dicts: Sequence[dict[str, str]]) -> None:
    documents, doc_ids = build_documents_from_dicts(document_dicts, args.user_id)

    if not documents:
        LOGGER.warning("No documents prepared for ingestion. Exiting.")
        return


    uri = args.mongodb_uri or os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "Missing MongoDB connection string. Supply via --mongodb-uri or MONGODB_URI env var."
        )

    if args.cohere_api_key:
        os.environ["COHERE_API_KEY"] = args.cohere_api_key

    LOGGER.info("Initialising Cohere embedding model: %s", args.embedding_model)
    embedding_model = CohereEmbeddings(model=args.embedding_model)

    probe_vector = embedding_model.embed_query("dimension probe")
    embedding_dim = len(probe_vector)
    LOGGER.info("Embedding dimension detected: %d", embedding_dim)
    
    
    if args.dry_run:
        LOGGER.info("Dry-run enabled; skipping MongoDB writes.")
        return

    namespace = f"{args.database}.{args.collection}"
    LOGGER.info("Connecting to MongoDB namespace: %s", namespace)

    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        namespace=namespace,
        embedding=embedding_model,
        index_name=args.index_name,
    )

    batch_size = max(args.batch_size, 1)
    total_docs = len(documents)
    LOGGER.info("Adding %d documents to MongoDB (batch size: %d)", total_docs, batch_size)
    with tqdm(total=total_docs, desc="Uploading", unit="doc") as progress:
        for start in range(0, total_docs, batch_size):
            end = min(start + batch_size, total_docs)
            vector_store.add_documents(documents[start:end], ids=doc_ids[start:end])
            progress.update(end - start)

    LOGGER.info("Creating vector search index '%s'", args.index_name)
    vector_store.create_vector_search_index(
        dimensions=embedding_dim,
        filter_fields=["metadata.user_id", "user_id"],
    )

    LOGGER.info("Ingestion complete.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest in-memory documents (dicts with text/title/doc_id) into MongoDB Atlas with vector search support."
        )
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Value stored in metadata.user_id for all ingested documents.",
    )
    parser.add_argument(
        "--database",
        default="langgraph_retrieval_agent",
        help="MongoDB database name (default: langgraph_retrieval_agent).",
    )
    parser.add_argument(
        "--collection",
        default="beir_fiqa",
        help="MongoDB collection name (default: beir_fiqa).",
    )
    parser.add_argument(
        "--index-name",
        default="vector_index",
        help="Atlas vector search index name (default: vector_index).",
    )
    parser.add_argument(
        "--embedding-model",
        default="embed-english-v3.0",
        help="Cohere embedding model identifier (default: embed-english-v3.0).",
    )
    parser.add_argument(
        "--cohere-api-key",
        dest="cohere_api_key",
        help="Cohere API key. Defaults to the COHERE_API_KEY environment variable if not provided.",
    )
    parser.add_argument(
        "--mongodb-uri",
        dest="mongodb_uri",
        help="MongoDB connection string. Defaults to the MONGODB_URI environment variable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare documents without writing to MongoDB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of documents to upload per batch (default: 64).",
    )
    return parser

def obtain_data_subset(corpus, queries, qrels, subset_size=10000):
    
    corpus_subset = {}
    queries_subset = {}
    qrels_subset = {}

    for i, c in enumerate(corpus):
        if i > subset_size:
            break
        
        corpus_subset[c] = corpus[c]
        
    for q in qrels:
        relevant_docs = qrels[q]
        
        all_relevant_present = True
        for doc_id in relevant_docs:
            if doc_id not in corpus_subset:
                all_relevant_present = False
                break
        
        if all_relevant_present:
            queries_subset[q] = queries[q]
            qrels_subset[q] = relevant_docs
            
    with open("corpus_subset.json", "w") as f:
        json.dump(corpus_subset, f, indent=2)
    with open("queries_subset.json", "w") as f:
        json.dump(queries_subset, f, indent=2)
    with open("qrels_subset.json", "w") as f:
        json.dump(qrels_subset, f, indent=2)
            
    return corpus_subset, queries_subset, qrels_subset

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Silence noisy lower-level client logs (e.g., httpx from Cohere SDK).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    
    data_path = "beir_data/fiqa"

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Obtain a smaller subset for quicker ingestion during testing
    corpus_small, _, _ = obtain_data_subset(corpus, queries, qrels, subset_size=10000)
    
    corpus_list = [{"text": corpus_small[c]['text'], "title": corpus_small[c]['title'], "doc_id": c} for c in corpus_small]

    try:
        ingest_documents(args, corpus_list)
    except Exception:  # pragma: no cover - surface errors for CLI users
        LOGGER.exception("Ingestion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
