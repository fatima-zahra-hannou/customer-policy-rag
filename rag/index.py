"""Embed chunks and index them in the vector store."""

import json
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import settings

logger = logging.getLogger("index")


def load_chunks():
    """Read the chunks produced by the chunking step."""
    with settings.chunks_file.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    """Embed every chunk and store it in the vector database."""
    chunks = load_chunks()
    logger.info("%s chunks to index", len(chunks))

    model = SentenceTransformer(settings.embedding_model, device=settings.device)

    texts = [f"passage: {c['text']}" for c in chunks]
    vectors = model.encode(texts, batch_size=settings.batch_size, show_progress_bar=True)
    logger.info("embeddings computed, shape %s", vectors.shape)

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if settings.collection_name in [c.name for c in client.list_collections()]:
        logger.warning("collection '%s' already exists, recreating", settings.collection_name)
        client.delete_collection(settings.collection_name)

    collection = client.create_collection(
        settings.collection_name,
        metadata={"hnsw:space": settings.distance_metric},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"page": c["page"], "section": c["section"], "doc_id": c["doc_id"]} for c in chunks
        ],
    )

    logger.info("%s vectors indexed in '%s'", collection.count(), settings.collection_name)


if __name__ == "__main__":
    main()
