"""Retrieve the most relevant chunks for a question."""

import logging
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import settings

logger = logging.getLogger("retrieve")

MODEL = SentenceTransformer(settings.embedding_model, device=settings.device)


def get_collection():
    """Open the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_collection(settings.collection_name)


def retrieve(question, k=None):
    """Return the k chunks closest to the question, best first."""
    k = k or settings.top_k
    collection = get_collection()

    vector = MODEL.encode([f"query: {question}"])
    res = collection.query(query_embeddings=vector.tolist(), n_results=k)

    passages = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        passages.append(
            {
                "rank": i + 1,
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "page": meta["page"],
                "section": meta["section"],
                "distance": res["distances"][0][i],
            }
        )

    return passages


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    for p in retrieve(question):
        logger.info(
            "[%s] distance %.3f | page %s | %s",
            p["rank"],
            p["distance"],
            p["page"],
            p["section"],
        )
        logger.info("    %s", p["text"][:250].replace("\n", " "))
