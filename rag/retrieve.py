"""Retrieve the most relevant chunks for a question."""

import logging
import sys

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.config import settings

logger = logging.getLogger("retrieve")

MODEL = SentenceTransformer(settings.embedding_model, device=settings.device)
RERANKER = CrossEncoder(settings.reranker_model, device=settings.device)


def get_collection():
    """Open the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_collection(settings.collection_name)


def dense_search(question, n):
    """Return the n chunks closest to the question in vector space."""
    collection = get_collection()
    vector = MODEL.encode([f"query: {question}"])
    res = collection.query(query_embeddings=vector.tolist(), n_results=n)

    candidates = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        candidates.append(
            {
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "page": meta["page"],
                "section": meta["section"],
                "distance": res["distances"][0][i],
            }
        )

    return candidates


def rerank(question, candidates, k):
    """Reorder candidates with a cross-encoder and keep the k best."""
    pairs = [(question, c["text"]) for c in candidates]
    scores = RERANKER.predict(pairs)

    for candidate, score in zip(candidates, scores, strict=True):
        candidate["score"] = float(score)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = candidates[:k]

    for i, candidate in enumerate(selected):
        candidate["rank"] = i + 1

    return selected


def retrieve(question, k=None, n_candidates=None):
    """Retrieve the k most relevant chunks for a question."""
    k = k or settings.top_k
    n_candidates = n_candidates or settings.n_candidates

    candidates = dense_search(question, n_candidates)
    return rerank(question, candidates, k)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    for p in retrieve(question):
        logger.info(
            "[%s] score %.3f | page %s | %s",
            p["rank"],
            p["score"],
            p["page"],
            p["section"],
        )
        logger.info("    %s", p["text"][:150].replace("\n", " "))
