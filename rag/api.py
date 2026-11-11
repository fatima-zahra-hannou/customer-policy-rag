"""HTTP API exposing the RAG pipeline."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from rag.generate import answer

logger = logging.getLogger("api")

app = FastAPI(title="Customer Policy RAG", version="0.1.0")


class Question(BaseModel):
    """Incoming request body."""

    question: str


class Source(BaseModel):
    """One cited passage."""

    rank: int
    page: int
    section: str


class Answer(BaseModel):
    """Response body returned to the client."""

    question: str
    answer: str
    abstained: bool
    sources: list[Source]


@app.get("/health")
def health():
    """Liveness check, used by Docker and monitoring."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    """Answer a question with citations, or abstain."""
    logger.info("question received: %s", payload.question[:80])
    result = answer(payload.question)
    return Answer(
        question=result["question"],
        answer=result["answer"],
        abstained=result["abstained"],
        sources=[Source(**s) for s in result["sources"]],
    )
