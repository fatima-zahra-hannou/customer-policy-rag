"""Generate a cited answer from retrieved passages, or abstain."""

import logging
import sys

from pydantic import BaseModel

from rag.config import settings
from rag.llm import call_llm
from rag.retrieve import retrieve

logger = logging.getLogger("generate")

NOT_FOUND = "Information non trouvée dans les conditions générales de vente."

PROMPT = """Tu assistes les conseillers du service client Sephora.

Règles :
- Réponds uniquement à partir des extraits ci-dessous, jamais de tes connaissances.
- Après chaque affirmation, cite le numéro de l'extrait entre crochets, par exemple [2].
- Si les extraits ne permettent pas de répondre, mets "abstained" à true
  et "answer" à "{not_found}".
- Réponds en français, en deux à quatre phrases.
- Ne déduis une impossibilité que si l'extrait définit explicitement une liste fermée ;
  sinon, dis que l'information n'est pas mentionnée.
- Reprends les exceptions et les conditions telles qu'écrites, sans les résumer.
- Formule la réponse pour un conseiller, sans parler au nom de Sephora.
- Si tu ne fais que renvoyer l'utilisateur ailleurs (site, service client) sans donner
  l'information demandée, considère que tu n'as pas répondu : mets "abstained" à true.

<extraits>
{passages}
</extraits>

Question : {question}
"""


class LLMAnswer(BaseModel):
    """Structure the LLM must return."""

    answer: str
    citations: list[int]
    abstained: bool


def format_passages(passages):
    """Number passages so the LLM can cite them."""
    return "\n\n".join(
        f"[{p['rank']}] (page {p['page']}, {p['section']})\n{p['text']}" for p in passages
    )


def abstain(question, reason, top_score):
    """Build the answer returned when the system declines to answer."""
    return {
        "question": question,
        "answer": NOT_FOUND,
        "abstained": True,
        "reason": reason,
        "top_score": top_score,
        "sources": [],
    }


def answer(question):
    """Answer a question with citations, or abstain if evidence is missing."""
    passages = retrieve(question)
    top_score = passages[0]["score"] if passages else 0.0

    if top_score < settings.abstention_threshold:
        logger.info("abstained before LLM, top score %.3f", top_score)
        return abstain(question, "low_retrieval_score", top_score)

    prompt = PROMPT.format(
        not_found=NOT_FOUND, passages=format_passages(passages), question=question
    )
    result = LLMAnswer.model_validate_json(call_llm(prompt, response_schema=LLMAnswer))

    if result.abstained:
        return abstain(question, "model_declined", top_score)

    by_rank = {p["rank"]: p for p in passages}
    valid = [c for c in result.citations if c in by_rank]
    invalid = set(result.citations) - set(valid)
    if invalid:
        logger.warning("model cited unknown passages %s", sorted(invalid))

    return {
        "question": question,
        "answer": result.answer,
        "abstained": False,
        "reason": None,
        "top_score": top_score,
        "sources": [
            {
                "rank": c,
                "page": by_rank[c]["page"],
                "section": by_rank[c]["section"],
                "text": by_rank[c]["text"],
            }
            for c in valid
        ],
    }


if __name__ == "__main__":
    response = answer(" ".join(sys.argv[1:]))
    logger.info("top score %.3f", response["top_score"])
    logger.info("answer: %s", response["answer"])
    for s in response["sources"]:
        logger.info("  [%s] page %s | %s", s["rank"], s["page"], s["section"])
    if response["abstained"]:
        logger.info("abstained (%s), top score %.3f", response["reason"], response["top_score"])
