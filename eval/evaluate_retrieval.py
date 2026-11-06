"""Measure retrieval quality against the golden set."""

import json
import logging
import re
from pathlib import Path

from rag.config import settings
from rag.retrieve import retrieve

logger = logging.getLogger("evaluate")

GOLDEN_FILE = Path(__file__).resolve().parent / "golden_set.jsonl"


def normalize(text):
    """Collapse all whitespace so line breaks do not break matching."""
    return re.sub(r"\s+", " ", text)


def load_golden_set():
    """Read the evaluation questions."""
    with GOLDEN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_rank(passages, expected_pages, evidence):
    """Return the rank of the first passage matching page and evidence."""
    evidence = normalize(evidence)
    for i, p in enumerate(passages):
        if p["page"] in expected_pages and evidence in normalize(p["text"]):
            return i + 1
    return None


def main():
    """Run every answerable question and report recall and MRR."""
    questions = [q for q in load_golden_set() if q["page"] is not None]
    k = settings.top_k

    hits = 0
    reciprocal_sum = 0.0

    for q in questions:
        passages = retrieve(q["question"], k=k)
        expected = q["page"] if isinstance(q["page"], list) else [q["page"]]
        rank = find_rank(passages, expected, q["preuve"])

        if rank:
            hits += 1
            reciprocal_sum += 1 / rank
            status = f"OK rank {rank}"
        else:
            status = "MISS      "

        logger.info(
            "%s | expected p%s | got %s | %s",
            status,
            q["page"],
            [p["page"] for p in passages],
            q["question"][:45],
        )

    n = len(questions)
    logger.info("")
    logger.info("Recall@%s : %s/%s = %.0f%%", k, hits, n, 100 * hits / n)
    logger.info("MRR@%s    : %.3f", k, reciprocal_sum / n)


if __name__ == "__main__":
    main()
