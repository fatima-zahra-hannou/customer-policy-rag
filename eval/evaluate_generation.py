"""Measure answer generation: response rate, abstention, citation validity."""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import settings  # noqa: E402
from rag.generate import answer  # noqa: E402

logger = logging.getLogger("evaluate_generation")

GOLDEN_FILE = Path(__file__).resolve().parent / "golden_set.jsonl"


def load_golden_set():
    """Read the evaluation questions."""
    with GOLDEN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_citation(response, expected_pages):
    """Return True if at least one cited page matches an expected page."""
    expected = expected_pages if isinstance(expected_pages, list) else [expected_pages]
    cited_pages = [s["page"] for s in response["sources"]]
    return any(p in expected for p in cited_pages)


def main():
    """Run every golden-set question through generation and report metrics."""
    questions = load_golden_set()
    results_file = Path(__file__).resolve().parent / "generation_results.jsonl"

    done = {}
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            done = {json.loads(line)["question"]: json.loads(line) for line in f}
        logger.info("resuming, %s questions already answered", len(done))

    answered_correctly = 0
    answered_wrongly_cited = 0
    wrongly_abstained = 0
    correctly_abstained = 0

    for q in questions:
        if q["question"] in done:
            response = done[q["question"]]
        else:
            response = answer(q["question"])
            response["question"] = q["question"]
            with results_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")
            time.sleep(settings.llm_delay_seconds)

        is_out_of_corpus = q["page"] is None

        if is_out_of_corpus:
            if response["abstained"]:
                correctly_abstained += 1
                status = "OK abstained"
            else:
                status = "FAIL answered (should abstain)"
        elif response["abstained"]:
            wrongly_abstained += 1
            status = f"FAIL abstained ({response['reason']}, score {response['top_score']:.3f})"
        elif check_citation(response, q["page"]):
            answered_correctly += 1
            status = "OK answered, correct citation"
        else:
            answered_wrongly_cited += 1
            status = "FAIL answered, wrong citation"

        logger.info("%-45s | %s", status, q["question"][:50])

    n_in_corpus = sum(1 for q in questions if q["page"] is not None)
    n_out_of_corpus = len(questions) - n_in_corpus

    logger.info("")
    logger.info("In-corpus questions   : %s", n_in_corpus)
    logger.info("  answered, correct citation : %s", answered_correctly)
    logger.info("  answered, wrong citation   : %s", answered_wrongly_cited)
    logger.info("  wrongly abstained          : %s", wrongly_abstained)
    logger.info("Out-of-corpus questions : %s", n_out_of_corpus)
    logger.info("  correctly abstained        : %s/%s", correctly_abstained, n_out_of_corpus)


if __name__ == "__main__":
    main()
