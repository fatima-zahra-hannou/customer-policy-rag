"""Add a short context sentence to each chunk to improve retrieval."""

import json
import logging
import time

from rag.config import settings
from rag.llm import call_llm

logger = logging.getLogger("contextualize")

PROMPT = """Voici une section d'un document juridique :

<section>
{section}
</section>

Voici un extrait de cette section :

<extrait>
{chunk}
</extrait>

Écris une seule phrase courte qui situe cet extrait dans le document et
reformule son sujet avec les mots qu'un client utiliserait.
Maximum 200 caractères. Réponds uniquement par cette phrase, sans préambule.
"""


def build_sections(chunks):
    """Group chunk texts by section title."""
    sections = {}
    for chunk in chunks:
        sections.setdefault(chunk["section"], []).append(chunk["text"])
    return {title: "\n".join(texts) for title, texts in sections.items()}


def main():
    """Generate a context sentence for every chunk and store the result."""
    with settings.chunks_file.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    sections = build_sections(chunks)
    logger.info("%s chunks across %s sections", len(chunks), len(sections))

    truncated = 0

    for i, chunk in enumerate(chunks, start=1):
        prompt = PROMPT.format(section=sections[chunk["section"]], chunk=chunk["text"])
        context = call_llm(prompt)

        if len(context) > settings.max_context_chars:
            truncated += 1
            context = context[: settings.max_context_chars]

        chunk["context"] = context

        if i % 20 == 0:
            logger.info("%s/%s contextualized", i, len(chunks))

        time.sleep(settings.llm_delay_seconds)

    if truncated:
        logger.warning(
            "%s contexts exceeded %s chars and were truncated",
            truncated,
            settings.max_context_chars,
        )

    with settings.contextual_chunks_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("written to %s", settings.contextual_chunks_file)


if __name__ == "__main__":
    main()
