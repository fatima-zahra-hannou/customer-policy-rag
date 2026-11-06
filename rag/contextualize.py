"""Add a short context sentence to each chunk to improve retrieval."""

import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.config import settings

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


def get_client():
    """Create the Gemini client from the API key in the environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def call_llm(client, prompt):
    """Call the LLM, retrying with exponential backoff on failure."""
    for attempt in range(settings.llm_retries):
        try:
            response = client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=settings.temperature),
            )
            return response.text.strip()
        except Exception as exc:
            if attempt == settings.llm_retries - 1:
                raise
            wait = 2**attempt
            logger.warning("LLM call failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)
    return None


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

    client = get_client()
    truncated = 0

    for i, chunk in enumerate(chunks, start=1):
        prompt = PROMPT.format(section=sections[chunk["section"]], chunk=chunk["text"])
        context = call_llm(client, prompt)

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
